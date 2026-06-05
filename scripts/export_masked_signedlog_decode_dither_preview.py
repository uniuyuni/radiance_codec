"""Export VST-base plus masked signed-log decode-dither previews.

The dither here is decoder-side: the saved signed-log index is unchanged, but
the reconstructed transform value is offset deterministically inside the
quantization bin.  This tests whether bits9 can hide shadow banding/mottling
without paying bits10 escape cost.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import OpenImageIO as oiio

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
OUTPUT_DIR = ROOT / "outputs" / "previews" / "vst_chroma_dark_protect"
sys.path.insert(0, str(ROOT / "scripts"))

from export_linear_index_preview_png import to_preview_u16, write_png  # noqa: E402
from probe_adaptive_ycocg_router import build_mask  # noqa: E402
from probe_darkbits_router_payload import read_exr, summarize_binary_mask  # noqa: E402
from probe_dithered_linear_index import inverse_signed_log, signed_log  # noqa: E402
from probe_vst_chroma_nr import safe_name  # noqa: E402


def read_u16_png(path: Path) -> np.ndarray:
    image_input = oiio.ImageInput.open(str(path))
    if image_input is None:
        raise RuntimeError(f"can't open {path}: {oiio.geterror()}")
    spec = image_input.spec()
    pixels = image_input.read_image(format=oiio.UINT16)
    image_input.close()
    if pixels is None:
        raise RuntimeError(f"can't read {path}: {oiio.geterror()}")
    return np.ascontiguousarray(
        np.asarray(pixels, dtype=np.uint16).reshape(
            spec.height,
            spec.width,
            spec.nchannels,
        )[:, :, :3]
    )


def hash_noise(shape: tuple[int, int], channel: int, seed: int) -> np.ndarray:
    yy, xx = np.indices(shape, dtype=np.uint32)
    with np.errstate(over="ignore"):
        h = xx * np.uint32(374761393)
        h ^= yy * np.uint32(668265263)
        h ^= np.uint32(channel + 1) * np.uint32(2246822519)
        h ^= np.uint32(seed) * np.uint32(3266489917)
        h ^= h >> np.uint32(13)
        h *= np.uint32(1274126177)
        h ^= h >> np.uint32(16)
    return h.astype(np.float64) / float(2**32) - 0.5


def write_mask_png(path: Path, mask: np.ndarray) -> None:
    image = np.repeat(mask[:, :, None], 3, axis=2).astype(np.uint16) * np.uint16(65535)
    write_png(path, image)


def reconstruct_signedlog_decode_dither(
    pixels: np.ndarray,
    bits: int,
    amplitude: float,
    seed: int,
) -> np.ndarray:
    levels = (1 << bits) - 1
    decoded = np.array(pixels[:, :, :3], copy=True)
    for channel in range(decoded.shape[2]):
        transformed = signed_log(pixels[:, :, channel])
        range_values = transformed.astype(np.float32).astype(np.float64)
        lo = float(np.min(range_values))
        hi = float(np.max(range_values))
        if not hi > lo:
            decoded[:, :, channel] = inverse_signed_log(np.asarray(lo)).astype(np.float32)
            continue
        q = np.rint((transformed - lo) / (hi - lo) * levels)
        q = np.clip(q, 0.0, float(levels))
        offset = hash_noise(transformed.shape, channel, seed) * float(amplitude)
        # Keep endpoints exact enough to avoid injecting sign/range artifacts at
        # hard clamps.  Interior bins get sub-step deterministic jitter.
        offset = np.where((q <= 0.0) | (q >= float(levels)), 0.0, offset)
        rec_t = lo + (q + offset) * (hi - lo) / levels
        decoded[:, :, channel] = inverse_signed_log(rec_t).astype(np.float32)
    out = np.array(pixels, copy=True)
    out[:, :, :3] = decoded
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="sample_DSCF0009.EXR")
    parser.add_argument("--crop-size", type=int, default=0)
    parser.add_argument(
        "--base-png",
        type=Path,
        default=(
            ROOT
            / "outputs"
            / "previews"
            / "vst_chroma_nr"
            / "sample_DSCF0009_full_vstchroma_gamma075_Y10_CL8_H5_s2_r2_ge0_1_tm2_5_w4_g2.2_decoded.png"
        ),
    )
    parser.add_argument("--bits", type=int, default=9)
    parser.add_argument("--amplitude", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260605)
    parser.add_argument("--mask-mode", default="dark-smooth")
    parser.add_argument("--dark-max", type=float, default=0.5)
    parser.add_argument("--mask-radius", type=int, default=2)
    parser.add_argument("--smooth-threshold", type=float, default=0.004)
    parser.add_argument("--white", type=float, default=4.0)
    parser.add_argument("--gamma", type=float, default=2.2)
    parser.add_argument("--label", default="candidate_slog9_dither")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    path = DATA_DIR / args.input
    pixels = read_exr(path, args.crop_size)
    base = read_u16_png(args.base_png)
    if base.shape[:2] != pixels.shape[:2]:
        raise RuntimeError(f"shape mismatch: exr={pixels.shape[:2]} base={base.shape[:2]}")
    mask = build_mask(
        pixels,
        args.mask_mode,
        args.dark_max,
        args.mask_radius,
        args.smooth_threshold,
    )
    dithered = reconstruct_signedlog_decode_dither(
        pixels,
        args.bits,
        args.amplitude,
        args.seed,
    )
    dithered_png = to_preview_u16(dithered, args.white, args.gamma)
    routed = np.array(base, copy=True)
    routed[mask, :3] = dithered_png[mask, :3]

    crop_label = "full" if args.crop_size == 0 else f"crop{args.crop_size}"
    mask_label = (
        f"{args.mask_mode}{args.dark_max:g}"
        f"_r{args.mask_radius}_st{args.smooth_threshold:g}"
    )
    label = safe_name(
        f"{args.label}_bits{args.bits}_amp{args.amplitude:g}_{mask_label}"
    )
    stem = safe_name(path.stem)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    decoded_png = args.output_dir / f"{stem}_{crop_label}_{label}.png"
    mask_png = args.output_dir / f"{stem}_{crop_label}_{label}_mask.png"
    write_png(decoded_png, routed)
    write_mask_png(mask_png, mask)

    mask_summary = summarize_binary_mask(mask)
    mask_payload = min(
        mask_summary["order0"],
        mask_summary["west_north"],
        key=lambda row: row["total_bytes"],
    )
    summary = {
        "image": args.input,
        "shape": list(pixels.shape),
        "label": label,
        "bits": args.bits,
        "amplitude": args.amplitude,
        "seed": args.seed,
        "mask_mode": args.mask_mode,
        "dark_max": args.dark_max,
        "mask_radius": args.mask_radius,
        "smooth_threshold": args.smooth_threshold,
        "mask_pixel_rate": float(np.mean(mask)),
        "mask_payload": mask_payload,
        "base_png": str(args.base_png),
        "decoded_png": str(decoded_png),
        "mask_png": str(mask_png),
        "note": "Decoder-side signed-log dither preview. Dither itself adds no encoded bits.",
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_json = RESULTS_DIR / f"masked_signedlog_decode_dither_{stem}_{crop_label}_{label}.json"
    output_json.write_text(json.dumps(summary, indent=2))
    print(
        f"{label} mask={summary['mask_pixel_rate']:.2%} "
        f"mask_bytes={int(mask_payload['total_bytes']):,}"
    )
    print(f"decoded_png={decoded_png}")
    print(f"mask_png={mask_png}")
    print(f"summary={output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
