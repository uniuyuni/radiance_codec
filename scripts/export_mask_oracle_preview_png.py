"""Compose display-space oracle previews for mask diagnostics.

This avoids rerunning the expensive VST-chroma full decode when only the
protection mask is changing.  Because the preview transfer is per-pixel,
choosing between already-rendered original/decoded PNG pixels is equivalent to
choosing before preview conversion for oracle visual checks.
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

from export_linear_index_preview_png import write_png  # noqa: E402
from probe_adaptive_ycocg_router import build_mask  # noqa: E402
from probe_darkbits_router_payload import read_exr, summarize_binary_mask  # noqa: E402
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
    image = np.asarray(pixels, dtype=np.uint16).reshape(
        spec.height,
        spec.width,
        spec.nchannels,
    )
    return np.ascontiguousarray(image[:, :, :3])


def write_mask_png(path: Path, mask: np.ndarray) -> None:
    image = np.repeat(mask[:, :, None], 3, axis=2).astype(np.uint16) * np.uint16(65535)
    write_png(path, image)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="sample_DSCF0009.EXR")
    parser.add_argument("--crop-size", type=int, default=0)
    parser.add_argument(
        "--original-png",
        type=Path,
        default=OUTPUT_DIR / "sample_DSCF0009_full_original_w4_g2.2.png",
    )
    parser.add_argument(
        "--decoded-png",
        type=Path,
        default=(
            ROOT
            / "outputs"
            / "previews"
            / "vst_chroma_nr"
            / "sample_DSCF0009_full_vstchroma_gamma075_Y10_CL8_H5_s2_r2_ge0_1_tm2_5_w4_g2.2_decoded.png"
        ),
    )
    parser.add_argument("--mask-mode", default="dark-smooth")
    parser.add_argument("--dark-max", type=float, default=0.5)
    parser.add_argument("--mask-radius", type=int, default=2)
    parser.add_argument("--smooth-threshold", type=float, default=0.01)
    parser.add_argument("--label", default="display_oracle")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    pixels = read_exr(DATA_DIR / args.input, args.crop_size)
    original = read_u16_png(args.original_png)
    decoded = read_u16_png(args.decoded_png)
    if original.shape[:2] != pixels.shape[:2] or decoded.shape[:2] != pixels.shape[:2]:
        raise RuntimeError(
            f"shape mismatch: exr={pixels.shape[:2]} original={original.shape[:2]} decoded={decoded.shape[:2]}"
        )

    mask = build_mask(
        pixels,
        args.mask_mode,
        args.dark_max,
        args.mask_radius,
        args.smooth_threshold,
    )
    routed = np.array(decoded, copy=True)
    routed[mask, :3] = original[mask, :3]

    crop_label = "full" if args.crop_size == 0 else f"crop{args.crop_size}"
    mask_label = (
        f"{args.mask_mode}{args.dark_max:g}"
        f"_r{args.mask_radius}_st{args.smooth_threshold:g}"
    )
    label = safe_name(f"{args.label}_{mask_label}")
    stem = safe_name((DATA_DIR / args.input).stem)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    decoded_png = args.output_dir / f"{stem}_{crop_label}_{label}_decoded.png"
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
        "mask_mode": args.mask_mode,
        "dark_max": args.dark_max,
        "mask_radius": args.mask_radius,
        "smooth_threshold": args.smooth_threshold,
        "mask_pixel_rate": float(np.mean(mask)),
        "mask_payload": mask_payload,
        "original_png": str(args.original_png),
        "base_decoded_png": str(args.decoded_png),
        "decoded_png": str(decoded_png),
        "mask_png": str(mask_png),
        "note": "Display-space oracle only. This is for mask diagnosis, not a codec size estimate.",
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_json = RESULTS_DIR / f"mask_oracle_{stem}_{crop_label}_{label}.json"
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
