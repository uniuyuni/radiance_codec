"""Export previews with an added signed-log10 edge guard mask.

This diagnoses edge artifacts seen in the VST base outside the dark-smooth
risk mask.  The preview starts from an already-rendered candidate PNG and
replaces edge pixels with the faithful signed-log10 preview.
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
from probe_log_signal_noise_resynth import box_mean_2d  # noqa: E402
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


def luma_linear(pixels: np.ndarray) -> np.ndarray:
    rgb = pixels[:, :, :3].astype(np.float64)
    return 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]


def dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.astype(bool, copy=True)
    mean = box_mean_2d(mask.astype(np.float64), radius)
    return mean > 0.0


def edge_mask_from_luma(pixels: np.ndarray, quantile: float, dilate_radius: int) -> np.ndarray:
    y = np.log2(1.0 + np.maximum(luma_linear(pixels), 0.0))
    gx = np.zeros(y.shape, dtype=np.float64)
    gy = np.zeros(y.shape, dtype=np.float64)
    gx[:, 1:] = np.abs(y[:, 1:] - y[:, :-1])
    gy[1:, :] = np.abs(y[1:, :] - y[:-1, :])
    grad = np.maximum(gx, gy)
    threshold = float(np.quantile(grad.reshape(-1), quantile))
    edge = grad >= threshold
    return dilate_mask(edge, dilate_radius)


def write_mask_png(path: Path, mask: np.ndarray) -> None:
    image = np.repeat(mask[:, :, None], 3, axis=2).astype(np.uint16) * np.uint16(65535)
    write_png(path, image)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="sample_DSCF0009.EXR")
    parser.add_argument("--crop-size", type=int, default=0)
    parser.add_argument(
        "--base-candidate-png",
        type=Path,
        default=OUTPUT_DIR / "candidate_Y8_slog10_st0025.png",
    )
    parser.add_argument(
        "--safe-png",
        type=Path,
        default=ROOT / "outputs" / "previews" / "sample_DSCF0009_crop0_signed-log_bits10_w4_g2.2_decoded.png",
    )
    parser.add_argument("--dark-max", type=float, default=0.5)
    parser.add_argument("--smooth-threshold", type=float, default=0.0025)
    parser.add_argument("--edge-quantile", type=float, default=0.98)
    parser.add_argument("--dilate-radius", type=int, default=1)
    parser.add_argument("--label", default="candidate_Y8_slog10_edge")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    path = DATA_DIR / args.input
    pixels = read_exr(path, args.crop_size)
    base = read_u16_png(args.base_candidate_png)
    safe = read_u16_png(args.safe_png)
    if base.shape[:2] != pixels.shape[:2] or safe.shape[:2] != pixels.shape[:2]:
        raise RuntimeError(
            f"shape mismatch: exr={pixels.shape[:2]} base={base.shape[:2]} safe={safe.shape[:2]}"
        )

    dark_mask = build_mask(
        pixels,
        "dark-smooth",
        args.dark_max,
        2,
        args.smooth_threshold,
    )
    edge_mask = edge_mask_from_luma(pixels, args.edge_quantile, args.dilate_radius)
    # The base candidate already contains the dark-smooth guard.  Measure and
    # apply only additional edge pixels for this diagnostic.
    extra_edge = edge_mask & ~dark_mask
    routed = np.array(base, copy=True)
    routed[extra_edge, :3] = safe[extra_edge, :3]

    mask_summary = summarize_binary_mask(extra_edge)
    mask_payload = min(
        mask_summary["order0"],
        mask_summary["west_north"],
        key=lambda row: row["total_bytes"],
    )
    crop_label = "full" if args.crop_size == 0 else f"crop{args.crop_size}"
    label = safe_name(
        f"{args.label}_q{args.edge_quantile:g}_d{args.dilate_radius}"
        f"_st{args.smooth_threshold:g}"
    )
    stem = safe_name(path.stem)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    decoded_png = args.output_dir / f"{stem}_{crop_label}_{label}.png"
    mask_png = args.output_dir / f"{stem}_{crop_label}_{label}_mask.png"
    write_png(decoded_png, routed)
    write_mask_png(mask_png, extra_edge)
    summary = {
        "image": args.input,
        "shape": list(pixels.shape),
        "label": label,
        "dark_mask_rate": float(np.mean(dark_mask)),
        "edge_mask_rate": float(np.mean(edge_mask)),
        "extra_edge_rate": float(np.mean(extra_edge)),
        "mask_payload": mask_payload,
        "decoded_png": str(decoded_png),
        "mask_png": str(mask_png),
        "base_candidate_png": str(args.base_candidate_png),
        "safe_png": str(args.safe_png),
        "note": "Display-space edge guard diagnostic; not a final bitstream estimate.",
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_json = RESULTS_DIR / f"edge_guard_{stem}_{crop_label}_{label}.json"
    output_json.write_text(json.dumps(summary, indent=2))
    print(
        f"{label} edge={summary['edge_mask_rate']:.2%} "
        f"extra={summary['extra_edge_rate']:.2%} "
        f"mask_bytes={int(mask_payload['total_bytes']):,}"
    )
    print(f"decoded_png={decoded_png}")
    print(f"mask_png={mask_png}")
    print(f"summary={output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
