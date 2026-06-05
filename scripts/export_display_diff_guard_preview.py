"""Export a preview with a signed-log10 guard selected by display-space diff.

The dark-smooth mask catches broad shadow banding, but it missed a subtle
shadow/highlight boundary artifact around a user-provided point.  This script
compares the current VST-routed preview against the faithful signed-log10
anchor in the same display space and routes only suspicious pixels to the
anchor.

This is a diagnostic/oracle-side mask generator, not a final bitstream format.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import OpenImageIO as oiio

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
OUTPUT_DIR = ROOT / "outputs" / "previews" / "vst_chroma_dark_protect"
sys.path.insert(0, str(ROOT / "scripts"))

from export_linear_index_preview_png import write_png  # noqa: E402
from probe_darkbits_router_payload import summarize_binary_mask  # noqa: E402
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


def display_luma_u16(pixels: np.ndarray) -> np.ndarray:
    rgb = pixels[:, :, :3].astype(np.float64) / 65535.0
    return 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]


def dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.astype(bool, copy=True)
    return box_mean_2d(mask.astype(np.float64), radius) > 0.0


def write_mask_png(path: Path, mask: np.ndarray) -> None:
    image = np.repeat(mask[:, :, None], 3, axis=2).astype(np.uint16) * np.uint16(65535)
    write_png(path, image)


def crop_around(image: np.ndarray, x: int, y: int, radius: int) -> np.ndarray:
    height, width = image.shape[:2]
    x0 = max(0, int(x) - int(radius))
    x1 = min(width, int(x) + int(radius))
    y0 = max(0, int(y) - int(radius))
    y1 = min(height, int(y) + int(radius))
    return np.ascontiguousarray(image[y0:y1, x0:x1])


def summarize_focus(mask: np.ndarray, luma_diff: np.ndarray, max_rgb_diff: np.ndarray, x: int, y: int) -> dict:
    height, width = mask.shape
    if not (0 <= x < width and 0 <= y < height):
        return {"inside_image": False}
    rows: dict[str, object] = {
        "inside_image": True,
        "x": x,
        "y": y,
        "mask_at_point": bool(mask[y, x]),
        "luma_diff_at_point": float(luma_diff[y, x]),
        "max_rgb_diff_at_point": float(max_rgb_diff[y, x]),
    }
    for radius in (8, 16, 32, 64):
        x0 = max(0, x - radius)
        x1 = min(width, x + radius + 1)
        y0 = max(0, y - radius)
        y1 = min(height, y + radius + 1)
        patch_mask = mask[y0:y1, x0:x1]
        patch_luma = luma_diff[y0:y1, x0:x1]
        patch_rgb = max_rgb_diff[y0:y1, x0:x1]
        rows[f"r{radius}_mask_rate"] = float(np.mean(patch_mask))
        rows[f"r{radius}_luma_diff_max"] = float(np.max(patch_luma))
        rows[f"r{radius}_max_rgb_diff_max"] = float(np.max(patch_rgb))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-png",
        type=Path,
        default=OUTPUT_DIR / "candidate_Y8_slog10_st0025.png",
    )
    parser.add_argument(
        "--safe-png",
        type=Path,
        default=ROOT / "outputs" / "previews" / "sample_DSCF0009_crop0_signed-log_bits10_w4_g2.2_decoded.png",
    )
    parser.add_argument("--luma-threshold", type=float, default=0.007)
    parser.add_argument("--rgb-threshold", type=float, default=0.0)
    parser.add_argument("--dilate-radius", type=int, default=2)
    parser.add_argument("--focus-x", type=int, default=-1)
    parser.add_argument("--focus-y", type=int, default=-1)
    parser.add_argument("--crop-radius", type=int, default=384)
    parser.add_argument("--crop-only", action="store_true")
    parser.add_argument("--label", default="candidate_Y8_slog10_displaydiff")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    base = read_u16_png(args.base_png)
    safe = read_u16_png(args.safe_png)
    if base.shape != safe.shape:
        raise RuntimeError(f"shape mismatch: base={base.shape} safe={safe.shape}")

    base_f = base.astype(np.float64) / 65535.0
    safe_f = safe.astype(np.float64) / 65535.0
    luma_diff = np.abs(display_luma_u16(base) - display_luma_u16(safe))
    max_rgb_diff = np.max(np.abs(base_f - safe_f), axis=2)

    raw_mask = luma_diff >= float(args.luma_threshold)
    if args.rgb_threshold > 0.0:
        raw_mask |= max_rgb_diff >= float(args.rgb_threshold)
    guard_mask = dilate_mask(raw_mask, args.dilate_radius)

    routed = np.array(base, copy=True)
    routed[guard_mask, :3] = safe[guard_mask, :3]

    mask_summary = summarize_binary_mask(guard_mask)
    mask_payload = min(
        mask_summary["order0"],
        mask_summary["west_north"],
        key=lambda row: row["total_bytes"],
    )

    label = safe_name(
        f"{args.label}_L{args.luma_threshold:g}"
        f"_R{args.rgb_threshold:g}_d{args.dilate_radius}"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    decoded_png = None if args.crop_only else args.output_dir / f"{label}.png"
    mask_png = args.output_dir / f"{label}_mask.png"
    if decoded_png is not None:
        write_png(decoded_png, routed)
    write_mask_png(mask_png, guard_mask)

    crop_png = None
    base_crop_png = None
    safe_crop_png = None
    if args.focus_x >= 0 and args.focus_y >= 0:
        crop_png = args.output_dir / f"{label}_crop_x{args.focus_x}_y{args.focus_y}.png"
        base_crop_png = args.output_dir / f"{label}_base_crop_x{args.focus_x}_y{args.focus_y}.png"
        safe_crop_png = args.output_dir / f"{label}_safe_crop_x{args.focus_x}_y{args.focus_y}.png"
        write_png(crop_png, crop_around(routed, args.focus_x, args.focus_y, args.crop_radius))
        write_png(base_crop_png, crop_around(base, args.focus_x, args.focus_y, args.crop_radius))
        write_png(safe_crop_png, crop_around(safe, args.focus_x, args.focus_y, args.crop_radius))

    quantiles = {}
    for q in (0.90, 0.95, 0.97, 0.98, 0.99, 0.995, 0.999):
        quantiles[f"luma_diff_q{q:g}"] = float(np.quantile(luma_diff.reshape(-1), q))
        quantiles[f"max_rgb_diff_q{q:g}"] = float(np.quantile(max_rgb_diff.reshape(-1), q))

    summary = {
        "label": label,
        "shape": list(base.shape),
        "luma_threshold": args.luma_threshold,
        "rgb_threshold": args.rgb_threshold,
        "dilate_radius": args.dilate_radius,
        "raw_mask_rate": float(np.mean(raw_mask)),
        "guard_mask_rate": float(np.mean(guard_mask)),
        "mask_payload": mask_payload,
        "diff_quantiles": quantiles,
        "focus": (
            None
            if args.focus_x < 0 or args.focus_y < 0
            else summarize_focus(guard_mask, luma_diff, max_rgb_diff, args.focus_x, args.focus_y)
        ),
        "base_png": str(args.base_png),
        "safe_png": str(args.safe_png),
        "decoded_png": None if decoded_png is None else str(decoded_png),
        "mask_png": str(mask_png),
        "crop_png": None if crop_png is None else str(crop_png),
        "base_crop_png": None if base_crop_png is None else str(base_crop_png),
        "safe_crop_png": None if safe_crop_png is None else str(safe_crop_png),
        "note": (
            "Display-space diff guard diagnostic. Mask cost is for the guard "
            "bitmap alone and does not include signed-log payload replacement."
        ),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_json = RESULTS_DIR / f"display_diff_guard_{label}.json"
    output_json.write_text(json.dumps(summary, indent=2))
    print(
        f"{label} raw={summary['raw_mask_rate']:.2%} "
        f"guard={summary['guard_mask_rate']:.2%} "
        f"mask_bytes={int(mask_payload['total_bytes']):,}"
    )
    if summary["focus"] is not None:
        focus = summary["focus"]
        print(
            f"focus=({args.focus_x},{args.focus_y}) "
            f"hit={focus['mask_at_point']} "
            f"L={focus['luma_diff_at_point']:.6f} "
            f"RGB={focus['max_rgb_diff_at_point']:.6f}"
        )
    if decoded_png is not None:
        print(f"decoded_png={decoded_png}")
    print(f"mask_png={mask_png}")
    if crop_png is not None:
        print(f"crop_png={crop_png}")
        print(f"base_crop_png={base_crop_png}")
        print(f"safe_crop_png={safe_crop_png}")
    print(f"summary={output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
