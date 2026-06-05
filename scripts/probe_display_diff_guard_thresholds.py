"""Sweep display-diff guard thresholds without exporting full previews."""
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


def focus_stats(mask: np.ndarray, luma_diff: np.ndarray, max_rgb_diff: np.ndarray, x: int, y: int) -> dict:
    height, width = mask.shape
    if not (0 <= x < width and 0 <= y < height):
        return {"inside_image": False}
    row: dict[str, object] = {
        "inside_image": True,
        "hit": bool(mask[y, x]),
        "luma_diff": float(luma_diff[y, x]),
        "max_rgb_diff": float(max_rgb_diff[y, x]),
    }
    for radius in (8, 16, 32, 64):
        x0 = max(0, x - radius)
        x1 = min(width, x + radius + 1)
        y0 = max(0, y - radius)
        y1 = min(height, y + radius + 1)
        patch = mask[y0:y1, x0:x1]
        row[f"r{radius}_mask_rate"] = float(np.mean(patch))
    return row


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
    parser.add_argument(
        "--thresholds",
        default="0.0065,0.007,0.00725,0.0075,0.008,0.0085,0.009,0.01,0.011,0.012",
    )
    parser.add_argument("--dilate-radius", type=int, default=0)
    parser.add_argument("--focus-x", type=int, default=2361)
    parser.add_argument("--focus-y", type=int, default=3811)
    parser.add_argument("--label", default="display_diff_guard_thresholds")
    args = parser.parse_args()

    base = read_u16_png(args.base_png)
    safe = read_u16_png(args.safe_png)
    if base.shape != safe.shape:
        raise RuntimeError(f"shape mismatch: base={base.shape} safe={safe.shape}")

    base_f = base.astype(np.float64) / 65535.0
    safe_f = safe.astype(np.float64) / 65535.0
    luma_diff = np.abs(display_luma_u16(base) - display_luma_u16(safe))
    max_rgb_diff = np.max(np.abs(base_f - safe_f), axis=2)
    thresholds = [float(item) for item in args.thresholds.split(",") if item.strip()]
    rows = []
    for threshold in thresholds:
        raw = luma_diff >= threshold
        mask = dilate_mask(raw, args.dilate_radius)
        mask_summary = summarize_binary_mask(mask)
        mask_payload = min(
            mask_summary["order0"],
            mask_summary["west_north"],
            key=lambda row: row["total_bytes"],
        )
        focus = focus_stats(mask, luma_diff, max_rgb_diff, args.focus_x, args.focus_y)
        rows.append({
            "luma_threshold": threshold,
            "dilate_radius": args.dilate_radius,
            "mask_rate": float(np.mean(mask)),
            "mask_bytes": int(mask_payload["total_bytes"]),
            "focus": focus,
        })

    summary = {
        "label": args.label,
        "shape": list(base.shape),
        "base_png": str(args.base_png),
        "safe_png": str(args.safe_png),
        "focus_x": args.focus_x,
        "focus_y": args.focus_y,
        "rows": rows,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = RESULTS_DIR / f"{safe_name(args.label)}.json"
    output.write_text(json.dumps(summary, indent=2))
    for row in rows:
        focus = row["focus"]
        print(
            f"L={row['luma_threshold']:.5f} "
            f"mask={row['mask_rate']:.2%} "
            f"bytes={row['mask_bytes']:,} "
            f"hit={focus['hit']} "
            f"r8={focus['r8_mask_rate']:.2%} "
            f"r32={focus['r32_mask_rate']:.2%}"
        )
    print(f"summary={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
