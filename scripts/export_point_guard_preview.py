"""Export a diagnostic preview with a manual signed-log10 guard around a point."""
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


def circular_mask(shape: tuple[int, int], x: int, y: int, radius: int) -> np.ndarray:
    height, width = shape
    yy, xx = np.indices((height, width), dtype=np.int32)
    return (xx - int(x)) * (xx - int(x)) + (yy - int(y)) * (yy - int(y)) <= int(radius) ** 2


def write_mask_png(path: Path, mask: np.ndarray) -> None:
    image = np.repeat(mask[:, :, None], 3, axis=2).astype(np.uint16) * np.uint16(65535)
    write_png(path, image)


def crop_around(image: np.ndarray, x: int, y: int, radius: int) -> np.ndarray:
    h, w = image.shape[:2]
    x0 = max(0, int(x) - int(radius))
    x1 = min(w, int(x) + int(radius))
    y0 = max(0, int(y) - int(radius))
    y1 = min(h, int(y) + int(radius))
    return np.ascontiguousarray(image[y0:y1, x0:x1])


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
    parser.add_argument("--x", type=int, required=True)
    parser.add_argument("--y", type=int, required=True)
    parser.add_argument("--radius", type=int, default=192)
    parser.add_argument("--crop-radius", type=int, default=384)
    parser.add_argument("--label", default="candidate_point_guard")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    base = read_u16_png(args.base_png)
    safe = read_u16_png(args.safe_png)
    if base.shape != safe.shape:
        raise RuntimeError(f"shape mismatch: base={base.shape} safe={safe.shape}")

    mask = circular_mask(base.shape[:2], args.x, args.y, args.radius)
    routed = np.array(base, copy=True)
    routed[mask, :3] = safe[mask, :3]

    mask_summary = summarize_binary_mask(mask)
    mask_payload = min(
        mask_summary["order0"],
        mask_summary["west_north"],
        key=lambda row: row["total_bytes"],
    )

    label = safe_name(f"{args.label}_x{args.x}_y{args.y}_r{args.radius}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    decoded_png = args.output_dir / f"{label}.png"
    mask_png = args.output_dir / f"{label}_mask.png"
    crop_png = args.output_dir / f"{label}_crop.png"
    safe_crop_png = args.output_dir / f"{label}_safe_crop.png"
    base_crop_png = args.output_dir / f"{label}_base_crop.png"
    write_png(decoded_png, routed)
    write_mask_png(mask_png, mask)
    write_png(crop_png, crop_around(routed, args.x, args.y, args.crop_radius))
    write_png(safe_crop_png, crop_around(safe, args.x, args.y, args.crop_radius))
    write_png(base_crop_png, crop_around(base, args.x, args.y, args.crop_radius))

    summary = {
        "label": label,
        "shape": list(base.shape),
        "x": args.x,
        "y": args.y,
        "radius": args.radius,
        "mask_pixel_rate": float(np.mean(mask)),
        "mask_payload": mask_payload,
        "base_png": str(args.base_png),
        "safe_png": str(args.safe_png),
        "decoded_png": str(decoded_png),
        "mask_png": str(mask_png),
        "crop_png": str(crop_png),
        "safe_crop_png": str(safe_crop_png),
        "base_crop_png": str(base_crop_png),
        "note": "Manual diagnostic guard around a user-provided point; not a final bitstream route.",
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_json = RESULTS_DIR / f"point_guard_{label}.json"
    output_json.write_text(json.dumps(summary, indent=2))
    print(
        f"{label} mask={summary['mask_pixel_rate']:.2%} "
        f"mask_bytes={int(mask_payload['total_bytes']):,}"
    )
    print(f"decoded_png={decoded_png}")
    print(f"crop_png={crop_png}")
    print(f"base_crop_png={base_crop_png}")
    print(f"safe_crop_png={safe_crop_png}")
    print(f"summary={output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
