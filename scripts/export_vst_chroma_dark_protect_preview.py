"""Export a full VST-chroma preview with dark smooth regions protected.

This is a visual routing probe for the denoised profile.  It keeps the
aggressive VST-chroma candidate in most of the image, but replaces dark smooth
pixels with a safer candidate to reduce mottled shadow gradients.

The size estimate is intentionally conservative/simple: it reports the base and
safe candidate payloads plus mask cost, but does not claim a final bitstream
size for the routed hybrid.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
OUTPUT_DIR = ROOT / "outputs" / "previews" / "vst_chroma_dark_protect"
sys.path.insert(0, str(ROOT / "scripts"))

from export_linear_index_preview_png import to_preview_u16, write_png  # noqa: E402
from probe_adaptive_ycocg_router import build_mask  # noqa: E402
from probe_darkbits_router_payload import read_exr, summarize_binary_mask  # noqa: E402
from probe_vst_chroma_nr import encode_decode_case, safe_name  # noqa: E402


def run_candidate(pixels: np.ndarray, args: argparse.Namespace, high_bits: int, threshold_mult: float):
    return encode_decode_case(
        pixels,
        args.vst_mode,
        args.vst_a,
        args.vst_b,
        args.vst_eps,
        args.y_bits,
        args.chroma_low_bits,
        high_bits,
        args.low_scale,
        args.guide_radius,
        args.guide_eps,
        threshold_mult,
    )


def write_mask_png(path: Path, mask: np.ndarray) -> None:
    image = np.repeat(mask[:, :, None], 3, axis=2).astype(np.uint16) * np.uint16(65535)
    write_png(path, image)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="sample_DSCF0009.EXR")
    parser.add_argument("--crop-size", type=int, default=0)
    parser.add_argument("--vst-mode", default="gamma075")
    parser.add_argument("--vst-a", type=float, default=1.0)
    parser.add_argument("--vst-b", type=float, default=0.01)
    parser.add_argument("--vst-eps", type=float, default=1.0e-4)
    parser.add_argument("--y-bits", type=int, default=10)
    parser.add_argument("--chroma-low-bits", type=int, default=8)
    parser.add_argument("--low-scale", type=int, default=2)
    parser.add_argument("--guide-radius", type=int, default=2)
    parser.add_argument("--guide-eps", type=float, default=0.1)
    parser.add_argument("--base-high-bits", type=int, default=5)
    parser.add_argument("--base-threshold", type=float, default=2.5)
    parser.add_argument("--safe-high-bits", type=int, default=6)
    parser.add_argument("--safe-threshold", type=float, default=1.75)
    parser.add_argument(
        "--safe-source",
        choices=("candidate", "original"),
        default="candidate",
        help="Use a safer candidate or original pixels inside the mask.",
    )
    parser.add_argument("--mask-mode", default="dark-smooth")
    parser.add_argument("--dark-max", type=float, default=0.25)
    parser.add_argument("--mask-radius", type=int, default=2)
    parser.add_argument("--smooth-threshold", type=float, default=0.002)
    parser.add_argument("--white", type=float, default=4.0)
    parser.add_argument("--lift-white", type=float, default=0.0)
    parser.add_argument("--gamma", type=float, default=2.2)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    path = DATA_DIR / args.input
    pixels = read_exr(path, args.crop_size)
    base_decoded, base_payload = run_candidate(
        pixels,
        args,
        args.base_high_bits,
        args.base_threshold,
    )
    if args.safe_source == "candidate":
        safe_decoded, safe_payload = run_candidate(
            pixels,
            args,
            args.safe_high_bits,
            args.safe_threshold,
        )
    else:
        safe_decoded = pixels
        safe_payload = {
            "estimated_bytes": 0,
            "estimated_ratio": 0.0,
            "note": "original pixels used for oracle visual routing only",
        }
    mask = build_mask(
        pixels,
        args.mask_mode,
        args.dark_max,
        args.mask_radius,
        args.smooth_threshold,
    )
    routed = np.array(base_decoded, copy=True)
    routed[mask, :3] = safe_decoded[mask, :3]

    mask_summary = summarize_binary_mask(mask)
    mask_payload = min(
        mask_summary["order0"],
        mask_summary["west_north"],
        key=lambda row: row["total_bytes"],
    )

    base_label = (
        f"H{args.base_high_bits}_tm{args.base_threshold:g}"
    )
    safe_label = (
        "original"
        if args.safe_source == "original"
        else f"H{args.safe_high_bits}_tm{args.safe_threshold:g}"
    )
    mask_label = (
        f"{args.mask_mode}{args.dark_max:g}"
        f"_r{args.mask_radius}_st{args.smooth_threshold:g}"
    )
    label = (
        f"vstchroma_darkprotect_{args.vst_mode}_Y{args.y_bits}"
        f"_CL{args.chroma_low_bits}_base{base_label}_safe{safe_label}"
        f"_{mask_label}"
    )
    crop_label = "full" if args.crop_size == 0 else f"crop{args.crop_size}"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = safe_name(path.stem)
    original_png = args.output_dir / f"{stem}_{crop_label}_original_w{args.white:g}_g{args.gamma:g}.png"
    routed_png = args.output_dir / f"{stem}_{crop_label}_{safe_name(label)}_w{args.white:g}_g{args.gamma:g}_decoded.png"
    mask_png = args.output_dir / f"{stem}_{crop_label}_{safe_name(label)}_mask.png"
    if not original_png.exists():
        write_png(original_png, to_preview_u16(pixels, args.white, args.gamma))
    write_png(routed_png, to_preview_u16(routed, args.white, args.gamma))
    write_mask_png(mask_png, mask)
    lift_original_png = None
    lift_routed_png = None
    if args.lift_white > 0.0:
        lift_original_png = (
            args.output_dir / f"{stem}_{crop_label}_original_w{args.lift_white:g}_g{args.gamma:g}.png"
        )
        lift_routed_png = (
            args.output_dir
            / f"{stem}_{crop_label}_{safe_name(label)}_w{args.lift_white:g}_g{args.gamma:g}_decoded.png"
        )
        if not lift_original_png.exists():
            write_png(lift_original_png, to_preview_u16(pixels, args.lift_white, args.gamma))
        write_png(lift_routed_png, to_preview_u16(routed, args.lift_white, args.gamma))

    summary = {
        "image": args.input,
        "shape": list(pixels.shape),
        "label": label,
        "raw_bytes": int(pixels.nbytes),
        "base_estimated_bytes": int(base_payload["estimated_bytes"]),
        "base_estimated_ratio": pixels.nbytes / float(base_payload["estimated_bytes"]),
        "safe_source": args.safe_source,
        "safe_estimated_bytes": int(safe_payload["estimated_bytes"]),
        "safe_estimated_ratio": (
            None
            if args.safe_source == "original"
            else pixels.nbytes / float(safe_payload["estimated_bytes"])
        ),
        "mask_pixel_rate": float(np.mean(mask)),
        "mask_payload": mask_payload,
        "note": (
            "Routed byte size is not final. This preview replaces dark smooth "
            "pixels with the safer decoded candidate for visual inspection."
        ),
        "original_png": str(original_png),
        "decoded_png": str(routed_png),
        "mask_png": str(mask_png),
        "lift_original_png": None if lift_original_png is None else str(lift_original_png),
        "lift_decoded_png": None if lift_routed_png is None else str(lift_routed_png),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_json = RESULTS_DIR / (
        f"vst_chroma_dark_protect_{safe_name(path.stem)}_{crop_label}_{safe_name(label)}.json"
    )
    output_json.write_text(json.dumps(summary, indent=2))
    print(
        f"{label} base={summary['base_estimated_bytes']:,} "
        f"({summary['base_estimated_ratio']:.2f}x) "
        f"safe={summary['safe_estimated_bytes']:,} "
        f"({summary['safe_estimated_ratio'] if summary['safe_estimated_ratio'] is not None else 'oracle'}"
        f"{'x' if summary['safe_estimated_ratio'] is not None else ''}) "
        f"mask={summary['mask_pixel_rate']:.2%}"
    )
    print(f"decoded_png={routed_png}")
    if lift_routed_png is not None:
        print(f"lift_decoded_png={lift_routed_png}")
    print(f"summary={output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
