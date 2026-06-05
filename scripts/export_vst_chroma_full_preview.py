"""Export one VST-chroma candidate without full proxy-gate evaluation.

Use this for full-resolution previews where the proxy visual gate is too heavy.
It writes a normal display PNG and a small JSON summary with the payload
estimate.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
OUTPUT_DIR = ROOT / "outputs" / "previews" / "vst_chroma_nr_full"
sys.path.insert(0, str(ROOT / "scripts"))

from export_linear_index_preview_png import to_preview_u16, write_png  # noqa: E402
from probe_darkbits_router_payload import read_exr  # noqa: E402
from probe_vst_chroma_nr import encode_decode_case, safe_name  # noqa: E402


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
    parser.add_argument("--high-bits", type=int, default=5)
    parser.add_argument("--low-scale", type=int, default=2)
    parser.add_argument("--guide-radius", type=int, default=2)
    parser.add_argument("--guide-eps", type=float, default=0.1)
    parser.add_argument("--threshold-mult", type=float, default=2.5)
    parser.add_argument("--white", type=float, default=4.0)
    parser.add_argument("--gamma", type=float, default=2.2)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    path = DATA_DIR / args.input
    pixels = read_exr(path, args.crop_size)
    decoded, payload = encode_decode_case(
        pixels,
        args.vst_mode,
        args.vst_a,
        args.vst_b,
        args.vst_eps,
        args.y_bits,
        args.chroma_low_bits,
        args.high_bits,
        args.low_scale,
        args.guide_radius,
        args.guide_eps,
        args.threshold_mult,
    )
    label = (
        f"vstchroma_{args.vst_mode}_Y{args.y_bits}_CL{args.chroma_low_bits}"
        f"_H{args.high_bits}_s{args.low_scale}_r{args.guide_radius}"
        f"_ge{args.guide_eps:g}_tm{args.threshold_mult:g}"
    )
    crop_label = "full" if args.crop_size == 0 else f"crop{args.crop_size}"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = safe_name(path.stem)
    original_png = args.output_dir / f"{stem}_{crop_label}_original_w{args.white:g}_g{args.gamma:g}.png"
    decoded_png = (
        args.output_dir
        / f"{stem}_{crop_label}_{safe_name(label)}_w{args.white:g}_g{args.gamma:g}_decoded.png"
    )
    if not original_png.exists():
        write_png(original_png, to_preview_u16(pixels, args.white, args.gamma))
    write_png(decoded_png, to_preview_u16(decoded, args.white, args.gamma))

    summary = {
        "image": args.input,
        "shape": list(pixels.shape),
        "label": label,
        "raw_bytes": int(pixels.nbytes),
        "estimated_bytes": int(payload["estimated_bytes"]),
        "estimated_ratio": pixels.nbytes / float(payload["estimated_bytes"]),
        "payload": payload,
        "original_png": str(original_png),
        "decoded_png": str(decoded_png),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_json = RESULTS_DIR / (
        f"vst_chroma_full_preview_{safe_name(path.stem)}_{crop_label}_{safe_name(label)}.json"
    )
    output_json.write_text(json.dumps(summary, indent=2))
    print(
        f"{label} est={summary['estimated_bytes']:,} "
        f"ratio={summary['estimated_ratio']:.2f}x"
    )
    print(f"decoded_png={decoded_png}")
    print(f"summary={output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
