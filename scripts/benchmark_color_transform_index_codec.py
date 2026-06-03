"""Benchmark color-decorrelated transform-index through the implemented codec.

This is still a research harness: color decorrelation is applied in Python,
StageLinearIndex encodes the decorrelated planes, then Python converts decoded
planes back to RGB for quality metrics.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "codec" / "python"))

import radiance_codec  # noqa: E402
from benchmark_linear_index_codec import error_stats  # noqa: E402
from probe_color_tile_transform_index import (  # noqa: E402
    DATA_DIR,
    RESULTS_DIR,
    forward_color,
    inverse_color,
    parse_csv,
    parse_ints,
    read_exr,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", default="sample_DSCF0009.EXR")
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--bits", default="7,8")
    parser.add_argument("--color-modes", default="rgb,g-diff,ycocg")
    parser.add_argument("--transforms", default="gamma075,asinh")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    bits_values = parse_ints(args.bits)
    color_modes = parse_csv(args.color_modes)
    transforms = parse_csv(args.transforms)
    results = []
    for path in sorted(DATA_DIR.glob(args.glob)):
        pixels = read_exr(path, args.crop_size)
        rows = []
        print(path.name)
        for color in color_modes:
            planes = forward_color(pixels, color)
            for transform in transforms:
                for bits in bits_values:
                    t0 = time.perf_counter()
                    encoded = radiance_codec.encode_linear_index_near_lossless(
                        planes,
                        bits=bits,
                        effort=9,
                        transform=transform,
                    )
                    t1 = time.perf_counter()
                    decoded_planes = radiance_codec.decode(encoded, planes.shape)
                    recon_rgb = inverse_color(decoded_planes, color)
                    t2 = time.perf_counter()
                    row = {
                        "color": color,
                        "transform": transform,
                        "bits": bits,
                        "encoded_bytes": len(encoded),
                        "ratio_vs_original": pixels.nbytes / len(encoded),
                        "encode_seconds": t1 - t0,
                        "decode_seconds": t2 - t1,
                        **error_stats(pixels, recon_rgb),
                    }
                    rows.append(row)
                    print(
                        f"  {color:<6} {transform:<10} bits{bits}: "
                        f"{row['ratio_vs_original']:6.2f}x "
                        f"bytes={row['encoded_bytes']} "
                        f"enc={row['encode_seconds']:.2f}s "
                        f"dec={row['decode_seconds']:.2f}s "
                        f"log={row['signed_log2_rmse']:.3e} "
                        f"p99={row['signed_log2_p99']:.3e} "
                        f"grad={row['gradient_signed_log2_nrmse']:.3e}")
        results.append({
            "image": path.name,
            "shape": list(pixels.shape),
            "crop_size": args.crop_size,
            "rows": rows,
        })

    if not args.no_save:
        safe_glob = args.glob.replace("*", "star").replace(".", "_")
        output = RESULTS_DIR / (
            f"color_transform_index_codec_{safe_glob}"
            f"_crop{args.crop_size}_bits{args.bits.replace(',', '-')}.json"
        )
        output.write_text(json.dumps(results, indent=2))
        print(f"\nSaved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
