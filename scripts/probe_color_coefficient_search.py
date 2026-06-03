"""Search simple color-decorrelation coefficients for transform-index.

The tested family is:

    P0 = G
    P1 = R - aG
    P2 = B - bG

After transform-index reconstruction, RGB is restored as:

    G = P0
    R = P1 + aG
    B = P2 + bG

This keeps the transform simple enough to implement later if it helps.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from probe_transform_index_quantization import (
    DATA_DIR,
    RESULTS_DIR,
    estimate_index_bits,
    parse_ints,
    parse_modes,
    quality_stats,
    quantize_transform,
    read_exr,
)


def parse_floats(text: str) -> tuple[float, ...]:
    return tuple(float(part) for part in text.split(",") if part.strip())


def forward_coeff(pixels: np.ndarray, a: float, b: float) -> np.ndarray:
    if pixels.shape[2] < 3:
        return np.ascontiguousarray(pixels)
    out = np.array(pixels, dtype=np.float32, copy=True)
    r = pixels[:, :, 0].astype(np.float64)
    g = pixels[:, :, 1].astype(np.float64)
    blue = pixels[:, :, 2].astype(np.float64)
    out[:, :, 0] = g.astype(np.float32)
    out[:, :, 1] = (r - a * g).astype(np.float32)
    out[:, :, 2] = (blue - b * g).astype(np.float32)
    return np.ascontiguousarray(out)


def inverse_coeff(planes: np.ndarray, a: float, b: float) -> np.ndarray:
    if planes.shape[2] < 3:
        return np.ascontiguousarray(planes)
    out = np.array(planes, dtype=np.float32, copy=True)
    g = planes[:, :, 0].astype(np.float64)
    r = planes[:, :, 1].astype(np.float64) + a * g
    blue = planes[:, :, 2].astype(np.float64) + b * g
    out[:, :, 0] = r.astype(np.float32)
    out[:, :, 1] = g.astype(np.float32)
    out[:, :, 2] = blue.astype(np.float32)
    return np.ascontiguousarray(out)


def summarize_image(
    path: Path,
    crop_size: int,
    bits_values: tuple[int, ...],
    modes: tuple[str, ...],
    coeffs: tuple[float, ...],
) -> dict:
    pixels = read_exr(path, crop_size)
    rows = []
    for bits in bits_values:
        for mode in modes:
            for a in coeffs:
                for b in coeffs:
                    planes = forward_coeff(pixels, a, b)
                    indices, recon_planes = quantize_transform(planes, bits, mode)
                    recon = inverse_coeff(recon_planes, a, b)
                    estimated_bits = estimate_index_bits(indices, bits)
                    rows.append({
                        "bits": bits,
                        "mode": mode,
                        "a": a,
                        "b": b,
                        "estimated_bytes": math.ceil(estimated_bits / 8.0),
                        "estimated_ratio_vs_original": (
                            pixels.nbytes * 8.0 / estimated_bits),
                        **quality_stats(pixels, recon),
                    })
    rows.sort(key=lambda row: (
        row["bits"],
        not (
            row["signed_log2_rmse"] <= 0.004
            and row["signed_log2_p99"] <= 0.018
            and row["gradient_signed_log2_nrmse"] <= 0.5
        ),
        -row["estimated_ratio_vs_original"],
    ))
    return {
        "image": path.name,
        "shape": list(pixels.shape),
        "crop_size": crop_size,
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", default="sample_DSCF0009.EXR")
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--bits", default="7,8")
    parser.add_argument("--modes", default="gamma075")
    parser.add_argument("--coeffs", default="0,0.25,0.5,0.75,1")
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    bits_values = parse_ints(args.bits)
    modes = parse_modes(args.modes)
    coeffs = parse_floats(args.coeffs)
    results = []
    for path in sorted(DATA_DIR.glob(args.glob)):
        row = summarize_image(path, args.crop_size, bits_values, modes, coeffs)
        results.append(row)
        print(path.name)
        for item in row["rows"][:args.top]:
            marker = "PASS" if (
                item["signed_log2_rmse"] <= 0.004
                and item["signed_log2_p99"] <= 0.018
                and item["gradient_signed_log2_nrmse"] <= 0.5
            ) else "fail"
            print(
                f"  {marker} bits{item['bits']:02d} {item['mode']:<10} "
                f"a={item['a']:.3f} b={item['b']:.3f} "
                f"est={item['estimated_ratio_vs_original']:6.2f}x "
                f"bytes={item['estimated_bytes']} "
                f"log={item['signed_log2_rmse']:.3e} "
                f"p99={item['signed_log2_p99']:.3e} "
                f"grad={item['gradient_signed_log2_nrmse']:.3e}")

    if not args.no_save:
        safe_glob = args.glob.replace("*", "star").replace(".", "_")
        output = RESULTS_DIR / (
            f"color_coefficient_search_{safe_glob}"
            f"_crop{args.crop_size}_bits{args.bits.replace(',', '-')}.json"
        )
        output.write_text(json.dumps(results, indent=2))
        print(f"\nSaved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
