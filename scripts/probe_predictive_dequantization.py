"""Probe decoder-side predictive dequantization for transform-index images.

The encoded index stream is unchanged.  Decode reconstructs each transform
value inside its quantization bin by nudging the bin center toward a causal
prediction from already decoded neighbors, clamped to the bin boundaries.
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
    inverse_transform,
    parse_ints,
    parse_modes,
    quality_stats,
    read_exr,
    transform_values,
)


def quantize_indices_with_ranges(
    pixels: np.ndarray,
    bits: int,
    mode: str,
) -> tuple[np.ndarray, list[tuple[float, float]]]:
    levels = (1 << bits) - 1
    indices = np.zeros(pixels.shape, dtype=np.uint16)
    ranges: list[tuple[float, float]] = []
    for channel in range(pixels.shape[2]):
        values = pixels[:, :, channel]
        transformed = transform_values(values, mode)
        finite = np.isfinite(values)
        finite_t = transformed[finite].astype(np.float64)
        lo = float(np.min(finite_t))
        hi = float(np.max(finite_t))
        ranges.append((lo, hi))
        if not hi > lo:
            continue
        q = np.floor((transformed - lo) / (hi - lo) * levels + 0.5)
        indices[:, :, channel] = np.clip(q, 0, levels).astype(np.uint16)
    return indices, ranges


def decode_predictive(
    indices: np.ndarray,
    ranges: list[tuple[float, float]],
    bits: int,
    mode: str,
    alpha: float,
) -> np.ndarray:
    levels = (1 << bits) - 1
    recon_t = np.zeros(indices.shape, dtype=np.float64)
    for channel in range(indices.shape[2]):
        lo, hi = ranges[channel]
        if not hi > lo:
            recon_t[:, :, channel] = lo
            continue
        step = (hi - lo) / levels
        for y in range(indices.shape[0]):
            for x in range(indices.shape[1]):
                idx = int(indices[y, x, channel])
                center = lo + idx * step
                bin_lo = lo if idx == 0 else center - 0.5 * step
                bin_hi = hi if idx == levels else center + 0.5 * step
                preds = []
                if x > 0:
                    preds.append(recon_t[y, x - 1, channel])
                if y > 0:
                    preds.append(recon_t[y - 1, x, channel])
                if not preds:
                    value = center
                else:
                    pred = float(sum(preds) / len(preds))
                    nudged = min(max(pred, bin_lo), bin_hi)
                    value = center * (1.0 - alpha) + nudged * alpha
                recon_t[y, x, channel] = value
    recon = inverse_transform(recon_t, mode).astype(np.float32)
    return np.ascontiguousarray(recon)


def decode_predictive_center(
    indices: np.ndarray,
    ranges: list[tuple[float, float]],
    bits: int,
    mode: str,
    alpha: float,
) -> np.ndarray:
    levels = (1 << bits) - 1
    recon_t = np.zeros(indices.shape, dtype=np.float64)
    for channel in range(indices.shape[2]):
        lo, hi = ranges[channel]
        if not hi > lo:
            recon_t[:, :, channel] = lo
            continue
        step = (hi - lo) / levels
        idx = indices[:, :, channel].astype(np.float64)
        center = lo + idx * step
        bin_lo = np.where(indices[:, :, channel] == 0, lo, center - 0.5 * step)
        bin_hi = np.where(indices[:, :, channel] == levels, hi, center + 0.5 * step)

        west = np.empty_like(center)
        north = np.empty_like(center)
        west[:, 0] = center[:, 0]
        west[:, 1:] = center[:, :-1]
        north[0, :] = center[0, :]
        north[1:, :] = center[:-1, :]
        pred = (west + north) * 0.5
        nudged = np.minimum(np.maximum(pred, bin_lo), bin_hi)
        recon_t[:, :, channel] = center * (1.0 - alpha) + nudged * alpha
    recon = inverse_transform(recon_t, mode).astype(np.float32)
    return np.ascontiguousarray(recon)


def summarize_image(
    path: Path,
    crop_size: int,
    bits_values: tuple[int, ...],
    modes: tuple[str, ...],
    alphas: tuple[float, ...],
    method: str,
) -> dict:
    pixels = read_exr(path, crop_size)
    rows = []
    for bits in bits_values:
        for mode in modes:
            indices, ranges = quantize_indices_with_ranges(pixels, bits, mode)
            for alpha in alphas:
                if method == "center":
                    recon = decode_predictive_center(indices, ranges, bits, mode, alpha)
                else:
                    recon = decode_predictive(indices, ranges, bits, mode, alpha)
                rows.append({
                    "bits": bits,
                    "mode": mode,
                    "alpha": alpha,
                    "method": method,
                    **quality_stats(pixels, recon),
                })
    rows.sort(key=lambda row: (
        row["bits"],
        row["mode"],
        row["signed_log2_rmse"],
    ))
    return {
        "image": path.name,
        "shape": list(pixels.shape),
        "crop_size": crop_size,
        "rows": rows,
    }


def parse_floats(text: str) -> tuple[float, ...]:
    return tuple(float(part) for part in text.split(",") if part.strip())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", default="sample_DSCF0009.EXR")
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--bits", default="7")
    parser.add_argument("--modes", default="signed-log,gamma075,asinh")
    parser.add_argument("--alphas", default="0,0.25,0.5,0.75,1")
    parser.add_argument("--method", choices=("causal", "center"), default="causal")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    bits_values = parse_ints(args.bits)
    modes = parse_modes(args.modes)
    alphas = parse_floats(args.alphas)
    results = []
    for path in sorted(DATA_DIR.glob(args.glob)):
        row = summarize_image(
            path,
            args.crop_size,
            bits_values,
            modes,
            alphas,
            args.method,
        )
        results.append(row)
        print(path.name)
        for item in row["rows"]:
            print(
                f"  bits{item['bits']:02d} {item['mode']:<10} "
                f"alpha={item['alpha']:.2f} "
                f"log={item['signed_log2_rmse']:.3e} "
                f"p99={item['signed_log2_p99']:.3e} "
                f"grad={item['gradient_signed_log2_nrmse']:.3e} "
                f"psnr={item['psnr_peak_abs']:.2f}dB")

    if not args.no_save:
        safe_glob = args.glob.replace("*", "star").replace(".", "_")
        output = RESULTS_DIR / (
            f"predictive_dequantization_{safe_glob}"
            f"_crop{args.crop_size}_bits{args.bits.replace(',', '-')}.json"
        )
        output.write_text(json.dumps(results, indent=2))
        print(f"\nSaved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
