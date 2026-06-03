"""Probe monotonic codebook quantization for transform-index near-lossless.

Uniform transform-index quantization stores per-channel min/max and maps values
to evenly spaced bins.  This probe keeps the same N-bit monotonic index plane,
but replaces uniform bin centers with per-channel Lloyd/quantile codebooks.

The goal is to see whether bits7 can approach bits8 quality without losing the
index-plane spatial structure that the entropy coder relies on.
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
    inverse_transform,
    parse_ints,
    parse_modes,
    quality_stats,
    quantize_transform,
    read_exr,
    transform_values,
)


def train_codebook(
    values: np.ndarray,
    codebook_size: int,
    iterations: int,
    sample_limit: int,
) -> np.ndarray:
    flat = values.reshape(-1)
    if flat.size == 0:
        return np.zeros(codebook_size, dtype=np.float64)
    if sample_limit > 0 and flat.size > sample_limit:
        step = max(1, flat.size // sample_limit)
        train = flat[::step][:sample_limit]
    else:
        train = flat

    if not bool(np.any(np.isfinite(train))):
        return np.zeros(codebook_size, dtype=np.float64)
    train = train[np.isfinite(train)].astype(np.float64)
    if train.size == 0:
        return np.zeros(codebook_size, dtype=np.float64)

    qs = (np.arange(codebook_size, dtype=np.float64) + 0.5) / codebook_size
    centers = np.quantile(train, qs).astype(np.float64)
    if not np.all(np.isfinite(centers)):
        centers = np.nan_to_num(centers, nan=0.0, posinf=0.0, neginf=0.0)

    for _ in range(max(0, iterations)):
        boundaries = (centers[:-1] + centers[1:]) * 0.5
        labels = np.searchsorted(boundaries, train, side="left")
        counts = np.bincount(labels, minlength=codebook_size)
        sums = np.bincount(labels, weights=train, minlength=codebook_size)
        nonzero = counts > 0
        centers[nonzero] = sums[nonzero] / counts[nonzero]
        centers = np.maximum.accumulate(centers)
    return centers.astype(np.float64)


def quantize_codebook(
    pixels: np.ndarray,
    bits: int,
    mode: str,
    iterations: int,
    sample_limit: int,
) -> tuple[np.ndarray, np.ndarray, list[list[float]]]:
    codebook_size = 1 << bits
    indices = np.zeros(pixels.shape, dtype=np.uint16)
    recon = np.array(pixels, copy=True)
    codebooks: list[list[float]] = []

    for channel in range(pixels.shape[2]):
        values = pixels[:, :, channel]
        finite = np.isfinite(values)
        transformed = transform_values(values, mode)
        finite_values = transformed[finite].astype(np.float64)
        centers = train_codebook(
            finite_values,
            codebook_size,
            iterations,
            sample_limit,
        )
        codebooks.append(centers.astype(float).tolist())
        if not bool(np.any(finite)):
            continue

        boundaries = (centers[:-1] + centers[1:]) * 0.5
        q = np.searchsorted(boundaries, transformed, side="left")
        q = np.clip(q, 0, codebook_size - 1)
        indices[:, :, channel] = q.astype(np.uint16)
        rec_t = centers[q]
        rec = inverse_transform(rec_t, mode)
        recon[:, :, channel][finite] = rec[finite].astype(np.float32)

    return indices, np.ascontiguousarray(recon), codebooks


def quantize_reconstruction_table(
    pixels: np.ndarray,
    bits: int,
    mode: str,
) -> tuple[np.ndarray, np.ndarray, list[list[float]]]:
    levels = (1 << bits) - 1
    table_size = 1 << bits
    indices = np.zeros(pixels.shape, dtype=np.uint16)
    recon = np.array(pixels, copy=True)
    tables: list[list[float]] = []

    for channel in range(pixels.shape[2]):
        values = pixels[:, :, channel]
        finite = np.isfinite(values)
        transformed = transform_values(values, mode)
        finite_transformed = transformed[finite].astype(np.float64)
        if finite_transformed.size == 0:
            table = np.zeros(table_size, dtype=np.float64)
            tables.append(table.astype(float).tolist())
            continue
        lo = float(np.min(finite_transformed))
        hi = float(np.max(finite_transformed))
        if not hi > lo:
            table = np.full(table_size, lo, dtype=np.float64)
            tables.append(table.astype(float).tolist())
            recon[:, :, channel][finite] = inverse_transform(
                np.asarray(lo, dtype=np.float64),
                mode,
            ).astype(np.float32)
            continue

        q = np.floor((transformed - lo) / (hi - lo) * levels + 0.5)
        q = np.clip(q, 0, levels).astype(np.uint16)
        indices[:, :, channel] = q

        flat_q = q[finite].reshape(-1).astype(np.int64)
        flat_t = finite_transformed.reshape(-1)
        counts = np.bincount(flat_q, minlength=table_size)
        sums = np.bincount(flat_q, weights=flat_t, minlength=table_size)
        uniform_centers = lo + np.arange(table_size) * (hi - lo) / levels
        table = uniform_centers.astype(np.float64)
        nonzero = counts > 0
        table[nonzero] = sums[nonzero] / counts[nonzero]
        tables.append(table.astype(float).tolist())

        rec_t = table[q]
        rec = inverse_transform(rec_t, mode)
        recon[:, :, channel][finite] = rec[finite].astype(np.float32)

    return indices, np.ascontiguousarray(recon), tables


def estimated_codebook_bits(indices: np.ndarray, bits: int) -> float:
    linear_metadata_bits = indices.shape[2] * 2 * 32 + 64
    payload_bits = estimate_index_bits(indices, bits) - linear_metadata_bits
    codebook_bits = indices.shape[2] * (1 << bits) * 32 + 64
    return payload_bits + codebook_bits


def summarize_image(
    path: Path,
    crop_size: int,
    bits_values: tuple[int, ...],
    modes: tuple[str, ...],
    iterations: int,
    sample_limit: int,
    skip_codebook: bool,
) -> dict:
    pixels = read_exr(path, crop_size)
    rows = []
    for bits in bits_values:
        for mode in modes:
            uniform_indices, uniform_recon = quantize_transform(pixels, bits, mode)
            uniform_bits = estimate_index_bits(uniform_indices, bits)
            rows.append({
                "quantizer": "uniform",
                "bits": bits,
                "mode": mode,
                "estimated_bytes": math.ceil(uniform_bits / 8.0),
                "estimated_ratio_vs_original": pixels.nbytes * 8.0 / uniform_bits,
                **quality_stats(pixels, uniform_recon),
            })

            table_indices, table_recon, _ = quantize_reconstruction_table(
                pixels,
                bits,
                mode,
            )
            table_bits = estimated_codebook_bits(table_indices, bits)
            rows.append({
                "quantizer": "recon-table",
                "bits": bits,
                "mode": mode,
                "estimated_bytes": math.ceil(table_bits / 8.0),
                "estimated_ratio_vs_original": pixels.nbytes * 8.0 / table_bits,
                **quality_stats(pixels, table_recon),
            })

            if not skip_codebook:
                codebook_indices, codebook_recon, _ = quantize_codebook(
                    pixels,
                    bits,
                    mode,
                    iterations,
                    sample_limit,
                )
                codebook_bits = estimated_codebook_bits(codebook_indices, bits)
                rows.append({
                    "quantizer": "codebook",
                    "bits": bits,
                    "mode": mode,
                    "estimated_bytes": math.ceil(codebook_bits / 8.0),
                    "estimated_ratio_vs_original": (
                        pixels.nbytes * 8.0 / codebook_bits),
                    **quality_stats(pixels, codebook_recon),
                })
    rows.sort(key=lambda row: (
        row["bits"],
        row["mode"],
        row["quantizer"],
    ))
    return {
        "image": path.name,
        "shape": list(pixels.shape),
        "crop_size": crop_size,
        "iterations": iterations,
        "sample_limit": sample_limit,
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", default="sample_DSCF0009.EXR")
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--bits", default="7,8")
    parser.add_argument("--modes", default="signed-log,gamma075,asinh")
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--sample-limit", type=int, default=400000)
    parser.add_argument("--skip-codebook", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    bits_values = parse_ints(args.bits)
    modes = parse_modes(args.modes)
    results = []
    for path in sorted(DATA_DIR.glob(args.glob)):
        row = summarize_image(
            path,
            args.crop_size,
            bits_values,
            modes,
            args.iterations,
            args.sample_limit,
            args.skip_codebook,
        )
        results.append(row)
        print(path.name)
        for item in row["rows"]:
            print(
                f"  {item['quantizer']:<8} bits{item['bits']:02d} "
                f"{item['mode']:<10} "
                f"est={item['estimated_ratio_vs_original']:6.2f}x "
                f"bytes={item['estimated_bytes']} "
                f"log={item['signed_log2_rmse']:.3e} "
                f"p99={item['signed_log2_p99']:.3e} "
                f"grad={item['gradient_signed_log2_nrmse']:.3e} "
                f"psnr={item['psnr_peak_abs']:.2f}dB")

    if not args.no_save:
        safe_glob = args.glob.replace("*", "star").replace(".", "_")
        output = RESULTS_DIR / (
            f"codebook_index_quantization_{safe_glob}"
            f"_crop{args.crop_size}_bits{args.bits.replace(',', '-')}.json"
        )
        output.write_text(json.dumps(results, indent=2))
        print(f"\nSaved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
