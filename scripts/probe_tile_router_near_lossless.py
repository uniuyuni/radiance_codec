"""Probe a quality-first tile router for high-ratio near-lossless.

This is not a production bitstream.  It estimates a future route:

* split the image into tiles,
* choose the smallest linear index bit-depth that passes quality thresholds,
* store per-tile/channel min/max plus quantized index planes,
* estimate index-plane coding with an avg predictor, context-coded zero mask,
  and a nonzero residual value stream.

The goal is to answer "how much ratio remains if quality gets first refusal?"
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import OpenImageIO as oiio

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"


def read_exr(path: Path, crop_size: int) -> np.ndarray:
    image_input = oiio.ImageInput.open(str(path))
    if image_input is None:
        raise RuntimeError(f"can't open {path}: {oiio.geterror()}")
    spec = image_input.spec()
    height = spec.height if crop_size == 0 else min(crop_size, spec.height)
    width = spec.width if crop_size == 0 else min(crop_size, spec.width)
    if crop_size:
        pixels = image_input.read_scanlines(
            0, height, 0, 0, spec.nchannels, oiio.FLOAT)
        image_input.close()
        if pixels is None:
            raise RuntimeError(f"can't read {path}: {oiio.geterror()}")
        return np.asarray(pixels, dtype=np.float32).reshape(
            height, spec.width, spec.nchannels)[:, :width, :]
    pixels = image_input.read_image(format=oiio.FLOAT)
    image_input.close()
    if pixels is None:
        raise RuntimeError(f"can't read {path}: {oiio.geterror()}")
    return np.asarray(pixels, dtype=np.float32).reshape(
        spec.height, spec.width, spec.nchannels)


def entropy_bits(values: np.ndarray, alphabet: int | None = None) -> float:
    flat = values.reshape(-1).astype(np.int64)
    if flat.size == 0:
        return 0.0
    counts = np.bincount(flat, minlength=alphabet or 0)
    counts = counts[counts > 0]
    if counts.size == 0:
        return 0.0
    probs = counts / counts.sum()
    return float(-(probs * np.log2(probs)).sum() * flat.size)


def avg_predictor(indices: np.ndarray) -> np.ndarray:
    west = np.zeros_like(indices)
    north = np.zeros_like(indices)
    west[:, 1:] = indices[:, :-1]
    north[1:] = indices[:-1]
    return ((west.astype(np.uint32) + north.astype(np.uint32)) // 2).astype(
        np.uint16)


def context_zero_mask_bits(mask: np.ndarray, context_tile: int) -> float:
    cost = 0.0
    for y0 in range(0, mask.shape[0], context_tile):
        for x0 in range(0, mask.shape[1], context_tile):
            tile = mask[y0:y0 + context_tile, x0:x0 + context_tile]
            counts = np.full((8, 2), 0.5, dtype=np.float64)
            for y in range(tile.shape[0]):
                for x in range(tile.shape[1]):
                    left = int(tile[y, x - 1]) if x else 0
                    up = int(tile[y - 1, x]) if y else 0
                    up_left = int(tile[y - 1, x - 1]) if y and x else 0
                    context = left | (up << 1) | (up_left << 2)
                    bit = int(tile[y, x])
                    total = counts[context].sum()
                    cost -= math.log2(counts[context, bit] / total)
                    counts[context, bit] += 1.0
    return cost


def quantize_linear_tile(tile: np.ndarray, bits: int) -> tuple[np.ndarray, np.ndarray]:
    levels = (1 << bits) - 1
    indices = np.zeros(tile.shape, dtype=np.uint16)
    recon = np.array(tile, copy=True)
    for channel in range(tile.shape[2]):
        values = tile[:, :, channel].astype(np.float64)
        finite = np.isfinite(values)
        if not bool(np.any(finite)):
            continue
        lo = float(values[finite].min())
        hi = float(values[finite].max())
        if hi <= lo:
            recon[:, :, channel][finite] = np.float32(lo)
            continue
        quantized = np.floor((values - lo) / (hi - lo) * levels + 0.5)
        quantized = np.clip(quantized, 0, levels)
        indices[:, :, channel] = quantized.astype(np.uint16)
        rec = lo + quantized * (hi - lo) / levels
        recon[:, :, channel][finite] = rec[finite].astype(np.float32)
    return indices, np.ascontiguousarray(recon)


def quality_stats(original: np.ndarray, quantized: np.ndarray) -> dict[str, float]:
    finite = np.isfinite(original) & np.isfinite(quantized)
    if not bool(np.any(finite)):
        return {
            "signed_log2_rmse": 0.0,
            "signed_log2_p99": 0.0,
            "abs_p99": 0.0,
            "gradient_signed_log2_nrmse": 0.0,
        }
    selected_original = original[finite].astype(np.float64)
    selected_quantized = quantized[finite].astype(np.float64)
    signed_log_original = (
        np.sign(selected_original) * np.log2(1.0 + np.abs(selected_original))
    )
    signed_log_quantized = (
        np.sign(selected_quantized) * np.log2(1.0 + np.abs(selected_quantized))
    )
    signed_log_abs = np.abs(signed_log_quantized - signed_log_original)
    diff_abs = np.abs(selected_quantized - selected_original)

    full_original = (
        np.sign(original.astype(np.float64))
        * np.log2(1.0 + np.abs(original.astype(np.float64)))
    )
    full_quantized = (
        np.sign(quantized.astype(np.float64))
        * np.log2(1.0 + np.abs(quantized.astype(np.float64)))
    )
    grad_original = np.concatenate(
        [
            np.diff(full_original, axis=0).reshape(-1),
            np.diff(full_original, axis=1).reshape(-1),
        ]
    )
    grad_quantized = np.concatenate(
        [
            np.diff(full_quantized, axis=0).reshape(-1),
            np.diff(full_quantized, axis=1).reshape(-1),
        ]
    )
    grad_finite = np.isfinite(grad_original) & np.isfinite(grad_quantized)
    gradient_nrmse = 0.0
    if bool(np.any(grad_finite)):
        grad_diff = grad_quantized[grad_finite] - grad_original[grad_finite]
        grad_rmse = float(np.sqrt(np.mean(grad_diff * grad_diff)))
        grad_rms = float(np.sqrt(np.mean(grad_original[grad_finite] ** 2)))
        if grad_rms > 0.0:
            gradient_nrmse = grad_rmse / grad_rms

    return {
        "signed_log2_rmse": float(np.sqrt(np.mean(signed_log_abs ** 2))),
        "signed_log2_p99": float(np.percentile(signed_log_abs, 99.0)),
        "abs_p99": float(np.percentile(diff_abs, 99.0)),
        "gradient_signed_log2_nrmse": gradient_nrmse,
    }


def estimate_index_bits(
    indices: np.ndarray,
    bits: int,
    context_tile: int,
) -> float:
    alphabet = 1 << bits
    total_bits = 0.0
    for channel in range(indices.shape[2]):
        channel_indices = indices[:, :, channel]
        residual = (
            (channel_indices.astype(np.int32)
             - avg_predictor(channel_indices).astype(np.int32))
            & (alphabet - 1)
        ).astype(np.uint16)
        nonzero = residual != 0
        total_bits += context_zero_mask_bits(nonzero, context_tile)
        nonzero_values = residual[nonzero] - 1
        if nonzero_values.size:
            total_bits += entropy_bits(nonzero_values, alphabet - 1)
    return total_bits


def choose_tile(
    tile: np.ndarray,
    candidate_bits: tuple[int, ...],
    context_tile: int,
    log_rmse_threshold: float,
    log_p99_threshold: float,
    gradient_threshold: float,
) -> dict:
    attempts = []
    for bits in candidate_bits:
        indices, quantized = quantize_linear_tile(tile, bits)
        quality = quality_stats(tile, quantized)
        payload_bits = estimate_index_bits(indices, bits, context_tile)
        metadata_bits = tile.shape[2] * 2 * 32 + 16
        row = {
            "bits": bits,
            "payload_bits": payload_bits,
            "metadata_bits": float(metadata_bits),
            "total_bits": payload_bits + metadata_bits,
            **quality,
        }
        attempts.append(row)
        if (
            quality["signed_log2_rmse"] <= log_rmse_threshold
            and quality["signed_log2_p99"] <= log_p99_threshold
            and quality["gradient_signed_log2_nrmse"] <= gradient_threshold
        ):
            row["selected"] = True
            row["attempts"] = attempts
            return row
    attempts[-1]["selected"] = True
    attempts[-1]["attempts"] = attempts
    return attempts[-1]


def summarize_image(
    path: Path,
    crop_size: int,
    tile_size: int,
    context_tile: int,
    candidate_bits: tuple[int, ...],
    log_rmse_threshold: float,
    log_p99_threshold: float,
    gradient_threshold: float,
) -> dict:
    pixels = np.ascontiguousarray(read_exr(path, crop_size), dtype=np.float32)
    total_bits = 0.0
    selected_counter: Counter[int] = Counter()
    tile_rows = []
    worst_log_p99 = 0.0
    worst_grad = 0.0
    for y in range(0, pixels.shape[0], tile_size):
        for x in range(0, pixels.shape[1], tile_size):
            tile = pixels[y:y + tile_size, x:x + tile_size, :]
            row = choose_tile(
                tile,
                candidate_bits,
                min(context_tile, tile.shape[0], tile.shape[1]),
                log_rmse_threshold,
                log_p99_threshold,
                gradient_threshold,
            )
            selected_counter[int(row["bits"])] += 1
            total_bits += float(row["total_bits"])
            worst_log_p99 = max(worst_log_p99, float(row["signed_log2_p99"]))
            worst_grad = max(worst_grad, float(row["gradient_signed_log2_nrmse"]))
            tile_rows.append({
                "x": x,
                "y": y,
                "bits": int(row["bits"]),
                "total_bits": float(row["total_bits"]),
                "ratio_vs_float32": (
                    tile.nbytes * 8.0 / float(row["total_bits"])
                    if row["total_bits"] else math.inf
                ),
                "signed_log2_rmse": float(row["signed_log2_rmse"]),
                "signed_log2_p99": float(row["signed_log2_p99"]),
                "gradient_signed_log2_nrmse": float(
                    row["gradient_signed_log2_nrmse"]),
            })
    ratio = pixels.nbytes * 8.0 / total_bits if total_bits else math.inf
    return {
        "image": path.name,
        "shape": list(pixels.shape),
        "crop_size": crop_size,
        "tile_size": tile_size,
        "context_tile": context_tile,
        "candidate_bits": list(candidate_bits),
        "log_rmse_threshold": log_rmse_threshold,
        "log_p99_threshold": log_p99_threshold,
        "gradient_threshold": gradient_threshold,
        "estimated_bits": total_bits,
        "estimated_bytes": math.ceil(total_bits / 8.0),
        "estimated_ratio_vs_float32": ratio,
        "selected_bits": dict(sorted(selected_counter.items())),
        "worst_signed_log2_p99": worst_log_p99,
        "worst_gradient_signed_log2_nrmse": worst_grad,
        "tiles": tile_rows,
    }


def parse_ints(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split(",") if part)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", default="sample_DSCF0009.EXR")
    parser.add_argument("--crop-size", type=int, default=1024)
    parser.add_argument("--tile-size", type=int, default=128)
    parser.add_argument("--context-tile", type=int, default=64)
    parser.add_argument("--candidate-bits", default="7,8,10,12")
    parser.add_argument("--log-rmse-threshold", type=float, default=0.004)
    parser.add_argument("--log-p99-threshold", type=float, default=0.018)
    parser.add_argument("--gradient-threshold", type=float, default=0.8)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    candidate_bits = parse_ints(args.candidate_bits)
    rows = []
    for path in sorted(DATA_DIR.glob(args.glob)):
        row = summarize_image(
            path,
            args.crop_size,
            args.tile_size,
            args.context_tile,
            candidate_bits,
            args.log_rmse_threshold,
            args.log_p99_threshold,
            args.gradient_threshold,
        )
        rows.append(row)
        print(path.stem)
        print(
            f"  estimated={row['estimated_ratio_vs_float32']:.2f}x "
            f"bytes={row['estimated_bytes']} "
            f"bits={row['selected_bits']} "
            f"worst_log_p99={row['worst_signed_log2_p99']:.3e} "
            f"worst_grad={row['worst_gradient_signed_log2_nrmse']:.3e}")
    if not args.no_save:
        safe_glob = args.glob.replace("*", "star").replace(".", "_")
        output = RESULTS_DIR / (
            f"tile_router_near_lossless_{safe_glob}_crop{args.crop_size}"
            f"_tile{args.tile_size}_ctx{args.context_tile}"
            f"_bits{args.candidate_bits.replace(',', '-')}"
            f"_lr{args.log_rmse_threshold:g}"
            f"_lp{args.log_p99_threshold:g}"
            f"_gr{args.gradient_threshold:g}.json"
        )
        output.write_text(json.dumps(rows, indent=2))
        print(f"\nSaved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
