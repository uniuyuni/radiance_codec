"""Probe transform-domain linear-index quantization for near-lossless images.

This is a decision probe for the 21MB target.  StageLinearIndex currently
quantizes values linearly per channel.  This script estimates whether the same
index codec would work better if the stored index were uniform in a reversible
monotonic transform domain such as signed-log, sqrt, or gamma.
"""
from __future__ import annotations

import argparse
import json
import math
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
            0,
            height,
            0,
            0,
            spec.nchannels,
            oiio.FLOAT,
        )
    else:
        pixels = image_input.read_image(format=oiio.FLOAT)
    image_input.close()
    if pixels is None:
        raise RuntimeError(f"can't read {path}: {oiio.geterror()}")
    return np.ascontiguousarray(
        np.asarray(pixels, dtype=np.float32).reshape(
            height,
            spec.width,
            spec.nchannels,
        )[:, :width, :]
    )


def signed_power(values: np.ndarray, power: float) -> np.ndarray:
    return np.sign(values) * np.power(np.abs(values), power)


def inverse_signed_power(values: np.ndarray, power: float) -> np.ndarray:
    return np.sign(values) * np.power(np.abs(values), 1.0 / power)


def transform_values(values: np.ndarray, mode: str) -> np.ndarray:
    values64 = values.astype(np.float64)
    if mode == "linear":
        return values64
    if mode == "signed-log":
        return np.sign(values64) * np.log2(1.0 + np.abs(values64))
    if mode == "sqrt":
        return signed_power(values64, 0.5)
    if mode == "gamma025":
        return signed_power(values64, 0.25)
    if mode == "gamma075":
        return signed_power(values64, 0.75)
    if mode == "asinh":
        return np.arcsinh(values64)
    raise ValueError(f"unknown mode: {mode}")


def inverse_transform(values: np.ndarray, mode: str) -> np.ndarray:
    if mode == "linear":
        return values
    if mode == "signed-log":
        return np.sign(values) * (np.exp2(np.abs(values)) - 1.0)
    if mode == "sqrt":
        return inverse_signed_power(values, 0.5)
    if mode == "gamma025":
        return inverse_signed_power(values, 0.25)
    if mode == "gamma075":
        return inverse_signed_power(values, 0.75)
    if mode == "asinh":
        return np.sinh(values)
    raise ValueError(f"unknown mode: {mode}")


def quantize_transform(
    pixels: np.ndarray,
    bits: int,
    mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    levels = (1 << bits) - 1
    indices = np.zeros(pixels.shape, dtype=np.uint16)
    recon = np.array(pixels, copy=True)
    for channel in range(pixels.shape[2]):
        values = pixels[:, :, channel]
        finite = np.isfinite(values)
        if not bool(np.any(finite)):
            continue
        transformed = transform_values(values, mode)
        finite_transformed = transformed[finite]
        lo = float(np.min(finite_transformed))
        hi = float(np.max(finite_transformed))
        if not hi > lo:
            recon[:, :, channel][finite] = np.float32(
                inverse_transform(np.array(lo), mode))
            continue
        q = np.floor((transformed - lo) / (hi - lo) * levels + 0.5)
        q = np.clip(q, 0, levels)
        indices[:, :, channel] = q.astype(np.uint16)
        rec_t = lo + q * (hi - lo) / levels
        rec = inverse_transform(rec_t, mode)
        recon[:, :, channel][finite] = rec[finite].astype(np.float32)
    return indices, np.ascontiguousarray(recon)


def avg_predictor(indices: np.ndarray) -> np.ndarray:
    west = np.zeros_like(indices)
    north = np.zeros_like(indices)
    west[:, 1:] = indices[:, :-1]
    north[1:] = indices[:-1]
    return ((west.astype(np.uint32) + north.astype(np.uint32)) // 2).astype(
        np.uint16)


def mask_context(mask: np.ndarray, y: int, x: int, channel: int) -> int:
    context = 0
    if x:
        context |= int(mask[y, x - 1, channel])
    if y:
        context |= int(mask[y - 1, x, channel]) << 1
    if x and y:
        context |= int(mask[y - 1, x - 1, channel]) << 2
    if x + 1 < mask.shape[1] and y:
        context |= int(mask[y - 1, x + 1, channel]) << 3
    if channel:
        context |= int(mask[y, x, channel - 1]) << 4
    phase = ((y & 1) << 1) | (x & 1)
    return context + 32 * (channel + mask.shape[2] * phase)


def context_zero_mask_bits(mask: np.ndarray) -> float:
    counts = np.full((32 * mask.shape[2] * 4, 2), 0.5, dtype=np.float64)
    cost = 0.0
    for channel in range(mask.shape[2]):
        for y in range(mask.shape[0]):
            for x in range(mask.shape[1]):
                context = mask_context(mask, y, x, channel)
                bit = int(mask[y, x, channel])
                total = counts[context].sum()
                cost -= math.log2(counts[context, bit] / total)
                counts[context, bit] += 1.0
    return cost


def entropy_bits(values: np.ndarray, alphabet: int) -> float:
    flat = values.reshape(-1).astype(np.int64)
    if flat.size == 0:
        return 0.0
    counts = np.bincount(flat, minlength=alphabet)
    counts = counts[counts > 0]
    if counts.size == 0:
        return 0.0
    probs = counts / counts.sum()
    return float(-(probs * np.log2(probs)).sum() * flat.size)


def estimate_index_bits(indices: np.ndarray, bits: int) -> float:
    alphabet = 1 << bits
    residual = np.zeros(indices.shape, dtype=np.uint16)
    for channel in range(indices.shape[2]):
        channel_indices = indices[:, :, channel]
        residual[:, :, channel] = (
            channel_indices.astype(np.int32)
            - avg_predictor(channel_indices).astype(np.int32)
        ) & (alphabet - 1)
    nonzero = residual != 0
    total_bits = context_zero_mask_bits(nonzero)
    for channel in range(indices.shape[2]):
        channel_residual = residual[:, :, channel]
        channel_nonzero = nonzero[:, :, channel]
        nonzero_values = channel_residual[channel_nonzero] - 1
        if nonzero_values.size:
            total_bits += entropy_bits(nonzero_values, alphabet - 1)
    metadata_bits = indices.shape[2] * 2 * 32 + 64
    return total_bits + metadata_bits


def quality_stats(original: np.ndarray, quantized: np.ndarray) -> dict[str, float]:
    diff = quantized.astype(np.float64) - original.astype(np.float64)
    rmse = float(np.sqrt(np.mean(diff * diff)))
    peak = float(np.max(np.abs(original)))
    psnr = math.inf
    if rmse > 0.0 and peak > 0.0:
        psnr = 20.0 * math.log10(peak / rmse)
    signed_log_original = (
        np.sign(original.astype(np.float64))
        * np.log2(1.0 + np.abs(original.astype(np.float64)))
    )
    signed_log_quantized = (
        np.sign(quantized.astype(np.float64))
        * np.log2(1.0 + np.abs(quantized.astype(np.float64)))
    )
    signed_log_diff = signed_log_quantized - signed_log_original
    grad_original = np.concatenate(
        [
            np.diff(signed_log_original, axis=0).reshape(-1),
            np.diff(signed_log_original, axis=1).reshape(-1),
        ]
    )
    grad_quantized = np.concatenate(
        [
            np.diff(signed_log_quantized, axis=0).reshape(-1),
            np.diff(signed_log_quantized, axis=1).reshape(-1),
        ]
    )
    grad_diff = grad_quantized - grad_original
    grad_signal = float(np.sqrt(np.mean(grad_original * grad_original)))
    grad_nrmse = 0.0
    if grad_signal > 0.0:
        grad_nrmse = float(np.sqrt(np.mean(grad_diff * grad_diff)) / grad_signal)
    return {
        "rmse": rmse,
        "psnr_peak_abs": psnr,
        "signed_log2_rmse": float(np.sqrt(np.mean(signed_log_diff ** 2))),
        "signed_log2_p99": float(np.percentile(np.abs(signed_log_diff), 99.0)),
        "gradient_signed_log2_nrmse": grad_nrmse,
    }


def parse_ints(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split(",") if part.strip())


def parse_modes(text: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in text.split(",") if part.strip())


def summarize_image(
    path: Path,
    crop_size: int,
    bits_values: tuple[int, ...],
    modes: tuple[str, ...],
) -> dict:
    pixels = read_exr(path, crop_size)
    rows = []
    for bits in bits_values:
        for mode in modes:
            indices, recon = quantize_transform(pixels, bits, mode)
            estimated_bits = estimate_index_bits(indices, bits)
            rows.append({
                "bits": bits,
                "mode": mode,
                "estimated_bytes": math.ceil(estimated_bits / 8.0),
                "estimated_ratio_vs_original": pixels.nbytes * 8.0 / estimated_bits,
                **quality_stats(pixels, recon),
            })
    rows.sort(key=lambda row: (
        row["bits"],
        row["signed_log2_rmse"],
        row["estimated_bytes"],
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
    parser.add_argument("--crop-size", type=int, default=1024)
    parser.add_argument("--bits", default="8,9")
    parser.add_argument(
        "--modes",
        default="linear,signed-log,sqrt,gamma025,gamma075,asinh",
    )
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    bits_values = parse_ints(args.bits)
    modes = parse_modes(args.modes)
    results = []
    for path in sorted(DATA_DIR.glob(args.glob)):
        row = summarize_image(path, args.crop_size, bits_values, modes)
        results.append(row)
        print(path.name)
        for item in row["rows"]:
            print(
                f"  bits{item['bits']:02d} {item['mode']:<10} "
                f"est={item['estimated_ratio_vs_original']:6.2f}x "
                f"bytes={item['estimated_bytes']} "
                f"log_rmse={item['signed_log2_rmse']:.3e} "
                f"log_p99={item['signed_log2_p99']:.3e} "
                f"grad={item['gradient_signed_log2_nrmse']:.3e} "
                f"psnr={item['psnr_peak_abs']:.2f}dB")

    if not args.no_save:
        safe_glob = args.glob.replace("*", "star").replace(".", "_")
        output = RESULTS_DIR / (
            f"transform_index_quantization_{safe_glob}"
            f"_crop{args.crop_size}_bits{args.bits.replace(',', '-')}.json"
        )
        output.write_text(json.dumps(results, indent=2))
        print(f"\nSaved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
