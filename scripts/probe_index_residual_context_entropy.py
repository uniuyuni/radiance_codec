"""Probe conditional entropy of transform-index residual symbols.

StageLinearIndex currently splits residuals into a nonzero mask plus an order-0
value stream.  This probe asks whether a direct multi-symbol residual coder with
decoder-visible spatial contexts could reduce the full residual entropy enough
to matter for the 21MB target.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import OpenImageIO as oiio

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
sys.path.insert(0, str(ROOT / "scripts"))

from probe_parametric_transform_index import quantize_power  # noqa: E402


def transform_values(values: np.ndarray, transform: str, gamma: float) -> np.ndarray:
    if transform == "power":
        return np.sign(values) * np.power(np.abs(values), gamma)
    if transform == "signed-log":
        return np.sign(values) * np.log2(1.0 + np.abs(values))
    if transform == "asinh":
        return np.arcsinh(values)
    raise ValueError(f"unknown transform: {transform}")


def inverse_transform_values(values: np.ndarray, transform: str, gamma: float) -> np.ndarray:
    if transform == "power":
        return np.sign(values) * np.power(np.abs(values), 1.0 / gamma)
    if transform == "signed-log":
        return np.sign(values) * (np.exp2(np.abs(values)) - 1.0)
    if transform == "asinh":
        return np.sinh(values)
    raise ValueError(f"unknown transform: {transform}")


def quantize_transform_indices(
    pixels: np.ndarray,
    bits: int,
    transform: str,
    gamma: float,
) -> tuple[np.ndarray, np.ndarray]:
    if transform == "power":
        return quantize_power(pixels, bits, gamma)
    levels = (1 << bits) - 1
    indices = np.zeros(pixels.shape, dtype=np.uint16)
    recon = np.array(pixels, copy=True)
    for channel in range(pixels.shape[2]):
        values = pixels[:, :, channel].astype(np.float64)
        transformed = transform_values(values, transform, gamma)
        range_values = transformed.astype(np.float32).astype(np.float64)
        lo = float(np.min(range_values))
        hi = float(np.max(range_values))
        if not hi > lo:
            recon[:, :, channel] = inverse_transform_values(
                np.asarray(lo),
                transform,
                gamma,
            ).astype(np.float32)
            continue
        q = np.floor((transformed - lo) / (hi - lo) * levels + 0.5)
        q = np.clip(q, 0, levels)
        indices[:, :, channel] = q.astype(np.uint16)
        rec_t = lo + q * (hi - lo) / levels
        recon[:, :, channel] = inverse_transform_values(
            rec_t,
            transform,
            gamma,
        ).astype(np.float32)
    return indices, np.ascontiguousarray(recon)


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


def med_predict_indices(indices: np.ndarray, bits: int) -> np.ndarray:
    west = np.zeros(indices.shape, dtype=np.int32)
    north = np.zeros(indices.shape, dtype=np.int32)
    northwest = np.zeros(indices.shape, dtype=np.int32)
    west[:, 1:, :] = indices[:, :-1, :]
    north[1:, :, :] = indices[:-1, :, :]
    northwest[1:, 1:, :] = indices[:-1, :-1, :]
    hi = np.maximum(west, north)
    lo = np.minimum(west, north)
    gradient = west + north - northwest
    pred = np.where(northwest >= hi, lo, np.where(northwest <= lo, hi, gradient))
    return np.clip(pred, 0, (1 << bits) - 1).astype(np.int32)


def signed_residual_from_symbol(symbols: np.ndarray, bits: int) -> np.ndarray:
    alphabet = 1 << bits
    half = alphabet // 2
    signed = symbols.astype(np.int16)
    signed[signed >= half] -= alphabet
    return signed


def residual_category(symbols: np.ndarray, bits: int) -> np.ndarray:
    signed = signed_residual_from_symbol(symbols, bits)
    category = np.full(symbols.shape, 6, dtype=np.uint16)
    category[signed == 0] = 0
    category[signed == 1] = 1
    category[signed == -1] = 2
    category[(signed >= 2) & (signed <= 3)] = 3
    category[(signed <= -2) & (signed >= -3)] = 4
    category[(signed >= 4) & (signed <= 7)] = 5
    category[(signed <= -4) & (signed >= -7)] = 6
    return category


def shifted_contexts(categories: np.ndarray) -> dict[str, np.ndarray]:
    height, width, channels = categories.shape
    west = np.zeros(categories.shape, dtype=np.uint16)
    north = np.zeros(categories.shape, dtype=np.uint16)
    northwest = np.zeros(categories.shape, dtype=np.uint16)
    prev_channel = np.zeros(categories.shape, dtype=np.uint16)
    west[:, 1:, :] = categories[:, :-1, :]
    north[1:, :, :] = categories[:-1, :, :]
    northwest[1:, 1:, :] = categories[:-1, :-1, :]
    if channels > 1:
        prev_channel[:, :, 1:] = categories[:, :, :-1]
    yy, xx = np.indices((height, width), dtype=np.uint16)
    phase = (((yy & 1) << 1) | (xx & 1))[:, :, None]
    channel = np.arange(channels, dtype=np.uint16)[None, None, :]
    return {
        "order0": np.zeros(categories.shape, dtype=np.uint16),
        "phase_channel": phase + 4 * channel,
        "westcat": west,
        "northcat": north,
        "west_north": west + 8 * north,
        "west_north_phase_channel": west + 8 * north + 64 * (phase + 4 * channel),
        "west_north_prev_channel": west + 8 * north + 64 * prev_channel,
        "wnnw_phase_channel": (
            west
            + 8 * north
            + 64 * northwest
            + 512 * (phase + 4 * channel)
        ),
    }


def conditional_entropy_bits(
    symbols: np.ndarray,
    contexts: np.ndarray,
    alphabet: int,
) -> float:
    flat_symbols = symbols.reshape(-1).astype(np.int64)
    flat_contexts = contexts.reshape(-1).astype(np.int64)
    context_count = int(flat_contexts.max()) + 1 if flat_contexts.size else 1
    joint = np.bincount(
        flat_contexts * alphabet + flat_symbols,
        minlength=context_count * alphabet,
    ).reshape(context_count, alphabet).astype(np.float64)
    context_totals = joint.sum(axis=1)
    nonzero = joint > 0
    probs = np.zeros_like(joint)
    probs[nonzero] = joint[nonzero] / np.repeat(context_totals, alphabet)[nonzero.reshape(-1)].reshape(joint.shape)[nonzero]
    # Avoid a huge temporary from log2 on zero entries.
    nz_probs = probs[nonzero]
    nz_counts = joint[nonzero]
    return float(-(nz_counts * np.log2(nz_probs)).sum())


def conditional_entropy_bits_fast(
    symbols: np.ndarray,
    contexts: np.ndarray,
    alphabet: int,
) -> float:
    flat_symbols = symbols.reshape(-1).astype(np.int64)
    flat_contexts = contexts.reshape(-1).astype(np.int64)
    context_count = int(flat_contexts.max()) + 1 if flat_contexts.size else 1
    joint = np.bincount(
        flat_contexts * alphabet + flat_symbols,
        minlength=context_count * alphabet,
    ).reshape(context_count, alphabet).astype(np.float64)
    context_totals = joint.sum(axis=1)
    nz_contexts, nz_symbols = np.nonzero(joint)
    counts = joint[nz_contexts, nz_symbols]
    probs = counts / context_totals[nz_contexts]
    return float(-(counts * np.log2(probs)).sum())


def summarize_image(
    path: Path,
    crop_size: int,
    bits: int,
    transform: str,
    gamma: float,
) -> dict:
    pixels = read_exr(path, crop_size)
    indices, _recon = quantize_transform_indices(pixels, bits, transform, gamma)
    alphabet = 1 << bits
    pred = med_predict_indices(indices, bits)
    residual = (indices.astype(np.int32) + alphabet - pred) & (alphabet - 1)
    residual = residual.astype(np.uint16)
    categories = residual_category(residual, bits)
    contexts = shifted_contexts(categories)
    rows = []
    total = residual.size
    for name, context in contexts.items():
        bits_total = conditional_entropy_bits_fast(residual, context, alphabet)
        rows.append({
            "context": name,
            "bits_per_sample": bits_total / total,
            "estimated_bytes": bits_total / 8.0 + 73.0,
            "estimated_ratio_vs_original": pixels.nbytes / (bits_total / 8.0 + 73.0),
            "context_count": int(context.max()) + 1 if context.size else 1,
        })
    rows.sort(key=lambda row: row["bits_per_sample"])
    return {
        "image": path.name,
        "shape": list(pixels.shape),
        "crop_size": crop_size,
        "bits": bits,
        "transform": transform,
        "gamma": gamma,
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", default="sample_DSCF0009.EXR")
    parser.add_argument("--crop-size", type=int, default=1024)
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--transform", choices=("power", "signed-log", "asinh"), default="power")
    parser.add_argument("--gamma", type=float, default=0.75)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    summaries = []
    for path in sorted(DATA_DIR.glob(args.glob)):
        summary = summarize_image(
            path,
            args.crop_size,
            args.bits,
            args.transform,
            args.gamma,
        )
        summaries.append(summary)
        print(path.name)
        for row in summary["rows"]:
            print(
                f"  {row['context']:26s} "
                f"bps={row['bits_per_sample']:.3f} "
                f"bytes={row['estimated_bytes']:,.0f} "
                f"ratio={row['estimated_ratio_vs_original']:.2f}x "
                f"contexts={row['context_count']}"
            )
        if args.limit and len(summaries) >= args.limit:
            break

    if not args.no_save:
        safe_glob = args.glob.replace("*", "star").replace(".", "_")
        output = RESULTS_DIR / (
            f"index_residual_context_entropy_{safe_glob}"
            f"_crop{args.crop_size}_{args.transform}"
            f"_gamma{args.gamma:g}_bits{args.bits}.json"
        )
        output.write_text(json.dumps(summaries, indent=2))
        print(f"\nSaved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
