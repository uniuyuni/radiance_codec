"""Probe small-residual + escape entropy models for index residuals.

This tests whether the dominant residuals (0, +/-1, +/-2, ...) are better
represented as a compact context-coded alphabet plus sparse escape magnitudes
than as one full residual symbol alphabet.
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

from benchmark_linear_index_codec import error_stats  # noqa: E402
from probe_index_residual_context_entropy import (  # noqa: E402
    conditional_entropy_bits_fast,
    med_predict_indices,
    quantize_transform_indices,
    signed_residual_from_symbol,
)


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


def quality_passes(quality: dict[str, float]) -> bool:
    return (
        quality["signed_log2_rmse"] <= 0.004
        and quality["signed_log2_p99"] <= 0.018
        and quality["gradient_signed_log2_nrmse"] <= 0.5
    )


def small_escape_symbols(signed: np.ndarray, small: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return category symbols and separate positive/negative escape magnitudes.

    Category layout:
      0                  residual 0
      1..small           +1 .. +small
      small+1..2*small   -1 .. -small
      2*small+1          positive escape
      2*small+2          negative escape
    Escape detail stores abs(residual) - small - 1.
    """
    category = np.zeros(signed.shape, dtype=np.uint16)
    for value in range(1, small + 1):
        category[signed == value] = value
        category[signed == -value] = small + value
    pos_escape = signed > small
    neg_escape = signed < -small
    category[pos_escape] = 2 * small + 1
    category[neg_escape] = 2 * small + 2
    pos_values = (signed[pos_escape] - small - 1).astype(np.uint16)
    neg_values = ((-signed[neg_escape]) - small - 1).astype(np.uint16)
    return category, pos_values, neg_values


def category_contexts(
    category: np.ndarray,
    pred_bins: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    height, width, channels = category.shape
    alphabet = int(category.max()) + 1
    west = np.zeros(category.shape, dtype=np.uint32)
    north = np.zeros(category.shape, dtype=np.uint32)
    northwest = np.zeros(category.shape, dtype=np.uint32)
    prev_channel = np.zeros(category.shape, dtype=np.uint32)
    west[:, 1:, :] = category[:, :-1, :]
    north[1:, :, :] = category[:-1, :, :]
    northwest[1:, 1:, :] = category[:-1, :-1, :]
    if channels > 1:
        prev_channel[:, :, 1:] = category[:, :, :-1]
    yy, xx = np.indices((height, width), dtype=np.uint32)
    phase = (((yy & 1) << 1) | (xx & 1))[:, :, None]
    channel = np.arange(channels, dtype=np.uint32)[None, None, :]
    phase_channel = phase + 4 * channel
    contexts = {
        "order0": np.zeros(category.shape, dtype=np.uint16),
        "phase_channel": phase_channel,
        "west": west,
        "north": north,
        "west_north": west + alphabet * north,
        "west_north_prev_channel":
            west + alphabet * north + alphabet * alphabet * prev_channel,
        "wnnw_phase_channel": (
            west
            + alphabet * north
            + alphabet * alphabet * northwest
            + alphabet * alphabet * alphabet * phase_channel
        ),
    }
    if pred_bins is not None:
        pred_bins = pred_bins.astype(np.uint32, copy=False)
        pred_count = int(pred_bins.max()) + 1 if pred_bins.size else 1
        base = west + alphabet * north + alphabet * alphabet * prev_channel
        contexts["west_north_prev_predbin"] = (
            base + alphabet * alphabet * alphabet * pred_bins
        )
        contexts["west_north_prev_phase_predbin"] = (
            base
            + alphabet * alphabet * alphabet * phase_channel
            + alphabet * alphabet * alphabet * 16 * pred_bins
        )
        contexts["west_north_predbin"] = (
            west + alphabet * north + alphabet * alphabet * pred_bins
        )
        contexts["predbin_phase_channel"] = pred_bins + pred_count * phase_channel
    return contexts


def filtered_entropy(
    values: np.ndarray,
    contexts: np.ndarray,
    mask: np.ndarray,
    alphabet: int,
) -> float:
    if not bool(np.any(mask)):
        return 0.0
    return conditional_entropy_bits_fast(values[mask], contexts[mask], alphabet)


def summarize_model(
    residual_symbols: np.ndarray,
    bits: int,
    small: int,
    category_context_name: str,
    detail_context_name: str,
    pred_bins: np.ndarray | None = None,
) -> dict:
    signed = signed_residual_from_symbol(residual_symbols, bits)
    category, pos_values, neg_values = small_escape_symbols(signed, small)
    contexts = category_contexts(category, pred_bins)
    category_context = contexts[category_context_name]
    detail_context = contexts[detail_context_name]
    category_alphabet = 2 * small + 3
    # Two's-complement-like signed residuals have one extra negative value
    # (-128 for bits8), so reserve enough symbols for the wider side.
    detail_alphabet = max(1, (1 << (bits - 1)) - small)

    category_bits = conditional_entropy_bits_fast(
        category,
        category_context,
        category_alphabet,
    )
    pos_mask = category == 2 * small + 1
    neg_mask = category == 2 * small + 2
    pos_bits = filtered_entropy(
        signed.astype(np.int16) - small - 1,
        detail_context,
        pos_mask,
        detail_alphabet,
    )
    neg_bits = filtered_entropy(
        (-signed.astype(np.int16)) - small - 1,
        detail_context,
        neg_mask,
        detail_alphabet,
    )
    total_bits = category_bits + pos_bits + neg_bits
    sample_count = residual_symbols.size
    return {
        "small": small,
        "category_context": category_context_name,
        "detail_context": detail_context_name,
        "category_bits_per_sample": category_bits / sample_count,
        "pos_escape_bits_per_sample": pos_bits / sample_count,
        "neg_escape_bits_per_sample": neg_bits / sample_count,
        "bits_per_sample": total_bits / sample_count,
        "escape_rate": float((pos_mask.sum() + neg_mask.sum()) / sample_count),
        "pos_escape_rate": float(pos_mask.sum() / sample_count),
        "neg_escape_rate": float(neg_mask.sum() / sample_count),
    }


def summarize_image(
    path: Path,
    crop_size: int,
    bits: int,
    transform: str,
    gamma: float,
    small_values: tuple[int, ...],
    category_contexts_to_try: tuple[str, ...],
    detail_contexts_to_try: tuple[str, ...],
    pred_bin_bits: int,
) -> dict:
    pixels = read_exr(path, crop_size)
    indices, recon = quantize_transform_indices(pixels, bits, transform, gamma)
    alphabet = 1 << bits
    pred = med_predict_indices(indices, bits)
    residual = (
        indices.astype(np.int32) + alphabet - pred
    ) & (alphabet - 1)
    residual = residual.astype(np.uint16)
    pred_bins = None
    if pred_bin_bits > 0:
        shift = max(0, bits - pred_bin_bits)
        pred_bins = (pred.astype(np.uint16) >> np.uint16(shift)).astype(np.uint16)
    full_context_category, _pos, _neg = small_escape_symbols(
        signed_residual_from_symbol(residual, bits),
        1,
    )
    full_contexts = category_contexts(full_context_category, pred_bins)
    full_rows = []
    for context_name, context in full_contexts.items():
        full_bits = conditional_entropy_bits_fast(residual, context, alphabet)
        full_rows.append({
            "context": context_name,
            "bits_per_sample": full_bits / residual.size,
        })
    full_rows.sort(key=lambda row: row["bits_per_sample"])

    rows = []
    for small in small_values:
        for category_context_name in category_contexts_to_try:
            for detail_context_name in detail_contexts_to_try:
                rows.append(
                    summarize_model(
                        residual,
                        bits,
                        small,
                        category_context_name,
                        detail_context_name,
                        pred_bins,
                    )
                )
    for row in rows:
        estimated_bytes = row["bits_per_sample"] * pixels.size / 8.0 + 73.0
        row["estimated_bytes"] = estimated_bytes
        row["estimated_ratio_vs_original"] = pixels.nbytes / estimated_bytes
    rows.sort(key=lambda row: row["bits_per_sample"])

    quality = error_stats(pixels, recon)
    return {
        "image": path.name,
        "shape": list(pixels.shape),
        "crop_size": crop_size,
        "bits": bits,
        "transform": transform,
        "gamma": gamma,
        "pred_bin_bits": pred_bin_bits,
        "quality": quality,
        "quality_passes": quality_passes(quality),
        "full_symbol_contexts": full_rows,
        "rows": rows,
    }


def parse_ints(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split(",") if part.strip())


def parse_strings(text: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in text.split(",") if part.strip())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", default="sample_DSCF0009.EXR")
    parser.add_argument("--crop-size", type=int, default=1024)
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--transform", choices=("power", "signed-log", "asinh"), default="power")
    parser.add_argument("--gamma", type=float, default=0.795)
    parser.add_argument("--small", default="1,2,3,4,7")
    parser.add_argument("--pred-bin-bits", type=int, default=0)
    parser.add_argument(
        "--category-contexts",
        default="west_north,west_north_prev_channel,wnnw_phase_channel",
    )
    parser.add_argument(
        "--detail-contexts",
        default="order0,phase_channel,west_north,west_north_prev_channel",
    )
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
            parse_ints(args.small),
            parse_strings(args.category_contexts),
            parse_strings(args.detail_contexts),
            args.pred_bin_bits,
        )
        summaries.append(summary)
        q = summary["quality"]
        print(path.name)
        print(
            f"  quality {'PASS' if summary['quality_passes'] else 'fail'} "
            f"log={q['signed_log2_rmse']:.3e} "
            f"p99={q['signed_log2_p99']:.3e} "
            f"grad={q['gradient_signed_log2_nrmse']:.3e}"
        )
        print("  full-symbol best:")
        for row in summary["full_symbol_contexts"][:4]:
            bytes_est = row["bits_per_sample"] * np.prod(summary["shape"]) / 8.0 + 73.0
            ratio = (np.prod(summary["shape"]) * 4.0) / bytes_est
            print(
                f"    {row['context']:24s} "
                f"bps={row['bits_per_sample']:.3f} "
                f"bytes={bytes_est:,.0f} ratio={ratio:.2f}x"
            )
        print("  small+escape best:")
        for row in summary["rows"][:12]:
            print(
                f"    small={row['small']} "
                f"cat={row['category_context']} "
                f"detail={row['detail_context']} "
                f"bps={row['bits_per_sample']:.3f} "
                f"bytes={row['estimated_bytes']:,.0f} "
                f"ratio={row['estimated_ratio_vs_original']:.2f}x "
                f"escape={row['escape_rate']:.2%}"
            )
        if args.limit and len(summaries) >= args.limit:
            break

    if not args.no_save:
        safe_glob = args.glob.replace("*", "star").replace(".", "_")
        output = RESULTS_DIR / (
            f"small_residual_escape_entropy_{safe_glob}"
            f"_crop{args.crop_size}_{args.transform}"
            f"_gamma{args.gamma:g}_bits{args.bits}.json"
        )
        output.write_text(json.dumps(summaries, indent=2))
        print(f"\nSaved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
