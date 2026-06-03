"""Estimate production-ish payload budgets for small residual + escape coding.

The earlier entropy probes reported an ideal lower bound for coding transform
index residuals directly.  This script keeps the reconstruction identical to
StageLinearIndex for a given transform/bits pair, then estimates the real cost
of replacing the current mask+value stream with:

  category stream: 0, +/-1..+/-small, positive escape, negative escape
  detail streams:  abs(residual) - small - 1 for each escape sign

For each stream it includes:
  * finite rANS frequencies quantized to PROB_SCALE=16384
  * rANS flush bytes
  * several simple model-table storage budgets

It is still a budget, not a bitstream implementation, but it is deliberately
closer to an implementation decision than a pure entropy number.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import OpenImageIO as oiio

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
sys.path.insert(0, str(ROOT / "scripts"))

from probe_index_residual_context_entropy import (  # noqa: E402
    med_predict_indices,
    transform_values,
)


PROB_BITS = 14
PROB_SCALE = 1 << PROB_BITS
BASE_STREAM_BYTES = 73 + 32


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


def quantize_channel(values: np.ndarray, bits: int, transform: str, gamma: float) -> np.ndarray:
    """Match StageLinearIndex's per-channel transform index quantization."""
    levels = (1 << bits) - 1
    transformed = transform_values(values.astype(np.float64), transform, gamma)
    range_values = transformed.astype(np.float32).astype(np.float64)
    lo = float(np.min(range_values))
    hi = float(np.max(range_values))
    if not hi > lo:
        return np.zeros(values.shape, dtype=np.uint16)
    q = np.floor((transformed - lo) / (hi - lo) * levels + 0.5)
    return np.clip(q, 0, levels).astype(np.uint16)


def signed_residual(indices: np.ndarray, bits: int) -> np.ndarray:
    alphabet = 1 << bits
    pred = med_predict_indices(indices[:, :, None], bits)[:, :, 0]
    residual = (indices.astype(np.int32) + alphabet - pred) & (alphabet - 1)
    signed = residual.astype(np.int16)
    signed[signed >= (alphabet // 2)] -= alphabet
    return signed


def residual_categories(signed: np.ndarray, small: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    category = np.zeros(signed.shape, dtype=np.uint16)
    abs_signed = np.abs(signed).astype(np.uint16)
    pos_small = (signed > 0) & (signed <= small)
    neg_small = (signed < 0) & (signed >= -small)
    category[pos_small] = abs_signed[pos_small]
    category[neg_small] = small + abs_signed[neg_small]

    pos_escape = signed > small
    neg_escape = signed < -small
    category[pos_escape] = 2 * small + 1
    category[neg_escape] = 2 * small + 2
    return category, pos_escape, neg_escape


def context_count_for(name: str, alphabet: int, channels: int) -> int:
    if name == "order0":
        return 1
    if name in {"west", "north"}:
        return alphabet
    if name == "west_north":
        return alphabet * alphabet
    if name == "west_north_prev_channel":
        return alphabet * alphabet * alphabet
    if name == "phase_channel":
        return 4 * channels
    if name == "west_north_prev_current_channel":
        return alphabet * alphabet * alphabet * channels
    if name == "west_north_prev_phase2":
        return alphabet * alphabet * alphabet * 4
    if name == "west_north_prev_phase2_channel":
        return alphabet * alphabet * alphabet * 4 * channels
    if name == "west_north_prev_xtrans6":
        return alphabet * alphabet * alphabet * 36
    if name == "west_north_prev_xtrans6_channel":
        return alphabet * alphabet * alphabet * 36 * channels
    raise ValueError(f"unsupported context: {name}")


def category_context_2d(
    category: np.ndarray,
    prev_category: np.ndarray | None,
    name: str,
    alphabet: int,
    channel: int,
    channels: int,
) -> np.ndarray:
    if name == "order0":
        return np.zeros(category.shape, dtype=np.uint32)

    if name == "phase_channel":
        height, width = category.shape
        yy, xx = np.indices((height, width), dtype=np.uint32)
        return (((yy & 1) << 1) | (xx & 1)) + 4 * channel

    west = np.zeros(category.shape, dtype=np.uint32)
    west[:, 1:] = category[:, :-1]
    if name == "west":
        return west

    north = np.zeros(category.shape, dtype=np.uint32)
    north[1:, :] = category[:-1, :]
    if name == "north":
        return north
    if name == "west_north":
        return west + alphabet * north
    if prev_category is None:
        prev = np.zeros(category.shape, dtype=np.uint32)
    else:
        prev = prev_category.astype(np.uint32, copy=False)
    base = west + alphabet * north + alphabet * alphabet * prev
    if name == "west_north_prev_channel":
        return base
    if name == "west_north_prev_current_channel":
        return base + alphabet * alphabet * alphabet * channel
    if name in {
        "west_north_prev_phase2",
        "west_north_prev_phase2_channel",
        "west_north_prev_xtrans6",
        "west_north_prev_xtrans6_channel",
    }:
        height, width = category.shape
        yy, xx = np.indices((height, width), dtype=np.uint32)
        if name.startswith("west_north_prev_phase2"):
            phase = ((yy & 1) << 1) | (xx & 1)
            phase_count = 4
        else:
            phase = (yy % 6) * 6 + (xx % 6)
            phase_count = 36
        context = base + alphabet * alphabet * alphabet * phase
        if name.endswith("_channel"):
            context = (
                context
                + alphabet * alphabet * alphabet * phase_count * channel
            )
        return context
    raise ValueError(f"unsupported context: {name}")


def add_joint(
    joint: np.ndarray,
    symbols: np.ndarray,
    contexts: np.ndarray,
    alphabet: int,
    mask: np.ndarray | None = None,
) -> None:
    if mask is not None:
        if not bool(np.any(mask)):
            return
        symbols = symbols[mask]
        contexts = contexts[mask]
    flat = contexts.reshape(-1).astype(np.int64) * alphabet
    flat += symbols.reshape(-1).astype(np.int64)
    counts = np.bincount(flat, minlength=joint.size).reshape(joint.shape)
    joint += counts


def build_quantized_freq(counts: np.ndarray) -> np.ndarray:
    total = int(counts.sum())
    if total <= 0:
        return np.zeros(counts.shape, dtype=np.uint16)
    seen = counts > 0
    freqs = np.zeros(counts.shape, dtype=np.int64)
    raw = (counts[seen] * PROB_SCALE + total // 2) // total
    raw = np.clip(raw, 1, PROB_SCALE - 1)
    freqs[seen] = raw
    assigned = int(freqs.sum())

    while assigned > PROB_SCALE:
        best = int(np.argmax(freqs))
        if freqs[best] <= 1:
            raise RuntimeError("cannot normalize frequency table")
        freqs[best] -= 1
        assigned -= 1
    while assigned < PROB_SCALE:
        best = int(np.argmax(freqs))
        freqs[best] += 1
        assigned += 1
    return freqs.astype(np.uint16)


def ideal_entropy_bits_for_counts(counts: np.ndarray) -> float:
    total = float(counts.sum())
    if total <= 0.0:
        return 0.0
    nz = counts[counts > 0].astype(np.float64)
    return float(-(nz * np.log2(nz / total)).sum())


def summarize_joint(joint: np.ndarray, alphabet: int, stream_name: str) -> dict:
    context_totals = joint.sum(axis=1)
    nonempty_contexts = np.flatnonzero(context_totals)
    sample_count = int(context_totals.sum())
    if sample_count == 0:
        return {
            "stream": stream_name,
            "sample_count": 0,
            "ideal_bits": 0.0,
            "finite_bits": 0.0,
            "payload_bytes": 0,
            "model_dense_bytes": 0,
            "model_sparse_full_bytes": 0,
            "model_sparse_pairs_bytes": 0,
            "context_count_possible": int(joint.shape[0]),
            "context_count_seen": 0,
            "symbol_pairs_seen": 0,
        }

    ideal_bits = 0.0
    finite_bits = 0.0
    symbol_pairs_seen = 0
    for cid in nonempty_contexts:
        counts = joint[cid]
        ideal_bits += ideal_entropy_bits_for_counts(counts)
        freqs = build_quantized_freq(counts)
        nz = counts > 0
        symbol_pairs_seen += int(np.count_nonzero(nz))
        finite_bits += float(
            (counts[nz].astype(np.float64)
             * (PROB_BITS - np.log2(freqs[nz].astype(np.float64)))).sum()
        )

    context_id_bytes = 2 if joint.shape[0] <= 65536 else 4
    symbol_id_bytes = 1 if alphabet <= 256 else 2
    context_count_seen = int(nonempty_contexts.size)
    payload_bytes = int(math.ceil(finite_bits / 8.0)) + 4
    dense_model_bytes = int(joint.shape[0] * alphabet * 2)
    sparse_full_bytes = int(context_count_seen * (context_id_bytes + alphabet * 2))
    sparse_pairs_bytes = int(
        context_count_seen * (context_id_bytes + 2)
        + symbol_pairs_seen * (symbol_id_bytes + 2)
    )
    return {
        "stream": stream_name,
        "sample_count": sample_count,
        "ideal_bits": ideal_bits,
        "finite_bits": finite_bits,
        "finite_redundancy_bits": finite_bits - ideal_bits,
        "bits_per_sample": finite_bits / sample_count,
        "payload_bytes": payload_bytes,
        "model_dense_bytes": dense_model_bytes,
        "model_sparse_full_bytes": sparse_full_bytes,
        "model_sparse_pairs_bytes": sparse_pairs_bytes,
        "context_count_possible": int(joint.shape[0]),
        "context_count_seen": context_count_seen,
        "symbol_pairs_seen": symbol_pairs_seen,
    }


def summarize_total(streams: list[dict], raw_bytes: int, sample_count: int) -> dict:
    ideal_bits = sum(float(row["ideal_bits"]) for row in streams)
    finite_bits = sum(float(row["finite_bits"]) for row in streams)
    payload_bytes = sum(int(row["payload_bytes"]) for row in streams)
    dense_model = sum(int(row["model_dense_bytes"]) for row in streams)
    sparse_full_model = sum(int(row["model_sparse_full_bytes"]) for row in streams)
    sparse_pairs_model = sum(int(row["model_sparse_pairs_bytes"]) for row in streams)

    def total_with(model_bytes: int) -> int:
        return BASE_STREAM_BYTES + payload_bytes + model_bytes

    dense_total = total_with(dense_model)
    sparse_full_total = total_with(sparse_full_model)
    sparse_pairs_total = total_with(sparse_pairs_model)
    return {
        "base_stream_bytes": BASE_STREAM_BYTES,
        "ideal_entropy_bytes_no_model": int(math.ceil(ideal_bits / 8.0)) + BASE_STREAM_BYTES,
        "finite_payload_bytes": payload_bytes,
        "finite_bits_per_sample": finite_bits / sample_count,
        "model_dense_bytes": dense_model,
        "model_sparse_full_bytes": sparse_full_model,
        "model_sparse_pairs_bytes": sparse_pairs_model,
        "total_dense_model_bytes": dense_total,
        "total_sparse_full_model_bytes": sparse_full_total,
        "total_sparse_pairs_model_bytes": sparse_pairs_total,
        "ratio_dense_model": raw_bytes / dense_total,
        "ratio_sparse_full_model": raw_bytes / sparse_full_total,
        "ratio_sparse_pairs_model": raw_bytes / sparse_pairs_total,
    }


def summarize_image(
    path: Path,
    crop_size: int,
    bits: int,
    transform: str,
    gamma: float,
    small: int,
    category_context_name: str,
    detail_context_names: tuple[str, ...],
) -> dict:
    started = time.perf_counter()
    pixels = read_exr(path, crop_size)
    height, width, channels = pixels.shape
    raw_bytes = int(pixels.nbytes)
    sample_count = int(pixels.size)
    category_alphabet = 2 * small + 3
    detail_alphabet = max(1, (1 << (bits - 1)) - small)

    category_joint = np.zeros(
        (context_count_for(category_context_name, category_alphabet, channels),
         category_alphabet),
        dtype=np.int64,
    )
    detail_joints: dict[str, dict[str, np.ndarray]] = {}
    for context_name in detail_context_names:
        context_count = context_count_for(context_name, category_alphabet, channels)
        detail_joints[context_name] = {
            "pos": np.zeros((context_count, detail_alphabet), dtype=np.int64),
            "neg": np.zeros((context_count, detail_alphabet), dtype=np.int64),
        }

    prev_category: np.ndarray | None = None
    escape_count = 0
    for channel in range(channels):
        print(f"  channel {channel + 1}/{channels}: quantize/predict/count", flush=True)
        indices = quantize_channel(pixels[:, :, channel], bits, transform, gamma)
        signed = signed_residual(indices, bits)
        category, pos_escape, neg_escape = residual_categories(signed, small)
        escape_count += int(np.count_nonzero(pos_escape) + np.count_nonzero(neg_escape))

        category_context = category_context_2d(
            category,
            prev_category,
            category_context_name,
            category_alphabet,
            channel,
            channels,
        )
        add_joint(category_joint, category, category_context, category_alphabet)

        for context_name, joints in detail_joints.items():
            detail_context = category_context_2d(
                category,
                prev_category,
                context_name,
                category_alphabet,
                channel,
                channels,
            )
            pos_detail = (signed.astype(np.int32) - small - 1).astype(np.int32)
            neg_detail = ((-signed.astype(np.int32)) - small - 1).astype(np.int32)
            add_joint(
                joints["pos"],
                pos_detail.astype(np.uint16, copy=False),
                detail_context,
                detail_alphabet,
                pos_escape,
            )
            add_joint(
                joints["neg"],
                neg_detail.astype(np.uint16, copy=False),
                detail_context,
                detail_alphabet,
                neg_escape,
            )
        prev_category = category

    category_summary = summarize_joint(
        category_joint,
        category_alphabet,
        f"category:{category_context_name}",
    )
    detail_rows = []
    candidates = []
    for context_name, joints in detail_joints.items():
        streams = [
            category_summary,
            summarize_joint(joints["pos"], detail_alphabet, f"pos_escape:{context_name}"),
            summarize_joint(joints["neg"], detail_alphabet, f"neg_escape:{context_name}"),
        ]
        total = summarize_total(streams, raw_bytes, sample_count)
        detail_rows.append({
            "detail_context": context_name,
            "streams": streams,
            "total": total,
        })
        candidates.append({
            "detail_context": context_name,
            **total,
        })
    candidates.sort(key=lambda row: row["total_sparse_pairs_model_bytes"])

    return {
        "image": path.name,
        "shape": [height, width, channels],
        "crop_size": crop_size,
        "raw_bytes": raw_bytes,
        "sample_count": sample_count,
        "bits": bits,
        "transform": transform,
        "gamma": gamma,
        "small": small,
        "category_context": category_context_name,
        "category_alphabet": category_alphabet,
        "detail_alphabet": detail_alphabet,
        "escape_rate": escape_count / sample_count,
        "elapsed_sec": time.perf_counter() - started,
        "detail_context_results": detail_rows,
        "candidates": candidates,
    }


def parse_strings(text: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in text.split(",") if part.strip())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", default="sample_DSCF0009.EXR")
    parser.add_argument("--crop-size", type=int, default=1024)
    parser.add_argument("--bits", type=int, default=10)
    parser.add_argument("--transform", choices=("linear", "power", "signed-log", "asinh"), default="signed-log")
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--small", type=int, default=15)
    parser.add_argument("--category-context", default="west_north_prev_channel")
    parser.add_argument("--detail-contexts", default="order0,west_north_prev_channel")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    summaries = []
    for path in sorted(DATA_DIR.glob(args.glob)):
        print(path.name, flush=True)
        summary = summarize_image(
            path,
            args.crop_size,
            args.bits,
            args.transform,
            args.gamma,
            args.small,
            args.category_context,
            parse_strings(args.detail_contexts),
        )
        summaries.append(summary)
        print(
            f"  escape={summary['escape_rate']:.2%} "
            f"elapsed={summary['elapsed_sec']:.1f}s",
            flush=True,
        )
        print("  candidates:", flush=True)
        for row in summary["candidates"]:
            print(
                f"    detail={row['detail_context']:24s} "
                f"payload={row['finite_payload_bytes']:,.0f} "
                f"model_pairs={row['model_sparse_pairs_bytes']:,.0f} "
                f"total_pairs={row['total_sparse_pairs_model_bytes']:,.0f} "
                f"ratio={row['ratio_sparse_pairs_model']:.2f}x "
                f"ideal_no_model={row['ideal_entropy_bytes_no_model']:,.0f}",
                flush=True,
            )
        if args.limit and len(summaries) >= args.limit:
            break

    if not args.no_save:
        safe_glob = args.glob.replace("*", "star").replace(".", "_")
        safe_category_context = args.category_context.replace(",", "-")
        safe_detail_contexts = args.detail_contexts.replace(",", "-")
        output = RESULTS_DIR / (
            f"small_escape_payload_budget_{safe_glob}"
            f"_crop{args.crop_size}_{args.transform}"
            f"_bits{args.bits}_small{args.small}"
            f"_cat{safe_category_context}_detail{safe_detail_contexts}.json"
        )
        output.write_text(json.dumps(summaries, indent=2))
        print(f"\nSaved: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
