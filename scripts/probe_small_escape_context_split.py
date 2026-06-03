"""Probe fallback-tree contexts for small residual category coding.

Directly multiplying a base context by extra axes (channel, phase, X-Trans
phase) often lowers entropy but loses to model-table cost.  This probe asks a
more practical question: for each base context, should we keep one parent model
or split that parent by an extra decoder-visible axis?

The reconstruction is still unchanged: signed-log bits10 indices, MED
prediction, small residual categories, and order-0 escape details.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
sys.path.insert(0, str(ROOT / "scripts"))

from probe_small_escape_payload_budget import (  # noqa: E402
    BASE_STREAM_BYTES,
    PROB_BITS,
    PROB_SCALE,
    add_joint,
    build_quantized_freq,
    category_context_2d,
    context_count_for,
    quantize_channel,
    read_exr,
    residual_categories,
    signed_residual,
)


def finite_bits_for_counts(counts: np.ndarray) -> float:
    total = int(counts.sum())
    if total <= 0:
        return 0.0
    freqs = build_quantized_freq(counts)
    nz = counts > 0
    return float(
        (counts[nz].astype(np.float64)
         * (PROB_BITS - np.log2(freqs[nz].astype(np.float64)))).sum()
    )


def parent_model_bytes(counts: np.ndarray) -> int:
    if int(counts.sum()) == 0:
        return 0
    # Existing context model format is: context u16 + seen u8 + (symbol u8,
    # freq u16) for each non-zero symbol.
    return 3 + 3 * int(np.count_nonzero(counts))


def split_model_bytes(child_counts: np.ndarray) -> int:
    nonempty_children = np.flatnonzero(child_counts.sum(axis=1))
    if nonempty_children.size == 0:
        return 0
    # Efficient split-tree estimate: parent id u16 + child-count u8, then
    # child id u8 + seen u8 + sparse symbol/freq pairs for each child model.
    total = 3
    for child in nonempty_children:
        counts = child_counts[child]
        total += 2 + 3 * int(np.count_nonzero(counts))
    return total


def dense_order0_model_bytes(counts: np.ndarray) -> int:
    return counts.size * 2 if int(counts.sum()) > 0 else 0


def detail_summary(hist: np.ndarray, name: str) -> dict:
    bits = finite_bits_for_counts(hist)
    payload_bytes = int(math.ceil(bits / 8.0)) + (4 if int(hist.sum()) else 0)
    return {
        "stream": name,
        "sample_count": int(hist.sum()),
        "finite_bits": bits,
        "payload_bytes": payload_bytes,
        "model_bytes": dense_order0_model_bytes(hist),
        "symbol_pairs_seen": int(np.count_nonzero(hist)),
    }


def axis_values(shape: tuple[int, int], channel: int, axis: str) -> tuple[np.ndarray, int]:
    height, width = shape
    if axis == "channel":
        return np.full((height, width), channel, dtype=np.uint32), 3
    yy, xx = np.indices((height, width), dtype=np.uint32)
    if axis == "phase2":
        return (((yy & 1) << 1) | (xx & 1)).astype(np.uint32), 4
    if axis == "phase2_channel":
        return ((((yy & 1) << 1) | (xx & 1)) + 4 * channel).astype(np.uint32), 12
    if axis == "xtrans6":
        return ((yy % 6) * 6 + (xx % 6)).astype(np.uint32), 36
    if axis == "xtrans6_channel":
        return (((yy % 6) * 6 + (xx % 6)) + 36 * channel).astype(np.uint32), 108
    raise ValueError(f"unsupported split axis: {axis}")


def choose_split_models(
    parent_joint: np.ndarray,
    child_joint: np.ndarray,
) -> dict:
    parent_count, alphabet = parent_joint.shape
    axis_count = child_joint.shape[0] // parent_count
    chosen_bits = 0.0
    chosen_model_bytes = 4  # parent-model-count u16 + split-parent-count u16
    baseline_bits = 0.0
    baseline_model_bytes = 2  # seen-context count u16
    split_parent_count = 0
    split_child_model_count = 0
    fallback_parent_count = 0
    gain_bits = 0.0
    extra_model_bytes = 0

    for parent in range(parent_count):
        parent_counts = parent_joint[parent]
        if int(parent_counts.sum()) == 0:
            continue
        children = child_joint[
            parent * axis_count:(parent + 1) * axis_count,
            :,
        ]
        parent_bits = finite_bits_for_counts(parent_counts)
        parent_model = parent_model_bytes(parent_counts)
        child_bits = 0.0
        for child in range(axis_count):
            child_bits += finite_bits_for_counts(children[child])
        child_model = split_model_bytes(children)

        baseline_bits += parent_bits
        baseline_model_bytes += parent_model

        parent_cost_bits = parent_bits + 8.0 * parent_model
        split_cost_bits = child_bits + 8.0 * child_model
        if split_cost_bits < parent_cost_bits:
            chosen_bits += child_bits
            chosen_model_bytes += child_model
            split_parent_count += 1
            split_child_model_count += int(np.count_nonzero(children.sum(axis=1)))
            gain_bits += parent_bits - child_bits
            extra_model_bytes += child_model - parent_model
        else:
            chosen_bits += parent_bits
            chosen_model_bytes += parent_model
            fallback_parent_count += 1

    return {
        "alphabet": alphabet,
        "axis_count": axis_count,
        "baseline_bits": baseline_bits,
        "baseline_model_bytes": baseline_model_bytes,
        "chosen_bits": chosen_bits,
        "chosen_model_bytes": chosen_model_bytes,
        "payload_bytes": int(math.ceil(chosen_bits / 8.0)) + 4,
        "baseline_payload_bytes": int(math.ceil(baseline_bits / 8.0)) + 4,
        "split_parent_count": split_parent_count,
        "split_child_model_count": split_child_model_count,
        "fallback_parent_count": fallback_parent_count,
        "gain_bits_before_model": gain_bits,
        "extra_model_bytes_for_splits": extra_model_bytes,
    }


def summarize_image(
    path: Path,
    crop_size: int,
    bits: int,
    transform: str,
    gamma: float,
    small: int,
    axis: str,
) -> dict:
    started = time.perf_counter()
    pixels = read_exr(path, crop_size)
    height, width, channels = pixels.shape
    category_alphabet = 2 * small + 3
    base_context_name = "west_north_prev_channel"
    base_count = context_count_for(base_context_name, category_alphabet, channels)

    # Ask axis_values once for the count.  The channel value does not matter
    # for phase-only axes.
    _, axis_count = axis_values((height, width), 0, axis)
    parent_joint = np.zeros((base_count, category_alphabet), dtype=np.int64)
    child_joint = np.zeros((base_count * axis_count, category_alphabet), dtype=np.int64)

    detail_alphabet = max(1, (1 << (bits - 1)) - small)
    pos_hist = np.zeros(detail_alphabet, dtype=np.int64)
    neg_hist = np.zeros(detail_alphabet, dtype=np.int64)
    escape_count = 0
    prev_category: np.ndarray | None = None

    for channel in range(channels):
        print(f"  channel {channel + 1}/{channels}: quantize/predict/count", flush=True)
        indices = quantize_channel(pixels[:, :, channel], bits, transform, gamma)
        signed = signed_residual(indices, bits)
        category, pos_escape, neg_escape = residual_categories(signed, small)
        escape_count += int(np.count_nonzero(pos_escape) + np.count_nonzero(neg_escape))

        base_context = category_context_2d(
            category,
            prev_category,
            base_context_name,
            category_alphabet,
            channel,
            channels,
        )
        axis_value, _axis_count = axis_values((height, width), channel, axis)
        add_joint(parent_joint, category, base_context, category_alphabet)
        add_joint(
            child_joint,
            category,
            base_context * axis_count + axis_value,
            category_alphabet,
        )

        if bool(np.any(pos_escape)):
            pos_detail = signed.astype(np.int32) - small - 1
            pos_hist += np.bincount(
                pos_detail[pos_escape].astype(np.int64),
                minlength=detail_alphabet,
            )
        if bool(np.any(neg_escape)):
            neg_detail = (-signed.astype(np.int32)) - small - 1
            neg_hist += np.bincount(
                neg_detail[neg_escape].astype(np.int64),
                minlength=detail_alphabet,
            )
        prev_category = category

    category = choose_split_models(parent_joint, child_joint)
    pos_detail = detail_summary(pos_hist, "pos_escape:order0")
    neg_detail = detail_summary(neg_hist, "neg_escape:order0")
    detail_payload_bytes = 4 + pos_detail["payload_bytes"] + neg_detail["payload_bytes"]
    detail_model_bytes = pos_detail["model_bytes"] + neg_detail["model_bytes"]
    total_bytes = (
        BASE_STREAM_BYTES
        + category["payload_bytes"]
        + category["chosen_model_bytes"]
        + detail_payload_bytes
        + detail_model_bytes
    )
    baseline_total_bytes = (
        BASE_STREAM_BYTES
        + category["baseline_payload_bytes"]
        + category["baseline_model_bytes"]
        + detail_payload_bytes
        + detail_model_bytes
    )
    raw_bytes = int(pixels.nbytes)

    return {
        "image": path.name,
        "shape": [height, width, channels],
        "crop_size": crop_size,
        "raw_bytes": raw_bytes,
        "bits": bits,
        "transform": transform,
        "gamma": gamma,
        "small": small,
        "split_axis": axis,
        "escape_rate": escape_count / int(pixels.size),
        "category": category,
        "detail": {
            "payload_bytes": detail_payload_bytes,
            "model_bytes": detail_model_bytes,
            "pos": pos_detail,
            "neg": neg_detail,
        },
        "total_bytes": total_bytes,
        "ratio_vs_original": raw_bytes / total_bytes,
        "baseline_total_bytes_same_cost_model": baseline_total_bytes,
        "baseline_ratio_vs_original": raw_bytes / baseline_total_bytes,
        "delta_vs_baseline_bytes": total_bytes - baseline_total_bytes,
        "elapsed_sec": time.perf_counter() - started,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", default="sample_DSCF0009.EXR")
    parser.add_argument("--crop-size", type=int, default=1024)
    parser.add_argument("--bits", type=int, default=10)
    parser.add_argument("--transform", default="signed-log")
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--small", type=int, default=7)
    parser.add_argument(
        "--axis",
        choices=("channel", "phase2", "phase2_channel", "xtrans6", "xtrans6_channel"),
        default="xtrans6",
    )
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
            args.axis,
        )
        summaries.append(summary)
        cat = summary["category"]
        print(
            f"  axis={args.axis} total={summary['total_bytes']:,.0f} "
            f"ratio={summary['ratio_vs_original']:.2f}x "
            f"delta={summary['delta_vs_baseline_bytes']:+,.0f} "
            f"splits={cat['split_parent_count']} "
            f"children={cat['split_child_model_count']} "
            f"elapsed={summary['elapsed_sec']:.1f}s",
            flush=True,
        )
        print(
            f"  category payload={cat['payload_bytes']:,.0f} "
            f"model={cat['chosen_model_bytes']:,.0f} "
            f"gain_bits={cat['gain_bits_before_model'] / 8.0:,.0f}B "
            f"extra_model={cat['extra_model_bytes_for_splits']:,.0f}B",
            flush=True,
        )
        if args.limit and len(summaries) >= args.limit:
            break

    if not args.no_save:
        safe_glob = args.glob.replace("*", "star").replace(".", "_")
        output = RESULTS_DIR / (
            f"small_escape_context_split_{safe_glob}"
            f"_crop{args.crop_size}_{args.transform}"
            f"_bits{args.bits}_small{args.small}_axis{args.axis}.json"
        )
        output.write_text(json.dumps(summaries, indent=2))
        print(f"\nSaved: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
