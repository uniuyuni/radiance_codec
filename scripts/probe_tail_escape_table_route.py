"""Probe practical top-K escape tables for hard float low tails.

Conditional entropy probes can show very low in-sample lower bounds because
they assume the decoder magically knows each context's symbol distribution.
This script pays for a smaller, explicit approximation: for each
decoder-visible context, send a top-K table of frequent tail symbols and encode
everything else as an escape/raw value.

It is still an estimate, but it asks a useful next question: can the
hash/context lower bound survive even a modest amount of side information?
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_structural_predictors import DATA_DIR, float_to_bits, read_exr  # noqa: E402
from evaluate_structural_delta_context import bits_to_ordered  # noqa: E402
from probe_grouped_tail_mlp import CONTEXT_FAMILIES, context_family_cost, kt_cost  # noqa: E402
from probe_tail_conditional_entropy import (  # noqa: E402
    conditional_symbol_entropy,
    context_variants,
)
from structural_delta_grouped_roundtrip import choose_group_payloads  # noqa: E402

RESULTS_DIR = ROOT / "results"


def entropy_bits_from_counts(counts: np.ndarray) -> float:
    total = float(np.sum(counts))
    if total == 0.0:
        return 0.0
    probs = counts.astype(np.float64) / total
    return float(-np.sum(probs * np.log2(probs)) * total)


def best_gdx_low_cost(payload: np.ndarray, width: int) -> float:
    total = 0.0
    for bit in range(width):
        total += min(
            context_family_cost(payload, bit, family)
            for family in CONTEXT_FAMILIES
        )
    return total


def tail_from_source(
    bits: np.ndarray,
    ordered: np.ndarray,
    body_payload: np.ndarray,
    tail_width: int,
    tail_source: str,
) -> np.ndarray:
    mask = np.uint32((1 << tail_width) - 1)
    if tail_source == "raw_mantissa":
        return (bits & np.uint32(0x7FFFFF)) & mask
    if tail_source == "body_payload":
        return body_payload & mask
    if tail_source == "ordered_low":
        return ordered & mask
    raise ValueError(f"unknown tail source: {tail_source}")


def topk_context_cost(
    tail: np.ndarray,
    context: np.ndarray,
    tail_width: int,
    k_values: tuple[int, ...],
    route_header_bits: int,
    context_header_bits: int,
) -> tuple[float, dict[str, float]]:
    flat_tail = tail.reshape(-1).astype(np.uint64)
    flat_context = context.reshape(-1).astype(np.uint64)
    order = np.argsort(flat_context, kind="stable")
    sorted_context = flat_context[order]
    sorted_tail = flat_tail[order]

    total_cost = float(route_header_bits)
    raw_total = float(flat_tail.size * tail_width)
    table_contexts = 0
    raw_contexts = 0
    dictionary_values = 0
    escaped_values = 0
    best_k_histogram: Counter[str] = Counter()

    start = 0
    while start < sorted_context.size:
        end = start + 1
        while end < sorted_context.size and sorted_context[end] == sorted_context[start]:
            end += 1
        values, counts = np.unique(sorted_tail[start:end], return_counts=True)
        sample_count = int(np.sum(counts))
        context_raw_cost = float(sample_count * tail_width)
        best_cost = context_raw_cost
        best_k = 0
        best_dict_size = 0
        best_escapes = sample_count

        count_order = np.argsort(counts)[::-1]
        for requested_k in k_values:
            k = min(int(requested_k), int(values.size))
            if k <= 0:
                continue
            selected_counts = counts[count_order[:k]]
            hits = int(np.sum(selected_counts))
            misses = sample_count - hits
            flag_cost = kt_cost(hits, sample_count)
            index_cost = entropy_bits_from_counts(selected_counts)
            dictionary_cost = k * tail_width
            candidate_cost = (
                context_header_bits
                + dictionary_cost
                + flag_cost
                + index_cost
                + misses * tail_width
            )
            if candidate_cost < best_cost:
                best_cost = candidate_cost
                best_k = k
                best_dict_size = k
                best_escapes = misses

        total_cost += best_cost
        if best_k:
            table_contexts += 1
            dictionary_values += best_dict_size
            escaped_values += best_escapes
            best_k_histogram[str(best_k)] += 1
        else:
            raw_contexts += 1
            best_k_histogram["raw"] += 1
        start = end

    return total_cost, {
        "raw_bits": raw_total,
        "table_contexts": float(table_contexts),
        "raw_contexts": float(raw_contexts),
        "dictionary_values": float(dictionary_values),
        "escaped_values": float(escaped_values),
        "best_k_histogram": dict(best_k_histogram),
    }


def global_topk_cost(
    tail: np.ndarray,
    tail_width: int,
    k_values: tuple[int, ...],
    route_header_bits: int,
    table_header_bits: int,
) -> tuple[float, dict[str, float]]:
    flat_tail = tail.reshape(-1).astype(np.uint64)
    values, counts = np.unique(flat_tail, return_counts=True)
    raw_cost = float(flat_tail.size * tail_width)
    best_cost = raw_cost
    best_k = 0
    best_escapes = int(flat_tail.size)
    order = np.argsort(counts)[::-1]
    for requested_k in k_values:
        k = min(int(requested_k), int(values.size))
        if k <= 0:
            continue
        selected_counts = counts[order[:k]]
        hits = int(np.sum(selected_counts))
        misses = int(flat_tail.size) - hits
        candidate_cost = (
            route_header_bits
            + table_header_bits
            + k * tail_width
            + kt_cost(hits, int(flat_tail.size))
            + entropy_bits_from_counts(selected_counts)
            + misses * tail_width
        )
        if candidate_cost < best_cost:
            best_cost = candidate_cost
            best_k = k
            best_escapes = misses
    return best_cost, {
        "raw_bits": raw_cost,
        "best_k": float(best_k),
        "escaped_values": float(best_escapes),
    }


def summarize_tail_width(
    bits: np.ndarray,
    ordered: np.ndarray,
    body_payload: np.ndarray,
    tail_width: int,
    tail_source: str,
    hash_bits_list: tuple[int, ...],
    k_values: tuple[int, ...],
    route_header_bits: int,
    context_header_bits: int,
    table_header_bits: int,
) -> dict:
    tail = tail_from_source(bits, ordered, body_payload, tail_width, tail_source)
    samples = int(tail.size)
    gdx_bits = best_gdx_low_cost(body_payload, tail_width)
    best_lower_name = ""
    best_lower_bits = float("inf")
    best_route_name = ""
    best_route_bits = float("inf")
    best_route_details: dict[str, float] = {}
    context_rows = []

    for hash_bits in hash_bits_list:
        for context_name, context in context_variants(
            bits,
            tail_width,
            hash_bits,
            body_payload,
        ).items():
            lower_bits = conditional_symbol_entropy(tail, context, tail_width)
            route_bits, details = topk_context_cost(
                tail,
                context,
                tail_width,
                k_values,
                route_header_bits,
                context_header_bits,
            )
            context_rows.append(
                {
                    "hash_bits": hash_bits,
                    "context": context_name,
                    "lower_bits": lower_bits,
                    "lower_bps": lower_bits / samples,
                    "route_bits": route_bits,
                    "route_bps": route_bits / samples,
                    "details": details,
                }
            )
            if lower_bits < best_lower_bits:
                best_lower_bits = lower_bits
                best_lower_name = f"hash{hash_bits}:{context_name}"
            if route_bits < best_route_bits:
                best_route_bits = route_bits
                best_route_name = f"hash{hash_bits}:{context_name}"
                best_route_details = details

    global_bits, global_details = global_topk_cost(
        tail,
        tail_width,
        k_values,
        route_header_bits,
        table_header_bits,
    )
    if global_bits < best_route_bits:
        best_route_bits = global_bits
        best_route_name = "global_topk"
        best_route_details = global_details

    return {
        "tail_width": tail_width,
        "tail_source": tail_source,
        "samples": samples,
        "gdx_bits": gdx_bits,
        "gdx_bps": gdx_bits / samples,
        "best_lower": best_lower_name,
        "best_lower_bits": best_lower_bits,
        "best_lower_bps": best_lower_bits / samples,
        "best_route": best_route_name,
        "best_route_bits": best_route_bits,
        "best_route_bps": best_route_bits / samples,
        "best_route_details": best_route_details,
        "global_topk_bits": global_bits,
        "global_topk_bps": global_bits / samples,
        "global_topk_details": global_details,
        "contexts": context_rows,
    }


def summarize_image(
    path: Path,
    crop_size: int,
    tile_size: int,
    previous_bits: int,
    tail_widths: tuple[int, ...],
    tail_source: str,
    hash_bits_list: tuple[int, ...],
    k_values: tuple[int, ...],
    route_header_bits: int,
    context_header_bits: int,
    table_header_bits: int,
) -> dict:
    pixels = read_exr(path)
    if crop_size:
        pixels = pixels[:crop_size, :crop_size]
    pixels = np.ascontiguousarray(pixels, dtype=np.float32)
    bits = float_to_bits(pixels).copy()
    ordered = bits_to_ordered(bits)
    _modes, payloads, _carries = choose_group_payloads(
        bits,
        pixels,
        tile_size,
        previous_bits,
        "body",
    )
    rows = [
        summarize_tail_width(
            bits,
            ordered,
            payloads["body"],
            tail_width,
            tail_source,
            hash_bits_list,
            k_values,
            route_header_bits,
            context_header_bits,
            table_header_bits,
        )
        for tail_width in tail_widths
    ]
    return {
        "image": path.name,
        "shape": list(bits.shape),
        "crop_size": crop_size,
        "tile_size": tile_size,
        "previous_bits": previous_bits,
        "tail_source": tail_source,
        "hash_bits_list": list(hash_bits_list),
        "k_values": list(k_values),
        "route_header_bits": route_header_bits,
        "context_header_bits": context_header_bits,
        "table_header_bits": table_header_bits,
        "rows": rows,
    }


def parse_int_tuple(text: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in text.split(",") if part.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", default="*.exr")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--tile-size", type=int, default=128)
    parser.add_argument("--previous-bits", type=int, default=4)
    parser.add_argument("--tail-widths", default="15")
    parser.add_argument(
        "--tail-source",
        choices=("raw_mantissa", "body_payload", "ordered_low"),
        default="body_payload",
    )
    parser.add_argument("--hash-bits", default="6,8,10,12")
    parser.add_argument("--k-values", default="1,2,4,8,16,32")
    parser.add_argument("--route-header-bits", type=int, default=8)
    parser.add_argument("--context-header-bits", type=int, default=12)
    parser.add_argument("--table-header-bits", type=int, default=16)
    parser.add_argument("--no-save", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = sorted(DATA_DIR.glob(args.glob))
    if args.limit:
        paths = paths[:args.limit]
    if not paths:
        raise RuntimeError(f"no images matched {DATA_DIR / args.glob}")

    tail_widths = parse_int_tuple(args.tail_widths)
    hash_bits_list = parse_int_tuple(args.hash_bits)
    k_values = parse_int_tuple(args.k_values)
    rows = []
    for path in paths:
        row = summarize_image(
            path,
            args.crop_size,
            args.tile_size,
            args.previous_bits,
            tail_widths,
            args.tail_source,
            hash_bits_list,
            k_values,
            args.route_header_bits,
            args.context_header_bits,
            args.table_header_bits,
        )
        rows.append(row)
        print(path.stem)
        for item in row["rows"]:
            print(
                f"  low{item['tail_width']:02d} {item['tail_source']}: "
                f"gdx={item['gdx_bps']:.3f} bps "
                f"lower={item['best_lower_bps']:.3f} bps {item['best_lower']} "
                f"topk={item['best_route_bps']:.3f} bps {item['best_route']} "
                f"global={item['global_topk_bps']:.3f} bps"
            )

    if not args.no_save:
        safe_glob = args.glob.replace("*", "star").replace(".", "_")
        output = RESULTS_DIR / (
            f"tail_escape_table_route_{safe_glob}_tile{args.tile_size}"
            f"_prev{args.previous_bits}_crop{args.crop_size}"
            f"_tail{args.tail_widths.replace(',', '-')}_{args.tail_source}.json"
        )
        output.write_text(json.dumps(rows, indent=2))
        print(f"\nSaved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
