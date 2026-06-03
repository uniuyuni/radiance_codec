"""Probe prefix-cascade coding for hard low-tail symbols.

Whole-symbol conditional entropy can look promising, but explicit dictionaries
are expensive.  This probe tries a dictionary-free middle ground: encode a
low-tail symbol from MSB to LSB, using decoder-visible high-float context plus
the already-decoded prefix of the same symbol as the context for the next bit.

Each context node pays KT universal binary cost, so the decoder can reproduce
the same adaptive model without side information.
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


def best_gdx_low_cost(payload: np.ndarray, width: int) -> float:
    total = 0.0
    for bit in range(width):
        total += min(
            context_family_cost(payload, bit, family)
            for family in CONTEXT_FAMILIES
        )
    return total


def sparse_context_bit_cost(context: np.ndarray, symbols: np.ndarray) -> float:
    context_flat = context.reshape(-1).astype(np.uint64)
    symbols_flat = symbols.reshape(-1).astype(np.uint8)
    _unique, inverse = np.unique(context_flat, return_inverse=True)
    totals = np.bincount(inverse)
    ones = np.bincount(inverse, weights=symbols_flat)
    total_cost = 0.0
    for index in np.nonzero(totals)[0]:
        total = int(totals[index])
        total_cost += kt_cost(int(ones[index]), total)
    return total_cost


def prefix_cascade_cost(
    tail: np.ndarray,
    visible_context: np.ndarray,
    width: int,
    max_prefix_bits: int,
) -> tuple[float, dict[str, float]]:
    tail_flat = tail.reshape(-1).astype(np.uint64)
    context_flat = visible_context.reshape(-1).astype(np.uint64)
    total = 0.0
    node_total = 0
    used_prefix_bits = 0
    for bit in range(width - 1, -1, -1):
        prefix_bits = min(width - bit - 1, max_prefix_bits)
        if prefix_bits:
            prefix = (tail_flat >> np.uint64(bit + 1)) & np.uint64(
                (1 << prefix_bits) - 1,
            )
        else:
            prefix = np.zeros_like(tail_flat)
        context = (context_flat << np.uint64(prefix_bits)) | prefix
        symbols = ((tail_flat >> np.uint64(bit)) & np.uint64(1)).astype(np.uint8)
        total += sparse_context_bit_cost(context, symbols)
        node_total += int(np.unique(context).size)
        used_prefix_bits = max(used_prefix_bits, prefix_bits)
    return total, {
        "node_total": float(node_total),
        "used_prefix_bits": float(used_prefix_bits),
    }


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


def summarize_scope(
    bits: np.ndarray,
    ordered: np.ndarray,
    body_payload: np.ndarray,
    tile_size: int,
    tail_width: int,
    tail_source: str,
    hash_bits_list: tuple[int, ...],
    max_prefix_bits_list: tuple[int, ...],
    context_filters: tuple[str, ...],
    model_scope: str,
    route_header_bits: float,
) -> dict:
    tail = tail_from_source(bits, ordered, body_payload, tail_width, tail_source)
    gdx_bits = 0.0
    best_bits = float("inf")
    best_name = ""
    best_details: dict[str, float] = {}
    best_lower = float("inf")
    best_lower_name = ""
    rows = []

    regions: list[tuple[slice, slice]]
    if model_scope == "image":
        regions = [(slice(0, bits.shape[0]), slice(0, bits.shape[1]))]
    elif model_scope == "tile":
        regions = [
            (slice(y, min(y + tile_size, bits.shape[0])), slice(x, min(x + tile_size, bits.shape[1])))
            for y in range(0, bits.shape[0], tile_size)
            for x in range(0, bits.shape[1], tile_size)
        ]
    else:
        raise ValueError(f"unknown model scope: {model_scope}")

    for y_slice, x_slice in regions:
        sl = np.s_[y_slice, x_slice]
        gdx_bits += best_gdx_low_cost(body_payload[sl], tail_width)

    for hash_bits in hash_bits_list:
        for y_slice, x_slice in regions:
            sl = np.s_[y_slice, x_slice]
            variants = context_variants(
                bits[sl],
                tail_width,
                hash_bits,
                body_payload[sl],
            )
            for context_name, context in variants.items():
                if context_filters and context_name not in context_filters:
                    continue
                lower = conditional_symbol_entropy(tail[sl], context, tail_width)
                lower_key = f"hash{hash_bits}:{context_name}"
                if lower < best_lower:
                    best_lower = lower
                    best_lower_name = lower_key
                for max_prefix_bits in max_prefix_bits_list:
                    cost, details = prefix_cascade_cost(
                        tail[sl],
                        context,
                        tail_width,
                        max_prefix_bits,
                    )
                    key = f"hash{hash_bits}:{context_name}:prefix{max_prefix_bits}"
                    rows.append(
                        {
                            "context": key,
                            "bits": cost,
                            "details": details,
                        }
                    )
                    if cost < best_bits:
                        best_bits = cost
                        best_name = key
                        best_details = details

    # Route header is paid once for image scope and once per tile for tile scope.
    route_bits = route_header_bits * len(regions)
    best_bits += route_bits
    return {
        "tail_width": tail_width,
        "tail_source": tail_source,
        "model_scope": model_scope,
        "samples": int(tail.size),
        "region_count": len(regions),
        "gdx_bits": gdx_bits,
        "gdx_bps": gdx_bits / tail.size,
        "best_prefix_bits": best_bits,
        "best_prefix_bps": best_bits / tail.size,
        "best_prefix": best_name,
        "best_prefix_details": best_details,
        "best_lower_bits": best_lower,
        "best_lower_bps": best_lower / tail.size,
        "best_lower": best_lower_name,
        "gain": gdx_bits / best_bits if best_bits else 1.0,
    }


def summarize_image(
    path: Path,
    crop_size: int,
    tile_size: int,
    previous_bits: int,
    tail_widths: tuple[int, ...],
    tail_source: str,
    hash_bits_list: tuple[int, ...],
    max_prefix_bits_list: tuple[int, ...],
    context_filters: tuple[str, ...],
    model_scopes: tuple[str, ...],
    route_header_bits: float,
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
    rows = []
    for tail_width in tail_widths:
        for model_scope in model_scopes:
            rows.append(
                summarize_scope(
                    bits,
                    ordered,
                    payloads["body"],
                    tile_size,
                    tail_width,
                    tail_source,
                    hash_bits_list,
                    max_prefix_bits_list,
                    context_filters,
                    model_scope,
                    route_header_bits,
                )
            )
    return {
        "image": path.name,
        "shape": list(bits.shape),
        "crop_size": crop_size,
        "tile_size": tile_size,
        "previous_bits": previous_bits,
        "tail_source": tail_source,
        "hash_bits_list": list(hash_bits_list),
        "max_prefix_bits_list": list(max_prefix_bits_list),
        "context_filters": list(context_filters),
        "model_scopes": list(model_scopes),
        "route_header_bits": route_header_bits,
        "rows": rows,
    }


def parse_int_tuple(text: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in text.split(",") if part.strip())


def parse_word_tuple(text: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in text.split(",") if part.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", default="sample_hilberts-mill-conference-room_2K.exr")
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
    parser.add_argument("--hash-bits", default="4,6,8,10,12")
    parser.add_argument("--max-prefix-bits", default="0,2,4,6,8,10,12,15")
    parser.add_argument(
        "--contexts",
        default="",
        help="Comma-separated context names. Empty tests all context variants.",
    )
    parser.add_argument("--model-scopes", default="tile,image")
    parser.add_argument("--route-header-bits", type=float, default=8.0)
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
    max_prefix_bits_list = parse_int_tuple(args.max_prefix_bits)
    context_filters = parse_word_tuple(args.contexts)
    model_scopes = parse_word_tuple(args.model_scopes)
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
            max_prefix_bits_list,
            context_filters,
            model_scopes,
            args.route_header_bits,
        )
        rows.append(row)
        print(path.stem)
        for item in row["rows"]:
            print(
                f"  low{item['tail_width']:02d} {item['model_scope']:5s}: "
                f"gdx={item['gdx_bps']:.3f} bps "
                f"prefix={item['best_prefix_bps']:.3f} bps "
                f"gain={item['gain']:.4f} "
                f"{item['best_prefix']} "
                f"lower={item['best_lower_bps']:.3f} bps {item['best_lower']}"
            )

    if not args.no_save:
        safe_glob = args.glob.replace("*", "star").replace(".", "_")
        output = RESULTS_DIR / (
            f"tail_prefix_cascade_{safe_glob}_tile{args.tile_size}"
            f"_prev{args.previous_bits}_crop{args.crop_size}"
            f"_tail{args.tail_widths.replace(',', '-')}_{args.tail_source}.json"
        )
        output.write_text(json.dumps(rows, indent=2))
        print(f"\nSaved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
