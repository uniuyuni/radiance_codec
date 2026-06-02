"""Probe hash/LSH-inspired exact routes for hard float tails.

This is a "god-domain" triage probe: it tests whether hash-like transforms can
create useful structure without hiding information in an external dictionary.

Two exact-compatible ideas are estimated:

* tail permutation route: apply a fixed reversible permutation to low-tail
  symbols, then measure whether causal residuals become simpler;
* decoder-visible hash/LSH context route: derive buckets only from already
  decoded high float fields and code tail symbols with an adaptive sparse
  alphabet model.

If these lose even as estimates, content-addressed/hash protocols should stay
as research notes rather than C++ codec work.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_structural_predictors import DATA_DIR, float_to_bits, read_exr  # noqa: E402
from evaluate_structural_delta_context import bits_to_ordered  # noqa: E402
from probe_grouped_tail_mlp import CONTEXT_FAMILIES, context_family_cost  # noqa: E402
from structural_delta_grouped_roundtrip import choose_group_payloads  # noqa: E402

RESULTS_DIR = ROOT / "results"


def entropy_bits_from_counts(counts: np.ndarray) -> float:
    total = float(np.sum(counts))
    if total == 0.0:
        return 0.0
    probs = counts.astype(np.float64) / total
    return float(-np.sum(probs * np.log2(probs)) * total)


def bit_entropy_cost(values: np.ndarray, width: int) -> float:
    flat = values.reshape(-1).astype(np.uint32)
    total = 0.0
    for bit in range(width):
        ones = int(np.count_nonzero((flat >> np.uint32(bit)) & np.uint32(1)))
        counts = np.array([flat.size - ones, ones], dtype=np.int64)
        total += entropy_bits_from_counts(counts)
    return total


def fold_hash(values: np.ndarray, bits: int) -> np.ndarray:
    mask = np.uint64((1 << bits) - 1)
    mixed = values.astype(np.uint64)
    mixed ^= mixed >> np.uint64(17)
    mixed *= np.uint64(0x9E3779B185EBCA87)
    mixed ^= mixed >> np.uint64(31)
    mixed *= np.uint64(0xC2B2AE3D27D4EB4F)
    mixed ^= mixed >> np.uint64(29)
    return (mixed & mask).astype(np.uint64)


def shift_zero(values: np.ndarray, dy: int, dx: int) -> np.ndarray:
    out = np.zeros_like(values)
    height, width, _channels = values.shape
    dst_y0 = max(0, -dy)
    dst_y1 = min(height, height - dy)
    dst_x0 = max(0, -dx)
    dst_x1 = min(width, width - dx)
    src_y0 = dst_y0 + dy
    src_y1 = dst_y1 + dy
    src_x0 = dst_x0 + dx
    src_x1 = dst_x1 + dx
    out[dst_y0:dst_y1, dst_x0:dst_x1] = values[src_y0:src_y1, src_x0:src_x1]
    return out


def best_gdx_low_cost(payload: np.ndarray, width: int) -> float:
    total = 0.0
    for bit in range(width):
        total += min(
            context_family_cost(payload, bit, family)
            for family in CONTEXT_FAMILIES
        )
    return total


def adaptive_symbol_cost(
    tail: np.ndarray,
    context: np.ndarray,
    width: int,
    alpha: float,
) -> float:
    """Sequential sparse Dirichlet model. Decoder can update after each symbol."""

    alphabet = 1 << width
    flat_tail = tail.reshape(-1).astype(np.uint32)
    flat_context = context.reshape(-1).astype(np.uint64)
    totals: defaultdict[int, int] = defaultdict(int)
    counts: defaultdict[int, Counter[int]] = defaultdict(Counter)
    cost = 0.0
    alpha_alphabet = alpha * alphabet
    for context_value, symbol_value in zip(flat_context, flat_tail):
        ctx = int(context_value)
        symbol = int(symbol_value)
        total = totals[ctx]
        count = counts[ctx][symbol]
        probability = (count + alpha) / (total + alpha_alphabet)
        cost -= math.log2(probability)
        totals[ctx] = total + 1
        counts[ctx][symbol] = count + 1
    return cost


def permute_tail(tail: np.ndarray, width: int, seed: int) -> np.ndarray:
    mask = np.uint32((1 << width) - 1)
    # Keep the multiplier odd so the map is bijective modulo 2**width.
    multiplier = np.uint32((0x9E37 ^ (seed * 0x45D9)) | 1)
    increment = np.uint32(seed * 0xA5A5 + 0x1F3D)
    return (((tail.astype(np.uint32) ^ np.uint32(seed)) * multiplier) + increment) & mask


def causal_residuals(values: np.ndarray, width: int) -> dict[str, np.ndarray]:
    mask = np.uint32((1 << width) - 1)
    west = shift_zero(values, 0, -1)
    north = shift_zero(values, -1, 0)
    previous_channel = np.zeros_like(values)
    if values.shape[2] > 1:
        previous_channel[..., 1:] = values[..., :-1]
    green = np.zeros_like(values)
    if values.shape[2] >= 3:
        green[..., 0] = values[..., 1]
        green[..., 1] = values[..., 1]
        green[..., 2] = values[..., 1]
        if values.shape[2] > 3:
            green[..., 3:] = values[..., 3:]
    predictors = {
        "zero": np.zeros_like(values),
        "west": west,
        "north": north,
        "previous_channel": previous_channel,
        "green": green,
    }
    residuals = {}
    for name, prediction in predictors.items():
        residuals[f"{name}_sub"] = (values - prediction) & mask
        residuals[f"{name}_xor"] = values ^ prediction
    return residuals


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


def prefix_bits(values: np.ndarray, value_bits: int, keep_bits: int) -> np.ndarray:
    if keep_bits <= 0:
        return np.zeros(values.shape, dtype=np.uint64)
    shift = max(0, value_bits - keep_bits)
    return (values.astype(np.uint64) >> np.uint64(shift)) & np.uint64((1 << keep_bits) - 1)


def visible_contexts(
    bits: np.ndarray,
    body_payload: np.ndarray,
    tail_width: int,
    hash_bits_list: tuple[int, ...],
    lsh_bits_list: tuple[int, ...],
) -> dict[str, np.ndarray]:
    height, width, channels = bits.shape
    exponent = ((bits >> np.uint32(23)) & np.uint32(0xFF)).astype(np.uint64)
    mantissa = (bits & np.uint32(0x7FFFFF)).astype(np.uint64)
    high = mantissa >> np.uint64(tail_width)
    high_bits = max(0, 23 - tail_width)
    payload_high = (
        (body_payload & np.uint32(0x7FFFFFFF)) >> np.uint32(tail_width)
    ).astype(np.uint64)
    payload_high_bits = max(0, 31 - tail_width)

    c = np.arange(channels, dtype=np.uint64)[None, None, :]
    channel = np.broadcast_to(c, bits.shape)
    yy, xx = np.mgrid[0:height, 0:width]
    x = np.broadcast_to(xx[:, :, None].astype(np.uint64), bits.shape)
    y = np.broadcast_to(yy[:, :, None].astype(np.uint64), bits.shape)
    west_high = shift_zero(high, 0, -1)
    north_high = shift_zero(high, -1, 0)

    contexts: dict[str, np.ndarray] = {
        "none": np.zeros(bits.shape, dtype=np.uint64),
        "channel": channel,
        "exponent_channel": (exponent << np.uint64(2)) | channel,
    }

    for keep_bits in lsh_bits_list:
        high_prefix = prefix_bits(high, high_bits, keep_bits)
        payload_prefix = prefix_bits(payload_high, payload_high_bits, keep_bits)
        contexts[f"lsh_high_prefix{keep_bits}_channel"] = (
            (high_prefix << np.uint64(2)) | channel
        )
        contexts[f"lsh_exp_high_prefix{keep_bits}_xy_channel"] = (
            (exponent << np.uint64(keep_bits + 8))
            | (high_prefix << np.uint64(8))
            | ((x & np.uint64(7)) << np.uint64(5))
            | ((y & np.uint64(7)) << np.uint64(2))
            | channel
        )
        contexts[f"lsh_payload_prefix{keep_bits}_xy_channel"] = (
            (payload_prefix << np.uint64(8))
            | ((x & np.uint64(7)) << np.uint64(5))
            | ((y & np.uint64(7)) << np.uint64(2))
            | channel
        )

    visible_code = (
        high
        ^ (exponent << np.uint64(17))
        ^ (channel << np.uint64(25))
        ^ ((x & np.uint64(15)) << np.uint64(29))
        ^ ((y & np.uint64(15)) << np.uint64(33))
        ^ (west_high << np.uint64(37))
        ^ (north_high << np.uint64(43))
    )
    payload_code = (
        payload_high
        ^ (channel << np.uint64(25))
        ^ ((x & np.uint64(15)) << np.uint64(29))
        ^ ((y & np.uint64(15)) << np.uint64(33))
    )
    for hash_bits in hash_bits_list:
        contexts[f"hash{hash_bits}_visible_high_xy_neighbors"] = fold_hash(
            visible_code,
            hash_bits,
        )
        contexts[f"hash{hash_bits}_payload_high_xy"] = fold_hash(
            payload_code,
            hash_bits,
        )
    return contexts


def summarize_tail_width(
    bits: np.ndarray,
    ordered: np.ndarray,
    body_payload: np.ndarray,
    tail_width: int,
    tail_source: str,
    hash_bits_list: tuple[int, ...],
    lsh_bits_list: tuple[int, ...],
    alpha: float,
    permutation_seeds: tuple[int, ...],
) -> dict:
    tail = tail_from_source(bits, ordered, body_payload, tail_width, tail_source)
    samples = int(tail.size)
    gdx_bits = best_gdx_low_cost(body_payload, tail_width)

    context_costs = {}
    for name, context in visible_contexts(
        bits,
        body_payload,
        tail_width,
        hash_bits_list,
        lsh_bits_list,
    ).items():
        context_costs[name] = adaptive_symbol_cost(tail, context, tail_width, alpha)

    residual_costs = {}
    for name, residual in causal_residuals(tail, tail_width).items():
        residual_costs[f"raw_tail/{name}"] = bit_entropy_cost(residual, tail_width)
    for seed in permutation_seeds:
        permuted = permute_tail(tail, tail_width, seed)
        for name, residual in causal_residuals(permuted, tail_width).items():
            residual_costs[f"perm{seed}/{name}"] = bit_entropy_cost(
                residual,
                tail_width,
            )

    best_context_name, best_context_bits = min(
        context_costs.items(),
        key=lambda item: item[1],
    )
    best_residual_name, best_residual_bits = min(
        residual_costs.items(),
        key=lambda item: item[1],
    )
    return {
        "tail_width": tail_width,
        "tail_source": tail_source,
        "samples": samples,
        "gdx_bits": gdx_bits,
        "gdx_bps": gdx_bits / samples,
        "best_adaptive_context": best_context_name,
        "best_adaptive_context_bits": best_context_bits,
        "best_adaptive_context_bps": best_context_bits / samples,
        "best_residual_route": best_residual_name,
        "best_residual_bits": best_residual_bits,
        "best_residual_bps": best_residual_bits / samples,
        "adaptive_context_bps": {
            name: cost / samples for name, cost in sorted(context_costs.items())
        },
        "residual_bps": {
            name: cost / samples for name, cost in sorted(residual_costs.items())
        },
    }


def summarize_image(
    path: Path,
    crop_size: int,
    tile_size: int,
    previous_bits: int,
    tail_widths: tuple[int, ...],
    tail_source: str,
    hash_bits_list: tuple[int, ...],
    lsh_bits_list: tuple[int, ...],
    alpha: float,
    permutation_seeds: tuple[int, ...],
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
        rows.append(
            summarize_tail_width(
                bits,
                ordered,
                payloads["body"],
                tail_width,
                tail_source,
                hash_bits_list,
                lsh_bits_list,
                alpha,
                permutation_seeds,
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
        "lsh_bits_list": list(lsh_bits_list),
        "alpha": alpha,
        "permutation_seeds": list(permutation_seeds),
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
    parser.add_argument("--lsh-bits", default="2,4,6,8")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--permutation-seeds", default="1,3,7")
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
    lsh_bits_list = parse_int_tuple(args.lsh_bits)
    permutation_seeds = parse_int_tuple(args.permutation_seeds)
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
            lsh_bits_list,
            args.alpha,
            permutation_seeds,
        )
        rows.append(row)
        print(path.stem)
        for item in row["rows"]:
            print(
                f"  low{item['tail_width']:02d} {item['tail_source']}: "
                f"gdx={item['gdx_bps']:.3f} bps "
                f"adaptive={item['best_adaptive_context_bps']:.3f} bps "
                f"{item['best_adaptive_context']} "
                f"residual={item['best_residual_bps']:.3f} bps "
                f"{item['best_residual_route']}"
            )

    if not args.no_save:
        safe_glob = args.glob.replace("*", "star").replace(".", "_")
        output = RESULTS_DIR / (
            f"tail_hash_lsh_protocol_{safe_glob}_tile{args.tile_size}"
            f"_prev{args.previous_bits}_crop{args.crop_size}"
            f"_tail{args.tail_widths.replace(',', '-')}_{args.tail_source}.json"
        )
        output.write_text(json.dumps(rows, indent=2))
        print(f"\nSaved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
