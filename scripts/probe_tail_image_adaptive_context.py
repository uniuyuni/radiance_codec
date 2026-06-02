"""Probe image-carried adaptive low-tail coding with high-float contexts.

The first adaptive high-context probe reset counts per tile.  That is a very
expensive learning regime for sparse high-float contexts.  This probe keeps the
adaptive counts across the whole image in raster-tile order.  It is still
decoder-shaped: every tile is decoded exactly once, and after a tile is decoded
its raw low mantissa bits can update the adaptive model even if that tile used
the existing GDX low-bit route.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.special import gammaln

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_structural_predictors import DATA_DIR, float_to_bits, read_exr  # noqa: E402
from probe_grouped_tail_mlp import CONTEXT_FAMILIES, context_family_cost  # noqa: E402
from probe_tail_adaptive_high_context import fold_hash, high_feature_seed, shift_plane  # noqa: E402
from structural_delta_grouped_roundtrip import choose_group_payloads  # noqa: E402

RESULTS_DIR = ROOT / "results"


def geomean(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def kt_cost_with_initial(
    initial_ones: int,
    initial_zeros: int,
    ones: int,
    zeros: int,
) -> float:
    total = ones + zeros
    if total == 0:
        return 0.0
    initial_total = initial_ones + initial_zeros
    log_probability = (
        math.lgamma(initial_ones + ones + 0.5)
        - math.lgamma(initial_ones + 0.5)
        + math.lgamma(initial_zeros + zeros + 0.5)
        - math.lgamma(initial_zeros + 0.5)
        + math.lgamma(initial_total + 1.0)
        - math.lgamma(initial_total + total + 1.0)
    )
    return -log_probability / math.log(2.0)


def kt_cost_many_with_initial(
    initial_ones: np.ndarray,
    initial_zeros: np.ndarray,
    ones: np.ndarray,
    zeros: np.ndarray,
) -> float:
    totals = ones + zeros
    if totals.size == 0:
        return 0.0
    initial_totals = initial_ones + initial_zeros
    log_probability = (
        gammaln(initial_ones + ones + 0.5)
        - gammaln(initial_ones + 0.5)
        + gammaln(initial_zeros + zeros + 0.5)
        - gammaln(initial_zeros + 0.5)
        + gammaln(initial_totals + 1.0)
        - gammaln(initial_totals + totals + 1.0)
    )
    return float(-np.sum(log_probability) / math.log(2.0))


def best_family_cost(payload: np.ndarray, bit_indices: range | tuple[int, ...]) -> float:
    total = 0.0
    for bit in bit_indices:
        total += min(
            context_family_cost(payload, bit, family)
            for family in CONTEXT_FAMILIES
        )
    return total


def context_for_bit(
    bits: np.ndarray,
    tail: np.ndarray,
    tail_width: int,
    bit: int,
    context_bits: int,
    xy_bits: int,
    prev_tail_bits: int,
    feature_mode: str,
    use_spatial: bool,
    use_previous_channel: bool,
) -> tuple[np.ndarray, np.ndarray]:
    symbols = ((tail >> np.uint32(bit)) & np.uint32(1)).astype(np.uint8)
    code = high_feature_seed(bits, tail_width, xy_bits, feature_mode)
    code ^= np.uint64(bit) << np.uint64(41)

    if prev_tail_bits > 0 and bit + 1 < tail_width:
        available = min(prev_tail_bits, tail_width - bit - 1)
        previous_tail = (tail >> np.uint32(bit + 1)) & np.uint32(
            (1 << available) - 1
        )
        code ^= previous_tail.astype(np.uint64) << np.uint64(44)

    if use_spatial:
        west = shift_plane(symbols, 0, -1).astype(np.uint64)
        north = shift_plane(symbols, -1, 0).astype(np.uint64)
        northwest = shift_plane(symbols, -1, -1).astype(np.uint64)
        northeast = shift_plane(symbols, -1, 1).astype(np.uint64)
        code ^= west << np.uint64(3)
        code ^= north << np.uint64(4)
        code ^= northwest << np.uint64(5)
        code ^= northeast << np.uint64(6)

    if use_previous_channel and bits.shape[2] > 1:
        pc = np.zeros_like(symbols, dtype=np.uint64)
        pc[..., 1:] = symbols[..., :-1]
        code ^= pc << np.uint64(7)

    return fold_hash(code, context_bits), symbols


def tile_adaptive_cost_and_update(
    bits_tile: np.ndarray,
    tail_width: int,
    context_bits: int,
    xy_bits: int,
    prev_tail_bits: int,
    feature_mode: str,
    use_spatial: bool,
    use_previous_channel: bool,
    one_counts: list[np.ndarray],
    zero_counts: list[np.ndarray],
) -> tuple[float, dict[str, float]]:
    mantissa = bits_tile & np.uint32(0x7fffff)
    tail = mantissa & np.uint32((1 << tail_width) - 1)
    context_count = 1 << context_bits
    total_cost = 0.0
    active_contexts = []
    learned_contexts = []
    for bit in range(tail_width - 1, -1, -1):
        context, symbols = context_for_bit(
            bits_tile,
            tail,
            tail_width,
            bit,
            context_bits,
            xy_bits,
            prev_tail_bits,
            feature_mode,
            use_spatial,
            use_previous_channel,
        )
        context_flat = context.reshape(-1)
        symbol_flat = symbols.reshape(-1)
        totals = np.bincount(context_flat, minlength=context_count).astype(np.int64)
        ones = np.bincount(
            context_flat,
            weights=symbol_flat,
            minlength=context_count,
        ).astype(np.int64)
        zeros = totals - ones
        active = np.nonzero(totals)[0]
        active_contexts.append(int(active.size))
        learned_contexts.append(
            int(np.count_nonzero(one_counts[bit][active] + zero_counts[bit][active]))
        )
        total_cost += kt_cost_many_with_initial(
            one_counts[bit][active],
            zero_counts[bit][active],
            ones[active],
            zeros[active],
        )
        one_counts[bit] += ones
        zero_counts[bit] += zeros
    return total_cost, {
        "mean_active_contexts": float(np.mean(active_contexts)),
        "mean_learned_active_contexts": float(np.mean(learned_contexts)),
    }


def current_tile_costs(
    body_payload_tile: np.ndarray,
    sign_payload_tile: np.ndarray,
    tail_width: int,
) -> dict[str, float]:
    low_gdx = best_family_cost(body_payload_tile, range(tail_width))
    high_gdx = best_family_cost(body_payload_tile, range(tail_width, 31))
    sign = best_family_cost(sign_payload_tile, (31,))
    return {
        "sign": sign,
        "high_gdx": high_gdx,
        "low_gdx": low_gdx,
        "current_total": sign + high_gdx + low_gdx,
    }


def evaluate_candidate(
    bits: np.ndarray,
    current_tiles: list[tuple[tuple[slice, slice], dict[str, float]]],
    tail_width: int,
    context_bits: int,
    xy_bits: int,
    prev_tail_bits: int,
    feature_mode: str,
    use_spatial: bool,
    use_previous_channel: bool,
    route_header_bits: int,
) -> dict:
    context_count = 1 << context_bits
    one_counts = [np.zeros(context_count, dtype=np.int64) for _ in range(tail_width)]
    zero_counts = [np.zeros(context_count, dtype=np.int64) for _ in range(tail_width)]
    totals = Counter()
    route_histogram = Counter()
    detail_totals = Counter()
    for sl, current in current_tiles:
        adaptive, details = tile_adaptive_cost_and_update(
            bits[sl],
            tail_width,
            context_bits,
            xy_bits,
            prev_tail_bits,
            feature_mode,
            use_spatial,
            use_previous_channel,
            one_counts,
            zero_counts,
        )
        adaptive += route_header_bits
        route = "image_adaptive_high_context" if adaptive < current["low_gdx"] else "current_low"
        route_histogram[route] += 1
        totals["current_total"] += current["current_total"]
        totals["hybrid_total"] += (
            current["sign"] + current["high_gdx"] + min(current["low_gdx"], adaptive)
        )
        totals["low_gdx"] += current["low_gdx"]
        totals["adaptive_low"] += adaptive
        totals["sign"] += current["sign"]
        totals["high_gdx"] += current["high_gdx"]
        for key, value in details.items():
            detail_totals[key] += float(value)
    current_total = float(totals["current_total"])
    hybrid_total = float(totals["hybrid_total"])
    return {
        "tail_width": tail_width,
        "context_bits": context_bits,
        "prev_tail_bits": prev_tail_bits,
        "feature_mode": feature_mode,
        "current_bits": current_total,
        "hybrid_bits": hybrid_total,
        "gain": current_total / hybrid_total if hybrid_total else 1.0,
        "route_histogram": dict(route_histogram),
        "totals": dict(totals),
        "detail_totals": dict(detail_totals),
    }


def summarize_image(
    path: Path,
    crop_size: int,
    tile_size: int,
    previous_bits: int,
    tail_widths: tuple[int, ...],
    context_bits_values: tuple[int, ...],
    xy_bits: int,
    prev_tail_bits_values: tuple[int, ...],
    feature_modes: tuple[str, ...],
    use_spatial: bool,
    use_previous_channel: bool,
    route_header_bits: int,
) -> dict:
    pixels = read_exr(path)
    if crop_size:
        pixels = pixels[:crop_size, :crop_size]
    pixels = np.ascontiguousarray(pixels, dtype=np.float32)
    bits = float_to_bits(pixels).copy()
    _modes, payloads, _carries = choose_group_payloads(
        bits,
        pixels,
        tile_size,
        previous_bits,
        "body",
    )
    raw_bits = int(bits.size * 32)
    rows = []
    tile_slices = [
        np.s_[y:y + tile_size, x:x + tile_size]
        for y in range(0, bits.shape[0], tile_size)
        for x in range(0, bits.shape[1], tile_size)
    ]
    for tail_width in tail_widths:
        best_row = None
        current_tiles = [
            (
                sl,
                current_tile_costs(
                    payloads["body"][sl],
                    payloads["sign"][sl],
                    tail_width,
                ),
            )
            for sl in tile_slices
        ]
        for context_bits in context_bits_values:
            for prev_tail_bits in prev_tail_bits_values:
                for feature_mode in feature_modes:
                    row = evaluate_candidate(
                        bits,
                        current_tiles,
                        tail_width,
                        context_bits,
                        xy_bits,
                        prev_tail_bits,
                        feature_mode,
                        use_spatial,
                        use_previous_channel,
                        route_header_bits,
                    )
                    row["current_ratio"] = raw_bits / row["current_bits"]
                    row["hybrid_ratio"] = raw_bits / row["hybrid_bits"]
                    if best_row is None or row["hybrid_bits"] < best_row["hybrid_bits"]:
                        best_row = row
        rows.append(best_row)
    return {
        "image": path.name,
        "shape": list(bits.shape),
        "crop_size": crop_size,
        "tile_size": tile_size,
        "previous_bits": previous_bits,
        "xy_bits": xy_bits,
        "use_spatial": use_spatial,
        "use_previous_channel": use_previous_channel,
        "route_header_bits": route_header_bits,
        "rows": rows,
    }


def parse_ints(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split(",") if part)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", default="ph_*puresky*.exr")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--tile-size", type=int, default=128)
    parser.add_argument("--previous-bits", type=int, default=4)
    parser.add_argument("--tail-widths", default="8,10,12,15")
    parser.add_argument("--context-bits", default="8,10,12")
    parser.add_argument("--xy-bits", type=int, default=3)
    parser.add_argument("--prev-tail-bits", default="0,2,4")
    parser.add_argument(
        "--feature-modes",
        default="channel,exponent_channel,high4_channel,high8_channel,exp_high4_channel,exp_high8_channel,hash_full",
    )
    parser.add_argument("--no-spatial", action="store_true")
    parser.add_argument("--no-previous-channel", action="store_true")
    parser.add_argument("--route-header-bits", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tail_widths = parse_ints(args.tail_widths)
    context_bits_values = parse_ints(args.context_bits)
    prev_tail_bits_values = parse_ints(args.prev_tail_bits)
    feature_modes = tuple(part for part in args.feature_modes.split(",") if part)
    rows = []
    for path in sorted(DATA_DIR.glob(args.glob)):
        row = summarize_image(
            path,
            args.crop_size,
            args.tile_size,
            args.previous_bits,
            tail_widths,
            context_bits_values,
            args.xy_bits,
            prev_tail_bits_values,
            feature_modes,
            not args.no_spatial,
            not args.no_previous_channel,
            args.route_header_bits,
        )
        rows.append(row)
        print(path.stem)
        for item in row["rows"]:
            print(
                f"  low{item['tail_width']:02d}/ctx{item['context_bits']:02d}"
                f"/prev{item['prev_tail_bits']:02d}/{item['feature_mode']}: "
                f"current={item['current_ratio']:.3f}x "
                f"hybrid={item['hybrid_ratio']:.3f}x "
                f"gain={item['gain']:.4f} routes={item['route_histogram']}"
            )
        if args.limit and len(rows) >= args.limit:
            break
    if rows:
        print("\ngeomean:")
        for tail_width in tail_widths:
            current_values = []
            hybrid_values = []
            for row in rows:
                item = next(r for r in row["rows"] if r["tail_width"] == tail_width)
                current_values.append(item["current_ratio"])
                hybrid_values.append(item["hybrid_ratio"])
            print(
                f"  low{tail_width:02d}: current={geomean(current_values):.3f}x "
                f"hybrid={geomean(hybrid_values):.3f}x"
            )
    safe_glob = args.glob.replace("*", "star").replace(".", "_")
    output = RESULTS_DIR / (
        f"tail_image_adaptive_context_{safe_glob}_tile{args.tile_size}"
        f"_prev{args.previous_bits}_crop{args.crop_size}"
        f"_tail{args.tail_widths.replace(',', '-')}"
        f"_ctx{args.context_bits.replace(',', '-')}"
        f"_pt{args.prev_tail_bits.replace(',', '-')}.json"
    )
    output.write_text(json.dumps(rows, indent=2))
    print(f"\nSaved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
