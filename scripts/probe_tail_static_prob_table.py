"""Probe an exact low-tail route with transmitted static probability tables.

The adaptive high-context probes lost much of the theoretical conditional
entropy gain to sparse KT counts.  This probe tries a different reversible
shape: build per-image/tile static probabilities for selected tail bit/context
pairs, transmit only entries whose savings exceed their side information, and
fall back to a global probability for the rest.

This is still an estimate, but it answers a concrete question: is the puresky
low-tail structure strong enough to pay for an explicit model table?
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
from probe_grouped_tail_mlp import CONTEXT_FAMILIES, context_family_cost  # noqa: E402
from structural_delta_grouped_roundtrip import choose_group_payloads  # noqa: E402

RESULTS_DIR = ROOT / "results"


def geomean(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def parse_ints(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split(",") if part)


def entropy_bits(ones: np.ndarray, totals: np.ndarray) -> np.ndarray:
    totals_f = totals.astype(np.float64)
    ones_f = ones.astype(np.float64)
    zeros_f = totals_f - ones_f
    out = np.zeros_like(totals_f, dtype=np.float64)
    active = totals_f > 0
    p = np.zeros_like(totals_f, dtype=np.float64)
    p[active] = ones_f[active] / totals_f[active]
    one_active = active & (ones_f > 0)
    zero_active = active & (zeros_f > 0)
    out[one_active] -= ones_f[one_active] * np.log2(p[one_active])
    out[zero_active] -= zeros_f[zero_active] * np.log2(1.0 - p[zero_active])
    return out


def cross_entropy_bits(
    ones: np.ndarray,
    totals: np.ndarray,
    probability: float,
) -> np.ndarray:
    probability = min(max(probability, 1.0e-9), 1.0 - 1.0e-9)
    ones_f = ones.astype(np.float64)
    zeros_f = totals.astype(np.float64) - ones_f
    return -ones_f * math.log2(probability) - zeros_f * math.log2(1.0 - probability)


def fold_hash(values: np.ndarray, bits: int) -> np.ndarray:
    mask = np.uint64((1 << bits) - 1)
    mixed = values.astype(np.uint64)
    mixed ^= mixed >> np.uint64(17)
    mixed *= np.uint64(0x9E3779B185EBCA87)
    mixed ^= mixed >> np.uint64(29)
    mixed *= np.uint64(0xC2B2AE3D27D4EB4F)
    mixed ^= mixed >> np.uint64(32)
    return (mixed & mask).astype(np.int64)


def best_family_cost(payload: np.ndarray, bit_indices: range | tuple[int, ...]) -> float:
    total = 0.0
    for bit in bit_indices:
        total += min(
            context_family_cost(payload, bit, family)
            for family in CONTEXT_FAMILIES
        )
    return total


def context_seed(
    bits: np.ndarray,
    ordered: np.ndarray,
    body_payload: np.ndarray,
    tail_width: int,
    xy_bits: int,
    mode: str,
) -> np.ndarray:
    height, width, channels = bits.shape
    exponent = ((bits >> np.uint32(23)) & np.uint32(0xff)).astype(np.uint64)
    mantissa = (bits & np.uint32(0x7fffff)).astype(np.uint64)
    ordered_high = (ordered >> np.uint32(tail_width)).astype(np.uint64)
    mantissa_high = (mantissa >> np.uint32(tail_width)).astype(np.uint64)
    payload_high = (
        (body_payload & np.uint32(0x7fffffff)) >> np.uint32(tail_width)
    ).astype(np.uint64)
    c = np.arange(channels, dtype=np.uint64)[None, None, :]
    channel = np.broadcast_to(c, bits.shape)
    yy, xx = np.mgrid[0:height, 0:width]
    x = np.broadcast_to(xx[:, :, None].astype(np.uint64), bits.shape)
    y = np.broadcast_to(yy[:, :, None].astype(np.uint64), bits.shape)

    if mode == "ordered_high":
        code = ordered_high ^ (channel << np.uint64(29))
    elif mode == "float_high":
        code = (
            mantissa_high
            ^ (exponent << np.uint64(17))
            ^ (channel << np.uint64(25))
        )
    elif mode == "payload_high":
        code = payload_high ^ (channel << np.uint64(25))
    elif mode == "payload_exp_high":
        code = (
            payload_high
            ^ (exponent << np.uint64(17))
            ^ (channel << np.uint64(25))
        )
    else:
        raise ValueError(f"unknown context mode: {mode}")

    if xy_bits:
        xy_mask = np.uint64((1 << xy_bits) - 1)
        code = code.copy()
        code ^= (x & xy_mask) << np.uint64(37)
        code ^= (y & xy_mask) << np.uint64(37 + xy_bits)
    return code


def static_probability_table_cost(
    tail: np.ndarray,
    seed: np.ndarray,
    tail_width: int,
    context_bits: int,
    previous_tail_bits: int,
    probability_bits: int,
    route_header_bits: int,
    entry_extra_bits: int,
) -> tuple[float, dict[str, float]]:
    flat_tail = tail.reshape(-1).astype(np.uint32)
    flat_seed = seed.reshape(-1).astype(np.uint64)
    context_count = 1 << context_bits

    total_cost = float(route_header_bits)
    selected_entries = 0
    active_contexts = 0
    global_model_bits = 0.0
    selected_savings = 0.0
    entropy_after_selection = 0.0

    for bit in range(tail_width - 1, -1, -1):
        symbols = ((flat_tail >> np.uint32(bit)) & np.uint32(1)).astype(np.int64)
        code = flat_seed ^ (np.uint64(bit) << np.uint64(45))
        if previous_tail_bits > 0 and bit + 1 < tail_width:
            available = min(previous_tail_bits, tail_width - bit - 1)
            previous = (flat_tail >> np.uint32(bit + 1)) & np.uint32(
                (1 << available) - 1
            )
            code = code ^ (previous.astype(np.uint64) << np.uint64(49))
        context = fold_hash(code, context_bits)
        totals = np.bincount(context, minlength=context_count).astype(np.int64)
        ones = np.bincount(
            context,
            weights=symbols,
            minlength=context_count,
        ).astype(np.int64)
        active = totals > 0
        active_contexts += int(np.count_nonzero(active))

        total = int(np.sum(totals))
        bit_ones = int(np.sum(ones))
        probability = (bit_ones + 0.5) / (total + 1.0)
        global_context_cost = cross_entropy_bits(ones, totals, probability)
        context_entropy = entropy_bits(ones, totals)
        entry_header = context_bits + probability_bits + entry_extra_bits
        savings = global_context_cost - context_entropy
        selected = active & (savings > entry_header)

        global_bit_cost = float(np.sum(global_context_cost))
        selected_bit_savings = float(np.sum(savings[selected] - entry_header))
        total_cost += probability_bits + global_bit_cost - selected_bit_savings
        global_model_bits += global_bit_cost
        selected_savings += selected_bit_savings
        selected_entries += int(np.count_nonzero(selected))
        entropy_after_selection += global_bit_cost - selected_bit_savings

    return total_cost, {
        "selected_entries": float(selected_entries),
        "active_contexts_per_bit": float(active_contexts / max(1, tail_width)),
        "global_model_bits": global_model_bits,
        "selected_savings": selected_savings,
        "entropy_after_selection": entropy_after_selection,
        "table_bits": float(
            route_header_bits
            + tail_width * probability_bits
            + selected_entries * (context_bits + probability_bits + entry_extra_bits)
        ),
    }


def summarize_image(
    path: Path,
    crop_size: int,
    tile_size: int,
    previous_bits: int,
    tail_widths: tuple[int, ...],
    context_bits_values: tuple[int, ...],
    previous_tail_bits_values: tuple[int, ...],
    context_modes: tuple[str, ...],
    xy_bits_values: tuple[int, ...],
    probability_bits: int,
    route_header_bits: int,
    entry_extra_bits: int,
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

    raw_bits = int(bits.size * 32)
    rows = []
    for tail_width in tail_widths:
        low_mask = np.uint32((1 << tail_width) - 1)
        ordered_tail = ordered & low_mask
        current_low = best_family_cost(payloads["body"], range(tail_width))
        current_high = best_family_cost(payloads["body"], range(tail_width, 31))
        sign = best_family_cost(payloads["sign"], (31,))
        best_row = None

        for context_mode in context_modes:
            for xy_bits in xy_bits_values:
                seed = context_seed(
                    bits,
                    ordered,
                    payloads["body"],
                    tail_width,
                    xy_bits,
                    context_mode,
                )
                for context_bits in context_bits_values:
                    for previous_tail_bits in previous_tail_bits_values:
                        static_low, details = static_probability_table_cost(
                            ordered_tail,
                            seed,
                            tail_width,
                            context_bits,
                            previous_tail_bits,
                            probability_bits,
                            route_header_bits,
                            entry_extra_bits,
                        )
                        selected_low = min(current_low, static_low)
                        route = (
                            "static_probability_table"
                            if static_low < current_low
                            else "current_low"
                        )
                        current_total = sign + current_high + current_low
                        hybrid_total = sign + current_high + selected_low
                        row = {
                            "tail_width": tail_width,
                            "context_mode": context_mode,
                            "xy_bits": xy_bits,
                            "context_bits": context_bits,
                            "previous_tail_bits": previous_tail_bits,
                            "current_bits": current_total,
                            "hybrid_bits": hybrid_total,
                            "current_low_bits": current_low,
                            "static_low_bits": static_low,
                            "current_ratio": raw_bits / current_total,
                            "hybrid_ratio": raw_bits / hybrid_total,
                            "gain": current_total / hybrid_total,
                            "route": route,
                            "details": details,
                        }
                        if best_row is None or row["hybrid_bits"] < best_row["hybrid_bits"]:
                            best_row = row
        rows.append(best_row)
    return {
        "image": path.name,
        "shape": list(bits.shape),
        "crop_size": crop_size,
        "tile_size": tile_size,
        "previous_bits": previous_bits,
        "probability_bits": probability_bits,
        "route_header_bits": route_header_bits,
        "entry_extra_bits": entry_extra_bits,
        "rows": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", default="ph_*puresky*.exr")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--tile-size", type=int, default=128)
    parser.add_argument("--previous-bits", type=int, default=4)
    parser.add_argument("--tail-widths", default="15")
    parser.add_argument("--context-bits", default="8,10,12")
    parser.add_argument("--previous-tail-bits", default="0,2,4")
    parser.add_argument(
        "--context-modes",
        default="payload_high,payload_exp_high,ordered_high,float_high",
    )
    parser.add_argument("--xy-bits", default="0,3")
    parser.add_argument("--probability-bits", type=int, default=12)
    parser.add_argument("--route-header-bits", type=int, default=24)
    parser.add_argument("--entry-extra-bits", type=int, default=2)
    parser.add_argument("--no-save", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tail_widths = parse_ints(args.tail_widths)
    context_bits_values = parse_ints(args.context_bits)
    previous_tail_bits_values = parse_ints(args.previous_tail_bits)
    context_modes = tuple(part for part in args.context_modes.split(",") if part)
    xy_bits_values = parse_ints(args.xy_bits)

    rows = []
    for path in sorted(DATA_DIR.glob(args.glob)):
        row = summarize_image(
            path,
            args.crop_size,
            args.tile_size,
            args.previous_bits,
            tail_widths,
            context_bits_values,
            previous_tail_bits_values,
            context_modes,
            xy_bits_values,
            args.probability_bits,
            args.route_header_bits,
            args.entry_extra_bits,
        )
        rows.append(row)
        print(path.stem)
        for item in row["rows"]:
            print(
                f"  low{item['tail_width']:02d}: "
                f"current={item['current_ratio']:.3f}x "
                f"hybrid={item['hybrid_ratio']:.3f}x "
                f"gain={item['gain']:.4f} route={item['route']} "
                f"ctx={item['context_mode']}/h{item['context_bits']}/"
                f"xy{item['xy_bits']}/prev{item['previous_tail_bits']} "
                f"table={item['details']['table_bits'] / 8.0:.1f}B"
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
        f"tail_static_prob_table_{safe_glob}_tile{args.tile_size}"
        f"_prev{args.previous_bits}_crop{args.crop_size}"
        f"_tail{args.tail_widths.replace(',', '-')}"
        f"_ctx{args.context_bits.replace(',', '-')}.json"
    )
    if not args.no_save:
        output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"\nSaved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
