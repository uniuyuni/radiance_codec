"""Probe exact bounded-value routes for float32 images.

For image pipelines that guarantee ranges such as 0.0..4.0, sign bits and the
exponent alphabet can be constrained.  This script estimates the remaining
compressed cost of sign+exponent streams versus a signless/exponent-alphabet
route.  Mantissa bits are intentionally left out: range bounds do not by
themselves solve the low-mantissa tail problem.
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
from probe_grouped_tail_mlp import CONTEXT_FAMILIES, context_family_cost, kt_cost  # noqa: E402
from structural_delta_grouped_roundtrip import choose_group_payloads  # noqa: E402

RESULTS_DIR = ROOT / "results"
EXPONENT_BITS = tuple(range(23, 31))


def entropy_bits_from_counts(counts: np.ndarray) -> float:
    total = float(np.sum(counts))
    if total == 0.0:
        return 0.0
    probs = counts.astype(np.float64) / total
    return float(-np.sum(probs * np.log2(probs)) * total)


def best_bit_range_cost(payload: np.ndarray, bits: tuple[int, ...]) -> float:
    total = 0.0
    for bit in bits:
        total += min(
            context_family_cost(payload, bit, family)
            for family in CONTEXT_FAMILIES
        )
    return total


def best_sign_cost(sign_payload: np.ndarray) -> float:
    return min(context_family_cost(sign_payload, 31, family) for family in CONTEXT_FAMILIES)


def exponent_alphabet_cost(
    exponents: np.ndarray,
    header_bits: float,
    symbol_bits: float,
) -> tuple[float, dict[str, float]]:
    flat = exponents.reshape(-1).astype(np.uint16)
    values, counts = np.unique(flat, return_counts=True)
    raw_cost = float(flat.size * 8)
    dictionary_cost = float(values.size * 8)
    index_cost = entropy_bits_from_counts(counts)
    route_cost = header_bits + dictionary_cost + index_cost
    # A constant exponent can be signaled even cheaper than a generic alphabet.
    if values.size == 1:
        route_cost = min(route_cost, header_bits + 8)
    route_cost = min(route_cost, header_bits + raw_cost)
    return route_cost, {
        "unique_exponents": float(values.size),
        "raw_exponent_bits": raw_cost,
        "dictionary_bits": dictionary_cost,
        "index_bits": index_cost,
        "min_exponent": float(values.min()),
        "max_exponent": float(values.max()),
        "dominant_exponent": float(values[np.argmax(counts)]),
        "dominant_count": float(np.max(counts)),
        "symbol_bits": symbol_bits,
    }


def summarize_image(
    path: Path,
    crop_size: int,
    tile_size: int,
    previous_bits: int,
    min_value: float,
    max_value: float,
    sign_flag_bits: float,
    exponent_header_bits: float,
    route_header_bits: float,
) -> dict:
    pixels = read_exr(path)
    if crop_size:
        pixels = pixels[:crop_size, :crop_size]
    pixels = np.ascontiguousarray(pixels, dtype=np.float32)
    bits = float_to_bits(pixels).copy()
    sign_bits = ((bits >> np.uint32(31)) & np.uint32(1)).astype(np.uint8)
    exponents = ((bits >> np.uint32(23)) & np.uint32(0xFF)).astype(np.uint8)
    numeric_min = float(np.min(pixels))
    numeric_max = float(np.max(pixels))
    all_sign_zero = bool(np.all(sign_bits == 0))
    range_eligible = numeric_min >= min_value and numeric_max <= max_value

    _modes, payloads, _carries = choose_group_payloads(
        bits,
        pixels,
        tile_size,
        previous_bits,
        "mantissa",
    )

    height, width, _channels = bits.shape
    current_bits = 0.0
    routed_bits = 0.0
    selected = Counter()
    unique_histogram = Counter()
    tile_rows = []
    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            sl = np.s_[y:y + tile_size, x:x + tile_size]
            current_sign = best_sign_cost(payloads["sign"][sl])
            current_exponent = best_bit_range_cost(payloads["exponent"][sl], EXPONENT_BITS)
            current = current_sign + current_exponent
            current_bits += current

            tile_sign_zero = bool(np.all(sign_bits[sl] == 0))
            tile_range_eligible = (
                float(np.min(pixels[sl])) >= min_value
                and float(np.max(pixels[sl])) <= max_value
            )
            exponent_cost, details = exponent_alphabet_cost(
                exponents[sl],
                exponent_header_bits,
                8.0,
            )
            candidate = route_header_bits + exponent_cost
            if tile_sign_zero:
                candidate += sign_flag_bits
            else:
                candidate += sign_flag_bits + current_sign

            if tile_sign_zero and tile_range_eligible and candidate < current:
                routed = candidate
                selected["bounded_alphabet"] += 1
            else:
                routed = current
                selected["gdx"] += 1
            routed_bits += routed
            unique_histogram[str(int(details["unique_exponents"]))] += 1
            tile_rows.append(
                {
                    "y": y,
                    "x": x,
                    "tile_sign_zero": tile_sign_zero,
                    "tile_range_eligible": tile_range_eligible,
                    "current_bits": current,
                    "candidate_bits": candidate,
                    "selected_bits": routed,
                    "exponent_details": details,
                }
            )

    samples = int(bits.size)
    return {
        "image": path.name,
        "shape": list(bits.shape),
        "crop_size": crop_size,
        "tile_size": tile_size,
        "previous_bits": previous_bits,
        "min_value": min_value,
        "max_value": max_value,
        "numeric_min": numeric_min,
        "numeric_max": numeric_max,
        "all_sign_zero": all_sign_zero,
        "range_eligible": range_eligible,
        "current_bits": current_bits,
        "routed_bits": routed_bits,
        "current_bps": current_bits / samples,
        "routed_bps": routed_bits / samples,
        "gain": current_bits / routed_bits if routed_bits else 1.0,
        "selected": dict(selected),
        "unique_exponent_histogram": dict(unique_histogram),
        "tiles": tile_rows,
    }


def geomean(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", default="*.exr")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--tile-size", type=int, default=128)
    parser.add_argument("--previous-bits", type=int, default=4)
    parser.add_argument("--min-value", type=float, default=0.0)
    parser.add_argument("--max-value", type=float, default=4.0)
    parser.add_argument("--sign-flag-bits", type=float, default=1.0)
    parser.add_argument("--exponent-header-bits", type=float, default=8.0)
    parser.add_argument("--route-header-bits", type=float, default=4.0)
    parser.add_argument("--no-save", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = sorted(DATA_DIR.glob(args.glob))
    if args.limit:
        paths = paths[:args.limit]
    if not paths:
        raise RuntimeError(f"no images matched {DATA_DIR / args.glob}")

    rows = []
    print(
        f"{'image':43s} {'eligible':>8s} {'current':>9s} "
        f"{'routed':>9s} {'gain':>8s} range selected"
    )
    print("-" * 128)
    for path in paths:
        row = summarize_image(
            path,
            args.crop_size,
            args.tile_size,
            args.previous_bits,
            args.min_value,
            args.max_value,
            args.sign_flag_bits,
            args.exponent_header_bits,
            args.route_header_bits,
        )
        rows.append(row)
        print(
            f"{path.stem:43s} "
            f"{str(row['range_eligible'] and row['all_sign_zero']):>8s} "
            f"{row['current_bps']:8.4f} "
            f"{row['routed_bps']:8.4f} "
            f"{row['gain']:7.4f} "
            f"[{row['numeric_min']:.5g}, {row['numeric_max']:.5g}] "
            f"{row['selected']}"
        )

    print("-" * 128)
    print(
        f"mean current={sum(row['current_bps'] for row in rows) / len(rows):.4f} "
        f"routed={sum(row['routed_bps'] for row in rows) / len(rows):.4f} "
        f"geomean_gain={geomean([row['gain'] for row in rows]):.4f}"
    )

    if not args.no_save:
        safe_glob = args.glob.replace("*", "star").replace(".", "_")
        output = RESULTS_DIR / (
            f"bounded_value_route_{safe_glob}_tile{args.tile_size}"
            f"_prev{args.previous_bits}_crop{args.crop_size}"
            f"_range{args.min_value:g}-{args.max_value:g}.json"
        )
        output.write_text(json.dumps(rows, indent=2))
        print(f"\nSaved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
