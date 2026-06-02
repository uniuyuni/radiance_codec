"""Estimate carry-aware ordered-float field deltas.

The field-wise ordered-delta probe is a useful upper bound, but plain
subtraction carries across float-field groups:

    actual_ordered = predicted_ordered + residual mod 2^32

This probe makes the field split reversible by decoding groups from low bits to
high bits.  For each group:

    residual_group = actual_group - prediction_group - incoming_carry mod base

The decoder can reconstruct the group and the outgoing carry before moving to
the next group.  Predictor modes may vary per field because the mixed prediction
word is defined group-by-group.
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

from analyze_structural_predictors import (  # noqa: E402
    DATA_DIR,
    FIELD_GROUPS,
    float_to_bits,
    load_blosc_ratios,
    predictors,
    read_exr,
)
from evaluate_structural_delta_context import (  # noqa: E402
    bits_to_ordered,
    context_bitplane_cost,
    delta_predictors,
)
from evaluate_structural_delta_valid import best_xor_tile_cost  # noqa: E402
from evaluate_structural_spatial_context import spatial_contexts  # noqa: E402

RESULTS_DIR = ROOT / "results"
LOW_TO_HIGH_FIELDS = tuple(
    sorted(FIELD_GROUPS.items(), key=lambda item: min(item[1]))
)


def field_layout(bit_indices: tuple[int, ...]) -> tuple[int, int, np.uint32]:
    low_bit = min(bit_indices)
    high_bit = max(bit_indices)
    width = high_bit - low_bit + 1
    if width != len(bit_indices):
        raise ValueError(f"field is not contiguous: {bit_indices}")
    return low_bit, width, np.uint32((1 << width) - 1)


def extract_field(values: np.ndarray, bit_indices: tuple[int, ...]) -> np.ndarray:
    low_bit, _width, mask = field_layout(bit_indices)
    return ((values >> np.uint32(low_bit)) & mask).astype(np.uint32, copy=False)


def context_bitplane_cost_low_to_high(
    bits: np.ndarray,
    bit_indices: tuple[int, ...],
    previous_bits: int,
) -> float:
    """KT-estimate bit cost for a low-to-high bit decode schedule.

    Lower residual bits are the only cross-plane context used here, matching the
    carry-aware reconstruction order.
    """

    cost = 0.0
    for bit_index in sorted(bit_indices):
        plane = ((bits >> bit_index) & 1).astype(np.uint8, copy=False)
        context = spatial_contexts(plane)
        for offset in range(1, previous_bits + 1):
            previous_bit = bit_index - offset
            if previous_bit < 0:
                continue
            previous_plane = ((bits >> previous_bit) & 1).astype(
                np.uint8,
                copy=False,
            )
            context |= previous_plane << (3 + offset)
        context_flat = context.reshape(-1)
        plane_flat = plane.reshape(-1)
        context_count = 16 << previous_bits
        totals = np.bincount(context_flat, minlength=context_count)
        ones = np.bincount(context_flat, weights=plane_flat, minlength=context_count)
        for context_id in range(context_count):
            total = int(totals[context_id])
            if total == 0:
                continue
            one_count = int(ones[context_id])
            zero_count = total - one_count
            log_probability = (
                math.lgamma(one_count + 0.5)
                + math.lgamma(zero_count + 0.5)
                - math.lgamma(total + 1.0)
                - 2.0 * math.lgamma(0.5)
            )
            cost += -log_probability / math.log(2.0)
    return cost


def carry_delta_field(
    actual_field: np.ndarray,
    prediction_field: np.ndarray,
    carry: np.ndarray,
    mask: np.uint32,
) -> np.ndarray:
    return (actual_field - prediction_field - carry).astype(np.uint32) & mask


def outgoing_carry(
    prediction_field: np.ndarray,
    residual_field: np.ndarray,
    carry: np.ndarray,
    width: int,
) -> np.ndarray:
    total = (
        prediction_field.astype(np.uint64)
        + residual_field.astype(np.uint64)
        + carry.astype(np.uint64)
    )
    return (total >> np.uint64(width)).astype(np.uint32)


def best_carry_delta_tile_cost(
    ordered_tile: np.ndarray,
    delta_prediction_bits: dict[str, np.ndarray],
    sl,
    previous_bits: int,
    mode_overhead_bits: int,
    final_high_context: bool,
) -> tuple[float, Counter]:
    carry = np.zeros_like(ordered_tile, dtype=np.uint32)
    residual_context_bits = np.zeros_like(ordered_tile, dtype=np.uint32)
    total_bits = 0.0
    modes = Counter()

    for field, bit_indices in LOW_TO_HIGH_FIELDS:
        low_bit, width, mask = field_layout(bit_indices)
        actual_field = extract_field(ordered_tile, bit_indices)
        candidates: dict[str, tuple[float, np.ndarray, np.ndarray]] = {}

        zero_prediction = np.zeros_like(actual_field, dtype=np.uint32)
        zero_residual = carry_delta_field(
            actual_field,
            zero_prediction,
            carry,
            mask,
        )
        zero_bits = residual_context_bits | (
            zero_residual.astype(np.uint32) << np.uint32(low_bit)
        )
        candidates["raw"] = (
            float(ordered_tile.size * width),
            zero_prediction,
            zero_residual,
        )
        candidates["delta_zero"] = (
            context_bitplane_cost_low_to_high(
                zero_bits,
                bit_indices,
                previous_bits,
            ),
            zero_prediction,
            zero_residual,
        )

        for name, prediction in delta_prediction_bits.items():
            prediction_field = extract_field(prediction[sl], bit_indices)
            residual_field = carry_delta_field(
                actual_field,
                prediction_field,
                carry,
                mask,
            )
            residual_bits = residual_context_bits | (
                residual_field.astype(np.uint32) << np.uint32(low_bit)
            )
            candidates[name] = (
                context_bitplane_cost_low_to_high(
                    residual_bits,
                    bit_indices,
                    previous_bits,
                ),
                prediction_field,
                residual_field,
            )

        best_mode, (best_cost, best_prediction, best_residual) = min(
            candidates.items(),
            key=lambda item: item[1][0],
        )
        total_bits += best_cost + mode_overhead_bits
        modes[f"{field}:{best_mode}"] += 1
        residual_context_bits |= best_residual.astype(np.uint32) << np.uint32(low_bit)
        carry = outgoing_carry(best_prediction, best_residual, carry, width)

    if final_high_context:
        total_bits = 0.0
        for _field, bit_indices in FIELD_GROUPS.items():
            total_bits += (
                context_bitplane_cost(
                    residual_context_bits,
                    bit_indices,
                    previous_bits,
                    channel_context=False,
                )
                + mode_overhead_bits
            )

    return total_bits, modes


def estimate_image(
    path: Path,
    tile_size: int,
    previous_bits: int,
    carry_mode_overhead_bits: int,
    xor_mode_overhead_bits: int,
    include_xor_fallback: bool,
    final_high_context: bool,
) -> dict:
    pixels = read_exr(path)
    bits = float_to_bits(pixels)
    ordered = bits_to_ordered(bits)
    delta_prediction_bits = delta_predictors(bits, pixels)
    xor_predictor_bits = predictors(pixels)
    height, width, channels = bits.shape
    raw_bits = bits.size * 32
    total_bits = 0.0
    pick_histogram = Counter()
    mode_histogram = Counter()

    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            sl = np.s_[y:y + tile_size, x:x + tile_size]
            tile = bits[sl]
            ordered_tile = ordered[sl]
            carry_cost, carry_modes = best_carry_delta_tile_cost(
                ordered_tile,
                delta_prediction_bits,
                sl,
                previous_bits,
                carry_mode_overhead_bits,
                final_high_context,
            )
            best_cost = carry_cost
            best_pick = "carry_delta"
            best_modes = carry_modes

            if include_xor_fallback:
                xor_cost, xor_modes = best_xor_tile_cost(
                    tile,
                    xor_predictor_bits,
                    sl,
                    previous_bits,
                    xor_mode_overhead_bits,
                )
                if xor_cost < best_cost:
                    best_cost = xor_cost
                    best_pick = "xor"
                    best_modes = xor_modes

            total_bits += best_cost
            pick_histogram[best_pick] += 1
            mode_histogram.update(best_modes)

    return {
        "image": path.name,
        "shape": [height, width, channels],
        "tile_size": tile_size,
        "previous_bits": previous_bits,
        "raw_bits": raw_bits,
        "estimated_bits": total_bits,
        "ratio": raw_bits / total_bits,
        "pick_histogram": dict(pick_histogram),
        "mode_histogram": dict(mode_histogram),
    }


def geomean(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", default="*.exr")
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--previous-bits", type=int, default=4)
    parser.add_argument("--carry-mode-overhead-bits", type=int, default=5)
    parser.add_argument("--xor-mode-overhead-bits", type=int, default=5)
    parser.add_argument("--no-xor-fallback", action="store_true")
    parser.add_argument("--final-high-context", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = sorted(DATA_DIR.glob(args.glob))
    if not paths:
        raise RuntimeError(f"no images matched {DATA_DIR / args.glob}")
    blosc = load_blosc_ratios()
    results = []
    print(
        f"{'image':43s} {'carry':>9s} {'Blosc':>9s} "
        f"{'hybrid':>9s} {'pick':>8s}"
    )
    print("-" * 86)
    for path in paths:
        result = estimate_image(
            path,
            args.tile_size,
            args.previous_bits,
            args.carry_mode_overhead_bits,
            args.xor_mode_overhead_bits,
            include_xor_fallback=not args.no_xor_fallback,
            final_high_context=args.final_high_context,
        )
        blosc_ratio = blosc.get(path.stem)
        hybrid_ratio = max(result["ratio"], blosc_ratio or 0.0)
        pick = "carry" if result["ratio"] >= (blosc_ratio or 0.0) else "Blosc"
        result["blosc_ratio"] = blosc_ratio
        result["hybrid_ratio"] = hybrid_ratio
        result["hybrid_pick"] = pick
        results.append(result)
        blosc_text = f"{blosc_ratio:8.2f}x" if blosc_ratio is not None else "       -"
        print(
            f"{path.stem:43s} {result['ratio']:8.2f}x "
            f"{blosc_text} {hybrid_ratio:8.2f}x {pick:>8s}"
        )
    print("-" * 86)
    for key in ["ratio", "blosc_ratio", "hybrid_ratio"]:
        values = [result[key] for result in results if result.get(key) is not None]
        print(f"geomean {key:12s}: {geomean(values):.3f}x")

    output = RESULTS_DIR / (
        f"structural_delta_carry_tile{args.tile_size}_prev{args.previous_bits}"
        f"{'_highctx' if args.final_high_context else ''}"
        f"{'_noxor' if args.no_xor_fallback else ''}.json"
    )
    output.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
