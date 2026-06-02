"""Estimate field-wise ordered deltas with explicit carry side bits.

For a predictor p and actual ordered value a:

    d = a - p mod 2^32

The upper-bound field-wise probe stores selected bit fields from d, but when a
different predictor is chosen per field the incoming carry for each field is no
longer implied by lower fields.  This probe stores that missing carry bit as
side information:

    actual_field = prediction_field + residual_field + carry_in mod base

The carry bit is coded with the same adaptive spatial binary model.  This keeps
field-wise predictor selection fully decodable while measuring the real price
of the carry ambiguity.
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
    ordered_delta,
    ordered_field_delta,
    ordered_xor_predictors,
)
from evaluate_structural_delta_valid import best_xor_tile_cost  # noqa: E402
from evaluate_structural_spatial_context import spatial_contexts  # noqa: E402

RESULTS_DIR = ROOT / "results"


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


def kt_binary_cost_from_counts(ones: np.ndarray, totals: np.ndarray) -> float:
    cost = 0.0
    for context_id in range(len(totals)):
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


def carry_plane_cost(carry: np.ndarray, channel_context: bool) -> float:
    plane = carry.astype(np.uint8, copy=False)
    context = spatial_contexts(plane)
    context_count = 16
    if channel_context:
        channel_plane = np.zeros_like(plane)
        channel_plane[..., 1:] = plane[..., :-1]
        context |= channel_plane << 4
        context_count <<= 1
    context_flat = context.reshape(-1)
    plane_flat = plane.reshape(-1)
    totals = np.bincount(context_flat, minlength=context_count)
    ones = np.bincount(context_flat, weights=plane_flat, minlength=context_count)
    return kt_binary_cost_from_counts(ones, totals)


def incoming_carry_bits(
    actual: np.ndarray,
    prediction: np.ndarray,
    residual: np.ndarray,
    bit_indices: tuple[int, ...],
) -> np.ndarray:
    _low_bit, _width, mask = field_layout(bit_indices)
    actual_field = extract_field(actual, bit_indices)
    prediction_field = extract_field(prediction, bit_indices)
    residual_field = extract_field(residual, bit_indices)
    without_carry = (prediction_field + residual_field) & mask
    with_carry = (without_carry + np.uint32(1)) & mask
    carry = np.where(without_carry == actual_field, 0, 1).astype(np.uint8)
    if bool(np.any((carry == 1) & (with_carry != actual_field))):
        raise RuntimeError("field residual cannot be repaired by a one-bit carry")
    return carry


def best_sidecarry_tile_cost(
    ordered_tile_source: np.ndarray,
    ordered_tile: np.ndarray,
    delta_prediction_bits: dict[str, np.ndarray],
    xor_predictor_bits: dict[str, np.ndarray],
    sl,
    previous_bits: int,
    mode_overhead_bits: int,
    carry_channel_context: bool,
) -> tuple[float, Counter, Counter]:
    tile_values = ordered_tile.size
    total = 0.0
    modes = Counter()
    carry_histogram = Counter()

    for field, bit_indices in FIELD_GROUPS.items():
        low_bit, width, _mask = field_layout(bit_indices)
        raw_cost = float(tile_values * width)
        candidates: dict[str, tuple[float, float, float, np.ndarray | None]] = {
            "raw": (raw_cost, raw_cost, 0.0, None),
        }
        for name, prediction in xor_predictor_bits.items():
            residual = ordered_tile_source ^ prediction[sl]
            residual_cost = context_bitplane_cost(
                residual,
                bit_indices,
                previous_bits,
                channel_context=False,
            )
            candidates[f"xor_{name}"] = (residual_cost, residual_cost, 0.0, None)
        for name, prediction in delta_prediction_bits.items():
            prediction_tile = prediction[sl]
            field_local = ordered_field_delta(
                ordered_tile,
                prediction_tile,
                bit_indices,
            )
            field_local_cost = context_bitplane_cost(
                field_local,
                bit_indices,
                previous_bits,
                channel_context=False,
            )
            candidates[f"field_{name}"] = (
                field_local_cost,
                field_local_cost,
                0.0,
                None,
            )

            residual = ordered_delta(ordered_tile, prediction_tile)
            residual_cost = context_bitplane_cost(
                residual,
                bit_indices,
                previous_bits,
                channel_context=False,
            )
            carry_cost = 0.0
            carry = None
            if low_bit > 0:
                carry = incoming_carry_bits(
                    ordered_tile,
                    prediction_tile,
                    residual,
                    bit_indices,
                )
                carry_cost = carry_plane_cost(carry, carry_channel_context)
            candidates[name] = (
                residual_cost + carry_cost,
                residual_cost,
                carry_cost,
                carry,
            )

        best_mode, (best_cost, _residual_cost, best_carry_cost, best_carry) = min(
            candidates.items(),
            key=lambda item: item[1][0],
        )
        total += best_cost + mode_overhead_bits
        modes[f"{field}:{best_mode}"] += 1
        if best_carry is not None:
            carry_histogram[f"{field}:cost_bits"] += best_carry_cost
            carry_histogram[f"{field}:ones"] += int(np.sum(best_carry))
            carry_histogram[f"{field}:total"] += int(best_carry.size)

    return total, modes, carry_histogram


def estimate_image(
    path: Path,
    tile_size: int,
    previous_bits: int,
    sidecarry_mode_overhead_bits: int,
    xor_mode_overhead_bits: int,
    include_xor_fallback: bool,
    carry_channel_context: bool,
) -> dict:
    pixels = read_exr(path)
    bits = float_to_bits(pixels)
    ordered = bits_to_ordered(bits)
    delta_prediction_bits = delta_predictors(bits, pixels, causal=True)
    xor_predictor_bits = ordered_xor_predictors(bits, pixels, causal=True)
    delta_prediction_bits.pop("delta_channel_green", None)
    xor_predictor_bits.pop("channel_green", None)
    height, width, channels = bits.shape
    raw_bits = bits.size * 32
    total_bits = 0.0
    pick_histogram = Counter()
    mode_histogram = Counter()
    carry_histogram = Counter()

    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            sl = np.s_[y:y + tile_size, x:x + tile_size]
            ordered_tile = ordered[sl]
            tile = ordered_tile
            sidecarry_cost, sidecarry_modes, sidecarry_carries = (
                best_sidecarry_tile_cost(
                    ordered_tile,
                    ordered_tile,
                    delta_prediction_bits,
                    xor_predictor_bits,
                    sl,
                    previous_bits,
                    sidecarry_mode_overhead_bits,
                    carry_channel_context,
                )
            )
            best_cost = sidecarry_cost
            best_pick = "sidecarry_delta"
            best_modes = sidecarry_modes
            best_carries = sidecarry_carries

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
                    best_carries = Counter()

            total_bits += best_cost
            pick_histogram[best_pick] += 1
            mode_histogram.update(best_modes)
            carry_histogram.update(best_carries)

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
        "carry_histogram": dict(carry_histogram),
    }


def geomean(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", default="*.exr")
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--previous-bits", type=int, default=4)
    parser.add_argument("--sidecarry-mode-overhead-bits", type=int, default=5)
    parser.add_argument("--xor-mode-overhead-bits", type=int, default=5)
    parser.add_argument("--no-xor-fallback", action="store_true")
    parser.add_argument("--carry-channel-context", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = sorted(DATA_DIR.glob(args.glob))
    if not paths:
        raise RuntimeError(f"no images matched {DATA_DIR / args.glob}")
    blosc = load_blosc_ratios()
    results = []
    print(
        f"{'image':43s} {'side':>9s} {'Blosc':>9s} "
        f"{'hybrid':>9s} {'pick':>8s}"
    )
    print("-" * 86)
    for path in paths:
        result = estimate_image(
            path,
            args.tile_size,
            args.previous_bits,
            args.sidecarry_mode_overhead_bits,
            args.xor_mode_overhead_bits,
            include_xor_fallback=not args.no_xor_fallback,
            carry_channel_context=args.carry_channel_context,
        )
        blosc_ratio = blosc.get(path.stem)
        hybrid_ratio = max(result["ratio"], blosc_ratio or 0.0)
        pick = "side" if result["ratio"] >= (blosc_ratio or 0.0) else "Blosc"
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
        f"structural_delta_sidecarry_tile{args.tile_size}_prev{args.previous_bits}"
        f"{'_carrychan' if args.carry_channel_context else ''}"
        f"{'_noxor' if args.no_xor_fallback else ''}.json"
    )
    output.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
