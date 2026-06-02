"""Estimate field-mixed ordered-float prediction with whole-word deltas.

The pure field-wise ordered-delta probe is not directly decodable because each
field may choose a different predictor while subtraction carries across fields.
This probe moves the field choice to the prediction word instead:

    mixed_prediction[field] = predictor_for_that_field[field]
    residual = actual_ordered - mixed_prediction mod 2^32

The residual is a normal whole-word delta, so reconstruction is unambiguous, but
the predictor can still use different modes for sign/exponent/mantissa groups.
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
)
from evaluate_structural_delta_valid import best_xor_tile_cost  # noqa: E402

RESULTS_DIR = ROOT / "results"


def field_mask(bit_indices: tuple[int, ...]) -> np.uint32:
    mask = np.uint32(0)
    for bit in bit_indices:
        mask |= np.uint32(1) << np.uint32(bit)
    return mask


FIELD_MASKS = {
    field: field_mask(bit_indices)
    for field, bit_indices in FIELD_GROUPS.items()
}


def build_mixed_prediction(
    prediction_tiles: dict[str, np.ndarray],
    field_modes: dict[str, str],
) -> np.ndarray:
    first = next(iter(prediction_tiles.values()))
    mixed = np.zeros_like(first, dtype=np.uint32)
    for field, mode in field_modes.items():
        mask = FIELD_MASKS[field]
        mixed |= prediction_tiles[mode] & mask
    return mixed


def residual_cost(
    residual: np.ndarray,
    previous_bits: int,
    mode_overhead_bits: int,
    raw_fallback: bool,
) -> tuple[float, Counter]:
    tile_values = residual.size
    total = 0.0
    coding_histogram = Counter()
    for field, bit_indices in FIELD_GROUPS.items():
        context_cost = context_bitplane_cost(
            residual,
            bit_indices,
            previous_bits,
            channel_context=False,
        )
        raw_cost = tile_values * len(bit_indices)
        if raw_fallback and raw_cost < context_cost:
            total += raw_cost + mode_overhead_bits
            coding_histogram[f"{field}:raw"] += 1
        else:
            total += context_cost + mode_overhead_bits
            coding_histogram[f"{field}:context"] += 1
    return total, coding_histogram


def score_modes(
    ordered_tile: np.ndarray,
    prediction_tiles: dict[str, np.ndarray],
    field_modes: dict[str, str],
    previous_bits: int,
    mode_overhead_bits: int,
    raw_fallback: bool,
) -> tuple[float, Counter]:
    mixed_prediction = build_mixed_prediction(prediction_tiles, field_modes)
    residual = ordered_delta(ordered_tile, mixed_prediction)
    return residual_cost(residual, previous_bits, mode_overhead_bits, raw_fallback)


def initial_field_modes(
    ordered_tile: np.ndarray,
    prediction_tiles: dict[str, np.ndarray],
    previous_bits: int,
) -> dict[str, str]:
    modes = {}
    for field, bit_indices in FIELD_GROUPS.items():
        candidates = {}
        for mode, prediction in prediction_tiles.items():
            residual = ordered_delta(ordered_tile, prediction)
            candidates[mode] = context_bitplane_cost(
                residual,
                bit_indices,
                previous_bits,
                channel_context=False,
            )
        modes[field] = min(candidates, key=candidates.__getitem__)
    return modes


def refine_field_modes(
    ordered_tile: np.ndarray,
    prediction_tiles: dict[str, np.ndarray],
    field_modes: dict[str, str],
    previous_bits: int,
    mode_overhead_bits: int,
    raw_fallback: bool,
    refine_iters: int,
) -> tuple[float, dict[str, str], Counter]:
    best_cost, best_coding = score_modes(
        ordered_tile,
        prediction_tiles,
        field_modes,
        previous_bits,
        mode_overhead_bits,
        raw_fallback,
    )
    mode_names = tuple(prediction_tiles)

    for _ in range(refine_iters):
        changed = False
        for field in FIELD_GROUPS:
            field_best_mode = field_modes[field]
            field_best_cost = best_cost
            field_best_coding = best_coding
            for mode in mode_names:
                if mode == field_modes[field]:
                    continue
                trial_modes = dict(field_modes)
                trial_modes[field] = mode
                trial_cost, trial_coding = score_modes(
                    ordered_tile,
                    prediction_tiles,
                    trial_modes,
                    previous_bits,
                    mode_overhead_bits,
                    raw_fallback,
                )
                if trial_cost < field_best_cost:
                    field_best_mode = mode
                    field_best_cost = trial_cost
                    field_best_coding = trial_coding
            if field_best_mode != field_modes[field]:
                field_modes[field] = field_best_mode
                best_cost = field_best_cost
                best_coding = field_best_coding
                changed = True
        if not changed:
            break

    return best_cost, field_modes, best_coding


def best_mixed_tile_cost(
    ordered_tile: np.ndarray,
    delta_prediction_bits: dict[str, np.ndarray],
    sl,
    previous_bits: int,
    mode_overhead_bits: int,
    raw_fallback: bool,
    refine_iters: int,
) -> tuple[float, Counter]:
    prediction_tiles = {
        "delta_zero": np.zeros_like(ordered_tile, dtype=np.uint32),
        **{name: prediction[sl] for name, prediction in delta_prediction_bits.items()},
    }
    field_modes = initial_field_modes(
        ordered_tile,
        prediction_tiles,
        previous_bits,
    )
    cost, field_modes, coding_histogram = refine_field_modes(
        ordered_tile,
        prediction_tiles,
        field_modes,
        previous_bits,
        mode_overhead_bits,
        raw_fallback,
        refine_iters,
    )
    modes = Counter(f"{field}:{mode}" for field, mode in field_modes.items())
    modes.update(coding_histogram)
    return cost, modes


def estimate_image(
    path: Path,
    tile_size: int,
    previous_bits: int,
    mixed_mode_overhead_bits: int,
    xor_mode_overhead_bits: int,
    include_xor_fallback: bool,
    raw_fallback: bool,
    refine_iters: int,
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
            mixed_cost, mixed_modes = best_mixed_tile_cost(
                ordered_tile,
                delta_prediction_bits,
                sl,
                previous_bits,
                mixed_mode_overhead_bits,
                raw_fallback,
                refine_iters,
            )
            best_cost = mixed_cost
            best_pick = "mixed_delta"
            best_modes = mixed_modes

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
    parser.add_argument("--mixed-mode-overhead-bits", type=int, default=5)
    parser.add_argument("--xor-mode-overhead-bits", type=int, default=5)
    parser.add_argument("--no-xor-fallback", action="store_true")
    parser.add_argument("--no-raw-fallback", action="store_true")
    parser.add_argument("--refine-iters", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = sorted(DATA_DIR.glob(args.glob))
    if not paths:
        raise RuntimeError(f"no images matched {DATA_DIR / args.glob}")
    blosc = load_blosc_ratios()
    results = []
    print(
        f"{'image':43s} {'mixed':>9s} {'Blosc':>9s} "
        f"{'hybrid':>9s} {'pick':>8s}"
    )
    print("-" * 86)
    for path in paths:
        result = estimate_image(
            path,
            args.tile_size,
            args.previous_bits,
            args.mixed_mode_overhead_bits,
            args.xor_mode_overhead_bits,
            include_xor_fallback=not args.no_xor_fallback,
            raw_fallback=not args.no_raw_fallback,
            refine_iters=args.refine_iters,
        )
        blosc_ratio = blosc.get(path.stem)
        hybrid_ratio = max(result["ratio"], blosc_ratio or 0.0)
        pick = "mixed" if result["ratio"] >= (blosc_ratio or 0.0) else "Blosc"
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
        f"structural_delta_mixed_tile{args.tile_size}_prev{args.previous_bits}"
        f"_refine{args.refine_iters}"
        f"{'_noxor' if args.no_xor_fallback else ''}"
        f"{'_noraw' if args.no_raw_fallback else ''}.json"
    )
    output.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
