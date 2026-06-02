"""Valid estimate for mixing XOR field modes with whole-word delta modes.

Ordered-float delta residuals are whole-word residuals:

    actual_ordered = predicted_ordered + residual mod 2^32

Therefore the delta predictor mode cannot vary independently by float field;
otherwise carries across bit groups would make reconstruction ambiguous. This
probe compares:

* field-wise XOR residual modes, which are valid per field; and
* tile-wise ordered-delta modes, which apply to all 32 bits in the tile.
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

RESULTS_DIR = ROOT / "results"
ALL_BITS = tuple(range(31, -1, -1))
INT32_MIN = -(1 << 31)
INT32_MAX = (1 << 31) - 1


def ordered_zigzag_delta(
    actual: np.ndarray,
    prediction: np.ndarray,
) -> np.ndarray | None:
    diff = actual.astype(np.int64) - prediction.astype(np.int64)
    if bool(np.any(diff < INT32_MIN)) or bool(np.any(diff > INT32_MAX)):
        return None
    encoded = (diff << 1) ^ (diff >> 31)
    return encoded.astype(np.uint32)


def best_xor_tile_cost(
    tile: np.ndarray,
    predictor_bits: dict[str, np.ndarray],
    sl,
    previous_bits: int,
    mode_overhead_bits: int,
) -> tuple[float, Counter]:
    tile_values = tile.size
    total = 0.0
    modes = Counter()
    for _field, bit_indices in FIELD_GROUPS.items():
        candidates = {"raw": tile_values * len(bit_indices)}
        for name, pred in predictor_bits.items():
            residual = tile ^ pred[sl]
            candidates[f"xor_{name}"] = context_bitplane_cost(
                residual,
                bit_indices,
                previous_bits,
                channel_context=False,
            )
        best = min(candidates, key=candidates.__getitem__)
        total += candidates[best] + mode_overhead_bits
        modes[best] += 1
    return total, modes


def best_delta_tile_cost(
    ordered_tile: np.ndarray,
    delta_prediction_bits: dict[str, np.ndarray],
    sl,
    previous_bits: int,
    mode_overhead_bits: int,
    zigzag: bool,
) -> tuple[float, str]:
    tile_values = ordered_tile.size
    candidates = {"raw": tile_values * 32}
    for name, pred in delta_prediction_bits.items():
        if zigzag:
            residual = ordered_zigzag_delta(ordered_tile, pred[sl])
            if residual is None:
                continue
        else:
            residual = ordered_delta(ordered_tile, pred[sl])
        candidates[name] = context_bitplane_cost(
            residual,
            ALL_BITS,
            previous_bits,
            channel_context=False,
        )
    best = min(candidates, key=candidates.__getitem__)
    return candidates[best] + mode_overhead_bits, best


def estimate_image(
    path: Path,
    tile_size: int,
    previous_bits: int,
    xor_mode_overhead_bits: int,
    delta_mode_overhead_bits: int,
    zigzag: bool,
) -> dict:
    pixels = read_exr(path)
    bits = float_to_bits(pixels)
    ordered = bits_to_ordered(bits)
    xor_predictor_bits = predictors(pixels)
    delta_prediction_bits = delta_predictors(bits, pixels)
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
            xor_cost, xor_modes = best_xor_tile_cost(
                tile,
                xor_predictor_bits,
                sl,
                previous_bits,
                xor_mode_overhead_bits,
            )
            delta_cost, delta_mode = best_delta_tile_cost(
                ordered_tile,
                delta_prediction_bits,
                sl,
                previous_bits,
                delta_mode_overhead_bits,
                zigzag,
            )
            if delta_cost < xor_cost:
                total_bits += delta_cost
                pick_histogram["delta"] += 1
                mode_histogram[delta_mode] += 1
            else:
                total_bits += xor_cost
                pick_histogram["xor"] += 1
                mode_histogram.update(xor_modes)

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
    parser.add_argument("--xor-mode-overhead-bits", type=int, default=5)
    parser.add_argument("--delta-mode-overhead-bits", type=int, default=4)
    parser.add_argument("--zigzag", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = sorted(DATA_DIR.glob(args.glob))
    if not paths:
        raise RuntimeError(f"no images matched {DATA_DIR / args.glob}")
    blosc = load_blosc_ratios()
    results = []
    print(
        f"{'image':43s} {'valid':>9s} {'Blosc':>9s} "
        f"{'hybrid':>9s} {'pick':>8s}"
    )
    print("-" * 86)
    for path in paths:
        result = estimate_image(
            path,
            args.tile_size,
            args.previous_bits,
            args.xor_mode_overhead_bits,
            args.delta_mode_overhead_bits,
            args.zigzag,
        )
        blosc_ratio = blosc.get(path.stem)
        hybrid_ratio = max(result["ratio"], blosc_ratio or 0.0)
        pick = "valid" if result["ratio"] >= (blosc_ratio or 0.0) else "Blosc"
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
        f"structural_delta_valid_tile{args.tile_size}_prev{args.previous_bits}"
        f"{'_zigzag' if args.zigzag else ''}.json"
    )
    output.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
