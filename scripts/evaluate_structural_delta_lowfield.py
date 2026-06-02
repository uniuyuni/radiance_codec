"""Estimate low-to-high field reconstruction for ordered-float deltas.

For each field we store bits of the whole-word residual:

    residual = actual_ordered - prediction_ordered mod 2^32

Fields may choose different predictors.  Decoding remains possible without a
carry side stream by reconstructing actual fields from low to high.  For the
current predictor, the incoming carry is computed from already reconstructed
lower actual bits and the predictor's lower bits.

The entropy model is restricted to field-local higher residual bits, because
fields are decoded low-to-high even though bits inside each field are decoded
high-to-low.
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
    delta_predictors,
    ordered_delta,
)
from evaluate_structural_spatial_context import spatial_contexts  # noqa: E402

RESULTS_DIR = ROOT / "results"


def kt_cost(ones: int, total: int) -> float:
    if total == 0:
        return 0.0
    zeros = total - ones
    log_probability = (
        math.lgamma(ones + 0.5)
        + math.lgamma(zeros + 0.5)
        - math.lgamma(total + 1.0)
        - 2.0 * math.lgamma(0.5)
    )
    return -log_probability / math.log(2.0)


def context_bitplane_cost_field_local(
    bits: np.ndarray,
    bit_indices: tuple[int, ...],
    previous_bits: int,
) -> float:
    field_bits = set(bit_indices)
    cost = 0.0
    for bit_index in bit_indices:
        plane = ((bits >> bit_index) & 1).astype(np.uint8, copy=False)
        context = spatial_contexts(plane)
        for offset in range(1, previous_bits + 1):
            previous_bit = bit_index + offset
            if previous_bit not in field_bits:
                continue
            previous_plane = ((bits >> previous_bit) & 1).astype(np.uint8, copy=False)
            context |= previous_plane << (3 + offset)
        context_flat = context.reshape(-1)
        plane_flat = plane.reshape(-1)
        context_count = 16 << previous_bits
        totals = np.bincount(context_flat, minlength=context_count)
        ones = np.bincount(context_flat, weights=plane_flat, minlength=context_count)
        for context_id in range(context_count):
            cost += kt_cost(int(ones[context_id]), int(totals[context_id]))
    return cost


def estimate_image(
    path: Path,
    tile_size: int,
    previous_bits: int,
    mode_overhead_bits: int,
) -> dict:
    pixels = read_exr(path)
    bits = float_to_bits(pixels)
    ordered = bits_to_ordered(bits)
    delta_prediction_bits = delta_predictors(bits, pixels)
    xor_predictor_bits = predictors(pixels)
    height, width, channels = bits.shape
    raw_bits = bits.size * 32
    total_bits = 0.0
    mode_histogram = Counter()
    field_mode_histogram = {field: Counter() for field in FIELD_GROUPS}

    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            sl = np.s_[y:y + tile_size, x:x + tile_size]
            tile = bits[sl]
            ordered_tile = ordered[sl]
            tile_values = tile.size
            for field, bit_indices in FIELD_GROUPS.items():
                candidates = {"raw": float(tile_values * len(bit_indices))}
                for name, prediction in xor_predictor_bits.items():
                    residual = tile ^ prediction[sl]
                    candidates[f"xor_{name}"] = context_bitplane_cost_field_local(
                        residual,
                        bit_indices,
                        previous_bits,
                    )
                for name, prediction in delta_prediction_bits.items():
                    residual = ordered_delta(ordered_tile, prediction[sl])
                    candidates[name] = context_bitplane_cost_field_local(
                        residual,
                        bit_indices,
                        previous_bits,
                    )
                best_mode = min(candidates, key=candidates.__getitem__)
                total_bits += candidates[best_mode] + mode_overhead_bits
                mode_histogram[best_mode] += 1
                field_mode_histogram[field][best_mode] += 1

    return {
        "image": path.name,
        "shape": [height, width, channels],
        "tile_size": tile_size,
        "previous_bits": previous_bits,
        "raw_bits": raw_bits,
        "estimated_bits": total_bits,
        "ratio": raw_bits / total_bits,
        "mode_histogram": dict(mode_histogram),
        "field_mode_histogram": {
            field: dict(histogram)
            for field, histogram in field_mode_histogram.items()
        },
    }


def geomean(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", default="*.exr")
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--previous-bits", type=int, default=4)
    parser.add_argument("--mode-overhead-bits", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = sorted(DATA_DIR.glob(args.glob))
    if not paths:
        raise RuntimeError(f"no images matched {DATA_DIR / args.glob}")
    blosc = load_blosc_ratios()
    results = []
    print(
        f"{'image':43s} {'lowfld':>9s} {'Blosc':>9s} "
        f"{'hybrid':>9s} {'pick':>8s}"
    )
    print("-" * 86)
    for path in paths:
        result = estimate_image(
            path,
            args.tile_size,
            args.previous_bits,
            args.mode_overhead_bits,
        )
        blosc_ratio = blosc.get(path.stem)
        hybrid_ratio = max(result["ratio"], blosc_ratio or 0.0)
        pick = "lowfld" if result["ratio"] >= (blosc_ratio or 0.0) else "Blosc"
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
        f"structural_delta_lowfield_tile{args.tile_size}_prev{args.previous_bits}.json"
    )
    output.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
