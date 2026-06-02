"""Estimate structural coding with decoder-reproducible adaptive bit models.

The earlier structural oracle picked a predictor per tile/field and charged
ideal binary entropy for each residual bit-plane. That is optimistic because a
real decoder also needs the bit probability. Storing probabilities per tile was
too expensive, while global probabilities missed the tile-specific density.

This probe uses a simple KT/Beta(1/2, 1/2) adaptive binary model for every
tile/field/mode/bit-plane. It needs no probability side information: encoder
and decoder start from the same prior and update counts as bits are coded.
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

RESULTS_DIR = ROOT / "results"
OUTPUT_JSON_TEMPLATE = "structural_adaptive_estimate_tile{tile_size}.json"


def kt_binary_cost(ones: int, total: int, alpha: float = 0.5) -> float:
    """Bits for an adaptive Bernoulli model with a symmetric Beta prior."""
    if total == 0:
        return 0.0
    zeros = total - ones
    log_probability = (
        math.lgamma(ones + alpha)
        + math.lgamma(zeros + alpha)
        - math.lgamma(total + 2.0 * alpha)
        - 2.0 * math.lgamma(alpha)
        + math.lgamma(2.0 * alpha)
    )
    return -log_probability / math.log(2.0)


def bitplane_adaptive_cost(bits: np.ndarray, bit_indices: tuple[int, ...]) -> float:
    flat = bits.reshape(-1)
    total = flat.size
    cost = 0.0
    for bit_index in bit_indices:
        ones = int(((flat >> bit_index) & 1).sum())
        cost += kt_binary_cost(ones, total)
    return cost


def estimate_image(path: Path, tile_size: int, mode_overhead_bits: int) -> dict:
    pixels = read_exr(path)
    bits = float_to_bits(pixels)
    predictor_bits = predictors(pixels)
    height, width, channels = bits.shape
    raw_bits = bits.size * 32
    total_bits = 0.0
    mode_histogram = Counter()
    field_mode_histogram = {field: Counter() for field in FIELD_GROUPS}
    tile_count = 0

    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            tile_count += 1
            sl = np.s_[y:y + tile_size, x:x + tile_size]
            tile_values = bits[sl].size
            for field, bit_indices in FIELD_GROUPS.items():
                candidates = {"raw": tile_values * len(bit_indices)}
                for name, pred in predictor_bits.items():
                    residual = bits[sl] ^ pred[sl]
                    candidates[name] = bitplane_adaptive_cost(residual, bit_indices)
                best_mode = min(candidates, key=candidates.__getitem__)
                total_bits += candidates[best_mode] + mode_overhead_bits
                mode_histogram[best_mode] += 1
                field_mode_histogram[field][best_mode] += 1

    return {
        "image": path.name,
        "shape": [height, width, channels],
        "tile_size": tile_size,
        "tile_count": tile_count,
        "raw_bits": raw_bits,
        "estimated_bits": total_bits,
        "estimated_bytes": math.ceil(total_bits / 8),
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
    parser.add_argument("--tile-size", type=int, default=32)
    parser.add_argument("--mode-overhead-bits", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = sorted(DATA_DIR.glob(args.glob))
    if not paths:
        raise RuntimeError(f"no images matched {DATA_DIR / args.glob}")
    blosc = load_blosc_ratios()
    results = []
    print(
        f"{'image':43s} {'adapt':>9s} {'Blosc':>9s} "
        f"{'hybrid':>9s} {'pick':>8s}"
    )
    print("-" * 86)
    for path in paths:
        result = estimate_image(path, args.tile_size, args.mode_overhead_bits)
        blosc_ratio = blosc.get(path.stem)
        if blosc_ratio is None:
            hybrid_ratio = result["ratio"]
            pick = "adapt"
        else:
            hybrid_ratio = max(result["ratio"], blosc_ratio)
            pick = "adapt" if result["ratio"] >= blosc_ratio else "Blosc"
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
        values = [result[key] for result in results if result[key] is not None]
        print(f"geomean {key:12s}: {geomean(values):.2f}x")
    output_json = RESULTS_DIR / OUTPUT_JSON_TEMPLATE.format(tile_size=args.tile_size)
    output_json.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
