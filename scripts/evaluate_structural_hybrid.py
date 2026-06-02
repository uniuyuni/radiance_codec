"""Estimate a practical structural+fallback codec for float32 HDR images.

The codec model is intentionally simple:

For each tile and each float field group:
  1. Try several reversible predictors.
  2. Estimate bit-plane entropy of the exact XOR residual.
  3. Also estimate raw field cost.
  4. Choose the cheapest mode.

Then compare the resulting image-level size with Blosc. This is still an
entropy estimate, not a final rANS stream, but it includes mode-map overhead and
field-level raw fallback. It is a closer Go/No-Go probe than a pure predictor
audit.
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
    binary_entropy,
    float_to_bits,
    load_blosc_ratios,
    predictors,
    read_exr,
)

RESULTS_DIR = ROOT / "results"
OUTPUT_JSON = RESULTS_DIR / "structural_hybrid_estimate.json"
MODE_NAMES = [
    "raw",
    "zero",
    "west",
    "north",
    "northwest",
    "northeast",
    "float_med",
    "float_planar",
]


def bit_entropy_cost(bits: np.ndarray, bit_indices: tuple[int, ...]) -> float:
    flat = bits.reshape(-1)
    n_values = flat.size
    cost = 0.0
    for bit_index in bit_indices:
        probability = float(np.mean((flat >> bit_index) & 1))
        cost += binary_entropy(probability) * n_values
    return cost


def estimate_image(
    path: Path,
    tile_size: int,
    mode_overhead_bits: int,
    entropy_table_overhead_bits: int,
) -> dict:
    pixels = read_exr(path)
    bits = float_to_bits(pixels)
    predictor_bits = predictors(pixels)
    height, width, channels = bits.shape
    raw_bits = bits.size * 32
    total_bits = 0.0
    mode_histogram = Counter()
    field_mode_histogram = {
        field: Counter()
        for field in FIELD_GROUPS
    }
    tile_count = 0

    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            tile_count += 1
            sl = np.s_[y:y + tile_size, x:x + tile_size]
            tile_values = bits[sl].size
            for field, bit_indices in FIELD_GROUPS.items():
                raw_cost = tile_values * len(bit_indices)
                candidates = {"raw": raw_cost}
                for name, pred in predictor_bits.items():
                    residual = bits[sl] ^ pred[sl]
                    candidates[name] = (
                        bit_entropy_cost(residual, bit_indices)
                        + entropy_table_overhead_bits * len(bit_indices)
                    )
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
    parser.add_argument("--entropy-table-overhead-bits", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = sorted(DATA_DIR.glob(args.glob))
    if not paths:
        raise RuntimeError(f"no images matched {DATA_DIR / args.glob}")
    blosc = load_blosc_ratios()
    results = []
    print(
        f"{'image':43s} {'struct':>9s} {'Blosc':>9s} "
        f"{'hybrid':>9s} {'pick':>8s}"
    )
    print("-" * 86)
    for path in paths:
        result = estimate_image(
            path,
            args.tile_size,
            args.mode_overhead_bits,
            args.entropy_table_overhead_bits,
        )
        blosc_ratio = blosc.get(path.stem)
        if blosc_ratio is None:
            hybrid_ratio = result["ratio"]
            pick = "struct"
        else:
            hybrid_ratio = max(result["ratio"], blosc_ratio)
            pick = "struct" if result["ratio"] >= blosc_ratio else "Blosc"
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
    OUTPUT_JSON.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {OUTPUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
