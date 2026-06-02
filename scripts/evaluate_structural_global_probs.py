"""Evaluate structural predictors with decoder-reproducible global probabilities.

Training images provide P(bit=1 | field, bit_index, predictor_mode). Test images
choose structural predictor modes per tile and field, but entropy coding uses
only these learned global probabilities, so no per-tile probability side
information is needed.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_structural_predictors import (  # noqa: E402
    FIELD_GROUPS,
    binary_entropy,
    float_to_bits,
    load_blosc_ratios,
    predictors,
    read_exr,
)

DATA_DIR = ROOT / "data"
TRAIN_DIR = ROOT / "ml" / "training_data"
OUTPUT_JSON = ROOT / "results" / "structural_global_probs_eval.json"


def binary_cost(ones: int, total: int, p_one: float) -> float:
    p_one = min(max(p_one, 1e-6), 1.0 - 1e-6)
    return -ones * math.log2(p_one) - (total - ones) * math.log2(1.0 - p_one)


def bit_counts(values: np.ndarray, bit_index: int) -> tuple[int, int]:
    flat = values.reshape(-1)
    ones = int(((flat >> bit_index) & 1).sum())
    return ones, int(flat.size)


def train_probabilities(paths: list[Path], limit: int | None) -> dict[str, float]:
    if limit is not None:
        paths = paths[:limit]
    counts = defaultdict(lambda: [1, 2])  # Laplace-smoothed [ones, total]
    for index, path in enumerate(paths, start=1):
        try:
            pixels = read_exr(path)
        except Exception as error:
            print(f"SKIP train {path.name}: {error}")
            continue
        bits = float_to_bits(pixels)
        pred_bits = predictors(pixels)
        for mode, pred in pred_bits.items():
            residual = bits ^ pred
            for field, bit_indices in FIELD_GROUPS.items():
                for bit_index in bit_indices:
                    ones, total = bit_counts(residual, bit_index)
                    key = f"{field}:{bit_index}:{mode}"
                    counts[key][0] += ones
                    counts[key][1] += total
        print(f"trained stats {index}/{len(paths)}: {path.name}")
    return {
        key: ones_total[0] / ones_total[1]
        for key, ones_total in counts.items()
    }


def estimate_image(
    path: Path,
    probabilities: dict[str, float],
    tile_size: int,
    mode_overhead_bits: int,
) -> dict:
    pixels = read_exr(path)
    bits = float_to_bits(pixels)
    pred_bits = predictors(pixels)
    height, width, channels = bits.shape
    raw_bits = bits.size * 32
    total_bits = 0.0
    oracle_bits = 0.0
    mode_hist = defaultdict(int)
    field_mode_hist = {field: defaultdict(int) for field in FIELD_GROUPS}

    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            sl = np.s_[y:y + tile_size, x:x + tile_size]
            tile_values = bits[sl].size
            for field, bit_indices in FIELD_GROUPS.items():
                raw_cost = tile_values * len(bit_indices)
                best_cost = raw_cost
                best_oracle = raw_cost
                best_mode = "raw"
                for mode, pred in pred_bits.items():
                    residual = bits[sl] ^ pred[sl]
                    cost = mode_overhead_bits
                    oracle = mode_overhead_bits
                    for bit_index in bit_indices:
                        ones, total = bit_counts(residual, bit_index)
                        key = f"{field}:{bit_index}:{mode}"
                        cost += binary_cost(ones, total, probabilities.get(key, 0.5))
                        if ones == 0 or ones == total:
                            oracle += 0.0
                        else:
                            oracle += binary_entropy(ones / total) * total
                    if cost < best_cost:
                        best_cost = cost
                        best_mode = mode
                    if oracle < best_oracle:
                        best_oracle = oracle
                total_bits += best_cost
                oracle_bits += best_oracle
                mode_hist[best_mode] += 1
                field_mode_hist[field][best_mode] += 1

    return {
        "image": path.name,
        "shape": [height, width, channels],
        "tile_size": tile_size,
        "raw_bits": raw_bits,
        "global_probability_bits": total_bits,
        "oracle_bits": oracle_bits,
        "global_probability_ratio": raw_bits / total_bits,
        "oracle_ratio": raw_bits / oracle_bits,
        "mode_histogram": dict(mode_hist),
        "field_mode_histogram": {
            field: dict(hist)
            for field, hist in field_mode_hist.items()
        },
    }


def geomean(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tile-size", type=int, default=16)
    parser.add_argument("--train-limit", type=int, default=12)
    parser.add_argument("--test-glob", default="*.exr")
    parser.add_argument("--stats-out", type=Path, default=ROOT / "ml" / "structural_global_probs.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    train_paths = sorted(TRAIN_DIR.glob("*.exr"))
    probabilities = train_probabilities(train_paths, args.train_limit)
    args.stats_out.write_text(json.dumps(probabilities, indent=2))
    print(f"Saved probabilities: {args.stats_out}")

    blosc = load_blosc_ratios()
    rows = []
    print(f"\n{'image':43s} {'globalP':>9s} {'oracle':>9s} {'Blosc':>9s} {'hybrid':>9s}")
    print("-" * 86)
    for path in sorted(DATA_DIR.glob(args.test_glob)):
        row = estimate_image(path, probabilities, args.tile_size, mode_overhead_bits=4)
        blosc_ratio = blosc[path.stem]
        row["blosc_ratio"] = blosc_ratio
        row["hybrid_ratio"] = max(row["global_probability_ratio"], blosc_ratio)
        rows.append(row)
        print(
            f"{path.stem:43s} {row['global_probability_ratio']:8.2f}x "
            f"{row['oracle_ratio']:8.2f}x {blosc_ratio:8.2f}x "
            f"{row['hybrid_ratio']:8.2f}x"
        )
    print("-" * 86)
    for key in ["global_probability_ratio", "oracle_ratio", "blosc_ratio", "hybrid_ratio"]:
        print(f"geomean {key:24s}: {geomean([row[key] for row in rows]):.2f}x")
    OUTPUT_JSON.write_text(json.dumps(rows, indent=2))
    print(f"\nSaved: {OUTPUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
