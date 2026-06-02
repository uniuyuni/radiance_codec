"""Estimate the actual mixed field payload for ordered-delta decoding.

Field-wise ordered deltas are reconstructable, but the entropy context sees the
final payload stream: higher residual bits may come from a different predictor
than the current field.  This estimator builds that payload word explicitly and
then scores contexts on the real stream.
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

from analyze_structural_predictors import DATA_DIR, FIELD_GROUPS, float_to_bits, load_blosc_ratios, read_exr  # noqa: E402
from evaluate_structural_delta_context import (  # noqa: E402
    bits_to_ordered,
    context_bitplane_cost,
    delta_predictors,
    ordered_delta,
    ordered_xor_predictors,
)

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


def candidate_payloads(
    ordered_tile: np.ndarray,
    xor_prediction_tiles: dict[str, np.ndarray],
    delta_prediction_tiles: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    candidates = {"raw": ordered_tile}
    for name, prediction in xor_prediction_tiles.items():
        candidates[f"xor_{name}"] = ordered_tile ^ prediction
    for name, prediction in delta_prediction_tiles.items():
        candidates[name] = ordered_delta(ordered_tile, prediction)
    return candidates


def initial_modes(
    candidates: dict[str, np.ndarray],
    previous_bits: int,
) -> dict[str, str]:
    modes = {}
    for field, bit_indices in FIELD_GROUPS.items():
        costs = {}
        tile_values = next(iter(candidates.values())).size
        for mode, payload in candidates.items():
            if mode == "raw":
                costs[mode] = float(tile_values * len(bit_indices))
            else:
                costs[mode] = context_bitplane_cost(
                    payload,
                    bit_indices,
                    previous_bits,
                    channel_context=False,
                )
        modes[field] = min(costs, key=costs.__getitem__)
    return modes


def build_payload_word(
    candidates: dict[str, np.ndarray],
    modes: dict[str, str],
) -> np.ndarray:
    first = next(iter(candidates.values()))
    payload = np.zeros_like(first, dtype=np.uint32)
    for field, mode in modes.items():
        payload |= candidates[mode] & FIELD_MASKS[field]
    return payload


def score_modes(
    candidates: dict[str, np.ndarray],
    modes: dict[str, str],
    previous_bits: int,
    mode_overhead_bits: int,
) -> float:
    payload = build_payload_word(candidates, modes)
    tile_values = payload.size
    total = 0.0
    for field, bit_indices in FIELD_GROUPS.items():
        if modes[field] == "raw":
            total += tile_values * len(bit_indices)
        else:
            total += context_bitplane_cost(
                payload,
                bit_indices,
                previous_bits,
                channel_context=False,
            )
        total += mode_overhead_bits
    return total


def refine_modes(
    candidates: dict[str, np.ndarray],
    modes: dict[str, str],
    previous_bits: int,
    mode_overhead_bits: int,
    refine_iters: int,
) -> tuple[float, dict[str, str]]:
    best = score_modes(candidates, modes, previous_bits, mode_overhead_bits)
    mode_names = tuple(candidates)
    for _ in range(refine_iters):
        changed = False
        for field in FIELD_GROUPS:
            field_best_mode = modes[field]
            field_best = best
            for mode in mode_names:
                if mode == modes[field]:
                    continue
                trial = dict(modes)
                trial[field] = mode
                cost = score_modes(
                    candidates,
                    trial,
                    previous_bits,
                    mode_overhead_bits,
                )
                if cost < field_best:
                    field_best = cost
                    field_best_mode = mode
            if field_best_mode != modes[field]:
                modes[field] = field_best_mode
                best = field_best
                changed = True
        if not changed:
            break
    return best, modes


def estimate_image(
    path: Path,
    tile_size: int,
    previous_bits: int,
    mode_overhead_bits: int,
    refine_iters: int,
) -> dict:
    pixels = read_exr(path)
    bits = float_to_bits(pixels)
    ordered = bits_to_ordered(bits)
    xor_predictor_bits = ordered_xor_predictors(bits, pixels, causal=True)
    delta_prediction_bits = delta_predictors(bits, pixels, causal=True)
    xor_predictor_bits.pop("channel_green", None)
    delta_prediction_bits.pop("delta_channel_green", None)
    height, width, channels = bits.shape
    raw_bits = bits.size * 32
    total_bits = 0.0
    mode_histogram = Counter()

    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            sl = np.s_[y:y + tile_size, x:x + tile_size]
            ordered_tile = ordered[sl]
            candidates = candidate_payloads(
                ordered_tile,
                {name: pred[sl] for name, pred in xor_predictor_bits.items()},
                {name: pred[sl] for name, pred in delta_prediction_bits.items()},
            )
            modes = initial_modes(candidates, previous_bits)
            cost, modes = refine_modes(
                candidates,
                modes,
                previous_bits,
                mode_overhead_bits,
                refine_iters,
            )
            total_bits += cost
            mode_histogram.update(f"{field}:{mode}" for field, mode in modes.items())

    return {
        "image": path.name,
        "shape": [height, width, channels],
        "tile_size": tile_size,
        "previous_bits": previous_bits,
        "refine_iters": refine_iters,
        "raw_bits": raw_bits,
        "estimated_bits": total_bits,
        "ratio": raw_bits / total_bits,
        "mode_histogram": dict(mode_histogram),
    }


def geomean(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", default="*.exr")
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--previous-bits", type=int, default=4)
    parser.add_argument("--mode-overhead-bits", type=int, default=5)
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
        f"{'image':43s} {'payload':>9s} {'Blosc':>9s} "
        f"{'hybrid':>9s} {'pick':>8s}"
    )
    print("-" * 86)
    for path in paths:
        result = estimate_image(
            path,
            args.tile_size,
            args.previous_bits,
            args.mode_overhead_bits,
            args.refine_iters,
        )
        blosc_ratio = blosc.get(path.stem)
        hybrid_ratio = max(result["ratio"], blosc_ratio or 0.0)
        pick = "payload" if result["ratio"] >= (blosc_ratio or 0.0) else "Blosc"
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
        f"structural_delta_payload_tile{args.tile_size}_prev{args.previous_bits}"
        f"_refine{args.refine_iters}.json"
    )
    output.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
