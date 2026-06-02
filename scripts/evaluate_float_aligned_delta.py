"""Estimate exponent-aligned float-field residuals.

Ordered-float deltas expose ULP locality, but carries across exponent/mantissa
fields complicate context alignment.  This probe keeps float fields separate:

* sign residual: xor with predicted sign;
* exponent residual: modular exponent delta;
* mantissa residual: actual mantissa minus the predictor's significand scaled
  into the actual exponent's units, modulo 23 mantissa bits.

The transform is reversible once sign/exponent have been decoded, and it avoids
cross-field arithmetic carries.
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
    float_med_prediction,
    float_planar_prediction,
    float_to_bits,
    load_blosc_ratios,
    read_exr,
)
from evaluate_structural_delta_context import (  # noqa: E402
    causal_shifted,
    context_bitplane_cost,
)

RESULTS_DIR = ROOT / "results"
MANTISSA_MASK = np.uint32(0x007FFFFF)
EXPONENT_MASK = np.uint32(0x7F800000)
SIGN_MASK = np.uint32(0x80000000)


def field_mask(bit_indices: tuple[int, ...]) -> np.uint32:
    mask = np.uint32(0)
    for bit in bit_indices:
        mask |= np.uint32(1) << np.uint32(bit)
    return mask


FIELD_MASKS = {field: field_mask(bits) for field, bits in FIELD_GROUPS.items()}


def predictor_bits(bits: np.ndarray, pixels: np.ndarray) -> dict[str, np.ndarray]:
    channel_previous = np.zeros_like(bits)
    if bits.shape[2] >= 2:
        channel_previous[..., 1:] = bits[..., :-1]
    return {
        "zero": np.zeros_like(bits),
        "west": causal_shifted(bits, 0, -1),
        "north": causal_shifted(bits, -1, 0),
        "northwest": causal_shifted(bits, -1, -1),
        "northeast": causal_shifted(bits, -1, 1),
        "channel_previous": channel_previous,
        "float_med": float_med_prediction(pixels),
        "float_planar": float_planar_prediction(pixels),
    }


def scaled_mantissa_prediction(actual_exp: np.ndarray, pred: np.ndarray) -> np.ndarray:
    pred_exp = ((pred & EXPONENT_MASK) >> np.uint32(23)).astype(np.int16)
    pred_mant = (pred & MANTISSA_MASK).astype(np.uint32)
    normal = (pred_exp != 0) & (pred_exp != 255)
    pred_sig = pred_mant | np.where(normal, np.uint32(1 << 23), np.uint32(0))
    shift = pred_exp.astype(np.int16) - actual_exp.astype(np.int16)
    scaled = np.zeros_like(pred_sig, dtype=np.uint32)

    for amount in range(0, 23):
        mask = shift == amount
        if np.any(mask):
            scaled[mask] = (pred_sig[mask] << np.uint32(amount)) & MANTISSA_MASK
    for amount in range(1, 25):
        mask = shift == -amount
        if np.any(mask):
            scaled[mask] = pred_sig[mask] >> np.uint32(amount)
    return scaled & MANTISSA_MASK


def aligned_residual(bits: np.ndarray, pred: np.ndarray) -> np.ndarray:
    actual_sign = (bits & SIGN_MASK) >> np.uint32(31)
    pred_sign = (pred & SIGN_MASK) >> np.uint32(31)
    sign_res = (actual_sign ^ pred_sign).astype(np.uint32)

    actual_exp = ((bits & EXPONENT_MASK) >> np.uint32(23)).astype(np.uint16)
    pred_exp = ((pred & EXPONENT_MASK) >> np.uint32(23)).astype(np.uint16)
    exp_res = ((actual_exp - pred_exp) & np.uint16(0xFF)).astype(np.uint32)

    actual_mant = bits & MANTISSA_MASK
    scaled_pred = scaled_mantissa_prediction(actual_exp, pred)
    mant_res = (actual_mant - scaled_pred) & MANTISSA_MASK

    return (
        (sign_res << np.uint32(31))
        | (exp_res << np.uint32(23))
        | mant_res
    ).astype(np.uint32)


def build_payload(candidates: dict[str, np.ndarray], modes: dict[str, str]) -> np.ndarray:
    first = next(iter(candidates.values()))
    payload = np.zeros_like(first, dtype=np.uint32)
    for field, mode in modes.items():
        payload |= candidates[mode] & FIELD_MASKS[field]
    return payload


def score(
    candidates: dict[str, np.ndarray],
    modes: dict[str, str],
    previous_bits: int,
    mode_overhead_bits: int,
) -> float:
    payload = build_payload(candidates, modes)
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


def choose_modes(
    candidates: dict[str, np.ndarray],
    previous_bits: int,
    mode_overhead_bits: int,
    refine_iters: int,
) -> tuple[float, dict[str, str]]:
    tile_values = next(iter(candidates.values())).size
    modes = {}
    for field, bit_indices in FIELD_GROUPS.items():
        costs = {}
        for mode, residual in candidates.items():
            if mode == "raw":
                costs[mode] = tile_values * len(bit_indices)
            else:
                costs[mode] = context_bitplane_cost(
                    residual,
                    bit_indices,
                    previous_bits,
                    channel_context=False,
                )
        modes[field] = min(costs, key=costs.__getitem__)

    best = score(candidates, modes, previous_bits, mode_overhead_bits)
    mode_names = tuple(candidates)
    for _ in range(refine_iters):
        changed = False
        for field in FIELD_GROUPS:
            field_best = best
            field_best_mode = modes[field]
            for mode in mode_names:
                if mode == modes[field]:
                    continue
                trial = dict(modes)
                trial[field] = mode
                cost = score(candidates, trial, previous_bits, mode_overhead_bits)
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
    preds = predictor_bits(bits, pixels)
    height, width, channels = bits.shape
    raw_bits = bits.size * 32
    total_bits = 0.0
    mode_histogram = Counter()

    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            sl = np.s_[y:y + tile_size, x:x + tile_size]
            tile = bits[sl]
            candidates = {"raw": tile}
            for name, pred in preds.items():
                candidates[name] = aligned_residual(tile, pred[sl])
            cost, modes = choose_modes(
                candidates,
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
        f"{'image':43s} {'align':>9s} {'Blosc':>9s} "
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
        pick = "align" if result["ratio"] >= (blosc_ratio or 0.0) else "Blosc"
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
        f"float_aligned_delta_tile{args.tile_size}_prev{args.previous_bits}"
        f"_refine{args.refine_iters}.json"
    )
    output.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
