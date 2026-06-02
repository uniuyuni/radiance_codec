"""Probe causal smooth-gradient predictors for hard puresky HDR tiles.

This is an upper-bound probe for a possible GDX7 body mode.  It keeps the
current grouped-delta sign route and compares the current body payload against
several deterministic second-order predictors that are cheap enough to move
into the C++ codec if they win.
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

from analyze_structural_predictors import DATA_DIR, float_to_bits, read_exr  # noqa: E402
from evaluate_structural_delta_context import bits_to_ordered  # noqa: E402
from probe_grouped_tail_mlp import CONTEXT_FAMILIES, context_family_cost  # noqa: E402
from structural_delta_grouped_roundtrip import choose_group_payloads  # noqa: E402

RESULTS_DIR = ROOT / "results"
BODY_MASK = np.uint32(0x7ffffffF)
SIGN_MASK = np.uint32(0x80000000)


def geomean(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def shift(values: np.ndarray, dy: int, dx: int) -> np.ndarray:
    out = np.zeros_like(values)
    height, width, _channels = values.shape
    dst_y0 = max(0, -dy)
    dst_y1 = min(height, height - dy)
    dst_x0 = max(0, -dx)
    dst_x1 = min(width, width - dx)
    src_y0 = dst_y0 + dy
    src_y1 = dst_y1 + dy
    src_x0 = dst_x0 + dx
    src_x1 = dst_x1 + dx
    out[dst_y0:dst_y1, dst_x0:dst_x1] = values[src_y0:src_y1, src_x0:src_x1]
    return out


def clamp_u32(values: np.ndarray) -> np.ndarray:
    return np.clip(values, 0, 0xffffffff).astype(np.uint32)


def ordered_second_x(ordered: np.ndarray) -> np.ndarray:
    west = shift(ordered, 0, -1).astype(np.int64)
    west2 = shift(ordered, 0, -2).astype(np.int64)
    pred = clamp_u32(2 * west - west2)
    pred[:, 0, :] = 0
    pred[:, 1, :] = west[:, 1, :]
    return pred


def ordered_second_y(ordered: np.ndarray) -> np.ndarray:
    north = shift(ordered, -1, 0).astype(np.int64)
    north2 = shift(ordered, -2, 0).astype(np.int64)
    pred = clamp_u32(2 * north - north2)
    pred[0, :, :] = 0
    pred[1, :, :] = north[1, :, :]
    return pred


def ordered_second_xy(ordered: np.ndarray) -> np.ndarray:
    xpred = ordered_second_x(ordered).astype(np.uint64)
    ypred = ordered_second_y(ordered).astype(np.uint64)
    pred = ((xpred + ypred) // 2).astype(np.uint32)
    pred[0, :, :] = ordered_second_x(ordered)[0, :, :]
    pred[:, 0, :] = ordered_second_y(ordered)[:, 0, :]
    pred[0, 0, :] = 0
    return pred


def float_to_ordered(values: np.ndarray) -> np.ndarray:
    return bits_to_ordered(float_to_bits(np.ascontiguousarray(values, dtype=np.float32)))


def float_second_x(pixels: np.ndarray) -> np.ndarray:
    west = shift(pixels, 0, -1).astype(np.float32)
    west2 = shift(pixels, 0, -2).astype(np.float32)
    pred = (np.float32(2.0) * west - west2).astype(np.float32)
    pred[:, 0, :] = np.float32(0.0)
    pred[:, 1, :] = west[:, 1, :]
    return float_to_ordered(pred)


def float_second_y(pixels: np.ndarray) -> np.ndarray:
    north = shift(pixels, -1, 0).astype(np.float32)
    north2 = shift(pixels, -2, 0).astype(np.float32)
    pred = (np.float32(2.0) * north - north2).astype(np.float32)
    pred[0, :, :] = np.float32(0.0)
    pred[1, :, :] = north[1, :, :]
    return float_to_ordered(pred)


def float_second_xy(pixels: np.ndarray) -> np.ndarray:
    xpred_bits = float_second_x(pixels)
    ypred_bits = float_second_y(pixels)
    xpred = xpred_bits.astype(np.uint64)
    ypred = ypred_bits.astype(np.uint64)
    pred = ((xpred + ypred) // 2).astype(np.uint32)
    pred[0, :, :] = xpred_bits[0, :, :]
    pred[:, 0, :] = ypred_bits[:, 0, :]
    pred[0, 0, :] = 0
    return pred


def body_delta_payload(ordered: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    actual = ordered & BODY_MASK
    pred = prediction & BODY_MASK
    return ((actual - pred) & BODY_MASK).astype(np.uint32)


def best_body_cost(payload: np.ndarray) -> tuple[float, dict[str, int]]:
    total = 0.0
    picks = Counter()
    masked = payload & BODY_MASK
    for bit in range(30, -1, -1):
        family, cost = min(
            (
                (family, context_family_cost(masked, bit, family))
                for family in CONTEXT_FAMILIES
            ),
            key=lambda item: item[1],
        )
        total += cost
        picks[family] += 1
    return total, dict(picks)


def sign_cost(payload: np.ndarray) -> float:
    masked = payload & SIGN_MASK
    return min(context_family_cost(masked, 31, family) for family in CONTEXT_FAMILIES)


def candidate_predictions(
    ordered: np.ndarray,
    pixels: np.ndarray,
) -> dict[str, np.ndarray]:
    return {
        "ordered_second_x": ordered_second_x(ordered),
        "ordered_second_y": ordered_second_y(ordered),
        "ordered_second_xy": ordered_second_xy(ordered),
        "float_second_x": float_second_x(pixels),
        "float_second_y": float_second_y(pixels),
        "float_second_xy": float_second_xy(pixels),
    }


def summarize_image(
    path: Path,
    crop_size: int,
    tile_size: int,
    previous_bits: int,
    mode_overhead_bits: float,
) -> dict:
    pixels = read_exr(path)
    if crop_size:
        pixels = pixels[:crop_size, :crop_size]
    pixels = np.ascontiguousarray(pixels, dtype=np.float32)
    bits = float_to_bits(pixels).copy()
    ordered = bits_to_ordered(bits)
    _modes, payloads, _carries = choose_group_payloads(
        bits,
        pixels,
        tile_size,
        previous_bits,
        "body",
    )
    predictions = candidate_predictions(ordered, pixels)
    candidate_payloads = {
        name: body_delta_payload(ordered, prediction)
        for name, prediction in predictions.items()
    }

    baseline_bits = 0.0
    hybrid_bits = 0.0
    candidate_bits = Counter()
    selected = Counter()
    mode_picks = Counter()
    height, width, _channels = bits.shape
    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            sl = np.s_[y:y + tile_size, x:x + tile_size]
            body_payload = payloads["body"][sl] & BODY_MASK
            sign_payload = payloads["sign"][sl] & SIGN_MASK
            tile_sign_bits = sign_cost(sign_payload)
            tile_body_bits, baseline_picks = best_body_cost(body_payload)
            baseline_bits += tile_sign_bits + tile_body_bits
            best_name = "current"
            best_body_bits = tile_body_bits
            mode_picks.update({f"current:{key}": value for key, value in baseline_picks.items()})
            for name, full_payload in candidate_payloads.items():
                candidate_body_bits, candidate_picks = best_body_cost(full_payload[sl])
                candidate_body_bits += mode_overhead_bits
                candidate_bits[name] += candidate_body_bits + tile_sign_bits
                if candidate_body_bits < best_body_bits:
                    best_name = name
                    best_body_bits = candidate_body_bits
                    mode_picks.update({f"{name}:{key}": value for key, value in candidate_picks.items()})
            hybrid_bits += tile_sign_bits + best_body_bits
            selected[best_name] += 1

    raw_bits = int(bits.size * 32)
    return {
        "image": path.name,
        "shape": list(bits.shape),
        "crop_size": crop_size,
        "tile_size": tile_size,
        "previous_bits": previous_bits,
        "mode_overhead_bits": mode_overhead_bits,
        "raw_bits": raw_bits,
        "baseline_bits": baseline_bits,
        "hybrid_bits": hybrid_bits,
        "baseline_ratio": raw_bits / baseline_bits if baseline_bits else 1.0,
        "hybrid_ratio": raw_bits / hybrid_bits if hybrid_bits else 1.0,
        "hybrid_gain": baseline_bits / hybrid_bits if hybrid_bits else 1.0,
        "candidate_ratios": {
            name: raw_bits / bits_value for name, bits_value in candidate_bits.items()
        },
        "selected": dict(selected),
        "mode_picks": dict(mode_picks),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", default="ph_*puresky*.exr")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--tile-size", type=int, default=128)
    parser.add_argument("--previous-bits", type=int, default=4)
    parser.add_argument("--mode-overhead-bits", type=float, default=8.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = []
    for path in sorted(DATA_DIR.glob(args.glob)):
        row = summarize_image(
            path,
            args.crop_size,
            args.tile_size,
            args.previous_bits,
            args.mode_overhead_bits,
        )
        rows.append(row)
        best_candidates = sorted(
            row["candidate_ratios"].items(),
            key=lambda item: item[1],
            reverse=True,
        )[:3]
        best_text = " ".join(f"{name}={ratio:.3f}x" for name, ratio in best_candidates)
        print(
            f"{path.stem:43s} "
            f"base={row['baseline_ratio']:.3f}x "
            f"hybrid={row['hybrid_ratio']:.3f}x "
            f"gain={row['hybrid_gain']:.4f} "
            f"selected={row['selected']} {best_text}"
        )
        if args.limit and len(rows) >= args.limit:
            break
    if rows:
        print(
            f"geomean base={geomean([r['baseline_ratio'] for r in rows]):.3f}x "
            f"hybrid={geomean([r['hybrid_ratio'] for r in rows]):.3f}x"
        )
    safe_glob = args.glob.replace("*", "star").replace(".", "_")
    output = RESULTS_DIR / (
        f"puresky_predictors_{safe_glob}_tile{args.tile_size}"
        f"_prev{args.previous_bits}_crop{args.crop_size}.json"
    )
    output.write_text(json.dumps(rows, indent=2))
    print(f"\nSaved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
