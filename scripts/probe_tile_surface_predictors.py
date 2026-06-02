"""Probe exact tile-surface predictors for smooth puresky/HDR regions.

Current GDX modes are causal local predictors.  Smooth skies may be better
described by a transmitted per-tile surface model: fit a small affine/quadratic
surface per channel, store its coefficients, then entropy-code the exact
ordered-float residual.  This is still exact lossless because the residual is
stored fully; the fitted model is only a predictor.
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
BODY_MASK = np.uint32(0x7fffffff)


def geomean(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def design_matrix(height: int, width: int, kind: str) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    x = (xx.reshape(-1).astype(np.float64) / max(1, width - 1)) * 2.0 - 1.0
    y = (yy.reshape(-1).astype(np.float64) / max(1, height - 1)) * 2.0 - 1.0
    if kind == "affine":
        return np.stack([np.ones_like(x), x, y], axis=1)
    if kind == "quadratic":
        return np.stack([np.ones_like(x), x, y, x * x, y * y, x * y], axis=1)
    raise ValueError(f"unknown surface kind: {kind}")


def fit_surface(values: np.ndarray, kind: str) -> np.ndarray:
    height, width = values.shape
    design = design_matrix(height, width, kind)
    coeffs, *_ = np.linalg.lstsq(design, values.reshape(-1).astype(np.float64), rcond=None)
    return coeffs.astype(np.float32)


def predict_surface(coeffs: np.ndarray, height: int, width: int, kind: str) -> np.ndarray:
    design = design_matrix(height, width, kind).astype(np.float32)
    pred = design @ coeffs.astype(np.float32)
    return pred.reshape(height, width)


def ordered_surface_prediction(ordered_tile: np.ndarray, kind: str) -> np.ndarray:
    height, width, channels = ordered_tile.shape
    pred = np.zeros_like(ordered_tile, dtype=np.uint32)
    for c in range(channels):
        coeffs = fit_surface(ordered_tile[:, :, c].astype(np.float64), kind)
        values = predict_surface(coeffs, height, width, kind)
        rounded = np.rint(values.astype(np.float64))
        pred[:, :, c] = np.clip(rounded, 0, 0xffffffff).astype(np.uint32)
    return pred


def float_surface_prediction(pixels_tile: np.ndarray, kind: str) -> np.ndarray:
    height, width, channels = pixels_tile.shape
    pred = np.zeros_like(pixels_tile, dtype=np.float32)
    for c in range(channels):
        coeffs = fit_surface(pixels_tile[:, :, c].astype(np.float64), kind)
        pred[:, :, c] = predict_surface(coeffs, height, width, kind).astype(np.float32)
    return bits_to_ordered(float_to_bits(np.ascontiguousarray(pred, dtype=np.float32)))


def positive_log_surface_prediction(pixels_tile: np.ndarray, kind: str) -> np.ndarray | None:
    if not bool(np.all(pixels_tile > 0.0)):
        return None
    height, width, channels = pixels_tile.shape
    pred = np.zeros_like(pixels_tile, dtype=np.float32)
    for c in range(channels):
        log_values = np.log2(pixels_tile[:, :, c].astype(np.float64))
        coeffs = fit_surface(log_values, kind)
        log_pred = predict_surface(coeffs, height, width, kind).astype(np.float64)
        pred[:, :, c] = np.exp2(log_pred).astype(np.float32)
    return bits_to_ordered(float_to_bits(np.ascontiguousarray(pred, dtype=np.float32)))


def body_delta_payload(ordered_tile: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    actual = ordered_tile & BODY_MASK
    pred = prediction & BODY_MASK
    return ((actual - pred) & BODY_MASK).astype(np.uint32)


def best_body_cost(payload: np.ndarray) -> tuple[float, dict[str, int]]:
    total = 0.0
    picks = Counter()
    for bit in range(30, -1, -1):
        family, cost = min(
            (
                (family, context_family_cost(payload, bit, family))
                for family in CONTEXT_FAMILIES
            ),
            key=lambda item: item[1],
        )
        total += cost
        picks[family] += 1
    return total, dict(picks)


def sign_cost(payload: np.ndarray) -> float:
    return min(context_family_cost(payload, 31, family) for family in CONTEXT_FAMILIES)


def tile_candidates(
    ordered_tile: np.ndarray,
    pixels_tile: np.ndarray,
) -> dict[str, np.ndarray]:
    candidates: dict[str, np.ndarray] = {}
    for kind in ("affine", "quadratic"):
        candidates[f"ordered_{kind}"] = body_delta_payload(
            ordered_tile,
            ordered_surface_prediction(ordered_tile, kind),
        )
        candidates[f"float_{kind}"] = body_delta_payload(
            ordered_tile,
            float_surface_prediction(pixels_tile, kind),
        )
        log_pred = positive_log_surface_prediction(pixels_tile, kind)
        if log_pred is not None:
            candidates[f"log_{kind}"] = body_delta_payload(ordered_tile, log_pred)
    return candidates


def coeff_overhead_bits(channels: int, mode: str) -> int:
    if mode == "current":
        return 0
    kind = "quadratic" if mode.endswith("quadratic") else "affine"
    coeffs = 6 if kind == "quadratic" else 3
    # 32-bit float coefficients per channel plus a small mode selector.
    return channels * coeffs * 32 + 8


def summarize_image(
    path: Path,
    crop_size: int,
    tile_size: int,
    previous_bits: int,
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
    height, width, channels = bits.shape
    raw_bits = int(bits.size * 32)
    current_bits = 0.0
    hybrid_bits = 0.0
    selected = Counter()
    candidate_totals = Counter()
    family_picks = Counter()
    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            sl = np.s_[y:y + tile_size, x:x + tile_size]
            current_body = payloads["body"][sl] & BODY_MASK
            current_sign = payloads["sign"][sl]
            tile_sign = sign_cost(current_sign)
            current_body_bits, current_picks = best_body_cost(current_body)
            current_tile = tile_sign + current_body_bits
            current_bits += current_tile
            best_name = "current"
            best_tile = current_tile
            family_picks.update({f"current:{k}": v for k, v in current_picks.items()})
            candidates = tile_candidates(ordered[sl], pixels[sl])
            for name, payload in candidates.items():
                body_bits, picks = best_body_cost(payload)
                tile_bits = tile_sign + body_bits + coeff_overhead_bits(channels, name)
                candidate_totals[name] += tile_bits
                if tile_bits < best_tile:
                    best_name = name
                    best_tile = tile_bits
                    family_picks.update({f"{name}:{k}": v for k, v in picks.items()})
            hybrid_bits += best_tile
            selected[best_name] += 1

    return {
        "image": path.name,
        "shape": list(bits.shape),
        "crop_size": crop_size,
        "tile_size": tile_size,
        "previous_bits": previous_bits,
        "raw_bits": raw_bits,
        "current_bits": current_bits,
        "hybrid_bits": hybrid_bits,
        "current_ratio": raw_bits / current_bits if current_bits else 1.0,
        "hybrid_ratio": raw_bits / hybrid_bits if hybrid_bits else 1.0,
        "gain": current_bits / hybrid_bits if hybrid_bits else 1.0,
        "selected": dict(selected),
        "candidate_ratios": {
            name: raw_bits / bits_value for name, bits_value in candidate_totals.items()
        },
        "family_picks": dict(family_picks),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", default="ph_*puresky*.exr")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--tile-size", type=int, default=128)
    parser.add_argument("--previous-bits", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = []
    for path in sorted(DATA_DIR.glob(args.glob)):
        row = summarize_image(path, args.crop_size, args.tile_size, args.previous_bits)
        rows.append(row)
        best_candidates = sorted(
            row["candidate_ratios"].items(),
            key=lambda item: item[1],
            reverse=True,
        )[:3]
        best_text = " ".join(f"{name}={ratio:.3f}x" for name, ratio in best_candidates)
        print(
            f"{path.stem:43s} current={row['current_ratio']:.3f}x "
            f"hybrid={row['hybrid_ratio']:.3f}x gain={row['gain']:.4f} "
            f"selected={row['selected']} {best_text}"
        )
        if args.limit and len(rows) >= args.limit:
            break
    if rows:
        print(
            f"geomean current={geomean([r['current_ratio'] for r in rows]):.3f}x "
            f"hybrid={geomean([r['hybrid_ratio'] for r in rows]):.3f}x"
        )
    safe_glob = args.glob.replace("*", "star").replace(".", "_")
    output = RESULTS_DIR / (
        f"tile_surface_predictors_{safe_glob}_tile{args.tile_size}"
        f"_prev{args.previous_bits}_crop{args.crop_size}.json"
    )
    output.write_text(json.dumps(rows, indent=2))
    print(f"\nSaved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
