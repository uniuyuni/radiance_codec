"""Probe a reversible 2x2 Lorenzo pyramid for ordered-float body bits.

GDX3 currently codes the ordered-float body with local predictors in raster
order.  This script tests a different lossless representation for the body:

    a = top-left anchor
    h = top-right - a
    v = bottom-left - a
    q = bottom-right - top-right - bottom-left + a

The residuals are signed modulo-2^31 body differences and are zigzag mapped
back into 31-bit unsigned symbols.  Anchors can be transformed recursively,
yielding a quadtree-like exact representation.  The sign bit stays on the
current GDX3 path; this probe is about whether upper-body structure benefits
from a block/hierarchy representation.
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
BODY_MASK = np.uint32(0x7FFFFFFF)
BODY_MOD = 1 << 31
BODY_HALF = 1 << 30


def wrap_body(values: np.ndarray) -> np.ndarray:
    return (values.astype(np.int64) & int(BODY_MASK)).astype(np.uint32)


def signed_body_diff(actual: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    raw = (
        actual.astype(np.int64)
        - prediction.astype(np.int64)
    ) & int(BODY_MASK)
    return np.where(raw >= BODY_HALF, raw - BODY_MOD, raw).astype(np.int64)


def zigzag31(values: np.ndarray) -> np.ndarray:
    encoded = (values << 1) ^ (values >> 30)
    return (encoded & int(BODY_MASK)).astype(np.uint32)


def unzigzag31(values: np.ndarray) -> np.ndarray:
    unsigned = values.astype(np.int64) & int(BODY_MASK)
    return ((unsigned >> 1) ^ -(unsigned & 1)).astype(np.int64)


def lowpass_from_anchor(
    anchor: np.ndarray,
    h_diff: np.ndarray,
    v_diff: np.ndarray,
    q_diff: np.ndarray,
    lowpass: str,
) -> np.ndarray:
    if lowpass == "anchor":
        return anchor
    if lowpass == "average":
        # Reversible S-transform-like lift.  The stored lowpass is close to the
        # block mean, but the exact anchor is recovered by subtracting the same
        # integer terms during inverse.
        return wrap_body(
            anchor.astype(np.int64)
            + np.floor_divide(h_diff, 2)
            + np.floor_divide(v_diff, 2)
            + np.floor_divide(q_diff, 4)
        )
    raise ValueError(f"unknown lowpass mode: {lowpass}")


def anchor_from_lowpass(
    low: np.ndarray,
    h_diff: np.ndarray,
    v_diff: np.ndarray,
    q_diff: np.ndarray,
    lowpass: str,
) -> np.ndarray:
    if lowpass == "anchor":
        return low
    if lowpass == "average":
        return wrap_body(
            low.astype(np.int64)
            - np.floor_divide(h_diff, 2)
            - np.floor_divide(v_diff, 2)
            - np.floor_divide(q_diff, 4)
        )
    raise ValueError(f"unknown lowpass mode: {lowpass}")


def lorenzo_pyramid_transform(
    body: np.ndarray,
    max_levels: int,
    lowpass: str,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Return transformed coefficients and the transformed region sizes."""

    coeff = (body & BODY_MASK).copy()
    height, width, _channels = coeff.shape
    current_h = height
    current_w = width
    levels: list[tuple[int, int]] = []

    for _level in range(max_levels):
        if current_h < 2 or current_w < 2:
            break
        if current_h % 2 or current_w % 2:
            break

        h2 = current_h // 2
        w2 = current_w // 2
        region = coeff[:current_h, :current_w]
        a = region[0::2, 0::2].copy()
        b = region[0::2, 1::2]
        c = region[1::2, 0::2]
        d = region[1::2, 1::2]

        h_diff = signed_body_diff(b, a)
        v_diff = signed_body_diff(c, a)
        diag_prediction = wrap_body(a.astype(np.int64) + h_diff + v_diff)
        q_diff = signed_body_diff(d, diag_prediction)

        coeff[:h2, :w2] = lowpass_from_anchor(a, h_diff, v_diff, q_diff, lowpass)
        coeff[:h2, w2:current_w] = zigzag31(h_diff)
        coeff[h2:current_h, :w2] = zigzag31(v_diff)
        coeff[h2:current_h, w2:current_w] = zigzag31(q_diff)

        levels.append((current_h, current_w))
        current_h = h2
        current_w = w2

    return coeff, levels


def lorenzo_pyramid_inverse(
    coeff: np.ndarray,
    levels: list[tuple[int, int]],
    lowpass: str,
) -> np.ndarray:
    body = (coeff & BODY_MASK).copy()
    for current_h, current_w in reversed(levels):
        h2 = current_h // 2
        w2 = current_w // 2
        h_diff = unzigzag31(body[:h2, w2:current_w])
        v_diff = unzigzag31(body[h2:current_h, :w2])
        q_diff = unzigzag31(body[h2:current_h, w2:current_w])
        a = anchor_from_lowpass(
            body[:h2, :w2].copy(),
            h_diff,
            v_diff,
            q_diff,
            lowpass,
        )

        b = wrap_body(a.astype(np.int64) + h_diff)
        c = wrap_body(a.astype(np.int64) + v_diff)
        d = wrap_body(a.astype(np.int64) + h_diff + v_diff + q_diff)

        restored = np.empty((current_h, current_w, body.shape[2]), dtype=np.uint32)
        restored[0::2, 0::2] = a
        restored[0::2, 1::2] = b
        restored[1::2, 0::2] = c
        restored[1::2, 1::2] = d
        body[:current_h, :current_w] = restored
    return body


def bitplane_best_family_cost(payload: np.ndarray, bit: int) -> tuple[str, float]:
    pairs = [(family, context_family_cost(payload, bit, family)) for family in CONTEXT_FAMILIES]
    return min(pairs, key=lambda item: item[1])


def body_cost(payload: np.ndarray) -> tuple[float, dict[str, int]]:
    total = 0.0
    picks = Counter()
    masked = payload & BODY_MASK
    for bit in range(30, -1, -1):
        family, cost = bitplane_best_family_cost(masked, bit)
        total += cost
        picks[family] += 1
    return total, dict(picks)


def sign_cost(payload: np.ndarray) -> float:
    return context_family_cost(payload, 31, "base")


def transform_tile_body(
    ordered_tile: np.ndarray,
    max_levels: int,
    lowpass: str,
) -> tuple[np.ndarray, int]:
    body = ordered_tile & BODY_MASK
    transformed, levels = lorenzo_pyramid_transform(body, max_levels, lowpass)
    restored = lorenzo_pyramid_inverse(transformed, levels, lowpass)
    if not np.array_equal(restored, body):
        mismatch = np.argwhere(restored != body)[0]
        y, x, c = (int(v) for v in mismatch)
        raise RuntimeError(
            f"lorenzo inverse mismatch at y={y} x={x} c={c}: "
            f"got=0x{int(restored[y, x, c]):08x} "
            f"expected=0x{int(body[y, x, c]):08x}"
        )
    return transformed, len(levels)


def summarize_image(
    path: Path,
    crop_size: int,
    tile_size: int,
    previous_bits: int,
    max_levels: int,
    lowpass: str,
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

    height, width, _channels = bits.shape
    baseline_bits = 0.0
    lorenzo_bits = 0.0
    hybrid_bits = 0.0
    sign_bits = 0.0
    selected = Counter()
    level_histogram = Counter()
    family_histogram = Counter()

    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            sl = np.s_[y:y + tile_size, x:x + tile_size]
            body_payload = payloads["body"][sl]
            sign_payload = payloads["sign"][sl]
            tile_sign_bits = sign_cost(sign_payload)
            tile_body_bits, body_picks = body_cost(body_payload)

            transformed_body, level_count = transform_tile_body(
                ordered[sl],
                max_levels,
                lowpass,
            )
            tile_lorenzo_bits, lorenzo_picks = body_cost(transformed_body)
            tile_lorenzo_bits += mode_overhead_bits

            sign_bits += tile_sign_bits
            baseline_bits += tile_sign_bits + tile_body_bits
            lorenzo_bits += tile_sign_bits + tile_lorenzo_bits
            if tile_lorenzo_bits < tile_body_bits:
                hybrid_bits += tile_sign_bits + tile_lorenzo_bits
                selected["lorenzo"] += 1
                family_histogram.update(lorenzo_picks)
            else:
                hybrid_bits += tile_sign_bits + tile_body_bits
                selected["gdx3"] += 1
                family_histogram.update(body_picks)
            level_histogram[str(level_count)] += 1

    raw_bits = int(bits.size * 32)
    return {
        "image": path.name,
        "shape": list(bits.shape),
        "crop_size": crop_size,
        "tile_size": tile_size,
        "previous_bits": previous_bits,
        "max_levels": max_levels,
        "lowpass": lowpass,
        "mode_overhead_bits": mode_overhead_bits,
        "raw_bits": raw_bits,
        "sign_bits": sign_bits,
        "baseline_bits": baseline_bits,
        "lorenzo_bits": lorenzo_bits,
        "hybrid_bits": hybrid_bits,
        "baseline_ratio": raw_bits / baseline_bits if baseline_bits else 1.0,
        "lorenzo_ratio": raw_bits / lorenzo_bits if lorenzo_bits else 1.0,
        "hybrid_ratio": raw_bits / hybrid_bits if hybrid_bits else 1.0,
        "hybrid_gain": baseline_bits / hybrid_bits if hybrid_bits else 1.0,
        "selected": dict(selected),
        "level_histogram": dict(level_histogram),
        "family_histogram": dict(family_histogram),
    }


def geomean(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", default="*.exr")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--crop-size", type=int, default=0)
    parser.add_argument("--tile-size", type=int, default=128)
    parser.add_argument("--previous-bits", type=int, default=4)
    parser.add_argument("--max-levels", type=int, default=4)
    parser.add_argument("--lowpass", choices=("anchor", "average"), default="average")
    parser.add_argument("--mode-overhead-bits", type=float, default=16.0)
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
            args.max_levels,
            args.lowpass,
            args.mode_overhead_bits,
        )
        rows.append(row)
        print(
            f"{path.stem:43s} "
            f"gdx={row['baseline_ratio']:.3f}x "
            f"lorenzo={row['lorenzo_ratio']:.3f}x "
            f"hybrid={row['hybrid_ratio']:.3f}x "
            f"gain={row['hybrid_gain']:.4f} "
            f"selected={row['selected']}"
        )
        if args.limit and len(rows) >= args.limit:
            break
    if rows:
        print(
            f"geomean gdx={geomean([r['baseline_ratio'] for r in rows]):.3f}x "
            f"lorenzo={geomean([r['lorenzo_ratio'] for r in rows]):.3f}x "
            f"hybrid={geomean([r['hybrid_ratio'] for r in rows]):.3f}x"
        )
    safe_glob = args.glob.replace("*", "star").replace(".", "_")
    output = RESULTS_DIR / (
        f"ordered_body_lorenzo_{safe_glob}_tile{args.tile_size}"
        f"_prev{args.previous_bits}_levels{args.max_levels}"
        f"_{args.lowpass}_crop{args.crop_size}.json"
    )
    output.write_text(json.dumps(rows, indent=2))
    print(f"\nSaved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
