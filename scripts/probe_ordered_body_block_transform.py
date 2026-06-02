"""Probe exact upper-body block transforms with lower residual fallback.

The earlier Lorenzo-pyramid probe transformed the whole ordered-float body.
That is a blunt instrument: for half/bfloat-like HDR images the lower body is
often already zero or well handled by GDX, while puresky lower tails are the
hard incompressible part.  This probe splits the exact body:

* high actual ordered-body bits are encoded through a reversible 2x2 block
  transform;
* low GDX residual bits are kept on the current payload route;
* sign stays on the current sign route.

This remains exact in principle: given decoder-reproducible prediction low bits,
the low residual reconstructs the actual low bits, while the transformed high
stream reconstructs the actual high bits.
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


def mask_for_width(width: int) -> int:
    if width <= 0 or width > 31:
        raise ValueError(f"width must be in 1..31, got {width}")
    return (1 << width) - 1


def wrap_width(values: np.ndarray, width: int) -> np.ndarray:
    return (values.astype(np.int64) & mask_for_width(width)).astype(np.uint32)


def signed_diff_width(actual: np.ndarray, prediction: np.ndarray, width: int) -> np.ndarray:
    mask = mask_for_width(width)
    half = 1 << (width - 1)
    modulus = 1 << width
    raw = (actual.astype(np.int64) - prediction.astype(np.int64)) & mask
    return np.where(raw >= half, raw - modulus, raw).astype(np.int64)


def zigzag_width(values: np.ndarray, width: int) -> np.ndarray:
    encoded = (values << 1) ^ (values >> (width - 1))
    return (encoded & mask_for_width(width)).astype(np.uint32)


def unzigzag_width(values: np.ndarray, width: int) -> np.ndarray:
    unsigned = values.astype(np.int64) & mask_for_width(width)
    return ((unsigned >> 1) ^ -(unsigned & 1)).astype(np.int64)


def lowpass_from_anchor(
    anchor: np.ndarray,
    h_diff: np.ndarray,
    v_diff: np.ndarray,
    q_diff: np.ndarray,
    width: int,
    lowpass: str,
) -> np.ndarray:
    if lowpass == "anchor":
        return anchor
    if lowpass == "average":
        return wrap_width(
            anchor.astype(np.int64)
            + np.floor_divide(h_diff, 2)
            + np.floor_divide(v_diff, 2)
            + np.floor_divide(q_diff, 4),
            width,
        )
    raise ValueError(f"unknown lowpass mode: {lowpass}")


def anchor_from_lowpass(
    low: np.ndarray,
    h_diff: np.ndarray,
    v_diff: np.ndarray,
    q_diff: np.ndarray,
    width: int,
    lowpass: str,
) -> np.ndarray:
    if lowpass == "anchor":
        return low
    if lowpass == "average":
        return wrap_width(
            low.astype(np.int64)
            - np.floor_divide(h_diff, 2)
            - np.floor_divide(v_diff, 2)
            - np.floor_divide(q_diff, 4),
            width,
        )
    raise ValueError(f"unknown lowpass mode: {lowpass}")


def transform_2x2_pyramid(
    values: np.ndarray,
    width: int,
    max_levels: int,
    lowpass: str,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    coeff = wrap_width(values, width)
    current_h, current_w, _channels = coeff.shape
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

        h_diff = signed_diff_width(b, a, width)
        v_diff = signed_diff_width(c, a, width)
        diag_prediction = wrap_width(a.astype(np.int64) + h_diff + v_diff, width)
        q_diff = signed_diff_width(d, diag_prediction, width)

        coeff[:h2, :w2] = lowpass_from_anchor(
            a,
            h_diff,
            v_diff,
            q_diff,
            width,
            lowpass,
        )
        coeff[:h2, w2:current_w] = zigzag_width(h_diff, width)
        coeff[h2:current_h, :w2] = zigzag_width(v_diff, width)
        coeff[h2:current_h, w2:current_w] = zigzag_width(q_diff, width)
        levels.append((current_h, current_w))
        current_h = h2
        current_w = w2
    return coeff, levels


def inverse_2x2_pyramid(
    coeff: np.ndarray,
    width: int,
    levels: list[tuple[int, int]],
    lowpass: str,
) -> np.ndarray:
    values = wrap_width(coeff, width)
    for current_h, current_w in reversed(levels):
        h2 = current_h // 2
        w2 = current_w // 2
        h_diff = unzigzag_width(values[:h2, w2:current_w], width)
        v_diff = unzigzag_width(values[h2:current_h, :w2], width)
        q_diff = unzigzag_width(values[h2:current_h, w2:current_w], width)
        a = anchor_from_lowpass(
            values[:h2, :w2].copy(),
            h_diff,
            v_diff,
            q_diff,
            width,
            lowpass,
        )
        b = wrap_width(a.astype(np.int64) + h_diff, width)
        c = wrap_width(a.astype(np.int64) + v_diff, width)
        d = wrap_width(a.astype(np.int64) + h_diff + v_diff + q_diff, width)

        restored = np.empty((current_h, current_w, values.shape[2]), dtype=np.uint32)
        restored[0::2, 0::2] = a
        restored[0::2, 1::2] = b
        restored[1::2, 0::2] = c
        restored[1::2, 1::2] = d
        values[:current_h, :current_w] = restored
    return values


def best_bitplane_cost(payload: np.ndarray, bit: int) -> tuple[str, float]:
    return min(
        ((family, context_family_cost(payload, bit, family)) for family in CONTEXT_FAMILIES),
        key=lambda item: item[1],
    )


def cost_bit_range(payload: np.ndarray, bit_start: int, bit_end: int) -> tuple[float, Counter[str]]:
    total = 0.0
    picks: Counter[str] = Counter()
    for bit in range(bit_end, bit_start - 1, -1):
        family, cost = best_bitplane_cost(payload, bit)
        total += cost
        picks[family] += 1
    return total, picks


def sign_cost(payload: np.ndarray) -> float:
    return context_family_cost(payload, 31, "base")


def split_transform_tile(
    body_actual: np.ndarray,
    body_payload: np.ndarray,
    split_bit: int,
    max_levels: int,
    lowpass: str,
) -> tuple[np.ndarray, np.ndarray, int]:
    low_mask = np.uint32((1 << split_bit) - 1) if split_bit else np.uint32(0)
    low_residual = body_payload & low_mask
    high_actual = body_actual >> np.uint32(split_bit)
    high_width = 31 - split_bit
    transformed, levels = transform_2x2_pyramid(
        high_actual,
        high_width,
        max_levels,
        lowpass,
    )
    restored_high = inverse_2x2_pyramid(transformed, high_width, levels, lowpass)
    if not np.array_equal(restored_high, high_actual):
        mismatch = np.argwhere(restored_high != high_actual)[0]
        y, x, c = (int(v) for v in mismatch)
        raise RuntimeError(
            f"block transform inverse mismatch at y={y} x={x} c={c}: "
            f"got=0x{int(restored_high[y, x, c]):08x} "
            f"expected=0x{int(high_actual[y, x, c]):08x}"
        )
    return transformed, low_residual, len(levels)


def prepare_image(
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
    body_actual = ordered & BODY_MASK
    _modes, payloads, _carries = choose_group_payloads(
        bits,
        pixels,
        tile_size,
        previous_bits,
        "body",
    )
    return {
        "image": path.name,
        "shape": list(bits.shape),
        "crop_size": crop_size,
        "tile_size": tile_size,
        "previous_bits": previous_bits,
        "raw_bits": int(bits.size * 32),
        "body_actual": body_actual,
        "payloads": payloads,
    }


def summarize_prepared_image(
    prepared: dict,
    split_bit: int,
    max_levels: int,
    lowpass: str,
    mode_overhead_bits: float,
) -> dict:
    body_actual = prepared["body_actual"]
    payloads = prepared["payloads"]
    tile_size = prepared["tile_size"]

    height, width, _channels = body_actual.shape
    baseline_bits = 0.0
    split_bits = 0.0
    hybrid_bits = 0.0
    selected: Counter[str] = Counter()
    level_histogram: Counter[str] = Counter()
    baseline_family_histogram: Counter[str] = Counter()
    split_family_histogram: Counter[str] = Counter()

    high_width = 31 - split_bit
    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            sl = np.s_[y:y + tile_size, x:x + tile_size]
            body_payload = payloads["body"][sl]
            sign_payload = payloads["sign"][sl]
            tile_sign = sign_cost(sign_payload)

            tile_body, body_picks = cost_bit_range(body_payload, 0, 30)
            baseline = tile_sign + tile_body

            transformed_high, low_residual, levels = split_transform_tile(
                body_actual[sl],
                body_payload,
                split_bit,
                max_levels,
                lowpass,
            )
            if split_bit:
                low_cost, low_picks = cost_bit_range(low_residual, 0, split_bit - 1)
            else:
                low_cost, low_picks = 0.0, Counter()
            high_cost, high_picks = cost_bit_range(transformed_high, 0, high_width - 1)
            split = tile_sign + low_cost + high_cost + mode_overhead_bits

            baseline_bits += baseline
            split_bits += split
            baseline_family_histogram.update(body_picks)
            if split < baseline:
                hybrid_bits += split
                selected["split_transform"] += 1
                split_family_histogram.update(low_picks)
                split_family_histogram.update(high_picks)
            else:
                hybrid_bits += baseline
                selected["gdx"] += 1
                split_family_histogram.update(body_picks)
            level_histogram[str(levels)] += 1

    return {
        "image": prepared["image"],
        "shape": prepared["shape"],
        "crop_size": prepared["crop_size"],
        "tile_size": tile_size,
        "previous_bits": prepared["previous_bits"],
        "split_bit": split_bit,
        "high_width": high_width,
        "max_levels": max_levels,
        "lowpass": lowpass,
        "mode_overhead_bits": mode_overhead_bits,
        "raw_bits": prepared["raw_bits"],
        "baseline_bits": baseline_bits,
        "split_bits": split_bits,
        "hybrid_bits": hybrid_bits,
        "baseline_ratio": prepared["raw_bits"] / baseline_bits if baseline_bits else 1.0,
        "split_ratio": prepared["raw_bits"] / split_bits if split_bits else 1.0,
        "hybrid_ratio": prepared["raw_bits"] / hybrid_bits if hybrid_bits else 1.0,
        "hybrid_gain": baseline_bits / hybrid_bits if hybrid_bits else 1.0,
        "selected": dict(selected),
        "level_histogram": dict(level_histogram),
        "baseline_family_histogram": dict(baseline_family_histogram),
        "split_family_histogram": dict(split_family_histogram),
    }


def summarize_image(
    path: Path,
    crop_size: int,
    tile_size: int,
    previous_bits: int,
    split_bit: int,
    max_levels: int,
    lowpass: str,
    mode_overhead_bits: float,
) -> dict:
    prepared = prepare_image(path, crop_size, tile_size, previous_bits)
    return summarize_prepared_image(
        prepared,
        split_bit,
        max_levels,
        lowpass,
        mode_overhead_bits,
    )


def geomean(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def parse_int_list(value: str | None, fallback: int) -> list[int]:
    if not value:
        return [fallback]
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_lowpass_list(value: str | None, fallback: str) -> list[str]:
    if not value:
        return [fallback]
    lowpasses = [part.strip() for part in value.split(",") if part.strip()]
    valid = {"anchor", "average"}
    unknown = sorted(set(lowpasses) - valid)
    if unknown:
        raise ValueError(f"unknown lowpass values: {', '.join(unknown)}")
    return lowpasses


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", default="*.exr")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--tile-size", type=int, default=128)
    parser.add_argument("--previous-bits", type=int, default=4)
    parser.add_argument("--split-bit", type=int, default=15)
    parser.add_argument("--split-bits", default="")
    parser.add_argument("--max-levels", type=int, default=4)
    parser.add_argument("--lowpass", choices=("anchor", "average"), default="average")
    parser.add_argument("--lowpasses", default="")
    parser.add_argument("--mode-overhead-bits", type=float, default=16.0)
    parser.add_argument("--no-save", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = sorted(DATA_DIR.glob(args.glob))
    if args.limit:
        paths = paths[:args.limit]
    if not paths:
        raise RuntimeError(f"no images matched {DATA_DIR / args.glob}")

    rows = []
    split_bits = parse_int_list(args.split_bits, args.split_bit)
    lowpasses = parse_lowpass_list(args.lowpasses, args.lowpass)
    print(
        f"{'image':43s} {'split':>5s} {'lowpass':>7s} {'gdx':>9s} {'route':>9s} "
        f"{'hybrid':>9s} {'gain':>8s} selected"
    )
    print("-" * 121)
    for path in paths:
        prepared = prepare_image(
            path,
            args.crop_size,
            args.tile_size,
            args.previous_bits,
        )
        for split_bit in split_bits:
            for lowpass in lowpasses:
                row = summarize_prepared_image(
                    prepared,
                    split_bit,
                    args.max_levels,
                    lowpass,
                    args.mode_overhead_bits,
                )
                rows.append(row)
                print(
                    f"{path.stem:43s} "
                    f"{split_bit:5d} "
                    f"{lowpass:>7s} "
                    f"{row['baseline_ratio']:8.3f}x "
                    f"{row['split_ratio']:8.3f}x "
                    f"{row['hybrid_ratio']:8.3f}x "
                    f"{row['hybrid_gain']:7.4f} {row['selected']}"
                )

    print("-" * 121)
    print(
        f"geomean gdx={geomean([row['baseline_ratio'] for row in rows]):.3f}x "
        f"route={geomean([row['split_ratio'] for row in rows]):.3f}x "
        f"hybrid={geomean([row['hybrid_ratio'] for row in rows]):.3f}x"
    )

    if not args.no_save:
        safe_glob = args.glob.replace("*", "star").replace(".", "_")
        split_label = (args.split_bits or str(args.split_bit)).replace(",", "-")
        lowpass_label = (args.lowpasses or args.lowpass).replace(",", "-")
        output = RESULTS_DIR / (
            f"ordered_body_block_transform_{safe_glob}_tile{args.tile_size}"
            f"_prev{args.previous_bits}_split{split_label}"
            f"_levels{args.max_levels}_{lowpass_label}_crop{args.crop_size}.json"
        )
        output.write_text(json.dumps(rows, indent=2))
        print(f"\nSaved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
