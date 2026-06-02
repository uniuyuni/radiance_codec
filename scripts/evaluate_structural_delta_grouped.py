"""Estimate grouped ordered-float delta payloads.

The optimistic field-wise ordered-delta estimate is strong because each field is
scored with higher residual bits from the same predictor.  The actual mixed
payload loses that alignment when adjacent fields use different predictors.

This probe couples adjacent float fields into groups.  A group chooses one
payload mode for all its fields, preserving candidate-specific context inside
the group.  Whole-word delta modes only need a carry side bit when the group
does not start at bit 0.
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
    read_exr,
)
from evaluate_structural_delta_context import (  # noqa: E402
    bits_to_ordered,
    context_bitplane_cost,
    delta_predictors,
    ordered_delta,
    ordered_xor_predictors,
)
from evaluate_structural_delta_sidecarry import (  # noqa: E402
    carry_plane_cost,
    field_layout,
    incoming_carry_bits,
)

RESULTS_DIR = ROOT / "results"

GROUP_PRESETS = {
    "body": (
        ("body", ("exponent", "mantissa_hi", "mantissa_mid", "mantissa_lo")),
        ("sign", ("sign",)),
    ),
    "mantissa": (
        ("mantissa", ("mantissa_hi", "mantissa_mid", "mantissa_lo")),
        ("exponent", ("exponent",)),
        ("sign", ("sign",)),
    ),
    "exp_hi_tail": (
        ("tail", ("mantissa_mid", "mantissa_lo")),
        ("exp_hi", ("exponent", "mantissa_hi")),
        ("sign", ("sign",)),
    ),
    "hi_mid_lo_exp": (
        ("lo", ("mantissa_lo",)),
        ("hi_mid_exp", ("exponent", "mantissa_hi", "mantissa_mid")),
        ("sign", ("sign",)),
    ),
    "word": (
        (
            "word",
            ("sign", "exponent", "mantissa_hi", "mantissa_mid", "mantissa_lo"),
        ),
    ),
}


def group_bit_indices(fields: tuple[str, ...]) -> tuple[int, ...]:
    bits = []
    for field in fields:
        bits.extend(FIELD_GROUPS[field])
    return tuple(sorted(bits, reverse=True))


def field_mask(bit_indices: tuple[int, ...]) -> np.uint32:
    mask = np.uint32(0)
    for bit in bit_indices:
        mask |= np.uint32(1) << np.uint32(bit)
    return mask


GROUP_MASK_CACHE = {
    preset: {
        group_name: field_mask(group_bit_indices(fields))
        for group_name, fields in groups
    }
    for preset, groups in GROUP_PRESETS.items()
}


def filter_group_predictors(
    group_name: str,
    xor_tiles: dict[str, np.ndarray],
    delta_tiles: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Keep only predictors supported by the production-shaped decode schedule."""

    if group_name == "sign":
        return (
            {
                name: value
                for name, value in xor_tiles.items()
                if name != "channel_green"
            },
            {},
        )
    return xor_tiles, delta_tiles


def grouped_delta(
    actual: np.ndarray,
    prediction: np.ndarray,
    bit_indices: tuple[int, ...],
) -> np.ndarray:
    low_bit, width, mask = field_layout(bit_indices)
    actual_group = (actual >> np.uint32(low_bit)) & mask
    prediction_group = (prediction >> np.uint32(low_bit)) & mask
    residual = (actual_group - prediction_group) & mask
    return (residual << np.uint32(low_bit)).astype(np.uint32)


def score_payload(
    payload: np.ndarray,
    fields: tuple[str, ...],
    previous_bits: int,
    raw: bool,
    bitplane_coding: str,
) -> float:
    bit_indices = group_bit_indices(fields)
    group_mask = field_mask(bit_indices)
    masked_payload = payload & group_mask
    tile_values = payload.size
    total = 0.0
    for field in fields:
        field_bits = FIELD_GROUPS[field]
        if raw:
            total += tile_values * len(field_bits)
        else:
            for bit_index in field_bits:
                context_cost = context_bitplane_cost(
                    masked_payload,
                    (bit_index,),
                    previous_bits,
                    channel_context=False,
                )
                if bitplane_coding == "sparse_hybrid":
                    total += min(
                        context_cost,
                        sparse_byte_bitplane_cost(masked_payload, bit_index),
                    ) + 1.0
                else:
                    total += context_cost
    return total


def sparse_byte_bitplane_cost(payload: np.ndarray, bit_index: int) -> float:
    """Falcon-style sparse byte storage estimate for one transposed bitplane."""

    plane = ((payload >> np.uint32(bit_index)) & np.uint32(1)).astype(
        np.uint8,
        copy=False,
    ).reshape(-1)
    packed = np.packbits(plane, bitorder="little")
    nonzero_bytes = int(np.count_nonzero(packed))
    # One bitmap bit per packed byte, then raw payload for non-zero bytes.
    return float(packed.size + 8 * nonzero_bytes)


def group_candidates(
    ordered_tile: np.ndarray,
    xor_tiles: dict[str, np.ndarray],
    delta_tiles: dict[str, np.ndarray],
    fields: tuple[str, ...],
    previous_bits: int,
    carry_channel_context: bool,
    bitplane_coding: str,
) -> dict[str, tuple[float, float, np.ndarray | None]]:
    bit_indices = group_bit_indices(fields)
    low_bit = min(bit_indices)
    candidates: dict[str, tuple[float, float, np.ndarray | None]] = {}

    raw_cost = score_payload(
        ordered_tile,
        fields,
        previous_bits,
        raw=True,
        bitplane_coding=bitplane_coding,
    )
    candidates["raw"] = (raw_cost, 0.0, None)

    for name, prediction in xor_tiles.items():
        residual = ordered_tile ^ prediction
        candidates[f"xor_{name}"] = (
            score_payload(
                residual,
                fields,
                previous_bits,
                raw=False,
                bitplane_coding=bitplane_coding,
            ),
            0.0,
            None,
        )

    for name, prediction in delta_tiles.items():
        whole = ordered_delta(ordered_tile, prediction)
        carry_cost = 0.0
        carry = None
        if low_bit > 0:
            carry = incoming_carry_bits(
                ordered_tile,
                prediction,
                whole,
                bit_indices,
            )
            carry_cost = carry_plane_cost(carry, carry_channel_context)
        candidates[name] = (
            score_payload(
                whole,
                fields,
                previous_bits,
                raw=False,
                bitplane_coding=bitplane_coding,
            ) + carry_cost,
            carry_cost,
            carry,
        )

        local = grouped_delta(ordered_tile, prediction, bit_indices)
        candidates[f"local_{name}"] = (
            score_payload(
                local,
                fields,
                previous_bits,
                raw=False,
                bitplane_coding=bitplane_coding,
            ),
            0.0,
            None,
        )

    return candidates


def estimate_image(
    path: Path,
    crop_size: int,
    tile_size: int,
    previous_bits: int,
    mode_overhead_bits: int,
    preset: str,
    carry_channel_context: bool,
    bitplane_coding: str,
) -> dict:
    pixels = read_exr(path)
    if crop_size:
        pixels = pixels[:crop_size, :crop_size]
    bits = float_to_bits(pixels)
    ordered = bits_to_ordered(bits)
    xor_predictor_bits = ordered_xor_predictors(bits, pixels, causal=True)
    delta_prediction_bits = delta_predictors(bits, pixels, causal=True)
    groups = GROUP_PRESETS[preset]
    height, width, channels = bits.shape
    raw_bits = bits.size * 32
    total_bits = 0.0
    mode_histogram = Counter()
    carry_histogram = Counter()

    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            sl = np.s_[y:y + tile_size, x:x + tile_size]
            ordered_tile = ordered[sl]
            xor_tiles = {name: pred[sl] for name, pred in xor_predictor_bits.items()}
            delta_tiles = {
                name: pred[sl] for name, pred in delta_prediction_bits.items()
            }
            for group_name, fields in groups:
                group_xor_tiles, group_delta_tiles = filter_group_predictors(
                    group_name,
                    xor_tiles,
                    delta_tiles,
                )
                candidates = group_candidates(
                    ordered_tile,
                    group_xor_tiles,
                    group_delta_tiles,
                    fields,
                    previous_bits,
                    carry_channel_context,
                    bitplane_coding,
                )
                mode, (cost, carry_cost, carry) = min(
                    candidates.items(),
                    key=lambda item: item[1][0],
                )
                total_bits += cost + mode_overhead_bits
                mode_histogram[f"{group_name}:{mode}"] += 1
                if carry is not None:
                    carry_histogram[f"{group_name}:cost_bits"] += carry_cost
                    carry_histogram[f"{group_name}:ones"] += int(np.sum(carry))
                    carry_histogram[f"{group_name}:total"] += int(carry.size)

    return {
        "image": path.name,
        "shape": [height, width, channels],
        "crop_size": crop_size,
        "tile_size": tile_size,
        "previous_bits": previous_bits,
        "preset": preset,
        "raw_bits": raw_bits,
        "estimated_bits": total_bits,
        "ratio": raw_bits / total_bits,
        "mode_histogram": dict(mode_histogram),
        "carry_histogram": dict(carry_histogram),
    }


def geomean(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", default="*.exr")
    parser.add_argument("--crop-size", type=int, default=0)
    parser.add_argument("--tile-size", type=int, default=128)
    parser.add_argument("--previous-bits", type=int, default=3)
    parser.add_argument("--mode-overhead-bits", type=int, default=5)
    parser.add_argument("--preset", choices=sorted(GROUP_PRESETS), default="body")
    parser.add_argument("--carry-channel-context", action="store_true")
    parser.add_argument(
        "--bitplane-coding",
        choices=("context", "sparse_hybrid"),
        default="context",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = sorted(DATA_DIR.glob(args.glob))
    if not paths:
        raise RuntimeError(f"no images matched {DATA_DIR / args.glob}")
    blosc = load_blosc_ratios()
    results = []
    print(
        f"{'image':43s} {'group':>9s} {'Blosc':>9s} "
        f"{'hybrid':>9s} {'pick':>8s}"
    )
    print("-" * 86)
    for path in paths:
        result = estimate_image(
            path,
            args.crop_size,
            args.tile_size,
            args.previous_bits,
            args.mode_overhead_bits,
            args.preset,
            args.carry_channel_context,
            args.bitplane_coding,
        )
        blosc_ratio = blosc.get(path.stem)
        hybrid_ratio = max(result["ratio"], blosc_ratio or 0.0)
        pick = "group" if result["ratio"] >= (blosc_ratio or 0.0) else "Blosc"
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

    coding_suffix = (
        "" if args.bitplane_coding == "context" else f"_{args.bitplane_coding}"
    )
    output = RESULTS_DIR / (
        f"structural_delta_grouped_{args.preset}"
        f"_tile{args.tile_size}_prev{args.previous_bits}{coding_suffix}"
        f"_crop{args.crop_size}.json"
    )
    output.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
