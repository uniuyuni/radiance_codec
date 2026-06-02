"""Probe per-channel context-family selection for GDX3 payloads.

The MANIAC-style tree often starts by splitting on channel features.  Current
GDX3 chooses one context family for the whole tile/group/bitplane.  This probe
tests a cheaper production-shaped variant: keep the same grouped-delta payload,
but allow a bitplane to either use one shared family or choose one family per
channel, while charging explicit side bits.
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

from analyze_structural_predictors import DATA_DIR, FIELD_GROUPS, float_to_bits, read_exr  # noqa: E402
from evaluate_structural_delta_grouped import GROUP_PRESETS, group_bit_indices  # noqa: E402
from probe_grouped_tail_mlp import (  # noqa: E402
    CONTEXT_FAMILIES,
    context_cost_from_flat,
    family_context,
)
from structural_delta_grouped_roundtrip import choose_group_payloads  # noqa: E402

RESULTS_DIR = ROOT / "results"


def group_mask(fields: tuple[str, ...]) -> np.uint32:
    mask = np.uint32(0)
    for bit in group_bit_indices(fields):
        mask |= np.uint32(1) << np.uint32(bit)
    return mask


def best_shared_and_channel_families(
    payload: np.ndarray,
    bit: int,
) -> tuple[str, float, tuple[str, ...], float]:
    symbols = ((payload >> np.uint32(bit)) & np.uint32(1))
    shared_pairs = []
    channel_pairs: list[list[tuple[str, float]]] = [
        [] for _channel in range(payload.shape[2])
    ]
    for family in CONTEXT_FAMILIES:
        context, context_count = family_context(payload, bit, family)
        shared_pairs.append(
            (
                family,
                context_cost_from_flat(
                    context.reshape(-1),
                    symbols.reshape(-1),
                    context_count,
                ),
            )
        )
        for channel in range(payload.shape[2]):
            channel_pairs[channel].append(
                (
                    family,
                    context_cost_from_flat(
                        context[:, :, channel].reshape(-1),
                        symbols[:, :, channel].reshape(-1),
                        context_count,
                    ),
                )
            )
    shared_name, shared_cost = min(shared_pairs, key=lambda item: item[1])
    channel_names = []
    channel_cost = 0.0
    for pairs in channel_pairs:
        family, cost = min(pairs, key=lambda item: item[1])
        channel_names.append(family)
        channel_cost += cost
    return shared_name, shared_cost, tuple(channel_names), channel_cost


def summarize_image(
    path: Path,
    crop_size: int,
    tile_size: int,
    previous_bits: int,
    preset: str,
    family_id_bits: int,
    split_flag_bits: int,
    min_gain_bits: float,
) -> dict:
    pixels = read_exr(path)
    if crop_size:
        pixels = pixels[:crop_size, :crop_size]
    pixels = np.ascontiguousarray(pixels, dtype=np.float32)
    bits = float_to_bits(pixels).copy()
    modes, payloads, _carries = choose_group_payloads(
        bits,
        pixels,
        tile_size,
        previous_bits,
        preset,
    )

    height, width, channels = bits.shape
    baseline_bits = 0.0
    channel_split_bits = 0.0
    hybrid_bits = 0.0
    pick_histogram = Counter()
    shared_family_histogram = Counter()
    split_family_histogram = Counter()
    split_bit_histogram = Counter()
    split_group_histogram = Counter()

    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            sl = np.s_[y:y + tile_size, x:x + tile_size]
            for group_name, fields in GROUP_PRESETS[preset]:
                payload = payloads[group_name][sl] & group_mask(fields)
                mode = modes[(y, x, group_name)]
                for bit in group_bit_indices(fields):
                    if mode == "raw":
                        raw_cost = float(payload.size)
                        baseline_bits += raw_cost
                        channel_split_bits += raw_cost
                        hybrid_bits += raw_cost
                        pick_histogram["raw"] += 1
                        continue

                    shared_name, shared_cost, split_names, split_cost = (
                        best_shared_and_channel_families(payload, bit)
                    )
                    shared_cost += family_id_bits

                    split_cost += channels * family_id_bits + split_flag_bits

                    baseline_bits += shared_cost
                    channel_split_bits += split_cost
                    if split_cost + min_gain_bits < shared_cost:
                        hybrid_bits += split_cost
                        pick_histogram["channel_split"] += 1
                        split_bit_histogram[str(bit)] += 1
                        split_group_histogram[group_name] += 1
                        for name in split_names:
                            split_family_histogram[name] += 1
                    else:
                        hybrid_bits += shared_cost
                        pick_histogram["shared"] += 1
                        shared_family_histogram[shared_name] += 1

    raw_bits = int(bits.size * 32)
    return {
        "image": path.name,
        "shape": [height, width, channels],
        "crop_size": crop_size,
        "tile_size": tile_size,
        "previous_bits": previous_bits,
        "preset": preset,
        "family_id_bits": family_id_bits,
        "split_flag_bits": split_flag_bits,
        "min_gain_bits": min_gain_bits,
        "raw_bits": raw_bits,
        "baseline_bits": baseline_bits,
        "channel_split_bits": channel_split_bits,
        "hybrid_bits": hybrid_bits,
        "baseline_ratio": raw_bits / baseline_bits,
        "channel_split_ratio": raw_bits / channel_split_bits,
        "hybrid_ratio": raw_bits / hybrid_bits,
        "hybrid_gain": baseline_bits / hybrid_bits,
        "pick_histogram": dict(pick_histogram),
        "shared_family_histogram": dict(shared_family_histogram),
        "split_family_histogram": dict(split_family_histogram),
        "split_bit_histogram": dict(split_bit_histogram),
        "split_group_histogram": dict(split_group_histogram),
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
    parser.add_argument("--preset", choices=sorted(GROUP_PRESETS), default="body")
    parser.add_argument("--family-id-bits", type=int, default=4)
    parser.add_argument("--split-flag-bits", type=int, default=1)
    parser.add_argument("--min-gain-bits", type=float, default=4.0)
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
            args.preset,
            args.family_id_bits,
            args.split_flag_bits,
            args.min_gain_bits,
        )
        rows.append(row)
        print(
            f"{path.stem:43s} "
            f"shared={row['baseline_ratio']:.3f}x "
            f"split={row['channel_split_ratio']:.3f}x "
            f"hybrid={row['hybrid_ratio']:.3f}x "
            f"gain={row['hybrid_gain']:.4f} "
            f"picks={row['pick_histogram']}"
        )
        if args.limit and len(rows) >= args.limit:
            break
    if rows:
        print(
            f"geomean shared={geomean([r['baseline_ratio'] for r in rows]):.3f}x "
            f"split={geomean([r['channel_split_ratio'] for r in rows]):.3f}x "
            f"hybrid={geomean([r['hybrid_ratio'] for r in rows]):.3f}x"
        )
    safe_glob = args.glob.replace("*", "star").replace(".", "_")
    output = RESULTS_DIR / (
        f"channel_split_families_{safe_glob}_{args.preset}"
        f"_tile{args.tile_size}_prev{args.previous_bits}_crop{args.crop_size}.json"
    )
    output.write_text(json.dumps(rows, indent=2))
    print(f"\nSaved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
