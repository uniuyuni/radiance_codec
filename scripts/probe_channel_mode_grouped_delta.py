"""Probe per-channel predictor modes for grouped ordered-float body payloads.

GDX4 can choose context families per channel, but the predictor mode is still
one mode per tile/group.  This probe asks whether choosing the body predictor
mode per channel is worth the extra mode side information.

The output is an estimate only.  It keeps the current sign payload/mode, builds
a body payload from independently selected channel modes, and scores the final
payload with the same channel-split family estimator used by the GDX4 probe.
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
from evaluate_structural_delta_context import (  # noqa: E402
    bits_to_ordered,
    delta_predictors,
    ordered_xor_predictors,
)
from evaluate_structural_delta_grouped import (  # noqa: E402
    GROUP_PRESETS,
    filter_group_predictors,
    group_bit_indices,
)
from probe_channel_split_families import best_shared_and_channel_families  # noqa: E402
from probe_grouped_tail_mlp import context_family_cost  # noqa: E402
from structural_delta_grouped_roundtrip import (  # noqa: E402
    choose_group_payloads,
    mode_payload,
)

RESULTS_DIR = ROOT / "results"
BODY_MASK = np.uint32(0x7FFFFFFF)
SIGN_MASK = np.uint32(0x80000000)


def body_candidates(
    ordered_tile: np.ndarray,
    xor_tiles: dict[str, np.ndarray],
    delta_tiles: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    group_xor_tiles, group_delta_tiles = filter_group_predictors(
        "body",
        xor_tiles,
        delta_tiles,
    )
    bit_indices = group_bit_indices(GROUP_PRESETS["body"][0][1])
    candidates = {"raw": ordered_tile & BODY_MASK}
    for name in sorted(group_xor_tiles):
        payload, _carry = mode_payload(
            f"xor_{name}",
            ordered_tile,
            group_xor_tiles,
            group_delta_tiles,
            bit_indices,
        )
        candidates[f"xor_{name}"] = payload & BODY_MASK
    for name in sorted(group_delta_tiles):
        payload, _carry = mode_payload(
            name,
            ordered_tile,
            group_xor_tiles,
            group_delta_tiles,
            bit_indices,
        )
        candidates[name] = payload & BODY_MASK
    return candidates


def channel_decode_rank(channels: int) -> dict[int, int]:
    if channels >= 3:
        order = [1, 0, 2, 3]
    else:
        order = [0, 1, 2, 3]
    return {channel: rank for rank, channel in enumerate(order[:channels])}


def mode_allowed_for_channel(
    mode: str,
    channel: int,
    channels: int,
    policy: str,
) -> bool:
    if policy == "oracle":
        return True
    if "channel" not in mode:
        return True
    if policy == "spatial":
        return False
    if policy != "green_order":
        raise ValueError(f"unknown mode policy: {policy}")

    ranks = channel_decode_rank(channels)
    if "channel_green" in mode:
        return channels >= 3 and channel in (0, 2) and ranks[1] < ranks[channel]
    if "channel_previous" in mode:
        if channel == 0:
            return True
        return ranks[channel - 1] < ranks[channel]
    return False


def score_channel_payload(
    payload_channel: np.ndarray,
    family_id_bits: int,
    mode_score: str,
) -> float:
    payload = payload_channel[:, :, None] & BODY_MASK
    total = 0.0
    for bit in range(30, -1, -1):
        if mode_score == "base":
            shared_cost = context_family_cost(payload, bit, "base")
        elif mode_score == "family":
            shared_name, shared_cost, _split_names, _split_cost = (
                best_shared_and_channel_families(payload, bit)
            )
            del shared_name
        else:
            raise ValueError(f"unknown mode score: {mode_score}")
        total += shared_cost + family_id_bits
    return total


def score_group_payload(
    payload: np.ndarray,
    bits: tuple[int, ...],
    allow_channel_split: bool,
    family_id_bits: int,
    split_flag_bits: int,
    min_split_gain_bits: float,
) -> tuple[float, dict[str, int]]:
    total = 0.0
    picks = Counter()
    masked = payload.copy()
    for bit in bits:
        shared_name, shared_cost, split_names, split_cost = (
            best_shared_and_channel_families(masked, bit)
        )
        del split_names
        shared_total = shared_cost + family_id_bits
        split_total = (
            split_cost
            + masked.shape[2] * family_id_bits
            + split_flag_bits
        )
        if allow_channel_split and split_total + min_split_gain_bits < shared_total:
            total += split_total
            picks["channel_split"] += 1
        else:
            total += shared_total
            picks[f"shared:{shared_name}"] += 1
    return total, dict(picks)


def summarize_image(
    path: Path,
    crop_size: int,
    tile_size: int,
    previous_bits: int,
    family_id_bits: int,
    split_flag_bits: int,
    min_split_gain_bits: float,
    extra_mode_bits_per_channel: int,
    mode_policy: str,
    mode_score: str,
) -> dict:
    pixels = read_exr(path)
    if crop_size:
        pixels = pixels[:crop_size, :crop_size]
    pixels = np.ascontiguousarray(pixels, dtype=np.float32)
    bits = float_to_bits(pixels).copy()
    ordered = bits_to_ordered(bits)

    modes, payloads, _carries = choose_group_payloads(
        bits,
        pixels,
        tile_size,
        previous_bits,
        "body",
    )
    xor_predictors = ordered_xor_predictors(bits, pixels, causal=True)
    delta_preds = delta_predictors(bits, pixels, causal=True)

    height, width, channels = bits.shape
    baseline_bits = 0.0
    channel_mode_bits = 0.0
    hybrid_bits = 0.0
    mode_histogram = Counter()
    shared_mode_histogram = Counter()
    pick_histogram = Counter()
    route_histogram = Counter()

    body_bits = group_bit_indices(GROUP_PRESETS["body"][0][1])
    sign_bits = group_bit_indices(GROUP_PRESETS["body"][1][1])

    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            sl = np.s_[y:y + tile_size, x:x + tile_size]

            baseline_body = payloads["body"][sl] & BODY_MASK
            baseline_sign = payloads["sign"][sl] & SIGN_MASK
            body_cost, body_picks = score_group_payload(
                baseline_body,
                body_bits,
                True,
                family_id_bits,
                split_flag_bits,
                min_split_gain_bits,
            )
            sign_cost, sign_picks = score_group_payload(
                baseline_sign,
                sign_bits,
                False,
                family_id_bits,
                split_flag_bits,
                min_split_gain_bits,
            )
            baseline_bits += body_cost + sign_cost
            baseline_tile_bits = body_cost + sign_cost
            pick_histogram.update(body_picks)
            pick_histogram.update(sign_picks)
            shared_mode_histogram[modes[(y, x, "body")]] += 1

            ordered_tile = ordered[sl]
            candidates = body_candidates(
                ordered_tile,
                {name: value[sl] for name, value in xor_predictors.items()},
                {name: value[sl] for name, value in delta_preds.items()},
            )
            selected_body = np.zeros_like(baseline_body)
            for c in range(channels):
                best_name = ""
                best_cost = float("inf")
                for name, candidate_payload in candidates.items():
                    if not mode_allowed_for_channel(name, c, channels, mode_policy):
                        continue
                    cost = score_channel_payload(
                        candidate_payload[:, :, c],
                        family_id_bits,
                        mode_score,
                    )
                    if cost < best_cost:
                        best_cost = cost
                        best_name = name
                selected_body[:, :, c] = candidates[best_name][:, :, c]
                mode_histogram[f"c{c}:{best_name}"] += 1

            selected_body_cost, selected_picks = score_group_payload(
                selected_body,
                body_bits,
                True,
                family_id_bits,
                split_flag_bits,
                min_split_gain_bits,
            )
            extra_mode_bits = max(0, channels - 1) * extra_mode_bits_per_channel
            channel_tile_bits = selected_body_cost + sign_cost + extra_mode_bits
            channel_mode_bits += channel_tile_bits
            if channel_tile_bits < baseline_tile_bits:
                hybrid_bits += channel_tile_bits
                route_histogram["channel_mode"] += 1
            else:
                hybrid_bits += baseline_tile_bits
                route_histogram["shared_mode"] += 1
            pick_histogram.update(selected_picks)

    raw_bits = int(bits.size * 32)
    return {
        "image": path.name,
        "shape": [height, width, channels],
        "crop_size": crop_size,
        "tile_size": tile_size,
        "previous_bits": previous_bits,
        "family_id_bits": family_id_bits,
        "split_flag_bits": split_flag_bits,
        "min_split_gain_bits": min_split_gain_bits,
        "extra_mode_bits_per_channel": extra_mode_bits_per_channel,
        "mode_policy": mode_policy,
        "mode_score": mode_score,
        "raw_bits": raw_bits,
        "baseline_bits": baseline_bits,
        "channel_mode_bits": channel_mode_bits,
        "hybrid_bits": hybrid_bits,
        "baseline_ratio": raw_bits / baseline_bits,
        "channel_mode_ratio": raw_bits / channel_mode_bits,
        "hybrid_ratio": raw_bits / hybrid_bits,
        "gain": baseline_bits / channel_mode_bits,
        "hybrid_gain": baseline_bits / hybrid_bits,
        "mode_histogram": dict(mode_histogram),
        "shared_mode_histogram": dict(shared_mode_histogram),
        "route_histogram": dict(route_histogram),
        "pick_histogram": dict(pick_histogram),
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
    parser.add_argument("--family-id-bits", type=int, default=4)
    parser.add_argument("--split-flag-bits", type=int, default=1)
    parser.add_argument("--min-split-gain-bits", type=float, default=4.0)
    parser.add_argument("--extra-mode-bits-per-channel", type=int, default=8)
    parser.add_argument(
        "--mode-policy",
        choices=("spatial", "green_order", "oracle"),
        default="spatial",
        help="Per-channel mode safety policy. 'oracle' can include cycles and "
        "is not directly decoder-safe.",
    )
    parser.add_argument("--mode-score", choices=("base", "family"), default="base")
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
            args.family_id_bits,
            args.split_flag_bits,
            args.min_split_gain_bits,
            args.extra_mode_bits_per_channel,
            args.mode_policy,
            args.mode_score,
        )
        rows.append(row)
        print(
            f"{path.stem:43s} "
            f"shared_mode={row['baseline_ratio']:.3f}x "
            f"channel_mode={row['channel_mode_ratio']:.3f}x "
            f"hybrid={row['hybrid_ratio']:.3f}x "
            f"gain={row['hybrid_gain']:.4f}"
        )
        if args.limit and len(rows) >= args.limit:
            break
    if rows:
        print(
            f"geomean shared_mode={geomean([r['baseline_ratio'] for r in rows]):.3f}x "
            f"channel_mode={geomean([r['channel_mode_ratio'] for r in rows]):.3f}x "
            f"hybrid={geomean([r['hybrid_ratio'] for r in rows]):.3f}x"
        )
    safe_glob = args.glob.replace("*", "star").replace(".", "_")
    output = RESULTS_DIR / (
        f"channel_mode_grouped_delta_{safe_glob}"
        f"_tile{args.tile_size}_prev{args.previous_bits}_{args.mode_policy}_{args.mode_score}"
        f"_crop{args.crop_size}.json"
    )
    output.write_text(json.dumps(rows, indent=2))
    print(f"\nSaved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
