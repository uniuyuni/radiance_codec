"""Probe a per-128-tile choice between one 128 tile and four 64 subtiles.

This is an estimator, not the C++ bitstream.  It reuses the existing grouped
delta/channel-mode scoring probes to answer a narrower design question: is a
signaled variable tile split likely to buy enough ratio to justify a GDX format
change?
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
from evaluate_structural_delta_grouped import GROUP_PRESETS, group_bit_indices  # noqa: E402
from probe_channel_mode_grouped_delta import (  # noqa: E402
    BODY_MASK,
    SIGN_MASK,
    body_candidates,
    mode_allowed_for_channel,
    score_channel_payload,
    score_group_payload,
)
from structural_delta_grouped_roundtrip import choose_group_payloads  # noqa: E402

RESULTS_DIR = ROOT / "results"


def geomean(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def tile_hybrid_cost(
    ordered: np.ndarray,
    payloads: dict[str, np.ndarray],
    modes: dict[tuple[int, int, str], str],
    xor_predictors: dict[str, np.ndarray],
    delta_preds: dict[str, np.ndarray],
    y: int,
    x: int,
    tile_size: int,
    family_id_bits: int,
    split_flag_bits: int,
    min_split_gain_bits: float,
    extra_mode_bits_per_channel: int,
    mode_policy: str,
    mode_score: str,
) -> tuple[float, str]:
    height, width, channels = ordered.shape
    sl = np.s_[y:min(y + tile_size, height), x:min(x + tile_size, width)]
    body_bits = group_bit_indices(GROUP_PRESETS["body"][0][1])
    sign_bits = group_bit_indices(GROUP_PRESETS["body"][1][1])

    baseline_body = payloads["body"][sl] & BODY_MASK
    baseline_sign = payloads["sign"][sl] & SIGN_MASK
    body_cost, _body_picks = score_group_payload(
        baseline_body,
        body_bits,
        True,
        family_id_bits,
        split_flag_bits,
        min_split_gain_bits,
    )
    sign_cost, _sign_picks = score_group_payload(
        baseline_sign,
        sign_bits,
        False,
        family_id_bits,
        split_flag_bits,
        min_split_gain_bits,
    )
    baseline_tile_bits = body_cost + sign_cost

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

    selected_body_cost, _selected_picks = score_group_payload(
        selected_body,
        body_bits,
        True,
        family_id_bits,
        split_flag_bits,
        min_split_gain_bits,
    )
    extra_mode_bits = max(0, channels - 1) * extra_mode_bits_per_channel
    channel_tile_bits = selected_body_cost + sign_cost + extra_mode_bits
    if channel_tile_bits < baseline_tile_bits:
        return channel_tile_bits, "channel_mode"
    return baseline_tile_bits, "shared_mode"


def prepare_cost_grid(
    bits: np.ndarray,
    pixels: np.ndarray,
    tile_size: int,
    previous_bits: int,
    family_id_bits: int,
    split_flag_bits: int,
    min_split_gain_bits: float,
    extra_mode_bits_per_channel: int,
    mode_policy: str,
    mode_score: str,
) -> tuple[dict[tuple[int, int], float], Counter[str]]:
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

    height, width, _channels = bits.shape
    costs: dict[tuple[int, int], float] = {}
    routes: Counter[str] = Counter()
    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            cost, route = tile_hybrid_cost(
                ordered,
                payloads,
                modes,
                xor_predictors,
                delta_preds,
                y,
                x,
                tile_size,
                family_id_bits,
                split_flag_bits,
                min_split_gain_bits,
                extra_mode_bits_per_channel,
                mode_policy,
                mode_score,
            )
            costs[(y, x)] = cost
            routes[route] += 1
    return costs, routes


def summarize_image(
    path: Path,
    crop_size: int,
    previous_bits: int,
    family_id_bits: int,
    split_flag_bits: int,
    min_split_gain_bits: float,
    extra_mode_bits_per_channel: int,
    mode_policy: str,
    mode_score: str,
    variable_split_bits: float,
) -> dict:
    pixels = read_exr(path)
    if crop_size:
        pixels = pixels[:crop_size, :crop_size]
    pixels = np.ascontiguousarray(pixels, dtype=np.float32)
    bits = float_to_bits(pixels).copy()
    height, width, channels = bits.shape

    costs128, routes128 = prepare_cost_grid(
        bits,
        pixels,
        128,
        previous_bits,
        family_id_bits,
        split_flag_bits,
        min_split_gain_bits,
        extra_mode_bits_per_channel,
        mode_policy,
        mode_score,
    )
    costs64, routes64 = prepare_cost_grid(
        bits,
        pixels,
        64,
        previous_bits,
        family_id_bits,
        split_flag_bits,
        min_split_gain_bits,
        extra_mode_bits_per_channel,
        mode_policy,
        mode_score,
    )

    bits128 = sum(costs128.values())
    bits64 = sum(costs64.values())
    dynamic_bits = 0.0
    split_histogram: Counter[str] = Counter()
    for y in range(0, height, 128):
        for x in range(0, width, 128):
            cost128 = costs128[(y, x)]
            cost64 = 0.0
            for sy in (y, y + 64):
                if sy >= height:
                    continue
                for sx in (x, x + 64):
                    if sx >= width:
                        continue
                    cost64 += costs64[(sy, sx)]
            cost64 += variable_split_bits
            if cost64 < cost128:
                dynamic_bits += cost64
                split_histogram["split64"] += 1
            else:
                dynamic_bits += cost128
                split_histogram["tile128"] += 1

    raw_bits = int(bits.size * 32)
    return {
        "image": path.name,
        "shape": [height, width, channels],
        "crop_size": crop_size,
        "raw_bits": raw_bits,
        "bits128": bits128,
        "bits64": bits64,
        "dynamic_bits": dynamic_bits,
        "ratio128": raw_bits / bits128,
        "ratio64": raw_bits / bits64,
        "dynamic_ratio": raw_bits / dynamic_bits,
        "dynamic_gain": bits128 / dynamic_bits,
        "routes128": dict(routes128),
        "routes64": dict(routes64),
        "split_histogram": dict(split_histogram),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", default="*.exr")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--previous-bits", type=int, default=4)
    parser.add_argument("--family-id-bits", type=int, default=4)
    parser.add_argument("--split-flag-bits", type=int, default=1)
    parser.add_argument("--min-split-gain-bits", type=float, default=4.0)
    parser.add_argument("--extra-mode-bits-per-channel", type=int, default=8)
    parser.add_argument("--mode-policy", choices=("spatial", "green_order", "oracle"), default="green_order")
    parser.add_argument("--mode-score", choices=("base", "family"), default="family")
    parser.add_argument("--variable-split-bits", type=float, default=1.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = []
    for path in sorted(DATA_DIR.glob(args.glob)):
        row = summarize_image(
            path,
            args.crop_size,
            args.previous_bits,
            args.family_id_bits,
            args.split_flag_bits,
            args.min_split_gain_bits,
            args.extra_mode_bits_per_channel,
            args.mode_policy,
            args.mode_score,
            args.variable_split_bits,
        )
        rows.append(row)
        print(
            f"{path.stem:43s} "
            f"tile128={row['ratio128']:.3f}x "
            f"tile64={row['ratio64']:.3f}x "
            f"dynamic={row['dynamic_ratio']:.3f}x "
            f"gain={row['dynamic_gain']:.4f} "
            f"splits={row['split_histogram']}"
        )
        if args.limit and len(rows) >= args.limit:
            break
    if rows:
        print(
            f"geomean tile128={geomean([r['ratio128'] for r in rows]):.3f}x "
            f"tile64={geomean([r['ratio64'] for r in rows]):.3f}x "
            f"dynamic={geomean([r['dynamic_ratio'] for r in rows]):.3f}x"
        )
    safe_glob = args.glob.replace("*", "star").replace(".", "_")
    output = RESULTS_DIR / (
        f"variable_tile_split_{safe_glob}_prev{args.previous_bits}"
        f"_crop{args.crop_size}.json"
    )
    output.write_text(json.dumps(rows, indent=2))
    print(f"\nSaved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
