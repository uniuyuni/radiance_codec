"""Probe cheap fixed context-family variants inspired by MANIAC splits.

The greedy tree probe showed that decoder-visible feature splits help hard
puresky images, but the tree itself is too slow and too much side information
for the current C++ stage. This script tests whether a small set of fixed
families can recover part of that gain.
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
from probe_grouped_tail_mlp import CONTEXT_FAMILIES, context_family_cost, kt_cost  # noqa: E402
from structural_delta_grouped_roundtrip import choose_group_payloads  # noqa: E402

RESULTS_DIR = ROOT / "results"


def shift_plane(plane: np.ndarray, dy: int, dx: int) -> np.ndarray:
    out = np.zeros_like(plane)
    height, width, _channels = plane.shape
    dst_y0 = max(0, -dy)
    dst_y1 = min(height, height - dy)
    dst_x0 = max(0, -dx)
    dst_x1 = min(width, width - dx)
    src_y0 = dst_y0 + dy
    src_y1 = dst_y1 + dy
    src_x0 = dst_x0 + dx
    src_x1 = dst_x1 + dx
    out[dst_y0:dst_y1, dst_x0:dst_x1] = plane[src_y0:src_y1, src_x0:src_x1]
    return out


def previous_plane(payload: np.ndarray, bit: int, offset: int) -> np.ndarray:
    previous_bit = bit + offset
    if previous_bit > 31:
        return np.zeros(payload.shape, dtype=np.uint16)
    return ((payload >> np.uint32(previous_bit)) & np.uint32(1)).astype(np.uint16)


def cost_from_context(context: np.ndarray, symbols: np.ndarray, count: int) -> float:
    context_flat = context.reshape(-1)
    symbols_flat = symbols.reshape(-1)
    totals = np.bincount(context_flat, minlength=count)
    ones = np.bincount(context_flat, weights=symbols_flat, minlength=count)
    return sum(
        kt_cost(int(ones[index]), int(totals[index]))
        for index in range(count)
        if totals[index]
    )


def variant_context_cost(payload: np.ndarray, bit: int, variant: str) -> float:
    plane = ((payload >> np.uint32(bit)) & np.uint32(1)).astype(np.uint16)
    w = shift_plane(plane, 0, -1)
    n = shift_plane(plane, -1, 0)
    nw = shift_plane(plane, -1, -1)
    ne = shift_plane(plane, -1, 1)
    pc = np.zeros_like(plane)
    pc[..., 1:] = plane[..., :-1]
    p1 = previous_plane(payload, bit, 1)
    p2 = previous_plane(payload, bit, 2)
    p3 = previous_plane(payload, bit, 3)
    p4 = previous_plane(payload, bit, 4)
    wp1 = shift_plane(p1, 0, -1)
    np1 = shift_plane(p1, -1, 0)
    wp2 = shift_plane(p2, 0, -1)
    np2 = shift_plane(p2, -1, 0)

    height, width, channels = payload.shape
    x_lsb = np.broadcast_to(
        (np.arange(width, dtype=np.uint16)[None, :, None] & 1),
        payload.shape,
    )
    y_lsb = np.broadcast_to(
        (np.arange(height, dtype=np.uint16)[:, None, None] & 1),
        payload.shape,
    )
    c = np.arange(channels, dtype=np.uint16)[None, None, :]
    c0 = np.broadcast_to((c == 0).astype(np.uint16), payload.shape)
    c1 = np.broadcast_to((c == 1).astype(np.uint16), payload.shape)
    c2 = np.broadcast_to((c == 2).astype(np.uint16), payload.shape)

    context = np.zeros(payload.shape, dtype=np.uint16)
    if variant == "hi_pc_xy":
        # Existing SpatialHiPc plus x/y parity. 13 feature bits.
        context |= w | (n << 1) | (nw << 2) | (ne << 3) | (pc << 4)
        context |= p1 << 5
        context |= wp1 << 6
        context |= np1 << 7
        context |= p2 << 8
        context |= wp2 << 9
        context |= np2 << 10
        context |= x_lsb << 11
        context |= y_lsb << 12
        return cost_from_context(context, plane, 1 << 13)
    if variant == "hi_channel_xy":
        # Existing SpatialHiChannel plus x/y parity. 13 feature bits.
        context |= w | (n << 1)
        context |= c0 << 2
        context |= c1 << 3
        context |= c2 << 4
        context |= p1 << 5
        context |= wp1 << 6
        context |= np1 << 7
        context |= p2 << 8
        context |= wp2 << 9
        context |= np2 << 10
        context |= x_lsb << 11
        context |= y_lsb << 12
        return cost_from_context(context, plane, 1 << 13)
    if variant == "prev_xy_channel":
        # Cheaper parity-aware fallback for low/random-looking planes.
        context |= p1
        context |= p2 << 1
        context |= x_lsb << 2
        context |= y_lsb << 3
        context |= c0 << 4
        context |= c1 << 5
        context |= c2 << 6
        return cost_from_context(context, plane, 1 << 7)
    if variant == "spatial_xy":
        context |= w | (n << 1) | (nw << 2) | (ne << 3)
        context |= x_lsb << 4
        context |= y_lsb << 5
        context |= p1 << 6
        context |= p2 << 7
        return cost_from_context(context, plane, 1 << 8)
    if variant == "prev4_channel":
        context |= p1
        context |= p2 << 1
        context |= p3 << 2
        context |= p4 << 3
        context |= c0 << 4
        context |= c1 << 5
        context |= c2 << 6
        return cost_from_context(context, plane, 1 << 7)
    if variant == "prev4_channel_xy":
        context |= p1
        context |= p2 << 1
        context |= p3 << 2
        context |= p4 << 3
        context |= x_lsb << 4
        context |= y_lsb << 5
        context |= c0 << 6
        context |= c1 << 7
        context |= c2 << 8
        return cost_from_context(context, plane, 1 << 9)
    if variant == "wn_prev4_channel":
        context |= w
        context |= n << 1
        context |= p1 << 2
        context |= p2 << 3
        context |= p3 << 4
        context |= p4 << 5
        context |= c0 << 6
        context |= c1 << 7
        context |= c2 << 8
        return cost_from_context(context, plane, 1 << 9)
    if variant == "wn_prev4_pc":
        context |= w
        context |= n << 1
        context |= pc << 2
        context |= p1 << 3
        context |= p2 << 4
        context |= p3 << 5
        context |= p4 << 6
        return cost_from_context(context, plane, 1 << 7)
    if variant == "spatial_prev4_lite":
        context |= w
        context |= n << 1
        context |= nw << 2
        context |= ne << 3
        context |= p1 << 4
        context |= p2 << 5
        context |= p3 << 6
        context |= p4 << 7
        return cost_from_context(context, plane, 1 << 8)
    raise ValueError(f"unknown variant: {variant}")


VARIANTS = (
    "hi_pc_xy",
    "hi_channel_xy",
    "spatial_xy",
    "prev4_channel",
    "prev4_channel_xy",
    "wn_prev4_channel",
    "wn_prev4_pc",
    "spatial_prev4_lite",
)


def best_existing_cost(payload: np.ndarray, bit: int) -> tuple[str, float]:
    pairs = [(family, context_family_cost(payload, bit, family)) for family in CONTEXT_FAMILIES]
    return min(pairs, key=lambda item: item[1])


def summarize_image(path: Path, crop_size: int, tile_size: int, previous_bits: int) -> dict:
    pixels = read_exr(path)
    if crop_size:
        pixels = pixels[:crop_size, :crop_size]
    pixels = np.ascontiguousarray(pixels, dtype=np.float32)
    bits = float_to_bits(pixels).copy()
    _modes, payloads, _carries = choose_group_payloads(
        bits,
        pixels,
        tile_size,
        previous_bits,
        "body",
    )
    height, width, _channels = bits.shape
    baseline_bits = 0.0
    variant_bits = 0.0
    pick_histogram = Counter()
    bit_pick_histogram = Counter()
    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            sl = np.s_[y:y + tile_size, x:x + tile_size]
            for group_name, fields in GROUP_PRESETS["body"]:
                payload = payloads[group_name][sl]
                for bit in group_bit_indices(fields):
                    base_name, base_cost = best_existing_cost(payload, bit)
                    best_name = f"existing:{base_name}"
                    best_cost = base_cost
                    for variant in VARIANTS:
                        # Sign group is only one plane and usually trivial;
                        # keep variants focused on body-like residuals.
                        if group_name == "sign" and variant != "spatial_xy":
                            continue
                        cost = variant_context_cost(payload, bit, variant)
                        if cost < best_cost:
                            best_name = f"variant:{variant}"
                            best_cost = cost
                    baseline_bits += base_cost
                    variant_bits += best_cost
                    pick_histogram[best_name] += 1
                    bit_pick_histogram[f"{bit}:{best_name}"] += 1
    raw_bits = int(bits.size * 32)
    return {
        "image": path.name,
        "shape": list(bits.shape),
        "crop_size": crop_size,
        "tile_size": tile_size,
        "previous_bits": previous_bits,
        "raw_bits": raw_bits,
        "baseline_bits": baseline_bits,
        "variant_bits": variant_bits,
        "baseline_ratio": raw_bits / baseline_bits if baseline_bits else 1.0,
        "variant_ratio": raw_bits / variant_bits if variant_bits else 1.0,
        "gain": baseline_bits / variant_bits if variant_bits else 1.0,
        "pick_histogram": dict(pick_histogram),
        "bit_pick_histogram": dict(bit_pick_histogram),
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = []
    for path in sorted(DATA_DIR.glob(args.glob)):
        row = summarize_image(path, args.crop_size, args.tile_size, args.previous_bits)
        rows.append(row)
        print(
            f"{path.stem:43s} "
            f"base={row['baseline_ratio']:.3f}x "
            f"variant={row['variant_ratio']:.3f}x "
            f"gain={row['gain']:.4f} picks={row['pick_histogram']}"
        )
        if args.limit and len(rows) >= args.limit:
            break
    if rows:
        print(
            f"geomean base={geomean([r['baseline_ratio'] for r in rows]):.3f}x "
            f"variant={geomean([r['variant_ratio'] for r in rows]):.3f}x"
        )
    safe_glob = args.glob.replace("*", "star").replace(".", "_")
    output = RESULTS_DIR / (
        f"context_family_variants_{safe_glob}_tile{args.tile_size}"
        f"_prev{args.previous_bits}_crop{args.crop_size}.json"
    )
    output.write_text(json.dumps(rows, indent=2))
    print(f"\nSaved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
