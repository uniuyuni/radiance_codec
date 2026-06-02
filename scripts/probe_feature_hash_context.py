"""Probe bounded feature-hash contexts for GDX3 payload bitplanes.

This is the next step after `PrevXYChannel`: instead of signaling a full
MANIAC-style tree, use one fixed hash of many decoder-visible features into a
bounded table.  If it wins, it can become another cheap C++ context family.
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
        return np.zeros(payload.shape, dtype=np.uint32)
    return ((payload >> np.uint32(previous_bit)) & np.uint32(1)).astype(np.uint32)


def cost_from_context(context: np.ndarray, symbols: np.ndarray, count: int) -> float:
    totals = np.bincount(context.reshape(-1), minlength=count)
    ones = np.bincount(context.reshape(-1), weights=symbols.reshape(-1), minlength=count)
    return sum(
        kt_cost(int(ones[index]), int(totals[index]))
        for index in range(count)
        if totals[index]
    )


def fold_context(code: np.ndarray, context_bits: int) -> np.ndarray:
    mask = np.uint32((1 << context_bits) - 1)
    mixed = code ^ (code >> np.uint32(context_bits))
    mixed ^= code >> np.uint32(context_bits * 2)
    mixed *= np.uint32(0x9E3779B1)
    mixed ^= mixed >> np.uint32(16)
    return (mixed & mask).astype(np.int64)


def add_feature(code: np.ndarray, feature: np.ndarray, shift: int) -> np.ndarray:
    return code | (feature.astype(np.uint32) << np.uint32(shift))


def hash_context_cost(
    payload: np.ndarray,
    ordered: np.ndarray,
    bit: int,
    variant: str,
    context_bits: int,
) -> float:
    plane = ((payload >> np.uint32(bit)) & np.uint32(1)).astype(np.uint32)
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
    x = np.arange(width, dtype=np.uint32)[None, :, None]
    y = np.arange(height, dtype=np.uint32)[:, None, None]
    c = np.arange(channels, dtype=np.uint32)[None, None, :]
    x_lsb = np.broadcast_to(x & 1, payload.shape)
    y_lsb = np.broadcast_to(y & 1, payload.shape)
    x2 = np.broadcast_to((x >> 1) & 1, payload.shape)
    y2 = np.broadcast_to((y >> 1) & 1, payload.shape)
    c0 = np.broadcast_to((c == 0).astype(np.uint32), payload.shape)
    c1 = np.broadcast_to((c == 1).astype(np.uint32), payload.shape)
    c2 = np.broadcast_to((c == 2).astype(np.uint32), payload.shape)
    high_value = (ordered >> np.uint32(15)).astype(np.uint32)

    code = np.zeros(payload.shape, dtype=np.uint32)
    if variant == "hash_prev_xy_channel":
        features = (p1, p2, p3, p4, x_lsb, y_lsb, x2, y2, c0, c1, c2)
    elif variant == "hash_spatial_prev_xy":
        features = (w, n, nw, ne, p1, p2, p3, p4, x_lsb, y_lsb, c0, c1, c2)
    elif variant == "hash_hi_neighbors_xy":
        features = (w, n, nw, ne, p1, wp1, np1, p2, wp2, np2, x_lsb, y_lsb, c0, c1, c2)
    elif variant == "hash_all_xy":
        features = (
            w,
            n,
            nw,
            ne,
            pc,
            p1,
            p2,
            p3,
            p4,
            wp1,
            np1,
            wp2,
            np2,
            x_lsb,
            y_lsb,
            c0,
            c1,
            c2,
        )
    elif variant == "hash_oracle_high_tail":
        features = (w, n, p1, p2, p3, p4, x_lsb, y_lsb, c0, c1, c2)
    else:
        raise ValueError(f"unknown hash context variant: {variant}")
    for shift, feature in enumerate(features):
        code = add_feature(code, feature, shift)
    if variant == "hash_oracle_high_tail":
        code ^= high_value * np.uint32(0x45d9f3b)
        code ^= (high_value >> np.uint32(7)) * np.uint32(0x119de1f3)
    context = fold_context(code, context_bits)
    return cost_from_context(context, plane, 1 << context_bits)


VARIANTS = (
    "hash_prev_xy_channel",
    "hash_spatial_prev_xy",
    "hash_hi_neighbors_xy",
    "hash_all_xy",
    "hash_oracle_high_tail",
)


def best_existing_cost(payload: np.ndarray, bit: int, effort: int, group_name: str) -> tuple[str, float]:
    families = CONTEXT_FAMILIES if (effort >= 10 or group_name != "sign") else ("base",)
    pairs = [(family, context_family_cost(payload, bit, family)) for family in families]
    return min(pairs, key=lambda item: item[1])


def summarize_image(
    path: Path,
    crop_size: int,
    tile_size: int,
    previous_bits: int,
    effort: int,
    context_bits: int,
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
    hash_bits = 0.0
    pick_histogram = Counter()
    bit_pick_histogram = Counter()
    variant_cost_totals = Counter()
    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            sl = np.s_[y:y + tile_size, x:x + tile_size]
            for group_name, fields in GROUP_PRESETS["body"]:
                payload = payloads[group_name][sl]
                for bit in group_bit_indices(fields):
                    base_name, base_cost = best_existing_cost(
                        payload,
                        bit,
                        effort,
                        group_name,
                    )
                    best_name = f"existing:{base_name}"
                    best_cost = base_cost
                    for variant in VARIANTS:
                        if group_name == "sign" and variant != "hash_spatial_prev_xy":
                            continue
                        if variant == "hash_oracle_high_tail" and bit > 14:
                            continue
                        cost = hash_context_cost(
                            payload,
                            ordered[sl],
                            bit,
                            variant,
                            context_bits,
                        )
                        variant_cost_totals[variant] += cost
                        if cost < best_cost:
                            best_name = f"hash:{variant}"
                            best_cost = cost
                    baseline_bits += base_cost
                    hash_bits += best_cost
                    pick_histogram[best_name] += 1
                    bit_pick_histogram[f"{bit}:{best_name}"] += 1
    raw_bits = int(bits.size * 32)
    return {
        "image": path.name,
        "shape": list(bits.shape),
        "crop_size": crop_size,
        "tile_size": tile_size,
        "previous_bits": previous_bits,
        "effort": effort,
        "context_bits": context_bits,
        "raw_bits": raw_bits,
        "baseline_bits": baseline_bits,
        "hash_bits": hash_bits,
        "baseline_ratio": raw_bits / baseline_bits if baseline_bits else 1.0,
        "hash_ratio": raw_bits / hash_bits if hash_bits else 1.0,
        "gain": baseline_bits / hash_bits if hash_bits else 1.0,
        "pick_histogram": dict(pick_histogram),
        "bit_pick_histogram": dict(bit_pick_histogram),
        "variant_cost_totals": dict(variant_cost_totals),
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
    parser.add_argument("--effort", type=int, default=9)
    parser.add_argument("--context-bits", type=int, default=11)
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
            args.effort,
            args.context_bits,
        )
        rows.append(row)
        print(
            f"{path.stem:43s} "
            f"base={row['baseline_ratio']:.3f}x "
            f"hash={row['hash_ratio']:.3f}x "
            f"gain={row['gain']:.4f} picks={row['pick_histogram']}"
        )
        if args.limit and len(rows) >= args.limit:
            break
    if rows:
        print(
            f"geomean base={geomean([r['baseline_ratio'] for r in rows]):.3f}x "
            f"hash={geomean([r['hash_ratio'] for r in rows]):.3f}x"
        )
    safe_glob = args.glob.replace("*", "star").replace(".", "_")
    output = RESULTS_DIR / (
        f"feature_hash_context_{safe_glob}_ctx{args.context_bits}_tile{args.tile_size}"
        f"_prev{args.previous_bits}_effort{args.effort}_crop{args.crop_size}.json"
    )
    output.write_text(json.dumps(rows, indent=2))
    print(f"\nSaved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
