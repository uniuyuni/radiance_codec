"""Probe an exact half-float route for float32 tiles.

Some EXR files are stored as float32 but every value is exactly representable as
IEEE binary16.  Current GDX coding can see the zero low mantissa bits, but it
still predicts and models the 32-bit ordered-float body/sign split.  This probe
asks whether a tile-local route that stores exact half bits, then applies the
same kind of reversible ordered-delta bitplane coding, is worth implementing.
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
from evaluate_structural_delta_grouped import GROUP_PRESETS, group_bit_indices  # noqa: E402
from probe_grouped_tail_mlp import CONTEXT_FAMILIES, context_family_cost  # noqa: E402
from structural_delta_grouped_roundtrip import choose_group_payloads  # noqa: E402

RESULTS_DIR = ROOT / "results"
U16_MASK = np.uint32(0xFFFF)


def half_bits(pixels: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(pixels.astype(np.float16)).view(np.uint16).astype(np.uint32)


def half_exact(pixels: np.ndarray) -> bool:
    with np.errstate(over="ignore", invalid="ignore"):
        reconstructed = pixels.astype(np.float16).astype(np.float32)
    return bool(np.array_equal(reconstructed, pixels, equal_nan=True))


def half_to_ordered(bits: np.ndarray) -> np.ndarray:
    sign = (bits >> np.uint32(15)) != 0
    return np.where(sign, (~bits) & U16_MASK, bits ^ np.uint32(0x8000)).astype(np.uint32)


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


def planar(ordered: np.ndarray) -> np.ndarray:
    west = shift(ordered, 0, -1).astype(np.int64)
    north = shift(ordered, -1, 0).astype(np.int64)
    northwest = shift(ordered, -1, -1).astype(np.int64)
    return np.clip(west + north - northwest, 0, 0xFFFF).astype(np.uint32)


def med(ordered: np.ndarray) -> np.ndarray:
    west = shift(ordered, 0, -1)
    north = shift(ordered, -1, 0)
    northwest = shift(ordered, -1, -1)
    mx = np.maximum(west, north)
    mn = np.minimum(west, north)
    return np.where(northwest >= mx, mn, np.where(northwest <= mn, mx, planar(ordered))).astype(np.uint32)


def select(ordered: np.ndarray) -> np.ndarray:
    west = shift(ordered, 0, -1).astype(np.int64)
    north = shift(ordered, -1, 0).astype(np.int64)
    northwest = shift(ordered, -1, -1).astype(np.int64)
    return np.where(np.abs(north - northwest) < np.abs(west - northwest), west, north).astype(np.uint32)


def paeth(ordered: np.ndarray) -> np.ndarray:
    west = shift(ordered, 0, -1).astype(np.int64)
    north = shift(ordered, -1, 0).astype(np.int64)
    northwest = shift(ordered, -1, -1).astype(np.int64)
    pred = west + north - northwest
    dist_w = np.abs(pred - west)
    dist_n = np.abs(pred - north)
    dist_nw = np.abs(pred - northwest)
    return np.where(
        (dist_w <= dist_n) & (dist_w <= dist_nw),
        west,
        np.where(dist_n <= dist_nw, north, northwest),
    ).astype(np.uint32)


def predictors(ordered: np.ndarray) -> dict[str, np.ndarray]:
    channel_previous = np.zeros_like(ordered)
    if ordered.shape[2] >= 2:
        channel_previous[..., 1:] = ordered[..., :-1]
    channel_green = np.zeros_like(ordered)
    if ordered.shape[2] >= 3:
        channel_green[..., 0] = ordered[..., 1]
        channel_green[..., 2] = ordered[..., 1]
    west = shift(ordered, 0, -1)
    north = shift(ordered, -1, 0)
    northwest = shift(ordered, -1, -1)
    northeast = shift(ordered, -1, 1)
    return {
        "raw": ordered,
        "xor_zero": ordered,
        "xor_west": ordered ^ west,
        "xor_north": ordered ^ north,
        "xor_channel_green": ordered ^ channel_green,
        "delta_west": (ordered - west) & U16_MASK,
        "delta_north": (ordered - north) & U16_MASK,
        "delta_northwest": (ordered - northwest) & U16_MASK,
        "delta_northeast": (ordered - northeast) & U16_MASK,
        "delta_channel_previous": (ordered - channel_previous) & U16_MASK,
        "delta_channel_green": (ordered - channel_green) & U16_MASK,
        "delta_med": (ordered - med(ordered)) & U16_MASK,
        "delta_planar": (ordered - planar(ordered)) & U16_MASK,
        "delta_select": (ordered - select(ordered)) & U16_MASK,
        "delta_paeth": (ordered - paeth(ordered)) & U16_MASK,
    }


def best_family_cost(payload: np.ndarray, bits: range) -> float:
    total = 0.0
    for bit in bits:
        total += min(context_family_cost(payload, bit, family) for family in CONTEXT_FAMILIES)
        total += 4.0
    return total


def current_tile_cost(
    body_payload: np.ndarray,
    sign_payload: np.ndarray,
) -> float:
    body = sum(
        best_family_cost(body_payload, range(bit, bit + 1))
        for bit in range(30, -1, -1)
    )
    sign = best_family_cost(sign_payload, range(31, 32))
    return body + sign


def half_route_tile_cost(pixels: np.ndarray, route_header_bits: int) -> tuple[float, str]:
    ordered = half_to_ordered(half_bits(pixels))
    best_cost = float("inf")
    best_mode = ""
    for mode, payload in predictors(ordered).items():
        cost = best_family_cost(payload, range(15, -1, -1))
        if cost < best_cost:
            best_cost = cost
            best_mode = mode
    return route_header_bits + best_cost, best_mode


def summarize_image(
    path: Path,
    crop_size: int,
    tile_size: int,
    previous_bits: int,
    route_header_bits: int,
) -> dict:
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

    height, width, _channels = pixels.shape
    current_bits = 0.0
    routed_bits = 0.0
    half_tiles = 0
    used_half_tiles = 0
    mode_histogram = Counter()
    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            sl = np.s_[y:y + tile_size, x:x + tile_size]
            current = current_tile_cost(payloads["body"][sl], payloads["sign"][sl])
            current_bits += current
            best = current
            if half_exact(pixels[sl]):
                half_tiles += 1
                half_cost, mode = half_route_tile_cost(pixels[sl], route_header_bits)
                if half_cost < best:
                    best = half_cost
                    used_half_tiles += 1
                    mode_histogram[mode] += 1
            routed_bits += best
    raw_bits = int(pixels.size * 32)
    return {
        "image": path.name,
        "shape": list(pixels.shape),
        "crop_size": crop_size,
        "tile_size": tile_size,
        "raw_bits": raw_bits,
        "current_bits": current_bits,
        "routed_bits": routed_bits,
        "current_ratio": raw_bits / current_bits,
        "routed_ratio": raw_bits / routed_bits,
        "gain": current_bits / routed_bits,
        "half_tiles": half_tiles,
        "used_half_tiles": used_half_tiles,
        "mode_histogram": dict(mode_histogram),
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
    parser.add_argument("--route-header-bits", type=int, default=8)
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
            args.route_header_bits,
        )
        rows.append(row)
        print(
            f"{path.stem:43s} "
            f"current={row['current_ratio']:.3f}x "
            f"half_route={row['routed_ratio']:.3f}x "
            f"gain={row['gain']:.4f} "
            f"half_tiles={row['used_half_tiles']}/{row['half_tiles']}"
        )
        if args.limit and len(rows) >= args.limit:
            break
    if rows:
        print(
            f"geomean current={geomean([r['current_ratio'] for r in rows]):.3f}x "
            f"half_route={geomean([r['routed_ratio'] for r in rows]):.3f}x"
        )
    safe_glob = args.glob.replace("*", "star").replace(".", "_")
    output = RESULTS_DIR / (
        f"half16_route_{safe_glob}_tile{args.tile_size}_crop{args.crop_size}.json"
    )
    output.write_text(json.dumps(rows, indent=2))
    print(f"\nSaved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
