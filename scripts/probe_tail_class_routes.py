"""Classify float32 mantissa-tail tiles and estimate route costs.

This is a reversible route probe, not a new codec bitstream.  It asks:

* Is a tile's low mantissa tail exactly zero?
* Is it on a fixed low-bit grid?
* Are individual bitplanes sparse enough to signal cheaply?
* Do low-tail values repeat enough for a palette/dictionary?
* Is the current grouped-delta payload already the best representation?
* Or is the tile effectively random-tail and should be stored with minimal
  modeling overhead?

The estimates are intentionally simple and side-information-aware enough to
rank ideas.  A route that does not beat the current GDX2-style family estimate
here is not worth moving to C++ yet.
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
from audit_float_tail_structure import entropy_bits  # noqa: E402
from probe_grouped_tail_mlp import CONTEXT_FAMILIES, context_family_cost, kt_cost  # noqa: E402
from structural_delta_grouped_roundtrip import choose_group_payloads  # noqa: E402

RESULTS_DIR = ROOT / "results"


def bitplane_kt_cost(values: np.ndarray, bits: range) -> float:
    flat = values.reshape(-1)
    cost = 0.0
    for bit in bits:
        plane = ((flat >> np.uint32(bit)) & np.uint32(1)).astype(np.uint8)
        ones = int(plane.sum())
        cost += kt_cost(ones, plane.size)
    return cost


def family_route_cost(payload: np.ndarray, bits: range) -> float:
    cost = 0.0
    for bit in bits:
        cost += min(context_family_cost(payload, bit, family) for family in CONTEXT_FAMILIES)
    return cost


def raw_entropy_cost(values: np.ndarray, width: int) -> float:
    return entropy_bits(values.reshape(-1), width) * values.size


def fixed_grid_cost(raw_tail: np.ndarray, tail_bits: int, metadata_bits: int) -> tuple[float, int]:
    best_cost = float("inf")
    best_zero_bits = 0
    for zero_bits in range(1, tail_bits + 1):
        mask = np.uint32((1 << zero_bits) - 1)
        if not bool(np.all((raw_tail & mask) == 0)):
            continue
        remaining_width = tail_bits - zero_bits
        shifted = raw_tail >> np.uint32(zero_bits)
        cost = metadata_bits + raw_entropy_cost(shifted, remaining_width)
        if cost < best_cost or (cost == best_cost and zero_bits > best_zero_bits):
            best_cost = cost
            best_zero_bits = zero_bits
    return best_cost, best_zero_bits


def plane_sparse_cost(values: np.ndarray, bits: range, metadata_bits: int) -> tuple[float, int]:
    flat = values.reshape(-1)
    cost = metadata_bits
    sparse_planes = 0
    for bit in bits:
        plane = ((flat >> np.uint32(bit)) & np.uint32(1)).astype(np.uint8)
        ones = int(plane.sum())
        zeros = plane.size - ones
        # Signal all-zero/all-one plane with one value bit; otherwise use KT.
        if ones == 0 or zeros == 0:
            cost += 1.0
            sparse_planes += 1
        else:
            kt = kt_cost(ones, plane.size)
            # Sparse index list estimate: store minority positions with
            # combinatorial entropy plus the minority value.
            minority = min(ones, zeros)
            if minority == 0:
                sparse = 1.0
            else:
                p = minority / plane.size
                sparse = 1.0 + plane.size * (
                    -p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p)
                )
            cost += min(kt, sparse)
            minority_rate = minority / plane.size if plane.size else 0.0
            if sparse < kt and minority_rate <= 0.05:
                sparse_planes += 1
    return cost, sparse_planes


def palette_cost(
    values: np.ndarray,
    width: int,
    metadata_bits: int,
    max_unique: int = 256,
) -> tuple[float, int]:
    flat = values.reshape(-1).astype(np.uint32)
    counts = Counter(int(v) for v in flat)
    unique = len(counts)
    if unique <= 1:
        return metadata_bits + width, unique
    if unique > max_unique:
        return float("inf"), unique
    probs = np.asarray(list(counts.values()), dtype=np.float64) / flat.size
    index_entropy = float(-(probs * np.log2(probs)).sum()) * flat.size
    dictionary_bits = unique * width
    # Crude side info for unique count and route metadata.
    return metadata_bits + 16 + dictionary_bits + index_entropy, unique


def classify_tile(
    raw_mantissa: np.ndarray,
    payload: np.ndarray,
    tail_bits: int,
    mid_bits: range,
    route_header_bits: int,
) -> dict:
    tail_mask = np.uint32((1 << tail_bits) - 1)
    raw_tail = raw_mantissa & tail_mask
    payload_tail = payload & tail_mask
    payload_mid = (payload >> np.uint32(mid_bits.start)) & np.uint32(
        (1 << (mid_bits.stop - mid_bits.start)) - 1
    )

    costs: dict[str, float] = {}
    details: dict[str, float | int] = {}

    costs["raw_tail_entropy"] = raw_entropy_cost(raw_tail, tail_bits)
    costs["payload_tail_entropy"] = raw_entropy_cost(payload_tail, tail_bits)
    costs["payload_tail_bitplane_kt"] = bitplane_kt_cost(payload_tail, range(tail_bits))
    costs["gdx2_family_tail"] = family_route_cost(payload, range(tail_bits))
    costs["gdx2_family_mid"] = family_route_cost(payload, mid_bits)

    fixed_metadata_bits = max(1, math.ceil(math.log2(tail_bits + 1)))
    fixed, fixed_zero_bits = fixed_grid_cost(raw_tail, tail_bits, fixed_metadata_bits)
    costs["fixed_grid_raw_tail"] = fixed
    details["fixed_zero_bits"] = fixed_zero_bits

    plane_metadata_bits = 2 * tail_bits
    plane_sparse, sparse_planes = plane_sparse_cost(
        payload_tail,
        range(tail_bits),
        plane_metadata_bits,
    )
    costs["plane_sparse_payload_tail"] = plane_sparse
    details["sparse_planes"] = sparse_planes

    palette_metadata_bits = 8
    palette, unique = palette_cost(payload_tail, tail_bits, palette_metadata_bits)
    costs["palette_payload_tail"] = palette
    details["palette_unique"] = unique
    details["fixed_metadata_bits"] = fixed_metadata_bits
    details["plane_metadata_bits"] = plane_metadata_bits
    details["palette_metadata_bits"] = palette_metadata_bits

    # Combined tail + mid estimate: current GDX2 family for mid, best tail route
    # for low tail.  This is the main reversible representation route estimate.
    tail_routes = {
        "fixed_grid": costs["fixed_grid_raw_tail"],
        "plane_sparse": costs["plane_sparse_payload_tail"],
        "palette": costs["palette_payload_tail"],
        "gdx2_tail": costs["gdx2_family_tail"],
        "payload_bitplane_kt": costs["payload_tail_bitplane_kt"],
    }
    tail_route, tail_cost = min(tail_routes.items(), key=lambda item: item[1])
    current_cost = costs["gdx2_family_tail"] + costs["gdx2_family_mid"]
    routed_cost = route_header_bits + tail_cost + costs["gdx2_family_mid"]
    costs["current_tail_mid"] = current_cost
    costs["routed_tail_mid"] = routed_cost
    details["tail_route"] = tail_route
    details["tail_route_gain"] = current_cost / routed_cost if routed_cost else 1.0

    raw_tail_entropy_per_bit = (
        costs["raw_tail_entropy"] / raw_tail.size / tail_bits if raw_tail.size else 0.0
    )
    payload_tail_entropy_per_bit = (
        costs["payload_tail_entropy"] / payload_tail.size / tail_bits
        if payload_tail.size
        else 0.0
    )
    mid_entropy_per_bit = (
        raw_entropy_cost(payload_mid, mid_bits.stop - mid_bits.start)
        / payload_mid.size
        / (mid_bits.stop - mid_bits.start)
        if payload_mid.size
        else 0.0
    )
    if fixed_zero_bits == tail_bits:
        klass = "zero_tail"
    elif fixed_zero_bits >= 8:
        klass = "fixed_grid"
    elif unique <= 16:
        klass = "palette_tail"
    elif payload_tail_entropy_per_bit > 0.92 and mid_entropy_per_bit > 0.85:
        klass = "random_tail"
    elif sparse_planes >= tail_bits // 2:
        klass = "plane_sparse"
    elif payload_tail_entropy_per_bit + 0.05 < raw_tail_entropy_per_bit:
        klass = "delta_structured"
    else:
        klass = "mixed_tail"
    details["class"] = klass
    details["raw_tail_entropy_per_bit"] = raw_tail_entropy_per_bit
    details["payload_tail_entropy_per_bit"] = payload_tail_entropy_per_bit
    details["payload_mid_entropy_per_bit"] = mid_entropy_per_bit
    return {
        "class": klass,
        "costs": costs,
        "details": details,
        "samples": int(raw_tail.size),
    }


def summarize_image(
    path: Path,
    crop_size: int,
    tile_size: int,
    previous_bits: int,
    tail_bits: int,
    mid_start: int,
    mid_stop: int,
    route_header_bits: int,
) -> dict:
    pixels = read_exr(path)
    if crop_size:
        pixels = pixels[:crop_size, :crop_size]
    pixels = np.ascontiguousarray(pixels, dtype=np.float32)
    bits = float_to_bits(pixels).copy()
    raw_mantissa = bits & np.uint32(0x7fffff)
    _modes, payloads, _carries = choose_group_payloads(
        bits,
        pixels,
        tile_size,
        previous_bits,
        "body",
    )
    payload = payloads["body"]
    height, width, _channels = bits.shape
    tiles = []
    class_histogram = Counter()
    route_histogram = Counter()
    totals = Counter()
    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            sl = np.s_[y:y + tile_size, x:x + tile_size]
            row = classify_tile(
                raw_mantissa[sl],
                payload[sl],
                tail_bits,
                range(mid_start, mid_stop),
                route_header_bits,
            )
            row["y"] = y
            row["x"] = x
            tiles.append(row)
            class_histogram[row["class"]] += 1
            route_histogram[str(row["details"]["tail_route"])] += 1
            for key, value in row["costs"].items():
                totals[key] += float(value)
    current = totals["current_tail_mid"]
    routed = totals["routed_tail_mid"]
    return {
        "image": path.name,
        "shape": list(bits.shape),
        "tile_size": tile_size,
        "tail_bits": tail_bits,
        "mid_bits": [mid_start, mid_stop],
        "class_histogram": dict(class_histogram),
        "route_histogram": dict(route_histogram),
        "current_tail_mid_bits": current,
        "routed_tail_mid_bits": routed,
        "tail_mid_gain": current / routed if routed else 1.0,
        "totals": dict(totals),
        "tiles": tiles,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", default="*.exr")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--crop-size", type=int, default=0)
    parser.add_argument("--tile-size", type=int, default=128)
    parser.add_argument("--previous-bits", type=int, default=4)
    parser.add_argument("--tail-bits", type=int, default=15)
    parser.add_argument("--mid-start", type=int, default=15)
    parser.add_argument("--mid-stop", type=int, default=21)
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
            args.tail_bits,
            args.mid_start,
            args.mid_stop,
            args.route_header_bits,
        )
        rows.append(row)
        print(
            f"{path.stem:43s} gain={row['tail_mid_gain']:.4f} "
            f"classes={row['class_histogram']} routes={row['route_histogram']}"
        )
        if args.limit and len(rows) >= args.limit:
            break
    safe_glob = args.glob.replace("*", "star").replace(".", "_")
    output = RESULTS_DIR / (
        f"tail_class_routes_{safe_glob}_tile{args.tile_size}"
        f"_tail{args.tail_bits}_mid{args.mid_start}-{args.mid_stop}"
        f"_crop{args.crop_size}.json"
    )
    output.write_text(json.dumps(rows, indent=2))
    print(f"\nSaved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
