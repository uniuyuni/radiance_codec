"""Probe MDL-costed signaled context trees for GDX payload bitplanes.

`evaluate_grouped_context_tree.py` proved that tiny MANIAC-like trees can help,
but it used a simple fixed split overhead.  This probe makes the accounting a
little harsher: a selected tree pays for its selector, topology, and split
feature ids.  Leaf probabilities are still estimated with the same KT universal
model used by the existing Python context-family probes, so the comparison is
not a final bitstream size.  It is a triage tool for deciding whether a C++
signaled-tree mode is worth building.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
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


def kt_cost_for_indices(target: np.ndarray, indices: np.ndarray) -> float:
    if indices.size == 0:
        return 0.0
    return kt_cost(int(np.sum(target[indices])), int(indices.size))


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


def coordinate_features(shape: tuple[int, int, int]) -> dict[str, np.ndarray]:
    height, width, channels = shape
    x = np.arange(width, dtype=np.uint8)[None, :, None]
    y = np.arange(height, dtype=np.uint8)[:, None, None]
    c = np.arange(channels, dtype=np.uint8)[None, None, :]
    return {
        "x0": np.broadcast_to(x & 1, shape),
        "x1": np.broadcast_to((x >> 1) & 1, shape),
        "y0": np.broadcast_to(y & 1, shape),
        "y1": np.broadcast_to((y >> 1) & 1, shape),
        "c0": np.broadcast_to((c == 0).astype(np.uint8), shape),
        "c1": np.broadcast_to((c == 1).astype(np.uint8), shape),
        "c2": np.broadcast_to((c == 2).astype(np.uint8), shape),
        "c_lsb": np.broadcast_to(c & 1, shape),
    }


def previous_bit(payload: np.ndarray, bit: int, offset: int) -> np.ndarray:
    previous = bit + offset
    if previous > 31:
        return np.zeros(payload.shape, dtype=np.uint8)
    return ((payload >> np.uint32(previous)) & np.uint32(1)).astype(np.uint8)


def feature_planes(
    payload: np.ndarray,
    ordered: np.ndarray,
    bit: int,
    previous_bits: int,
    feature_set: str,
) -> dict[str, np.ndarray]:
    plane = ((payload >> np.uint32(bit)) & np.uint32(1)).astype(np.uint8)
    features: dict[str, np.ndarray] = {
        "w": shift_plane(plane, 0, -1),
        "n": shift_plane(plane, -1, 0),
        "nw": shift_plane(plane, -1, -1),
        "ne": shift_plane(plane, -1, 1),
    }
    pc = np.zeros_like(plane)
    pc[..., 1:] = plane[..., :-1]
    features["pc"] = pc

    for offset in range(1, previous_bits + 1):
        prev = previous_bit(payload, bit, offset)
        features[f"p{offset}"] = prev
        features[f"wp{offset}"] = shift_plane(prev, 0, -1)
        features[f"np{offset}"] = shift_plane(prev, -1, 0)

    features.update(coordinate_features(plane.shape))

    if feature_set == "with_ordered_high":
        for high_bit in (30, 29, 28, 27, 26, 25, 24, 23, 21, 19, 17, 15):
            if high_bit > bit:
                value = ((ordered >> np.uint32(high_bit)) & np.uint32(1)).astype(
                    np.uint8,
                )
                features[f"ord{high_bit}"] = value
                features[f"word{high_bit}"] = shift_plane(value, 0, -1)
                features[f"nord{high_bit}"] = shift_plane(value, -1, 0)
    elif feature_set != "strict":
        raise ValueError(f"unknown feature set: {feature_set}")

    return features


def mdl_tree_cost(
    payload: np.ndarray,
    ordered: np.ndarray,
    bit: int,
    previous_bits: int,
    feature_set: str,
    max_leaves: int,
    selector_bits: int,
    header_bits: int,
    topology_bit_cost: int,
    split_extra_bits: int,
    min_gain_bits: float,
) -> tuple[float, int, dict[str, int]]:
    target = ((payload >> np.uint32(bit)) & np.uint32(1)).astype(np.uint8).reshape(-1)
    features = {
        name: value.reshape(-1)
        for name, value in feature_planes(
            payload,
            ordered,
            bit,
            previous_bits,
            feature_set,
        ).items()
    }
    feature_id_bits = max(1, math.ceil(math.log2(max(2, len(features)))))
    split_penalty = (
        feature_id_bits
        + split_extra_bits
        + 2 * topology_bit_cost
    )

    indices = np.arange(target.size)
    leaves: list[tuple[np.ndarray, frozenset[str], float]] = [
        (indices, frozenset(), kt_cost_for_indices(target, indices)),
    ]
    split_histogram: Counter[str] = Counter()

    while len(leaves) < max_leaves:
        best: tuple[float, int, str, np.ndarray, np.ndarray, float, float] | None = None
        for leaf_index, (leaf_indices, used, leaf_cost) in enumerate(leaves):
            if leaf_indices.size <= 1:
                continue
            for name, feature in features.items():
                if name in used:
                    continue
                values = feature[leaf_indices]
                if bool(np.all(values == values[0])):
                    continue
                left = leaf_indices[values == 0]
                right = leaf_indices[values == 1]
                left_cost = kt_cost_for_indices(target, left)
                right_cost = kt_cost_for_indices(target, right)
                gain = leaf_cost - left_cost - right_cost - split_penalty
                if gain > min_gain_bits and (best is None or gain > best[0]):
                    best = (
                        gain,
                        leaf_index,
                        name,
                        left,
                        right,
                        left_cost,
                        right_cost,
                    )
        if best is None:
            break
        _gain, leaf_index, name, left, right, left_cost, right_cost = best
        _old_indices, used, _old_cost = leaves.pop(leaf_index)
        next_used = frozenset((*used, name))
        leaves.append((left, next_used, left_cost))
        leaves.append((right, next_used, right_cost))
        split_histogram[name] += 1

    split_count = len(leaves) - 1
    topology_bits = (2 * split_count + 1) * topology_bit_cost
    split_bits = split_count * (feature_id_bits + split_extra_bits)
    leaf_bits = sum(leaf_cost for _indices, _used, leaf_cost in leaves)
    total = selector_bits + header_bits + topology_bits + split_bits + leaf_bits
    return total, len(leaves), dict(split_histogram)


def best_existing_cost(payload: np.ndarray, bit: int, selector_bits: int) -> tuple[str, float]:
    family, cost = min(
        (
            (name, context_family_cost(payload, bit, name))
            for name in CONTEXT_FAMILIES
        ),
        key=lambda item: item[1],
    )
    return family, cost + selector_bits


def selected_bits(fields: tuple[str, ...], bit_min: int, bit_max: int) -> tuple[int, ...]:
    return tuple(
        bit
        for bit in group_bit_indices(fields)
        if bit_min <= bit <= bit_max
    )


def region_for_bit(bit: int) -> str:
    if bit <= 14:
        return "tail_0_14"
    if bit <= 20:
        return "mid_15_20"
    if bit <= 30:
        return "high_21_30"
    return "sign"


def summarize_image(
    path: Path,
    crop_size: int,
    tile_size: int,
    previous_bits: int,
    preset: str,
    bit_min: int,
    bit_max: int,
    feature_set: str,
    max_leaves: int,
    existing_selector_bits: int,
    tree_selector_bits: int,
    tree_header_bits: int,
    topology_bit_cost: int,
    split_extra_bits: int,
    min_gain_bits: float,
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
        preset,
    )
    height, width, _channels = bits.shape
    raw_bits = int(bits.size * 32)
    existing_bits = 0.0
    tree_bits = 0.0
    hybrid_bits = 0.0
    existing_by_region: defaultdict[str, float] = defaultdict(float)
    hybrid_by_region: defaultdict[str, float] = defaultdict(float)
    pick_histogram: Counter[str] = Counter()
    bit_pick_histogram: Counter[str] = Counter()
    region_pick_histogram: Counter[str] = Counter()
    leaf_histogram: Counter[str] = Counter()
    split_histogram: Counter[str] = Counter()

    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            sl = np.s_[y:y + tile_size, x:x + tile_size]
            ordered_tile = ordered[sl]
            for group_name, fields in GROUP_PRESETS[preset]:
                payload = payloads[group_name][sl]
                for bit in selected_bits(fields, bit_min, bit_max):
                    region = region_for_bit(bit)
                    family_name, family_cost = best_existing_cost(
                        payload,
                        bit,
                        existing_selector_bits,
                    )
                    tree_cost, leaves, splits = mdl_tree_cost(
                        payload,
                        ordered_tile,
                        bit,
                        previous_bits,
                        feature_set,
                        max_leaves,
                        tree_selector_bits,
                        tree_header_bits,
                        topology_bit_cost,
                        split_extra_bits,
                        min_gain_bits,
                    )
                    existing_bits += family_cost
                    tree_bits += tree_cost
                    existing_by_region[region] += family_cost
                    if tree_cost < family_cost:
                        best_name = "tree"
                        best_cost = tree_cost
                        leaf_histogram[str(leaves)] += 1
                        split_histogram.update(splits)
                    else:
                        best_name = f"family:{family_name}"
                        best_cost = family_cost
                    hybrid_bits += best_cost
                    hybrid_by_region[region] += best_cost
                    pick_histogram[best_name] += 1
                    bit_pick_histogram[f"{bit}:{best_name}"] += 1
                    region_pick_histogram[f"{region}:{best_name}"] += 1

    return {
        "image": path.name,
        "shape": list(bits.shape),
        "crop_size": crop_size,
        "tile_size": tile_size,
        "previous_bits": previous_bits,
        "preset": preset,
        "bit_min": bit_min,
        "bit_max": bit_max,
        "feature_set": feature_set,
        "max_leaves": max_leaves,
        "existing_selector_bits": existing_selector_bits,
        "tree_selector_bits": tree_selector_bits,
        "tree_header_bits": tree_header_bits,
        "topology_bit_cost": topology_bit_cost,
        "split_extra_bits": split_extra_bits,
        "min_gain_bits": min_gain_bits,
        "raw_bits": raw_bits,
        "existing_bits": existing_bits,
        "tree_bits": tree_bits,
        "hybrid_bits": hybrid_bits,
        "existing_ratio": raw_bits / existing_bits if existing_bits else 0.0,
        "tree_ratio": raw_bits / tree_bits if tree_bits else 0.0,
        "hybrid_ratio": raw_bits / hybrid_bits if hybrid_bits else 0.0,
        "gain": existing_bits / hybrid_bits if hybrid_bits else 1.0,
        "existing_bits_by_region": dict(existing_by_region),
        "hybrid_bits_by_region": dict(hybrid_by_region),
        "pick_histogram": dict(pick_histogram),
        "bit_pick_histogram": dict(bit_pick_histogram),
        "region_pick_histogram": dict(region_pick_histogram),
        "tree_leaf_histogram": dict(leaf_histogram),
        "tree_split_histogram": dict(split_histogram),
    }


def geomean(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values if value > 0) / len(values))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", default="*.exr")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--tile-size", type=int, default=128)
    parser.add_argument("--previous-bits", type=int, default=4)
    parser.add_argument("--preset", choices=sorted(GROUP_PRESETS), default="body")
    parser.add_argument("--bit-min", type=int, default=0)
    parser.add_argument("--bit-max", type=int, default=30)
    parser.add_argument(
        "--feature-set",
        choices=("strict", "with_ordered_high"),
        default="strict",
    )
    parser.add_argument("--max-leaves", type=int, default=16)
    parser.add_argument("--existing-selector-bits", type=int, default=4)
    parser.add_argument("--tree-selector-bits", type=int, default=4)
    parser.add_argument("--tree-header-bits", type=int, default=8)
    parser.add_argument("--topology-bit-cost", type=int, default=1)
    parser.add_argument("--split-extra-bits", type=int, default=0)
    parser.add_argument("--min-gain-bits", type=float, default=0.0)
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
    print(
        f"{'image':43s} {'existing':>9s} {'tree':>9s} "
        f"{'hybrid':>9s} {'gain':>8s} picks"
    )
    print("-" * 104)
    for path in paths:
        row = summarize_image(
            path,
            args.crop_size,
            args.tile_size,
            args.previous_bits,
            args.preset,
            args.bit_min,
            args.bit_max,
            args.feature_set,
            args.max_leaves,
            args.existing_selector_bits,
            args.tree_selector_bits,
            args.tree_header_bits,
            args.topology_bit_cost,
            args.split_extra_bits,
            args.min_gain_bits,
        )
        rows.append(row)
        print(
            f"{path.stem:43s} "
            f"{row['existing_ratio']:8.3f}x "
            f"{row['tree_ratio']:8.3f}x "
            f"{row['hybrid_ratio']:8.3f}x "
            f"{row['gain']:7.4f} {row['pick_histogram']}"
        )

    existing_geomean = geomean([row["existing_ratio"] for row in rows])
    tree_geomean = geomean([row["tree_ratio"] for row in rows])
    hybrid_geomean = geomean([row["hybrid_ratio"] for row in rows])
    print("-" * 104)
    print(
        f"geomean existing={existing_geomean:.3f}x "
        f"tree={tree_geomean:.3f}x hybrid={hybrid_geomean:.3f}x"
    )

    if not args.no_save:
        safe_glob = args.glob.replace("*", "star").replace(".", "_")
        output = RESULTS_DIR / (
            f"signaled_context_tree_mdl_{safe_glob}_{args.preset}"
            f"_bits{args.bit_min}-{args.bit_max}_tile{args.tile_size}"
            f"_prev{args.previous_bits}_leaves{args.max_leaves}"
            f"_{args.feature_set}_crop{args.crop_size}.json"
        )
        output.write_text(json.dumps(rows, indent=2))
        print(f"\nSaved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
