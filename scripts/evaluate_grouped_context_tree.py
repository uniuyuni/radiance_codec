"""Probe MANIAC-like context trees for grouped-delta payload bitplanes.

This keeps the existing grouped-delta predictor mode map, then asks whether the
payload entropy coder improves if each expensive bitplane can use a small
signaled binary context tree instead of the fixed context id.
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

from analyze_structural_predictors import DATA_DIR, FIELD_GROUPS, float_to_bits, load_blosc_ratios, read_exr  # noqa: E402
from evaluate_structural_delta_context import context_bitplane_cost  # noqa: E402
from evaluate_structural_delta_grouped import GROUP_PRESETS, group_bit_indices  # noqa: E402
from probe_grouped_tail_mlp import CONTEXT_FAMILIES, context_family_cost as gdx_context_family_cost  # noqa: E402
from structural_delta_grouped_roundtrip import choose_group_payloads  # noqa: E402

RESULTS_DIR = ROOT / "results"


def kt_cost(ones: int, total: int) -> float:
    if total == 0:
        return 0.0
    zeros = total - ones
    log_probability = (
        math.lgamma(ones + 0.5)
        + math.lgamma(zeros + 0.5)
        - math.lgamma(total + 1.0)
        - 2.0 * math.lgamma(0.5)
    )
    return -log_probability / math.log(2.0)


def kt_cost_for_indices(target: np.ndarray, indices: np.ndarray) -> float:
    if indices.size == 0:
        return 0.0
    ones = int(np.sum(target[indices]))
    return kt_cost(ones, int(indices.size))


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
    y_lsb = (np.arange(height, dtype=np.uint8)[:, None, None] & 1)
    x_lsb = (np.arange(width, dtype=np.uint8)[None, :, None] & 1)
    c = np.arange(channels, dtype=np.uint8)[None, None, :]
    return {
        "x_lsb": np.broadcast_to(x_lsb, shape),
        "y_lsb": np.broadcast_to(y_lsb, shape),
        "c_lsb": np.broadcast_to(c & 1, shape),
        "c_is_0": np.broadcast_to((c == 0).astype(np.uint8), shape),
        "c_is_1": np.broadcast_to((c == 1).astype(np.uint8), shape),
        "c_is_2": np.broadcast_to((c == 2).astype(np.uint8), shape),
    }


def feature_planes(
    payload: np.ndarray,
    bit_index: int,
    previous_bits: int,
) -> dict[str, np.ndarray]:
    plane = ((payload >> np.uint32(bit_index)) & np.uint32(1)).astype(
        np.uint8,
        copy=False,
    )
    features = {
        "w": shift_plane(plane, 0, -1),
        "n": shift_plane(plane, -1, 0),
        "nw": shift_plane(plane, -1, -1),
        "ne": shift_plane(plane, -1, 1),
    }
    channel = np.zeros_like(plane)
    channel[..., 1:] = plane[..., :-1]
    features["pc"] = channel

    for offset in range(1, previous_bits + 1):
        previous_bit = bit_index + offset
        if previous_bit > 31:
            continue
        prev = ((payload >> np.uint32(previous_bit)) & np.uint32(1)).astype(
            np.uint8,
            copy=False,
        )
        features[f"p{offset}"] = prev
        features[f"w_p{offset}"] = shift_plane(prev, 0, -1)
        features[f"n_p{offset}"] = shift_plane(prev, -1, 0)

    features.update(coordinate_features(plane.shape))
    return features


def fixed_context_cost(
    payload: np.ndarray,
    bit_index: int,
    previous_bits: int,
) -> float:
    return context_bitplane_cost(
        payload,
        (bit_index,),
        previous_bits,
        channel_context=False,
    )


def context_family_costs(
    payload: np.ndarray,
    bit_index: int,
    previous_bits: int,
    mode_overhead_bits: int,
) -> dict[str, float]:
    del previous_bits
    return {
        family: gdx_context_family_cost(payload, bit_index, family) + mode_overhead_bits
        for family in CONTEXT_FAMILIES
    }


def legacy_context_family_costs(
    payload: np.ndarray,
    bit_index: int,
    previous_bits: int,
    mode_overhead_bits: int,
) -> dict[str, float]:
    plane = ((payload >> np.uint32(bit_index)) & np.uint32(1)).astype(
        np.uint8,
        copy=False,
    )
    features = feature_planes(payload, bit_index, previous_bits)
    families = {
        "spatial_prev": ["w", "n", "nw", "ne"] + [
            f"p{i}" for i in range(1, previous_bits + 1)
            if f"p{i}" in features
        ],
        "spatial_prev_pc": ["w", "n", "nw", "ne", "pc"] + [
            f"p{i}" for i in range(1, previous_bits + 1)
            if f"p{i}" in features
        ],
        "wn_prev": ["w", "n"] + [
            f"p{i}" for i in range(1, previous_bits + 1)
            if f"p{i}" in features
        ],
        "prev_only": [
            f"p{i}" for i in range(1, previous_bits + 1)
            if f"p{i}" in features
        ],
        "spatial_prev_channel": ["w", "n", "nw", "ne", "c_is_0", "c_is_1", "c_is_2"] + [
            f"p{i}" for i in range(1, previous_bits + 1)
            if f"p{i}" in features
        ],
        "spatial_hi_neighbors": ["w", "n", "nw", "ne"] + [
            name
            for i in range(1, min(previous_bits, 2) + 1)
            for name in (f"p{i}", f"w_p{i}", f"n_p{i}")
            if name in features
        ],
    }
    costs: dict[str, float] = {}
    for name, feature_names in families.items():
        context = np.zeros_like(plane, dtype=np.uint16)
        for shift, feature_name in enumerate(feature_names):
            context |= features[feature_name].astype(np.uint16) << shift
        context_flat = context.reshape(-1)
        plane_flat = plane.reshape(-1)
        context_count = 1 << len(feature_names)
        totals = np.bincount(context_flat, minlength=context_count)
        ones = np.bincount(context_flat, weights=plane_flat, minlength=context_count)
        cost = 0.0
        for context_id in range(context_count):
            total = int(totals[context_id])
            if total:
                cost += kt_cost(int(ones[context_id]), total)
        costs[name] = cost + mode_overhead_bits
    return costs


def greedy_tree_cost(
    payload: np.ndarray,
    bit_index: int,
    previous_bits: int,
    max_leaves: int,
    split_overhead_bits: int,
    tree_header_bits: int,
    min_gain_bits: float,
) -> tuple[float, int, dict[str, int]]:
    target = ((payload >> np.uint32(bit_index)) & np.uint32(1)).astype(
        np.uint8,
        copy=False,
    ).reshape(-1)
    features = {
        name: value.reshape(-1)
        for name, value in feature_planes(payload, bit_index, previous_bits).items()
    }
    indices = np.arange(target.size)
    leaves = [(indices, set[str](), kt_cost_for_indices(target, indices))]
    split_count = 0
    split_histogram = Counter()

    while len(leaves) < max_leaves:
        best: tuple[float, int, str, np.ndarray, np.ndarray, float, float] | None = None
        for leaf_index, (leaf_indices, used, leaf_cost) in enumerate(leaves):
            if leaf_indices.size <= 1:
                continue
            for name, feature in features.items():
                if name in used:
                    continue
                split_values = feature[leaf_indices]
                if bool(np.all(split_values == split_values[0])):
                    continue
                left = leaf_indices[split_values == 0]
                right = leaf_indices[split_values == 1]
                left_cost = kt_cost_for_indices(target, left)
                right_cost = kt_cost_for_indices(target, right)
                gain = leaf_cost - left_cost - right_cost - split_overhead_bits
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
        old_indices, used, _old_cost = leaves.pop(leaf_index)
        del old_indices
        next_used = set(used)
        next_used.add(name)
        leaves.append((left, next_used, left_cost))
        leaves.append((right, next_used, right_cost))
        split_count += 1
        split_histogram[name] += 1

    cost = tree_header_bits + split_count * split_overhead_bits
    cost += sum(leaf_cost for _indices, _used, leaf_cost in leaves)
    return cost, len(leaves), dict(split_histogram)


def group_mask(fields: tuple[str, ...]) -> np.uint32:
    mask = np.uint32(0)
    for bit in group_bit_indices(fields):
        mask |= np.uint32(1) << np.uint32(bit)
    return mask


def estimate_image(
    path: Path,
    tile_size: int,
    previous_bits: int,
    preset: str,
    crop_size: int,
    max_leaves: int,
    split_overhead_bits: int,
    tree_header_bits: int,
    family_mode_overhead_bits: int,
    min_gain_bits: float,
    mode_overhead_bits: int,
) -> dict:
    pixels = read_exr(path)
    if crop_size:
        pixels = pixels[:crop_size, :crop_size]
    bits = float_to_bits(pixels)
    modes, payloads, _carries = choose_group_payloads(
        bits,
        pixels,
        tile_size,
        previous_bits,
        preset,
    )
    groups = GROUP_PRESETS[preset]
    height, width, channels = bits.shape
    raw_bits = bits.size * 32
    baseline_bits = 0.0
    family_bits = 0.0
    tree_bits = 0.0
    hybrid_bits = 0.0
    pick_histogram = Counter()
    bit_pick_histogram = Counter()
    group_pick_histogram = Counter()
    tree_leaf_histogram = Counter()
    tree_split_histogram = Counter()
    tree_bit_split_histogram = Counter()

    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            sl = np.s_[y:y + tile_size, x:x + tile_size]
            for group_name, fields in groups:
                mode = modes[(y, x, group_name)]
                payload = payloads[group_name][sl] & group_mask(fields)
                for field in fields:
                    for bit_index in FIELD_GROUPS[field]:
                        if mode == "raw":
                            cost = float(payload.size)
                            baseline_bits += cost
                            family_bits += cost
                            tree_bits += cost
                            hybrid_bits += cost
                            pick_histogram["raw"] += 1
                            bit_pick_histogram[f"{bit_index}:raw"] += 1
                            group_pick_histogram[f"{group_name}:raw"] += 1
                            continue
                        base = fixed_context_cost(payload, bit_index, previous_bits)
                        family_costs = context_family_costs(
                            payload,
                            bit_index,
                            previous_bits,
                            family_mode_overhead_bits,
                        )
                        family_name, family = min(
                            family_costs.items(),
                            key=lambda item: item[1],
                        )
                        if max_leaves > 0:
                            tree, leaves, tree_splits = greedy_tree_cost(
                                payload,
                                bit_index,
                                previous_bits,
                                max_leaves,
                                split_overhead_bits,
                                tree_header_bits,
                                min_gain_bits,
                            )
                        else:
                            tree, leaves, tree_splits = float("inf"), 0, {}
                        best_name, best = min(
                            [
                                ("base", base),
                                (f"family:{family_name}", family),
                                ("tree", tree),
                            ],
                            key=lambda item: item[1],
                        )
                        baseline_bits += base
                        family_bits += min(base, family)
                        tree_bits += min(base, tree)
                        hybrid_bits += best
                        pick_histogram[best_name] += 1
                        bit_pick_histogram[f"{bit_index}:{best_name}"] += 1
                        group_pick_histogram[f"{group_name}:{best_name}"] += 1
                        if best_name == "tree":
                            tree_leaf_histogram[str(leaves)] += 1
                            tree_split_histogram.update(tree_splits)
                            for feature_name, count in tree_splits.items():
                                tree_bit_split_histogram[f"{bit_index}:{feature_name}"] += count

    tiles = ((height + tile_size - 1) // tile_size) * (
        (width + tile_size - 1) // tile_size
    )
    overhead = tiles * len(groups) * mode_overhead_bits
    baseline_bits += overhead
    family_bits += overhead
    tree_bits += overhead
    hybrid_bits += overhead
    return {
        "image": path.name,
        "shape": [height, width, channels],
        "tile_size": tile_size,
        "previous_bits": previous_bits,
        "preset": preset,
        "crop_size": crop_size,
        "raw_bits": raw_bits,
        "baseline_bits": baseline_bits,
        "family_bits": family_bits,
        "tree_bits": tree_bits,
        "hybrid_bits": hybrid_bits,
        "baseline_ratio": raw_bits / baseline_bits,
        "family_ratio": raw_bits / family_bits,
        "tree_ratio": raw_bits / tree_bits,
        "hybrid_ratio": raw_bits / hybrid_bits,
        "pick_histogram": dict(pick_histogram),
        "bit_pick_histogram": dict(bit_pick_histogram),
        "group_pick_histogram": dict(group_pick_histogram),
        "tree_leaf_histogram": dict(tree_leaf_histogram),
        "tree_split_histogram": dict(tree_split_histogram),
        "tree_bit_split_histogram": dict(tree_bit_split_histogram),
    }


def geomean(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", default="*.exr")
    parser.add_argument("--tile-size", type=int, default=128)
    parser.add_argument("--previous-bits", type=int, default=4)
    parser.add_argument("--preset", choices=sorted(GROUP_PRESETS), default="body")
    parser.add_argument("--crop-size", type=int, default=0)
    parser.add_argument("--max-leaves", type=int, default=16)
    parser.add_argument("--split-overhead-bits", type=int, default=8)
    parser.add_argument("--tree-header-bits", type=int, default=16)
    parser.add_argument("--family-mode-overhead-bits", type=int, default=4)
    parser.add_argument("--min-gain-bits", type=float, default=4.0)
    parser.add_argument("--mode-overhead-bits", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = sorted(DATA_DIR.glob(args.glob))
    if not paths:
        raise RuntimeError(f"no images matched {DATA_DIR / args.glob}")
    blosc = load_blosc_ratios()
    rows = []
    print(
        f"{'image':43s} {'base':>8s} {'family':>8s} "
        f"{'tree':>8s} {'hybrid':>8s} {'Blosc':>8s}"
    )
    print("-" * 94)
    for path in paths:
        row = estimate_image(
            path,
            args.tile_size,
            args.previous_bits,
            args.preset,
            args.crop_size,
            args.max_leaves,
            args.split_overhead_bits,
            args.tree_header_bits,
            args.family_mode_overhead_bits,
            args.min_gain_bits,
            args.mode_overhead_bits,
        )
        row["blosc_ratio"] = blosc.get(path.stem)
        rows.append(row)
        blosc_text = (
            f"{row['blosc_ratio']:7.2f}x"
            if row["blosc_ratio"] is not None
            else "      -"
        )
        print(
            f"{path.stem:43s} {row['baseline_ratio']:7.2f}x "
            f"{row['family_ratio']:7.2f}x {row['tree_ratio']:7.2f}x "
            f"{row['hybrid_ratio']:7.2f}x {blosc_text}"
        )
    print("-" * 94)
    for key in ["baseline_ratio", "family_ratio", "tree_ratio", "hybrid_ratio", "blosc_ratio"]:
        values = [row[key] for row in rows if row.get(key) is not None]
        print(f"geomean {key:14s}: {geomean(values):.3f}x")

    safe_glob = args.glob.replace("*", "star").replace(".", "_")
    output = RESULTS_DIR / (
        f"grouped_context_tree_{safe_glob}_{args.preset}_tile{args.tile_size}"
        f"_prev{args.previous_bits}_leaves{args.max_leaves}_crop{args.crop_size}.json"
    )
    output.write_text(json.dumps(rows, indent=2))
    print(f"\nSaved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
