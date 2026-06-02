"""Probe fixed context families extracted from MDL tree split hints.

The signaled-tree probe can find local structure, but a dynamic tree has
metadata and implementation cost.  This script asks the cheaper follow-up
question: if we turn frequent tree features into fixed GDX-style context
families, do they still beat the current fixed families under the same KT
bitplane estimate?
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
from probe_grouped_tail_mlp import kt_cost  # noqa: E402
from probe_signaled_context_tree_mdl import feature_planes, region_for_bit  # noqa: E402
from structural_delta_grouped_roundtrip import choose_group_payloads  # noqa: E402

RESULTS_DIR = ROOT / "results"

CURRENT_CPP_FAMILIES: dict[str, tuple[str, ...]] = {
    "base": ("w", "n", "nw", "ne", "p1", "p2", "p3", "p4"),
    "prev_only": ("p1", "p2", "p3", "p4"),
    "wn_prev": ("w", "n", "p1", "p2", "p3", "p4"),
    "spatial_prev_pc": ("w", "n", "nw", "ne", "pc", "p1", "p2", "p3", "p4"),
    "spatial_prev_channel": (
        "w",
        "n",
        "nw",
        "ne",
        "c0",
        "c1",
        "c2",
        "p1",
        "p2",
        "p3",
        "p4",
    ),
    "spatial_hi_neighbors": (
        "w",
        "n",
        "nw",
        "ne",
        "p1",
        "wp1",
        "np1",
        "p2",
        "wp2",
        "np2",
    ),
    "spatial_hi_channel": (
        "w",
        "n",
        "c0",
        "c1",
        "c2",
        "p1",
        "wp1",
        "np1",
        "p2",
        "wp2",
        "np2",
    ),
    "spatial_hi_pc": (
        "w",
        "n",
        "nw",
        "ne",
        "pc",
        "p1",
        "wp1",
        "np1",
        "p2",
        "wp2",
        "np2",
    ),
    "prev_xy_channel": ("p1", "p2", "x0", "y0", "c0", "c1", "c2"),
    "prev4_channel": ("p1", "p2", "p3", "p4", "c0", "c1", "c2"),
    "wn_prev4_channel": ("w", "n", "p1", "p2", "p3", "p4", "c0", "c1", "c2"),
    "wn_prev4_pc": ("w", "n", "pc", "p1", "p2", "p3", "p4"),
    "spatial_xy": ("w", "n", "nw", "ne", "x0", "y0", "p1", "p2"),
    "constant_zero": (),
}

TREE_DERIVED_CANDIDATES: dict[str, tuple[str, ...]] = {
    "hi_neighbors_channel": ("p1", "wp1", "np1", "p2", "wp2", "np2", "c0", "c1", "c2"),
    "wn_hi1_channel": ("w", "n", "p1", "wp1", "np1", "c0", "c1", "c2"),
    "wn_pc_hi1_channel": ("w", "n", "pc", "p1", "wp1", "np1", "c0", "c1", "c2"),
    "wn_prev_xy_channel": ("w", "n", "p1", "p2", "x0", "y0", "c0", "c1", "c2"),
    "wn_hi1_xy": ("w", "n", "p1", "wp1", "np1", "x0", "y0"),
    "hi_neighbors_xy": ("p1", "wp1", "np1", "p2", "wp2", "np2", "x0", "y0"),
    "spatial_hi1_xy": ("w", "n", "nw", "ne", "p1", "wp1", "np1", "x0", "y0"),
}


def selected_bits(fields: tuple[str, ...], bit_min: int, bit_max: int) -> tuple[int, ...]:
    return tuple(bit for bit in group_bit_indices(fields) if bit_min <= bit <= bit_max)


def context_cost_from_parts(
    target: np.ndarray,
    features: dict[str, np.ndarray],
    parts: tuple[str, ...],
) -> float | None:
    if not parts:
        return kt_cost(int(np.sum(target)), int(target.size))
    missing = [part for part in parts if part not in features]
    if missing:
        return None

    context = np.zeros(target.shape, dtype=np.uint32)
    for shift, part in enumerate(parts):
        context |= features[part].astype(np.uint32).reshape(-1) << np.uint32(shift)
    context_count = 1 << len(parts)
    totals = np.bincount(context, minlength=context_count)
    ones = np.bincount(context, weights=target, minlength=context_count)
    cost = 0.0
    for context_id in np.nonzero(totals)[0]:
        total = int(totals[context_id])
        cost += kt_cost(int(ones[context_id]), total)
    return cost


def best_family_cost(
    target: np.ndarray,
    features: dict[str, np.ndarray],
    families: dict[str, tuple[str, ...]],
    selector_bits: int,
    max_context_bits: int,
) -> tuple[str, float]:
    best_name = ""
    best_cost = float("inf")
    for name, parts in families.items():
        if len(parts) > max_context_bits:
            continue
        cost = context_cost_from_parts(target, features, parts)
        if cost is None:
            continue
        cost += selector_bits
        if cost < best_cost:
            best_name = name
            best_cost = cost
    if not best_name:
        raise RuntimeError("no valid context family candidates")
    return best_name, best_cost


def parse_candidate_text(text: str) -> dict[str, tuple[str, ...]]:
    if not text:
        return dict(TREE_DERIVED_CANDIDATES)
    candidates: dict[str, tuple[str, ...]] = {}
    for index, item in enumerate(part for part in text.split(",") if part):
        if "=" in item:
            name, parts_text = item.split("=", 1)
        else:
            name, parts_text = f"candidate_{index}", item
        candidates[name] = tuple(part for part in parts_text.split("+") if part)
    return candidates


def summarize_image(
    path: Path,
    crop_size: int,
    tile_size: int,
    previous_bits: int,
    preset: str,
    bit_min: int,
    bit_max: int,
    feature_set: str,
    selector_bits: int,
    max_context_bits: int,
    candidates: dict[str, tuple[str, ...]],
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
    baseline_bits = 0.0
    candidate_bits = 0.0
    extended_bits = 0.0
    baseline_by_region: defaultdict[str, float] = defaultdict(float)
    extended_by_region: defaultdict[str, float] = defaultdict(float)
    baseline_histogram: Counter[str] = Counter()
    candidate_histogram: Counter[str] = Counter()
    extended_histogram: Counter[str] = Counter()
    bit_pick_histogram: Counter[str] = Counter()

    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            sl = np.s_[y:y + tile_size, x:x + tile_size]
            ordered_tile = ordered[sl]
            for group_name, fields in GROUP_PRESETS[preset]:
                payload = payloads[group_name][sl]
                for bit in selected_bits(fields, bit_min, bit_max):
                    target = ((payload >> np.uint32(bit)) & np.uint32(1)).astype(
                        np.uint8,
                    ).reshape(-1)
                    features = {
                        name: value.reshape(-1)
                        for name, value in feature_planes(
                            payload,
                            ordered_tile,
                            bit,
                            previous_bits,
                            feature_set,
                        ).items()
                    }
                    baseline_name, baseline_cost = best_family_cost(
                        target,
                        features,
                        CURRENT_CPP_FAMILIES,
                        selector_bits,
                        max_context_bits,
                    )
                    candidate_name, candidate_cost = best_family_cost(
                        target,
                        features,
                        candidates,
                        selector_bits,
                        max_context_bits,
                    )
                    region = region_for_bit(bit)
                    baseline_bits += baseline_cost
                    candidate_bits += candidate_cost
                    baseline_by_region[region] += baseline_cost
                    baseline_histogram[baseline_name] += 1
                    candidate_histogram[candidate_name] += 1

                    if candidate_cost < baseline_cost:
                        extended_cost = candidate_cost
                        extended_name = f"candidate:{candidate_name}"
                    else:
                        extended_cost = baseline_cost
                        extended_name = f"current:{baseline_name}"
                    extended_bits += extended_cost
                    extended_by_region[region] += extended_cost
                    extended_histogram[extended_name] += 1
                    bit_pick_histogram[f"{bit}:{extended_name}"] += 1

    raw_bits = int(bits.size * 32)
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
        "selector_bits": selector_bits,
        "max_context_bits": max_context_bits,
        "candidate_parts": {name: list(parts) for name, parts in candidates.items()},
        "raw_bits": raw_bits,
        "baseline_bits": baseline_bits,
        "candidate_bits": candidate_bits,
        "extended_bits": extended_bits,
        "baseline_ratio": raw_bits / baseline_bits if baseline_bits else 0.0,
        "candidate_ratio": raw_bits / candidate_bits if candidate_bits else 0.0,
        "extended_ratio": raw_bits / extended_bits if extended_bits else 0.0,
        "gain": baseline_bits / extended_bits if extended_bits else 1.0,
        "baseline_bits_by_region": dict(baseline_by_region),
        "extended_bits_by_region": dict(extended_by_region),
        "baseline_histogram": dict(baseline_histogram),
        "candidate_histogram": dict(candidate_histogram),
        "extended_histogram": dict(extended_histogram),
        "bit_pick_histogram": dict(bit_pick_histogram),
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
    parser.add_argument("--bit-min", type=int, default=15)
    parser.add_argument("--bit-max", type=int, default=20)
    parser.add_argument(
        "--feature-set",
        choices=("strict", "with_ordered_high"),
        default="strict",
    )
    parser.add_argument("--selector-bits", type=int, default=4)
    parser.add_argument("--max-context-bits", type=int, default=12)
    parser.add_argument(
        "--candidates",
        default="",
        help=(
            "Comma-separated candidate specs. Use name=f1+f2 or f1+f2. "
            "Empty uses built-in tree-derived candidates."
        ),
    )
    parser.add_argument("--no-save", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = sorted(DATA_DIR.glob(args.glob))
    if args.limit:
        paths = paths[:args.limit]
    if not paths:
        raise RuntimeError(f"no images matched {DATA_DIR / args.glob}")

    candidates = parse_candidate_text(args.candidates)
    rows = []
    print(
        f"{'image':43s} {'current':>9s} {'cand':>9s} "
        f"{'extended':>9s} {'gain':>8s} picks"
    )
    print("-" * 116)
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
            args.selector_bits,
            args.max_context_bits,
            candidates,
        )
        rows.append(row)
        print(
            f"{path.stem:43s} "
            f"{row['baseline_ratio']:8.3f}x "
            f"{row['candidate_ratio']:8.3f}x "
            f"{row['extended_ratio']:8.3f}x "
            f"{row['gain']:7.4f} {row['extended_histogram']}"
        )

    current_geomean = geomean([row["baseline_ratio"] for row in rows])
    candidate_geomean = geomean([row["candidate_ratio"] for row in rows])
    extended_geomean = geomean([row["extended_ratio"] for row in rows])
    print("-" * 116)
    print(
        f"geomean current={current_geomean:.3f}x "
        f"candidate={candidate_geomean:.3f}x extended={extended_geomean:.3f}x"
    )

    if not args.no_save:
        safe_glob = args.glob.replace("*", "star").replace(".", "_")
        output = RESULTS_DIR / (
            f"fixed_context_family_candidates_{safe_glob}_{args.preset}"
            f"_bits{args.bit_min}-{args.bit_max}_tile{args.tile_size}"
            f"_prev{args.previous_bits}_{args.feature_set}_crop{args.crop_size}.json"
        )
        output.write_text(json.dumps(rows, indent=2))
        print(f"\nSaved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
