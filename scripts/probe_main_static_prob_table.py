"""Probe transmitted static probability tables for the main GDX payload.

GDX8 encodes each selected payload bitplane with adaptive KT counts inside the
chosen context family.  That costs no side information, but it pays a learning
penalty for every tile/bit/context.  This probe asks whether it is worth sending
small per-bitplane probability tables for the expensive main payload, falling
back to the current adaptive route when the table overhead is not recovered.

The estimate keeps the current grouped-delta payload and context-family set.  It
does not change predictors; it only tests the entropy-coder side of the main
payload.
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import math
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_structural_predictors import DATA_DIR, float_to_bits, read_exr  # noqa: E402
from evaluate_structural_delta_grouped import GROUP_PRESETS, group_bit_indices  # noqa: E402
from probe_channel_split_families import best_shared_and_channel_families  # noqa: E402
from probe_grouped_tail_mlp import (  # noqa: E402
    CONTEXT_FAMILIES,
    context_cost_from_flat,
    family_context,
)
from structural_delta_grouped_roundtrip import choose_group_payloads  # noqa: E402

RESULTS_DIR = ROOT / "results"


CORPUS_PATTERNS = {
    "custom": (),
    "all": ("*.exr",),
    "target-no-noise": ("*.exr",),
    "realistic-core": ("ph_*.exr", "oexr_ScanLines_*.exr"),
    "realistic-no-puresky": ("ph_*.exr", "oexr_ScanLines_*.exr"),
    "highres-sample": ("sample_*.exr",),
    "puresky-hard": ("ph_*puresky*.exr",),
    "easy": ("synth_gradient_1k.exr", "oexr_TestImages_GrayRampsHorizontal.exr"),
    "synthetic": ("synth_*.exr",),
    "noise-stress": ("synth_noise_1k.exr",),
}

CORPUS_EXCLUDES = {
    "target-no-noise": ("synth_noise_1k.exr",),
    "realistic-no-puresky": ("ph_*puresky*.exr",),
}


@dataclass
class StaticChoice:
    adaptive_bits: float
    static_bits: float
    selected_bits: float
    selected_static: bool
    selected_entries: int
    table_bits: float


def parse_ints(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split(",") if part)


def geomean(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def selected_paths(args: argparse.Namespace) -> list[Path]:
    if args.corpus == "custom":
        paths = sorted(DATA_DIR.glob(args.glob))
    else:
        paths = sorted(
            {
                path
                for pattern in CORPUS_PATTERNS[args.corpus]
                for path in DATA_DIR.glob(pattern)
            }
        )
    excludes = list(CORPUS_EXCLUDES.get(args.corpus, ())) + args.exclude
    return [
        path
        for path in paths
        if not any(
            fnmatch.fnmatch(path.name, pattern)
            or fnmatch.fnmatch(path.stem, pattern)
            for pattern in excludes
        )
    ]


def entropy_bits(ones: np.ndarray, totals: np.ndarray) -> np.ndarray:
    totals_f = totals.astype(np.float64)
    ones_f = ones.astype(np.float64)
    zeros_f = totals_f - ones_f
    out = np.zeros_like(totals_f)
    active = totals_f > 0
    p = np.zeros_like(totals_f)
    p[active] = ones_f[active] / totals_f[active]
    one_active = active & (ones_f > 0)
    zero_active = active & (zeros_f > 0)
    out[one_active] -= ones_f[one_active] * np.log2(p[one_active])
    out[zero_active] -= zeros_f[zero_active] * np.log2(1.0 - p[zero_active])
    return out


def cross_entropy_bits(
    ones: np.ndarray,
    totals: np.ndarray,
    probability: float,
) -> np.ndarray:
    probability = min(max(probability, 1.0e-9), 1.0 - 1.0e-9)
    ones_f = ones.astype(np.float64)
    zeros_f = totals.astype(np.float64) - ones_f
    return -ones_f * math.log2(probability) - zeros_f * math.log2(1.0 - probability)


def table_cost_from_counts(
    ones: np.ndarray,
    totals: np.ndarray,
    context_id_bits: int,
    probability_bits: int,
    route_header_bits: int,
    entry_extra_bits: int,
) -> tuple[float, int, float]:
    active = totals > 0
    total = int(np.sum(totals))
    if total == 0:
        return 0.0, 0, 0.0

    bit_ones = int(np.sum(ones))
    probability = (bit_ones + 0.5) / (total + 1.0)
    global_cost = cross_entropy_bits(ones, totals, probability)
    context_entropy = entropy_bits(ones, totals)

    # Store one global probability for the bitplane, then only contexts whose
    # custom probability pays for its own index/probability entry.
    entry_header = context_id_bits + probability_bits + entry_extra_bits
    savings = global_cost - context_entropy
    selected = active & (savings > entry_header)
    selected_entries = int(np.count_nonzero(selected))
    selected_savings = float(np.sum(savings[selected] - entry_header))
    table_bits = (
        route_header_bits
        + probability_bits
        + selected_entries * entry_header
    )
    static_bits = route_header_bits + probability_bits + float(np.sum(global_cost))
    static_bits -= selected_savings
    return static_bits, selected_entries, table_bits


def static_cost_for_family(
    payload: np.ndarray,
    bit: int,
    family: str,
    probability_bits: int,
    route_header_bits: int,
    entry_extra_bits: int,
) -> StaticChoice:
    context, context_count = family_context(payload, bit, family)
    symbols = ((payload >> np.uint32(bit)) & np.uint32(1)).reshape(-1)
    context_flat = context.reshape(-1)
    adaptive = context_cost_from_flat(context_flat, symbols, context_count)
    totals = np.bincount(context_flat, minlength=context_count).astype(np.int64)
    ones = np.bincount(
        context_flat,
        weights=symbols,
        minlength=context_count,
    ).astype(np.int64)
    context_id_bits = max(1, int(math.ceil(math.log2(context_count))))
    static, selected_entries, table_bits = table_cost_from_counts(
        ones,
        totals,
        context_id_bits,
        probability_bits,
        route_header_bits,
        entry_extra_bits,
    )
    return StaticChoice(
        adaptive_bits=adaptive,
        static_bits=static,
        selected_bits=min(adaptive, static),
        selected_static=static < adaptive,
        selected_entries=selected_entries,
        table_bits=table_bits if static < adaptive else 0.0,
    )


def static_cost_for_channel_family(
    payload: np.ndarray,
    bit: int,
    channel: int,
    family: str,
    probability_bits: int,
    route_header_bits: int,
    entry_extra_bits: int,
) -> StaticChoice:
    context, context_count = family_context(payload, bit, family)
    symbols = ((payload[:, :, channel] >> np.uint32(bit)) & np.uint32(1)).reshape(-1)
    context_flat = context[:, :, channel].reshape(-1)
    adaptive = context_cost_from_flat(context_flat, symbols, context_count)
    totals = np.bincount(context_flat, minlength=context_count).astype(np.int64)
    ones = np.bincount(
        context_flat,
        weights=symbols,
        minlength=context_count,
    ).astype(np.int64)
    context_id_bits = max(1, int(math.ceil(math.log2(context_count))))
    static, selected_entries, table_bits = table_cost_from_counts(
        ones,
        totals,
        context_id_bits,
        probability_bits,
        route_header_bits,
        entry_extra_bits,
    )
    return StaticChoice(
        adaptive_bits=adaptive,
        static_bits=static,
        selected_bits=min(adaptive, static),
        selected_static=static < adaptive,
        selected_entries=selected_entries,
        table_bits=table_bits if static < adaptive else 0.0,
    )


def group_mask(fields: tuple[str, ...]) -> np.uint32:
    mask = np.uint32(0)
    for bit in group_bit_indices(fields):
        mask |= np.uint32(1) << np.uint32(bit)
    return mask


def best_static_shared(
    payload: np.ndarray,
    bit: int,
    probability_bits: int,
    route_header_bits: int,
    entry_extra_bits: int,
) -> tuple[str, StaticChoice]:
    best_family = ""
    best_choice: StaticChoice | None = None
    for family in CONTEXT_FAMILIES:
        choice = static_cost_for_family(
            payload,
            bit,
            family,
            probability_bits,
            route_header_bits,
            entry_extra_bits,
        )
        if best_choice is None or choice.selected_bits < best_choice.selected_bits:
            best_family = family
            best_choice = choice
    assert best_choice is not None
    return best_family, best_choice


def best_static_channel_split(
    payload: np.ndarray,
    bit: int,
    probability_bits: int,
    route_header_bits: int,
    entry_extra_bits: int,
) -> tuple[tuple[str, ...], StaticChoice]:
    families = []
    adaptive = 0.0
    static = 0.0
    selected = 0.0
    selected_entries = 0
    table_bits = 0.0
    selected_static = False
    for channel in range(payload.shape[2]):
        best_family = ""
        best_choice: StaticChoice | None = None
        for family in CONTEXT_FAMILIES:
            choice = static_cost_for_channel_family(
                payload,
                bit,
                channel,
                family,
                probability_bits,
                route_header_bits,
                entry_extra_bits,
            )
            if best_choice is None or choice.selected_bits < best_choice.selected_bits:
                best_family = family
                best_choice = choice
        assert best_choice is not None
        families.append(best_family)
        adaptive += best_choice.adaptive_bits
        static += best_choice.static_bits
        selected += best_choice.selected_bits
        selected_entries += best_choice.selected_entries
        table_bits += best_choice.table_bits
        selected_static = selected_static or best_choice.selected_static
    return tuple(families), StaticChoice(
        adaptive_bits=adaptive,
        static_bits=static,
        selected_bits=selected,
        selected_static=selected_static,
        selected_entries=selected_entries,
        table_bits=table_bits,
    )


def summarize_image(
    path: Path,
    crop_size: int,
    tile_size: int,
    previous_bits: int,
    probability_bits: int,
    route_header_bits: int,
    entry_extra_bits: int,
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
        "body",
    )

    height, width, channels = bits.shape
    raw_bits = int(bits.size * 32)
    adaptive_bits = 0.0
    hybrid_bits = 0.0
    ideal_static_bits = 0.0
    selected_entries = 0
    table_bits = 0.0
    route_histogram = Counter()
    group_histogram = Counter()
    bit_histogram = Counter()
    family_histogram = Counter()

    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            sl = np.s_[y : y + tile_size, x : x + tile_size]
            for group_name, fields in GROUP_PRESETS["body"]:
                payload = payloads[group_name][sl] & group_mask(fields)
                mode = modes[(y, x, group_name)]
                for bit in group_bit_indices(fields):
                    if mode == "raw":
                        raw_cost = float(payload.size)
                        adaptive_bits += raw_cost
                        hybrid_bits += raw_cost
                        ideal_static_bits += raw_cost
                        route_histogram["raw"] += 1
                        continue

                    shared_name, shared_choice = best_static_shared(
                        payload,
                        bit,
                        probability_bits,
                        route_header_bits,
                        entry_extra_bits,
                    )
                    shared_adaptive = shared_choice.adaptive_bits + 4.0
                    shared_selected = shared_choice.selected_bits + 4.0
                    shared_ideal = shared_choice.static_bits + 4.0

                    best_adaptive = shared_adaptive
                    best_hybrid = shared_adaptive
                    best_ideal = min(shared_adaptive, shared_ideal)
                    best_route = "shared_adaptive"
                    best_selected_entries = 0
                    best_table_bits = 0.0
                    best_family = shared_name

                    if shared_choice.selected_static:
                        candidate = shared_choice.static_bits + 4.0
                        if candidate + min_gain_bits < best_hybrid:
                            best_hybrid = candidate
                            best_route = "shared_static"
                            best_selected_entries = shared_choice.selected_entries
                            best_table_bits = shared_choice.table_bits

                    if group_name == "body" and channels > 1:
                        split_names, split_choice = best_static_channel_split(
                            payload,
                            bit,
                            probability_bits,
                            route_header_bits,
                            entry_extra_bits,
                        )
                        split_adaptive = split_choice.adaptive_bits + channels * 4.0 + 1.0
                        split_static = split_choice.static_bits + channels * 4.0 + 1.0
                        best_adaptive = min(best_adaptive, split_adaptive)
                        best_ideal = min(best_ideal, split_adaptive, split_static)
                        if split_adaptive + min_gain_bits < best_hybrid:
                            best_hybrid = split_adaptive
                            best_route = "channel_split_adaptive"
                            best_family = "+".join(split_names)
                            best_selected_entries = 0
                            best_table_bits = 0.0
                        if split_choice.selected_static:
                            if split_static + min_gain_bits < best_hybrid:
                                best_hybrid = split_static
                                best_route = "channel_split_static"
                                best_family = "+".join(split_names)
                                best_selected_entries = split_choice.selected_entries
                                best_table_bits = split_choice.table_bits

                    adaptive_bits += best_adaptive
                    hybrid_bits += best_hybrid
                    ideal_static_bits += best_ideal
                    selected_entries += best_selected_entries
                    table_bits += best_table_bits
                    route_histogram[best_route] += 1
                    group_histogram[group_name] += 1
                    bit_histogram[str(bit)] += 1
                    family_histogram[best_family] += 1

    return {
        "image": path.name,
        "shape": [height, width, channels],
        "crop_size": crop_size,
        "tile_size": tile_size,
        "previous_bits": previous_bits,
        "probability_bits": probability_bits,
        "route_header_bits": route_header_bits,
        "entry_extra_bits": entry_extra_bits,
        "min_gain_bits": min_gain_bits,
        "raw_bits": raw_bits,
        "adaptive_bits": adaptive_bits,
        "hybrid_bits": hybrid_bits,
        "ideal_static_bits": ideal_static_bits,
        "adaptive_ratio": raw_bits / adaptive_bits,
        "hybrid_ratio": raw_bits / hybrid_bits,
        "ideal_static_ratio": raw_bits / ideal_static_bits,
        "hybrid_gain": adaptive_bits / hybrid_bits,
        "ideal_static_gain": adaptive_bits / ideal_static_bits,
        "selected_entries": selected_entries,
        "table_bits": table_bits,
        "route_histogram": dict(route_histogram),
        "group_histogram": dict(group_histogram),
        "bit_histogram": dict(bit_histogram),
        "family_histogram": dict(family_histogram),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", default="ph_studio_small_03_1k.exr")
    parser.add_argument(
        "--corpus",
        choices=sorted(CORPUS_PATTERNS),
        default="custom",
    )
    parser.add_argument("--exclude", action="append", default=[])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--tile-size", type=int, default=128)
    parser.add_argument("--previous-bits", type=int, default=4)
    parser.add_argument("--probability-bits", type=int, default=12)
    parser.add_argument("--route-header-bits", type=int, default=2)
    parser.add_argument("--entry-extra-bits", type=int, default=2)
    parser.add_argument("--min-gain-bits", type=float, default=4.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = selected_paths(args)
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        raise RuntimeError(
            f"no images matched corpus={args.corpus!r} glob={args.glob!r}"
        )

    rows = []
    for path in paths:
        row = summarize_image(
            path,
            args.crop_size,
            args.tile_size,
            args.previous_bits,
            args.probability_bits,
            args.route_header_bits,
            args.entry_extra_bits,
            args.min_gain_bits,
        )
        rows.append(row)
        print(
            f"{path.stem:43s} "
            f"adaptive={row['adaptive_ratio']:.3f}x "
            f"hybrid={row['hybrid_ratio']:.3f}x "
            f"ideal={row['ideal_static_ratio']:.3f}x "
            f"gain={row['hybrid_gain']:.4f} "
            f"routes={row['route_histogram']}"
        )

    print(
        f"geomean adaptive={geomean([r['adaptive_ratio'] for r in rows]):.3f}x "
        f"hybrid={geomean([r['hybrid_ratio'] for r in rows]):.3f}x "
        f"ideal={geomean([r['ideal_static_ratio'] for r in rows]):.3f}x"
    )
    safe = args.glob.replace("*", "star").replace(".", "_")
    selection = args.corpus if args.corpus != "custom" else safe
    output = RESULTS_DIR / (
        f"main_static_prob_table_{selection}_tile{args.tile_size}"
        f"_prev{args.previous_bits}_crop{args.crop_size}.json"
    )
    output.write_text(
        json.dumps([dict(row) for row in rows], indent=2),
        encoding="utf-8",
    )
    print(f"\nSaved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
