"""Probe exact context-palette routes for raw low mantissa tails.

The conditional entropy probe found that puresky low tails become much simpler
when grouped by decoder-visible high float features.  This script adds the
missing side information: per-context palettes/dictionaries plus index entropy.
It is still an estimate, but it is much closer to a real exact codec route than
pure conditional entropy.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_structural_predictors import DATA_DIR, float_to_bits, read_exr  # noqa: E402
from probe_grouped_tail_mlp import CONTEXT_FAMILIES, context_family_cost, kt_cost  # noqa: E402
from structural_delta_grouped_roundtrip import choose_group_payloads  # noqa: E402

RESULTS_DIR = ROOT / "results"


def entropy_bits_from_counts(counts: np.ndarray) -> float:
    total = float(np.sum(counts))
    if total == 0.0:
        return 0.0
    probs = counts.astype(np.float64) / total
    return float(-np.sum(probs * np.log2(probs)) * total)


def fold_hash(values: np.ndarray, bits: int) -> np.ndarray:
    mask = np.uint64((1 << bits) - 1)
    mixed = values.astype(np.uint64)
    mixed ^= mixed >> np.uint64(bits)
    mixed ^= mixed >> np.uint64(bits * 2)
    mixed *= np.uint64(0x9E3779B1)
    mixed ^= mixed >> np.uint64(16)
    return (mixed & mask).astype(np.uint64)


def high_context(bits: np.ndarray, tail_width: int, hash_bits: int, xy_bits: int) -> np.ndarray:
    height, width, channels = bits.shape
    exponent = ((bits >> np.uint32(23)) & np.uint32(0xff)).astype(np.uint64)
    mantissa = (bits & np.uint32(0x7fffff)).astype(np.uint64)
    high = mantissa >> np.uint64(tail_width)
    c = np.arange(channels, dtype=np.uint64)[None, None, :]
    channel = np.broadcast_to(c, bits.shape)
    yy, xx = np.mgrid[0:height, 0:width]
    x = np.broadcast_to(xx[:, :, None].astype(np.uint64), bits.shape)
    y = np.broadcast_to(yy[:, :, None].astype(np.uint64), bits.shape)
    xy_mask = np.uint64((1 << xy_bits) - 1)
    code = high ^ (exponent << np.uint64(17)) ^ (channel << np.uint64(25))
    if xy_bits:
        code ^= (x & xy_mask) << np.uint64(27)
        code ^= (y & xy_mask) << np.uint64(27 + xy_bits)
    return fold_hash(code, hash_bits)


def best_gdx_low_cost(payload: np.ndarray, width: int) -> float:
    total = 0.0
    for bit in range(width):
        total += min(
            context_family_cost(payload, bit, family)
            for family in CONTEXT_FAMILIES
        )
    return total


def best_gdx_high_cost(payload: np.ndarray, low_width: int) -> float:
    total = 0.0
    for bit in range(low_width, 31):
        total += min(
            context_family_cost(payload, bit, family)
            for family in CONTEXT_FAMILIES
        )
    return total


def sign_cost(payload: np.ndarray) -> float:
    return min(context_family_cost(payload, 31, family) for family in CONTEXT_FAMILIES)


def transformed_tail(tail: np.ndarray, tail_width: int, transform: str) -> np.ndarray:
    mask = np.uint32((1 << tail_width) - 1)
    out = tail.copy()
    if transform == "raw":
        return out
    if transform == "xor_prev":
        if tail.shape[2] > 1:
            out[..., 1:] = tail[..., 1:] ^ tail[..., :-1]
        return out
    if transform == "sub_prev":
        if tail.shape[2] > 1:
            out[..., 1:] = (tail[..., 1:] - tail[..., :-1]) & mask
        return out
    if transform == "xor_green":
        if tail.shape[2] >= 3:
            out[..., 0] = tail[..., 0] ^ tail[..., 1]
            out[..., 1] = tail[..., 1]
            out[..., 2] = tail[..., 2] ^ tail[..., 1]
        return out
    if transform == "sub_green":
        if tail.shape[2] >= 3:
            out[..., 0] = (tail[..., 0] - tail[..., 1]) & mask
            out[..., 1] = tail[..., 1]
            out[..., 2] = (tail[..., 2] - tail[..., 1]) & mask
        return out
    raise ValueError(f"unknown tail transform: {transform}")


def palette_context_cost(
    tail: np.ndarray,
    context: np.ndarray,
    tail_width: int,
    context_header_bits: int,
    route_header_bits: int,
    max_palette: int,
) -> tuple[float, dict[str, float]]:
    flat_tail = tail.reshape(-1).astype(np.uint64)
    flat_context = context.reshape(-1).astype(np.uint64)
    combined = (flat_context << np.uint64(tail_width)) | flat_tail
    unique, counts = np.unique(combined, return_counts=True)
    contexts = unique >> np.uint64(tail_width)
    values = unique & np.uint64((1 << tail_width) - 1)
    total_cost = float(route_header_bits)
    raw_cost = float(flat_tail.size * tail_width)
    used_palette_contexts = 0
    raw_contexts = 0
    dictionary_values = 0
    start = 0
    while start < unique.size:
        end = start + 1
        while end < unique.size and contexts[end] == contexts[start]:
            end += 1
        context_counts = counts[start:end]
        sample_count = int(np.sum(context_counts))
        unique_count = end - start
        context_raw = float(sample_count * tail_width)
        if unique_count <= max_palette and sample_count > unique_count:
            dictionary_bits = unique_count * tail_width
            index_bits = entropy_bits_from_counts(context_counts)
            palette_bits = context_header_bits + dictionary_bits + index_bits
            if palette_bits < context_raw:
                total_cost += palette_bits
                used_palette_contexts += 1
                dictionary_values += unique_count
            else:
                total_cost += context_raw
                raw_contexts += 1
        else:
            total_cost += context_raw
            raw_contexts += 1
        start = end
    return total_cost, {
        "raw_cost": raw_cost,
        "used_palette_contexts": float(used_palette_contexts),
        "raw_contexts": float(raw_contexts),
        "dictionary_values": float(dictionary_values),
        "unique_contexts": float(len(np.unique(contexts))),
    }


def adaptive_dictionary_cost(
    tail: np.ndarray,
    context: np.ndarray,
    tail_width: int,
    route_header_bits: int,
) -> tuple[float, dict[str, float]]:
    flat_tail = tail.reshape(-1).astype(np.uint32)
    flat_context = context.reshape(-1).astype(np.uint64)
    seen: dict[int, dict[int, int]] = defaultdict(dict)
    totals = Counter()
    repeat_counts = Counter()
    raw_value_bits = 0.0
    index_bits = 0.0
    for ctx_value, tail_value in zip(flat_context, flat_tail):
        ctx = int(ctx_value)
        value = int(tail_value)
        table = seen[ctx]
        totals[ctx] += 1
        if value in table:
            repeat_counts[ctx] += 1
            index_bits += math.log2(max(1, len(table)))
            table[value] += 1
        else:
            raw_value_bits += tail_width
            table[value] = 1

    flag_bits = 0.0
    for ctx, total in totals.items():
        flag_bits += kt_cost(int(repeat_counts[ctx]), int(total))

    cost = route_header_bits + flag_bits + raw_value_bits + index_bits
    return cost, {
        "adaptive_contexts": float(len(totals)),
        "adaptive_repeats": float(sum(repeat_counts.values())),
        "adaptive_raw_values": float(sum(len(table) for table in seen.values())),
        "adaptive_flag_bits": flag_bits,
        "adaptive_raw_value_bits": raw_value_bits,
        "adaptive_index_bits": index_bits,
    }


def summarize_tile(
    bits_tile: np.ndarray,
    raw_mantissa_tile: np.ndarray,
    body_payload_tile: np.ndarray,
    sign_payload_tile: np.ndarray,
    tail_width: int,
    hash_bits: int,
    xy_bits: int,
    context_header_bits: int,
    route_header_bits: int,
    max_palette: int,
    tail_transform: str,
) -> tuple[dict[str, float], dict[str, float]]:
    tail = transformed_tail(
        raw_mantissa_tile & np.uint32((1 << tail_width) - 1),
        tail_width,
        tail_transform,
    )
    context = high_context(bits_tile, tail_width, hash_bits, xy_bits)
    palette_cost, details = palette_context_cost(
        tail,
        context,
        tail_width,
        context_header_bits,
        route_header_bits,
        max_palette,
    )
    adaptive_cost, adaptive_details = adaptive_dictionary_cost(
        tail,
        context,
        tail_width,
        route_header_bits,
    )
    low = best_gdx_low_cost(body_payload_tile, tail_width)
    high = best_gdx_high_cost(body_payload_tile, tail_width)
    sign = sign_cost(sign_payload_tile)
    route_costs = {
        "current_low": low,
        "context_palette": palette_cost,
        "adaptive_dictionary": adaptive_cost,
    }
    route, selected_low = min(route_costs.items(), key=lambda item: item[1])
    costs = {
        "current_total": sign + high + low,
        "hybrid_total": sign + high + selected_low,
        "gdx_low": low,
        "palette_low": palette_cost,
        "adaptive_low": adaptive_cost,
        "high": high,
        "sign": sign,
    }
    details.update(adaptive_details)
    details["selected_route"] = route
    return costs, details


def summarize_image(
    path: Path,
    crop_size: int,
    tile_size: int,
    previous_bits: int,
    tail_widths: tuple[int, ...],
    hash_bits_values: tuple[int, ...],
    xy_bits: int,
    context_header_bits: int,
    route_header_bits: int,
    max_palette: int,
    tail_transforms: tuple[str, ...],
    scope: str,
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
    raw_bits = int(bits.size * 32)
    rows = []
    for tail_width in tail_widths:
        best_row = None
        tile_totals = Counter()
        for y in range(0, bits.shape[0], tile_size):
            for x in range(0, bits.shape[1], tile_size):
                sl = np.s_[y:y + tile_size, x:x + tile_size]
                tile_totals["gdx_low"] += best_gdx_low_cost(
                    payloads["body"][sl],
                    tail_width,
                )
                tile_totals["high"] += best_gdx_high_cost(
                    payloads["body"][sl],
                    tail_width,
                )
                tile_totals["sign"] += sign_cost(payloads["sign"][sl])
        for hash_bits in hash_bits_values:
            for tail_transform in tail_transforms:
                totals = Counter()
                detail_totals = Counter()
                route_histogram = Counter()
                if scope == "image":
                    tail = transformed_tail(
                        raw_mantissa & np.uint32((1 << tail_width) - 1),
                        tail_width,
                        tail_transform,
                    )
                    context = high_context(bits, tail_width, hash_bits, xy_bits)
                    palette_cost, details = palette_context_cost(
                        tail,
                        context,
                        tail_width,
                        context_header_bits,
                        route_header_bits,
                        max_palette,
                    )
                    adaptive_cost, adaptive_details = adaptive_dictionary_cost(
                        tail,
                        context,
                        tail_width,
                        route_header_bits,
                    )
                    route_costs = {
                        "current_low": float(tile_totals["gdx_low"]),
                        "context_palette": palette_cost,
                        "adaptive_dictionary": adaptive_cost,
                    }
                    route, selected_low = min(
                        route_costs.items(),
                        key=lambda item: item[1],
                    )
                    totals.update(
                        {
                            "current_total": (
                                tile_totals["sign"]
                                + tile_totals["high"]
                                + tile_totals["gdx_low"]
                            ),
                            "hybrid_total": (
                                tile_totals["sign"]
                                + tile_totals["high"]
                                + selected_low
                            ),
                            "gdx_low": tile_totals["gdx_low"],
                            "palette_low": palette_cost,
                            "adaptive_low": adaptive_cost,
                            "high": tile_totals["high"],
                            "sign": tile_totals["sign"],
                        },
                    )
                    details.update(adaptive_details)
                    route_histogram[route] += 1
                    for key, value in details.items():
                        if isinstance(value, (int, float)):
                            detail_totals[key] += float(value)
                elif scope == "tile":
                    for y in range(0, bits.shape[0], tile_size):
                        for x in range(0, bits.shape[1], tile_size):
                            sl = np.s_[y:y + tile_size, x:x + tile_size]
                            costs, details = summarize_tile(
                                bits[sl],
                                raw_mantissa[sl],
                                payloads["body"][sl],
                                payloads["sign"][sl],
                                tail_width,
                                hash_bits,
                                xy_bits,
                                context_header_bits,
                                route_header_bits,
                                max_palette,
                                tail_transform,
                            )
                            totals.update(costs)
                            route_histogram[str(details["selected_route"])] += 1
                            for key, value in details.items():
                                if isinstance(value, (int, float)):
                                    detail_totals[key] += float(value)
                else:
                    raise ValueError(f"unknown scope: {scope}")
                current = float(totals["current_total"])
                hybrid = float(totals["hybrid_total"])
                row = {
                    "tail_width": tail_width,
                    "hash_bits": hash_bits,
                    "tail_transform": tail_transform,
                    "scope": scope,
                    "current_bits": current,
                    "hybrid_bits": hybrid,
                    "current_ratio": raw_bits / current if current else 1.0,
                    "hybrid_ratio": raw_bits / hybrid if hybrid else 1.0,
                    "gain": current / hybrid if hybrid else 1.0,
                    "route_histogram": dict(route_histogram),
                    "totals": dict(totals),
                    "detail_totals": dict(detail_totals),
                }
                if best_row is None or row["hybrid_bits"] < best_row["hybrid_bits"]:
                    best_row = row
        rows.append(best_row)
    return {
        "image": path.name,
        "shape": list(bits.shape),
        "crop_size": crop_size,
        "tile_size": tile_size,
        "previous_bits": previous_bits,
        "xy_bits": xy_bits,
        "context_header_bits": context_header_bits,
        "route_header_bits": route_header_bits,
        "max_palette": max_palette,
        "scope": scope,
        "rows": rows,
    }


def parse_ints(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split(",") if part)


def geomean(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", default="ph_*puresky*.exr")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--tile-size", type=int, default=128)
    parser.add_argument("--previous-bits", type=int, default=4)
    parser.add_argument("--tail-widths", default="8,10,12,15")
    parser.add_argument("--hash-bits", default="8,10,12")
    parser.add_argument("--xy-bits", type=int, default=3)
    parser.add_argument("--context-header-bits", type=int, default=16)
    parser.add_argument("--route-header-bits", type=int, default=8)
    parser.add_argument("--max-palette", type=int, default=256)
    parser.add_argument(
        "--tail-transforms",
        default="raw",
        help="Comma-separated low-tail transforms: raw,xor_prev,sub_prev,xor_green,sub_green",
    )
    parser.add_argument("--scope", choices=("tile", "image"), default="tile")
    parser.add_argument("--no-save", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tail_widths = parse_ints(args.tail_widths)
    hash_bits_values = parse_ints(args.hash_bits)
    tail_transforms = tuple(part for part in args.tail_transforms.split(",") if part)
    rows = []
    for path in sorted(DATA_DIR.glob(args.glob)):
        row = summarize_image(
            path,
            args.crop_size,
            args.tile_size,
            args.previous_bits,
            tail_widths,
            hash_bits_values,
            args.xy_bits,
            args.context_header_bits,
            args.route_header_bits,
            args.max_palette,
            tail_transforms,
            args.scope,
        )
        rows.append(row)
        print(path.stem)
        for item in row["rows"]:
            print(
                f"  low{item['tail_width']:02d}/hash{item['hash_bits']:02d}: "
                f"current={item['current_ratio']:.3f}x "
                f"hybrid={item['hybrid_ratio']:.3f}x "
                f"gain={item['gain']:.4f} tf={item['tail_transform']} "
                f"routes={item['route_histogram']}"
            )
        if args.limit and len(rows) >= args.limit:
            break
    if rows:
        print("\ngeomean:")
        for tail_width in tail_widths:
            current_values = []
            hybrid_values = []
            for row in rows:
                item = next(r for r in row["rows"] if r["tail_width"] == tail_width)
                current_values.append(item["current_ratio"])
                hybrid_values.append(item["hybrid_ratio"])
            print(
                f"  low{tail_width:02d}: current={geomean(current_values):.3f}x "
                f"hybrid={geomean(hybrid_values):.3f}x"
            )
    safe_glob = args.glob.replace("*", "star").replace(".", "_")
    output = RESULTS_DIR / (
        f"tail_context_palette_{safe_glob}_tile{args.tile_size}"
        f"_prev{args.previous_bits}_crop{args.crop_size}"
        f"_tail{args.tail_widths.replace(',', '-')}_hash{args.hash_bits.replace(',', '-')}.json"
    )
    if not args.no_save:
        output.write_text(json.dumps(rows, indent=2))
        print(f"\nSaved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
