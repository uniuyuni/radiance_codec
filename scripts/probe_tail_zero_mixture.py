"""Probe exact low-tail routes that exploit per-sample zero/trailing-zero runs.

The previous tail route probe only looked for whole-tile fixed grids.  Puresky
does not satisfy that: many samples have low mantissa bits equal to zero, but
not all samples in the tile.  This probe asks whether an exact route that stores
a zero/nonzero mask, or a capped trailing-zero count, can beat the current GDX
bitplane route for the low raw mantissa tail.
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
from probe_grouped_tail_mlp import CONTEXT_FAMILIES, context_family_cost, kt_cost  # noqa: E402
from structural_delta_grouped_roundtrip import choose_group_payloads  # noqa: E402

RESULTS_DIR = ROOT / "results"
BODY_BITS = tuple(range(30, -1, -1))


def multinomial_kt_cost(counts: list[int], alpha: float = 0.5) -> float:
    total = sum(counts)
    if total == 0:
        return 0.0
    log_probability = math.lgamma(len(counts) * alpha) - math.lgamma(
        total + len(counts) * alpha
    )
    for count in counts:
        log_probability += math.lgamma(count + alpha) - math.lgamma(alpha)
    return -log_probability / math.log(2.0)


def best_family_cost(payload: np.ndarray, bits: range | tuple[int, ...]) -> float:
    total = 0.0
    for bit in bits:
        total += min(
            context_family_cost(payload, bit, family)
            for family in CONTEXT_FAMILIES
        )
    return total


def bitplane_kt_cost(values: np.ndarray, width: int) -> float:
    if values.size == 0 or width == 0:
        return 0.0
    total = 0.0
    flat = values.reshape(-1)
    for bit in range(width):
        ones = int(((flat >> np.uint32(bit)) & np.uint32(1)).sum())
        total += kt_cost(ones, flat.size)
    return total


def flag_context_cost(flag: np.ndarray) -> float:
    payload = flag.astype(np.uint32, copy=False)
    return min(
        context_family_cost(payload, 0, family)
        for family in CONTEXT_FAMILIES
    )


def zero_mask_route_cost(
    raw_tail: np.ndarray,
    width: int,
    route_header_bits: int,
) -> tuple[float, dict[str, float]]:
    nonzero = raw_tail != 0
    nonzero_count = int(np.count_nonzero(nonzero))
    total = int(raw_tail.size)
    flag = nonzero.astype(np.uint32)
    flag_kt = kt_cost(nonzero_count, total)
    flag_context = flag_context_cost(flag)
    selected = raw_tail[nonzero]
    raw_values = float(nonzero_count * width)
    bitplane_values = bitplane_kt_cost(selected, width)
    value_cost = min(raw_values, bitplane_values)
    cost = route_header_bits + min(flag_kt, flag_context) + value_cost
    return cost, {
        "nonzero_rate": nonzero_count / total if total else 0.0,
        "flag_kt": flag_kt,
        "flag_context": flag_context,
        "raw_values": raw_values,
        "bitplane_values": bitplane_values,
    }


def capped_trailing_zeros(values: np.ndarray, width: int) -> np.ndarray:
    flat = values.reshape(-1)
    out = np.zeros(flat.shape, dtype=np.uint8)
    active = np.ones(flat.shape, dtype=bool)
    for bit in range(width):
        zero = ((flat >> np.uint32(bit)) & np.uint32(1)) == 0
        out += (active & zero).astype(np.uint8)
        active &= zero
    return out.reshape(values.shape)


def trailing_zero_route_cost(
    raw_tail: np.ndarray,
    width: int,
    route_header_bits: int,
) -> tuple[float, dict[str, float]]:
    tzc = capped_trailing_zeros(raw_tail, width)
    counts = [int(np.count_nonzero(tzc == value)) for value in range(width + 1)]
    count_cost = multinomial_kt_cost(counts)
    remaining_raw = 0.0
    remaining_bitplane = 0.0
    for zeros in range(width):
        selected = raw_tail[tzc == zeros] >> np.uint32(zeros)
        remaining_width = width - zeros
        remaining_raw += float(selected.size * remaining_width)
        remaining_bitplane += bitplane_kt_cost(selected, remaining_width)
    cost = route_header_bits + count_cost + min(remaining_raw, remaining_bitplane)
    return cost, {
        "count_cost": count_cost,
        "remaining_raw": remaining_raw,
        "remaining_bitplane": remaining_bitplane,
        "full_zero_rate": counts[width] / raw_tail.size if raw_tail.size else 0.0,
        "mean_trailing_zeros": float(np.mean(tzc)) if tzc.size else 0.0,
    }


def tile_route_costs(
    raw_mantissa: np.ndarray,
    body_payload: np.ndarray,
    sign_payload: np.ndarray,
    low_width: int,
    route_header_bits: int,
) -> tuple[dict[str, float], dict[str, float]]:
    low_mask = np.uint32((1 << low_width) - 1)
    raw_tail = raw_mantissa & low_mask
    current_low = best_family_cost(body_payload, range(low_width))
    high = best_family_cost(body_payload, range(low_width, 31))
    sign = best_family_cost(sign_payload, (31,))
    zero_mask, zero_details = zero_mask_route_cost(
        raw_tail,
        low_width,
        route_header_bits,
    )
    trailing_zero, trailing_details = trailing_zero_route_cost(
        raw_tail,
        low_width,
        route_header_bits,
    )
    low_routes = {
        "current_low": current_low,
        "zero_mask_raw_tail": zero_mask,
        "trailing_zero_raw_tail": trailing_zero,
    }
    route, low = min(low_routes.items(), key=lambda item: item[1])
    details = {
        **{f"zero_{key}": value for key, value in zero_details.items()},
        **{f"tz_{key}": value for key, value in trailing_details.items()},
    }
    details["selected_route"] = route
    details["current_low_bits"] = current_low
    details["selected_low_bits"] = low
    return {
        "current_total": sign + high + current_low,
        "hybrid_total": sign + high + low,
        "sign": sign,
        "high": high,
        **low_routes,
    }, details


def summarize_image(
    path: Path,
    crop_size: int,
    tile_size: int,
    previous_bits: int,
    low_widths: tuple[int, ...],
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
    height, width, _channels = bits.shape
    raw_bits = int(bits.size * 32)
    width_rows = []
    for low_width in low_widths:
        totals = Counter()
        route_histogram = Counter()
        detail_totals = Counter()
        for y in range(0, height, tile_size):
            for x in range(0, width, tile_size):
                sl = np.s_[y:y + tile_size, x:x + tile_size]
                costs, details = tile_route_costs(
                    raw_mantissa[sl],
                    payloads["body"][sl],
                    payloads["sign"][sl],
                    low_width,
                    route_header_bits,
                )
                totals.update(costs)
                route_histogram[str(details["selected_route"])] += 1
                for key, value in details.items():
                    if isinstance(value, (int, float)):
                        detail_totals[key] += float(value)
        current = float(totals["current_total"])
        hybrid = float(totals["hybrid_total"])
        width_rows.append(
            {
                "low_width": low_width,
                "current_bits": current,
                "hybrid_bits": hybrid,
                "current_ratio": raw_bits / current if current else 1.0,
                "hybrid_ratio": raw_bits / hybrid if hybrid else 1.0,
                "gain": current / hybrid if hybrid else 1.0,
                "route_histogram": dict(route_histogram),
                "totals": dict(totals),
                "detail_totals": dict(detail_totals),
            }
        )
    return {
        "image": path.name,
        "shape": list(bits.shape),
        "crop_size": crop_size,
        "tile_size": tile_size,
        "previous_bits": previous_bits,
        "route_header_bits": route_header_bits,
        "rows": width_rows,
    }


def parse_widths(text: str) -> tuple[int, ...]:
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
    parser.add_argument("--low-widths", default="4,8,10,12,15")
    parser.add_argument("--route-header-bits", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    low_widths = parse_widths(args.low_widths)
    rows = []
    for path in sorted(DATA_DIR.glob(args.glob)):
        row = summarize_image(
            path,
            args.crop_size,
            args.tile_size,
            args.previous_bits,
            low_widths,
            args.route_header_bits,
        )
        rows.append(row)
        print(path.stem)
        for item in row["rows"]:
            print(
                f"  low{item['low_width']:02d}: "
                f"current={item['current_ratio']:.3f}x "
                f"hybrid={item['hybrid_ratio']:.3f}x "
                f"gain={item['gain']:.4f} routes={item['route_histogram']}"
            )
        if args.limit and len(rows) >= args.limit:
            break
    if rows:
        print("\ngeomean:")
        for low_width in low_widths:
            current_values = []
            hybrid_values = []
            for row in rows:
                item = next(r for r in row["rows"] if r["low_width"] == low_width)
                current_values.append(item["current_ratio"])
                hybrid_values.append(item["hybrid_ratio"])
            print(
                f"  low{low_width:02d}: current={geomean(current_values):.3f}x "
                f"hybrid={geomean(hybrid_values):.3f}x"
            )
    safe_glob = args.glob.replace("*", "star").replace(".", "_")
    output = RESULTS_DIR / (
        f"tail_zero_mixture_{safe_glob}_tile{args.tile_size}"
        f"_prev{args.previous_bits}_crop{args.crop_size}"
        f"_low{args.low_widths.replace(',', '-')}.json"
    )
    output.write_text(json.dumps(rows, indent=2))
    print(f"\nSaved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
