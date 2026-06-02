"""Probe exact low-tail reversible 2x2 transform routes.

This keeps the current grouped-delta sign/high-body coding and only replaces
the low ordered-body tail with a reversible 2x2 pyramid transform.  It is a
tail-only version of the ordered-body block transform probe.
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
from probe_grouped_tail_mlp import CONTEXT_FAMILIES, context_family_cost  # noqa: E402
from probe_ordered_body_block_transform import (  # noqa: E402
    inverse_2x2_pyramid,
    transform_2x2_pyramid,
)
from structural_delta_grouped_roundtrip import choose_group_payloads  # noqa: E402

RESULTS_DIR = ROOT / "results"


def geomean(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def best_bitplane_cost(payload: np.ndarray, bit: int) -> tuple[str, float]:
    return min(
        ((family, context_family_cost(payload, bit, family)) for family in CONTEXT_FAMILIES),
        key=lambda item: item[1],
    )


def cost_bit_range(payload: np.ndarray, bit_start: int, bit_end: int) -> tuple[float, Counter[str]]:
    total = 0.0
    picks: Counter[str] = Counter()
    for bit in range(bit_end, bit_start - 1, -1):
        family, cost = best_bitplane_cost(payload, bit)
        total += cost
        picks[family] += 1
    return total, picks


def tail_source_payload(
    ordered_tile: np.ndarray,
    body_payload_tile: np.ndarray,
    tail_width: int,
    source: str,
) -> np.ndarray:
    mask = np.uint32((1 << tail_width) - 1)
    if source == "ordered_tail":
        return (ordered_tile & mask).astype(np.uint32)
    if source == "body_payload_low":
        return (body_payload_tile & mask).astype(np.uint32)
    raise ValueError(f"unknown tail source: {source}")


def transformed_tail_payload(
    tail: np.ndarray,
    tail_width: int,
    max_levels: int,
    lowpass: str,
) -> tuple[np.ndarray, int]:
    transformed, levels = transform_2x2_pyramid(
        tail,
        tail_width,
        max_levels,
        lowpass,
    )
    restored = inverse_2x2_pyramid(transformed, tail_width, levels, lowpass)
    if not np.array_equal(restored, tail):
        mismatch = np.argwhere(restored != tail)[0]
        y, x, c = (int(value) for value in mismatch)
        raise RuntimeError(
            f"tail transform inverse mismatch at y={y} x={x} c={c}: "
            f"got=0x{int(restored[y, x, c]):08x} "
            f"expected=0x{int(tail[y, x, c]):08x}",
        )
    return transformed, len(levels)


def summarize_image(
    path: Path,
    crop_size: int,
    tile_size: int,
    previous_bits: int,
    tail_widths: tuple[int, ...],
    max_levels: int,
    lowpass: str,
    tail_sources: tuple[str, ...],
    route_header_bits: float,
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

    raw_bits = int(bits.size * 32)
    rows = []
    for tail_width in tail_widths:
        best_row = None
        for tail_source in tail_sources:
            totals = Counter()
            selected = Counter()
            level_histogram = Counter()
            transform_family_histogram = Counter()
            for y in range(0, bits.shape[0], tile_size):
                for x in range(0, bits.shape[1], tile_size):
                    sl = np.s_[y:y + tile_size, x:x + tile_size]
                    body_tile = payloads["body"][sl]
                    sign_tile = payloads["sign"][sl]
                    sign, _sign_picks = cost_bit_range(sign_tile, 31, 31)
                    high, _high_picks = cost_bit_range(body_tile, tail_width, 30)
                    current_low, _current_picks = cost_bit_range(
                        body_tile,
                        0,
                        tail_width - 1,
                    )
                    tail = tail_source_payload(
                        ordered[sl],
                        body_tile,
                        tail_width,
                        tail_source,
                    )
                    transformed, levels = transformed_tail_payload(
                        tail,
                        tail_width,
                        max_levels,
                        lowpass,
                    )
                    transformed_low, transform_picks = cost_bit_range(
                        transformed,
                        0,
                        tail_width - 1,
                    )
                    route_low = transformed_low + route_header_bits
                    if route_low < current_low:
                        selected["tail_transform"] += 1
                        selected_low = route_low
                        transform_family_histogram.update(transform_picks)
                    else:
                        selected["current_low"] += 1
                        selected_low = current_low
                    totals["current_total"] += sign + high + current_low
                    totals["hybrid_total"] += sign + high + selected_low
                    totals["current_low"] += current_low
                    totals["transformed_low"] += route_low
                    totals["sign"] += sign
                    totals["high"] += high
                    level_histogram[str(levels)] += 1

            current_bits = float(totals["current_total"])
            hybrid_bits = float(totals["hybrid_total"])
            row = {
                "tail_width": tail_width,
                "tail_source": tail_source,
                "max_levels": max_levels,
                "lowpass": lowpass,
                "current_bits": current_bits,
                "hybrid_bits": hybrid_bits,
                "current_ratio": raw_bits / current_bits if current_bits else 1.0,
                "hybrid_ratio": raw_bits / hybrid_bits if hybrid_bits else 1.0,
                "gain": current_bits / hybrid_bits if hybrid_bits else 1.0,
                "selected": dict(selected),
                "totals": dict(totals),
                "level_histogram": dict(level_histogram),
                "transform_family_histogram": dict(transform_family_histogram),
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
        "route_header_bits": route_header_bits,
        "rows": rows,
    }


def parse_ints(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split(",") if part)


def parse_words(text: str) -> tuple[str, ...]:
    return tuple(part for part in text.split(",") if part)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", default="sample_hilberts-mill-conference-room_2K.exr")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--tile-size", type=int, default=128)
    parser.add_argument("--previous-bits", type=int, default=4)
    parser.add_argument("--tail-widths", default="8,10,12,15")
    parser.add_argument("--max-levels", type=int, default=4)
    parser.add_argument("--lowpass", choices=("anchor", "average"), default="average")
    parser.add_argument(
        "--tail-sources",
        default="ordered_tail,body_payload_low",
        help="Comma-separated sources: ordered_tail,body_payload_low",
    )
    parser.add_argument("--route-header-bits", type=float, default=16.0)
    parser.add_argument("--no-save", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tail_widths = parse_ints(args.tail_widths)
    tail_sources = parse_words(args.tail_sources)
    rows = []
    for path in sorted(DATA_DIR.glob(args.glob)):
        row = summarize_image(
            path,
            args.crop_size,
            args.tile_size,
            args.previous_bits,
            tail_widths,
            args.max_levels,
            args.lowpass,
            tail_sources,
            args.route_header_bits,
        )
        rows.append(row)
        print(path.stem)
        for item in row["rows"]:
            print(
                f"  low{item['tail_width']:02d}/{item['tail_source']}: "
                f"current={item['current_ratio']:.3f}x "
                f"hybrid={item['hybrid_ratio']:.3f}x "
                f"gain={item['gain']:.4f} selected={item['selected']}"
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

    if not args.no_save:
        safe_glob = args.glob.replace("*", "star").replace(".", "_")
        output = RESULTS_DIR / (
            f"tail_wavelet_transform_{safe_glob}_tile{args.tile_size}"
            f"_prev{args.previous_bits}_crop{args.crop_size}"
            f"_tail{args.tail_widths.replace(',', '-')}_{args.lowpass}.json"
        )
        output.write_text(json.dumps(rows, indent=2))
        print(f"\nSaved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
