"""Probe exact adaptive low-tail coding with high-float feature contexts.

The conditional-entropy probe showed a large lower bound if low raw mantissa
tails are grouped by already-decodable high float features.  Palette/dictionary
routes lost that gain to side information.  This probe removes the dictionary:
it codes low-tail bits with adaptive KT counts in a fixed hash table.

The intended decoder-shaped route is:

1. code sign + high body bits with the current GDX payload route;
2. reconstruct exponent/high mantissa;
3. code raw low mantissa bits from high to low with contexts built from
   high float features, already decoded higher low-tail bits, and causal
   spatial/channel neighbors.
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


def geomean(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


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


def fold_hash(values: np.ndarray, bits: int) -> np.ndarray:
    mask = np.uint64((1 << bits) - 1)
    mixed = values.astype(np.uint64)
    mixed ^= mixed >> np.uint64(bits)
    mixed ^= mixed >> np.uint64(bits * 2)
    mixed *= np.uint64(0x9E3779B1)
    mixed ^= mixed >> np.uint64(16)
    return (mixed & mask).astype(np.int64)


def best_family_cost(payload: np.ndarray, bit_indices: range | tuple[int, ...]) -> float:
    total = 0.0
    for bit in bit_indices:
        total += min(
            context_family_cost(payload, bit, family)
            for family in CONTEXT_FAMILIES
        )
    return total


def context_cost(context: np.ndarray, symbols: np.ndarray, context_bits: int) -> float:
    context_count = 1 << context_bits
    context_flat = context.reshape(-1)
    symbol_flat = symbols.reshape(-1).astype(np.uint8)
    totals = np.bincount(context_flat, minlength=context_count)
    ones = np.bincount(context_flat, weights=symbol_flat, minlength=context_count)
    cost = 0.0
    active = np.nonzero(totals)[0]
    for context_id in active:
        cost += kt_cost(int(ones[context_id]), int(totals[context_id]))
    return cost


def high_feature_seed(
    bits: np.ndarray,
    tail_width: int,
    xy_bits: int,
    feature_mode: str,
) -> np.ndarray:
    height, width, channels = bits.shape
    exponent = ((bits >> np.uint32(23)) & np.uint32(0xff)).astype(np.uint64)
    mantissa = (bits & np.uint32(0x7fffff)).astype(np.uint64)
    high = mantissa >> np.uint64(tail_width)
    c = np.arange(channels, dtype=np.uint64)[None, None, :]
    channel = np.broadcast_to(c, bits.shape)
    yy, xx = np.mgrid[0:height, 0:width]
    x = np.broadcast_to(xx[:, :, None].astype(np.uint64), bits.shape)
    y = np.broadcast_to(yy[:, :, None].astype(np.uint64), bits.shape)
    if feature_mode == "channel":
        code = channel
    elif feature_mode == "exponent_channel":
        code = (exponent << np.uint64(2)) | channel
    elif feature_mode == "high4_channel":
        code = ((high & np.uint64(0x0f)) << np.uint64(2)) | channel
    elif feature_mode == "high8_channel":
        code = ((high & np.uint64(0xff)) << np.uint64(2)) | channel
    elif feature_mode == "exp_high4_channel":
        code = (
            (exponent << np.uint64(6))
            | ((high & np.uint64(0x0f)) << np.uint64(2))
            | channel
        )
    elif feature_mode == "exp_high8_channel":
        code = (
            (exponent << np.uint64(10))
            | ((high & np.uint64(0xff)) << np.uint64(2))
            | channel
        )
    elif feature_mode == "hash_full":
        code = high ^ (exponent << np.uint64(17)) ^ (channel << np.uint64(25))
    else:
        raise ValueError(f"unknown feature mode: {feature_mode}")
    code = code.copy()
    if xy_bits:
        mask = np.uint64((1 << xy_bits) - 1)
        code ^= (x & mask) << np.uint64(37)
        code ^= (y & mask) << np.uint64(37 + xy_bits)
    return code


def adaptive_low_tail_cost(
    bits: np.ndarray,
    tail_width: int,
    context_bits: int,
    xy_bits: int,
    prev_tail_bits: int,
    feature_mode: str,
    use_spatial: bool,
    use_previous_channel: bool,
) -> tuple[float, dict[str, float]]:
    mantissa = bits & np.uint32(0x7fffff)
    tail = mantissa & np.uint32((1 << tail_width) - 1)
    seed = high_feature_seed(bits, tail_width, xy_bits, feature_mode)
    total = 0.0
    active_contexts = Counter()
    for bit in range(tail_width - 1, -1, -1):
        symbols = ((tail >> np.uint32(bit)) & np.uint32(1)).astype(np.uint8)
        code = seed ^ (np.uint64(bit) << np.uint64(41))

        if prev_tail_bits > 0 and bit + 1 < tail_width:
            available = min(prev_tail_bits, tail_width - bit - 1)
            previous_tail = (tail >> np.uint32(bit + 1)) & np.uint32(
                (1 << available) - 1
            )
            code ^= previous_tail.astype(np.uint64) << np.uint64(44)

        if use_spatial:
            west = shift_plane(symbols, 0, -1).astype(np.uint64)
            north = shift_plane(symbols, -1, 0).astype(np.uint64)
            northwest = shift_plane(symbols, -1, -1).astype(np.uint64)
            northeast = shift_plane(symbols, -1, 1).astype(np.uint64)
            code ^= west << np.uint64(3)
            code ^= north << np.uint64(4)
            code ^= northwest << np.uint64(5)
            code ^= northeast << np.uint64(6)

        if use_previous_channel and bits.shape[2] > 1:
            pc = np.zeros_like(symbols, dtype=np.uint64)
            pc[..., 1:] = symbols[..., :-1]
            code ^= pc << np.uint64(7)

        context = fold_hash(code, context_bits)
        active_contexts[str(bit)] = int(np.unique(context).size)
        total += context_cost(context, symbols, context_bits)
    return total, {
        "mean_active_contexts": float(
            sum(active_contexts.values()) / max(1, len(active_contexts))
        ),
        "max_active_contexts": float(max(active_contexts.values() or [0])),
    }


def summarize_tile(
    bits_tile: np.ndarray,
    body_payload_tile: np.ndarray,
    sign_payload_tile: np.ndarray,
    tail_width: int,
    context_bits: int,
    xy_bits: int,
    prev_tail_bits: int,
    feature_mode: str,
    use_spatial: bool,
    use_previous_channel: bool,
    route_header_bits: int,
) -> tuple[dict[str, float], dict[str, float]]:
    low_gdx = best_family_cost(body_payload_tile, range(tail_width))
    high_gdx = best_family_cost(body_payload_tile, range(tail_width, 31))
    sign = best_family_cost(sign_payload_tile, (31,))
    adaptive, details = adaptive_low_tail_cost(
        bits_tile,
        tail_width,
        context_bits,
        xy_bits,
        prev_tail_bits,
        feature_mode,
        use_spatial,
        use_previous_channel,
    )
    adaptive += route_header_bits
    selected = min(low_gdx, adaptive)
    route = "adaptive_high_context" if adaptive < low_gdx else "current_low"
    return {
        "current_total": sign + high_gdx + low_gdx,
        "hybrid_total": sign + high_gdx + selected,
        "sign": sign,
        "high_gdx": high_gdx,
        "low_gdx": low_gdx,
        "adaptive_low": adaptive,
    }, {
        **details,
        "selected_route": route,
    }


def summarize_image(
    path: Path,
    crop_size: int,
    tile_size: int,
    previous_bits: int,
    tail_widths: tuple[int, ...],
    context_bits_values: tuple[int, ...],
    xy_bits: int,
    prev_tail_bits_values: tuple[int, ...],
    feature_modes: tuple[str, ...],
    use_spatial: bool,
    use_previous_channel: bool,
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
    raw_bits = int(bits.size * 32)
    rows = []
    for tail_width in tail_widths:
        best_row = None
        for context_bits in context_bits_values:
            for prev_tail_bits in prev_tail_bits_values:
                for feature_mode in feature_modes:
                    totals = Counter()
                    route_histogram = Counter()
                    detail_totals = Counter()
                    for y in range(0, bits.shape[0], tile_size):
                        for x in range(0, bits.shape[1], tile_size):
                            sl = np.s_[y:y + tile_size, x:x + tile_size]
                            costs, details = summarize_tile(
                                bits[sl],
                                payloads["body"][sl],
                                payloads["sign"][sl],
                                tail_width,
                                context_bits,
                                xy_bits,
                                prev_tail_bits,
                                feature_mode,
                                use_spatial,
                                use_previous_channel,
                                route_header_bits,
                            )
                            totals.update(costs)
                            route_histogram[str(details["selected_route"])] += 1
                            for key, value in details.items():
                                if isinstance(value, (int, float)):
                                    detail_totals[key] += float(value)
                    current = float(totals["current_total"])
                    hybrid = float(totals["hybrid_total"])
                    row = {
                        "tail_width": tail_width,
                        "context_bits": context_bits,
                        "prev_tail_bits": prev_tail_bits,
                        "feature_mode": feature_mode,
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
        "use_spatial": use_spatial,
        "use_previous_channel": use_previous_channel,
        "route_header_bits": route_header_bits,
        "rows": rows,
    }


def parse_ints(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split(",") if part)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", default="ph_*puresky*.exr")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--tile-size", type=int, default=128)
    parser.add_argument("--previous-bits", type=int, default=4)
    parser.add_argument("--tail-widths", default="8,10,12,15")
    parser.add_argument("--context-bits", default="10,12,14")
    parser.add_argument("--xy-bits", type=int, default=3)
    parser.add_argument("--prev-tail-bits", default="0,2,4,8")
    parser.add_argument(
        "--feature-modes",
        default="hash_full,exp_high8_channel,exp_high4_channel,exponent_channel",
    )
    parser.add_argument("--no-spatial", action="store_true")
    parser.add_argument("--no-previous-channel", action="store_true")
    parser.add_argument("--route-header-bits", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tail_widths = parse_ints(args.tail_widths)
    context_bits_values = parse_ints(args.context_bits)
    prev_tail_bits_values = parse_ints(args.prev_tail_bits)
    feature_modes = tuple(part for part in args.feature_modes.split(",") if part)
    rows = []
    for path in sorted(DATA_DIR.glob(args.glob)):
        row = summarize_image(
            path,
            args.crop_size,
            args.tile_size,
            args.previous_bits,
            tail_widths,
            context_bits_values,
            args.xy_bits,
            prev_tail_bits_values,
            feature_modes,
            not args.no_spatial,
            not args.no_previous_channel,
            args.route_header_bits,
        )
        rows.append(row)
        print(path.stem)
        for item in row["rows"]:
            print(
                f"  low{item['tail_width']:02d}/ctx{item['context_bits']:02d}"
                f"/prev{item['prev_tail_bits']:02d}/{item['feature_mode']}: "
                f"current={item['current_ratio']:.3f}x "
                f"hybrid={item['hybrid_ratio']:.3f}x "
                f"gain={item['gain']:.4f} routes={item['route_histogram']}"
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
        f"tail_adaptive_high_context_{safe_glob}_tile{args.tile_size}"
        f"_prev{args.previous_bits}_crop{args.crop_size}"
        f"_tail{args.tail_widths.replace(',', '-')}"
        f"_ctx{args.context_bits.replace(',', '-')}"
        f"_pt{args.prev_tail_bits.replace(',', '-')}.json"
    )
    output.write_text(json.dumps(rows, indent=2))
    print(f"\nSaved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
