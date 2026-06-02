"""Probe a decoder-shaped high-body/low-tail split.

Earlier zero-mask probes treated the raw low mantissa tail as if it could be
separated from the current grouped body residual for free.  This script checks
the actual representation claim:

* code the high body bits using the already selected grouped-delta body mode;
* code the low ordered-body tail directly, using exact low-tail routes;
* reconstruct the body by combining predictor high bits, encoded high residual,
  and the decoded low tail.

For delta modes the missing low residual is not a problem: the decoder can
derive the low-part borrow from the decoded low tail and the predictor low tail.
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
from probe_grouped_tail_mlp import kt_cost  # noqa: E402
from probe_tail_zero_mixture import trailing_zero_route_cost, zero_mask_route_cost  # noqa: E402
from structural_delta_field_roundtrip import predictor_scalar  # noqa: E402
from structural_delta_grouped_roundtrip import choose_group_payloads, decode_grouped  # noqa: E402

RESULTS_DIR = ROOT / "results"
BODY_MASK = np.uint32(0x7fffffff)


def geomean(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def best_family_cost(payload: np.ndarray, bit_indices: range | tuple[int, ...]) -> float:
    total = 0.0
    for bit in bit_indices:
        total += min(
            context_family_cost(payload, bit, family)
            for family in CONTEXT_FAMILIES
        )
    return total


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


def masked_tail_context_cost(
    tail: np.ndarray,
    active: np.ndarray,
    width: int,
    previous_bits: int,
) -> float:
    context_count = 16 << previous_bits
    total_cost = 0.0
    decoded_bits = np.zeros_like(tail, dtype=np.uint32)
    active_flat = active.reshape(-1)
    for bit in range(width):
        plane = ((tail >> np.uint32(bit)) & np.uint32(1)).astype(np.uint8)
        west = shift_plane(plane, 0, -1)
        north = shift_plane(plane, -1, 0)
        northwest = shift_plane(plane, -1, -1)
        northeast = shift_plane(plane, -1, 1)
        context = (
            west.astype(np.uint16)
            | (north.astype(np.uint16) << 1)
            | (northwest.astype(np.uint16) << 2)
            | (northeast.astype(np.uint16) << 3)
        )
        for offset in range(1, previous_bits + 1):
            previous_bit = bit - offset
            if previous_bit < 0:
                continue
            previous = ((decoded_bits >> np.uint32(previous_bit)) & 1).astype(
                np.uint16
            )
            context |= previous << (3 + offset)
        context_flat = context.reshape(-1)[active_flat]
        plane_flat = plane.reshape(-1)[active_flat]
        totals = np.bincount(context_flat, minlength=context_count)
        ones = np.bincount(context_flat, weights=plane_flat, minlength=context_count)
        for context_id in np.nonzero(totals)[0]:
            total = int(totals[context_id])
            total_cost += kt_cost(int(ones[context_id]), total)
        decoded_bits |= plane.astype(np.uint32) << np.uint32(bit)
    return total_cost


def decode_split_ordered(
    modes: dict[tuple[int, int, str], str],
    body_payload: np.ndarray,
    sign_payload: np.ndarray,
    carries: dict[str, np.ndarray],
    ordered_low_tail: np.ndarray,
    shape: tuple[int, int, int],
    tile_size: int,
    tail_width: int,
) -> np.ndarray:
    height, width, channels = shape
    low_base = 1 << tail_width
    low_mask = low_base - 1
    high_mask = (1 << (31 - tail_width)) - 1
    decoded = np.zeros(shape, dtype=np.uint32)
    for y in range(height):
        tile_y = (y // tile_size) * tile_size
        for x in range(width):
            tile_x = (x // tile_size) * tile_size
            body_mode = modes[(tile_y, tile_x, "body")]
            if body_mode.endswith("channel_green") and channels >= 3:
                channel_order = (1, 0, 2, 3)[:channels]
            else:
                channel_order = tuple(range(channels))
            for c in channel_order:
                low = int(ordered_low_tail[y, x, c]) & low_mask
                payload_high = (int(body_payload[y, x, c]) >> tail_width) & high_mask
                if body_mode == "raw":
                    high = payload_high
                elif body_mode.startswith("xor_"):
                    pred = predictor_scalar(
                        body_mode[4:],
                        decoded,
                        y,
                        x,
                        c,
                    ) & int(BODY_MASK)
                    pred_high = (pred >> tail_width) & high_mask
                    high = pred_high ^ payload_high
                else:
                    pred_mode = (
                        body_mode[6:] if body_mode.startswith("local_") else body_mode
                    )
                    pred = predictor_scalar(pred_mode, decoded, y, x, c) & int(BODY_MASK)
                    pred_low = pred & low_mask
                    pred_high = (pred >> tail_width) & high_mask
                    borrow = 1 if low < pred_low else 0
                    high = (pred_high + payload_high + borrow) & high_mask
                decoded[y, x, c] = np.uint32((high << tail_width) | low)

            sign_mode = modes[(tile_y, tile_x, "sign")]
            for c in range(channels):
                payload_digit = (int(sign_payload[y, x, c]) >> 31) & 1
                if sign_mode == "raw":
                    digit = payload_digit
                elif sign_mode.startswith("xor_"):
                    pred = predictor_scalar(sign_mode[4:], decoded, y, x, c)
                    digit = ((pred >> 31) & 1) ^ payload_digit
                else:
                    pred_mode = (
                        sign_mode[6:]
                        if sign_mode.startswith("local_")
                        else sign_mode
                    )
                    pred = predictor_scalar(pred_mode, decoded, y, x, c)
                    carry = int(carries["sign"][y, x, c])
                    digit = (((pred >> 31) & 1) + payload_digit + carry) & 1
                decoded[y, x, c] = (
                    decoded[y, x, c] & np.uint32(0x7fffffff)
                ) | np.uint32(digit << 31)
    return decoded


def summarize_image(
    path: Path,
    crop_size: int,
    tile_size: int,
    previous_bits: int,
    tail_widths: tuple[int, ...],
    route_header_bits: int,
) -> dict:
    pixels = read_exr(path)
    if crop_size:
        pixels = pixels[:crop_size, :crop_size]
    pixels = np.ascontiguousarray(pixels, dtype=np.float32)
    bits = float_to_bits(pixels).copy()
    ordered = bits_to_ordered(bits)
    modes, payloads, carries = choose_group_payloads(
        bits,
        pixels,
        tile_size,
        previous_bits,
        "body",
    )
    reference = decode_grouped(
        modes,
        payloads,
        carries,
        ordered.shape,
        tile_size,
        "body",
    )
    if not np.array_equal(reference, ordered):
        raise RuntimeError(f"{path.name}: baseline grouped decode mismatch")

    raw_bits = int(bits.size * 32)
    rows = []
    for tail_width in tail_widths:
        low_mask = np.uint32((1 << tail_width) - 1)
        ordered_tail = ordered & low_mask
        body_with_low_zero = payloads["body"] & np.uint32(0x7fffffff ^ int(low_mask))
        split_ordered = decode_split_ordered(
            modes,
            body_with_low_zero,
            payloads["sign"],
            carries,
            ordered_tail,
            ordered.shape,
            tile_size,
            tail_width,
        )
        if not np.array_equal(split_ordered, ordered):
            mismatch = np.argwhere(split_ordered != ordered)[0]
            y, x, c = (int(value) for value in mismatch)
            raise RuntimeError(
                f"{path.name}: split mismatch tail={tail_width} "
                f"y={y} x={x} c={c} "
                f"got=0x{int(split_ordered[y, x, c]):08x} "
                f"want=0x{int(ordered[y, x, c]):08x}"
            )

        totals = Counter()
        routes = Counter()
        for y in range(0, ordered.shape[0], tile_size):
            for x in range(0, ordered.shape[1], tile_size):
                sl = np.s_[y:y + tile_size, x:x + tile_size]
                body_tile = payloads["body"][sl]
                sign_tile = payloads["sign"][sl]
                tail_tile = ordered_tail[sl]
                sign = best_family_cost(sign_tile, (31,))
                current_low = best_family_cost(body_tile, range(tail_width))
                current_high = best_family_cost(body_tile, range(tail_width, 31))
                zero_mask, _zero_details = zero_mask_route_cost(
                    tail_tile,
                    tail_width,
                    route_header_bits,
                )
                nonzero = tail_tile != 0
                zero_context = (
                    route_header_bits
                    + best_family_cost(nonzero.astype(np.uint32), (0,))
                    + masked_tail_context_cost(
                        tail_tile,
                        nonzero,
                        tail_width,
                        previous_bits,
                    )
                )
                trailing_zero, _trailing_details = trailing_zero_route_cost(
                    tail_tile,
                    tail_width,
                    route_header_bits,
                )
                low_routes = {
                    "current_low": current_low,
                    "zero_mask_ordered_tail": zero_mask,
                    "zero_mask_context_ordered_tail": zero_context,
                    "trailing_zero_ordered_tail": trailing_zero,
                }
                route, split_low = min(low_routes.items(), key=lambda item: item[1])
                routes[route] += 1
                totals["sign"] += sign
                totals["current_high"] += current_high
                totals["current_low"] += current_low
                totals["split_low"] += split_low
                totals["current_total"] += sign + current_high + current_low
                totals["split_total"] += sign + current_high + split_low
                for name, cost in low_routes.items():
                    totals[name] += cost
        current = float(totals["current_total"])
        split = float(totals["split_total"])
        rows.append(
            {
                "tail_width": tail_width,
                "current_bits": current,
                "split_bits": split,
                "current_ratio": raw_bits / current if current else 1.0,
                "split_ratio": raw_bits / split if split else 1.0,
                "gain": current / split if split else 1.0,
                "route_histogram": dict(routes),
                "totals": dict(totals),
            }
        )
    return {
        "image": path.name,
        "shape": list(ordered.shape),
        "crop_size": crop_size,
        "tile_size": tile_size,
        "previous_bits": previous_bits,
        "route_header_bits": route_header_bits,
        "rows": rows,
    }


def parse_ints(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split(",") if part)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", default="ph_*puresky*.exr")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--tile-size", type=int, default=128)
    parser.add_argument("--previous-bits", type=int, default=4)
    parser.add_argument("--tail-widths", default="15")
    parser.add_argument("--route-header-bits", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tail_widths = parse_ints(args.tail_widths)
    rows = []
    for path in sorted(DATA_DIR.glob(args.glob)):
        row = summarize_image(
            path,
            args.crop_size,
            args.tile_size,
            args.previous_bits,
            tail_widths,
            args.route_header_bits,
        )
        rows.append(row)
        print(path.stem)
        for item in row["rows"]:
            print(
                f"  low{item['tail_width']:02d}: current={item['current_ratio']:.3f}x "
                f"split={item['split_ratio']:.3f}x gain={item['gain']:.4f} "
                f"routes={item['route_histogram']}"
            )
        if args.limit and len(rows) >= args.limit:
            break
    if rows:
        print("\ngeomean:")
        for tail_width in tail_widths:
            current_values = []
            split_values = []
            for row in rows:
                item = next(r for r in row["rows"] if r["tail_width"] == tail_width)
                current_values.append(item["current_ratio"])
                split_values.append(item["split_ratio"])
            print(
                f"  low{tail_width:02d}: current={geomean(current_values):.3f}x "
                f"split={geomean(split_values):.3f}x"
            )
    safe_glob = args.glob.replace("*", "star").replace(".", "_")
    output = RESULTS_DIR / (
        f"low_tail_split_roundtrip_{safe_glob}_tile{args.tile_size}"
        f"_prev{args.previous_bits}_crop{args.crop_size}"
        f"_tail{args.tail_widths.replace(',', '-')}.json"
    )
    output.write_text(json.dumps(rows, indent=2))
    print(f"\nSaved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
