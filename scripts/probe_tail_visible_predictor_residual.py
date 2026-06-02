"""Probe deterministic decoder-visible predictors for low-tail symbols.

Context entropy showed that high float fields contain information about hard
low tails, but per-context dictionaries are too expensive.  This script tries a
cheaper shape: predict the low-tail value directly from decoder-visible high
fields with fixed functions, then code only the exact residual.
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
from structural_delta_grouped_roundtrip import choose_group_payloads  # noqa: E402

RESULTS_DIR = ROOT / "results"
BODY_MASK = np.uint32(0x7FFFFFFF)


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


def shift_zero(values: np.ndarray, dy: int, dx: int) -> np.ndarray:
    out = np.zeros_like(values)
    height, width, _channels = values.shape
    dst_y0 = max(0, -dy)
    dst_y1 = min(height, height - dy)
    dst_x0 = max(0, -dx)
    dst_x1 = min(width, width - dx)
    src_y0 = dst_y0 + dy
    src_y1 = dst_y1 + dy
    src_x0 = dst_x0 + dx
    src_x1 = dst_x1 + dx
    out[dst_y0:dst_y1, dst_x0:dst_x1] = values[src_y0:src_y1, src_x0:src_x1]
    return out


def fold_hash(values: np.ndarray, width: int) -> np.ndarray:
    mask = np.uint64((1 << width) - 1)
    mixed = values.astype(np.uint64)
    mixed ^= mixed >> np.uint64(17)
    mixed *= np.uint64(0x9E3779B185EBCA87)
    mixed ^= mixed >> np.uint64(31)
    mixed *= np.uint64(0xC2B2AE3D27D4EB4F)
    mixed ^= mixed >> np.uint64(29)
    return (mixed & mask).astype(np.uint32)


def tail_from_source(
    bits: np.ndarray,
    ordered: np.ndarray,
    body_payload: np.ndarray,
    tail_width: int,
    tail_source: str,
) -> np.ndarray:
    mask = np.uint32((1 << tail_width) - 1)
    if tail_source == "raw_mantissa":
        return (bits & np.uint32(0x7FFFFF)) & mask
    if tail_source == "body_payload":
        return body_payload & mask
    if tail_source == "ordered_low":
        return ordered & mask
    raise ValueError(f"unknown tail source: {tail_source}")


def visible_predictors(
    bits: np.ndarray,
    ordered: np.ndarray,
    body_payload: np.ndarray,
    tail_width: int,
) -> dict[str, np.ndarray]:
    mask = np.uint32((1 << tail_width) - 1)
    height, width, channels = bits.shape
    mantissa = (bits & np.uint32(0x7FFFFF)).astype(np.uint64)
    exponent = ((bits >> np.uint32(23)) & np.uint32(0xFF)).astype(np.uint64)
    ordered_body = (ordered & BODY_MASK).astype(np.uint64)
    payload_body = (body_payload & BODY_MASK).astype(np.uint64)
    mantissa_high = mantissa >> np.uint64(tail_width)
    ordered_high = ordered_body >> np.uint64(tail_width)
    payload_high = payload_body >> np.uint64(tail_width)

    yy, xx = np.mgrid[0:height, 0:width]
    x = np.broadcast_to(xx[:, :, None].astype(np.uint64), bits.shape)
    y = np.broadcast_to(yy[:, :, None].astype(np.uint64), bits.shape)
    c = np.arange(channels, dtype=np.uint64)[None, None, :]
    channel = np.broadcast_to(c, bits.shape)

    predictors: dict[str, np.ndarray] = {
        "zero": np.zeros(bits.shape, dtype=np.uint32),
        "mantissa_high_low": (mantissa_high & np.uint64(mask)).astype(np.uint32),
        "ordered_high_low": (ordered_high & np.uint64(mask)).astype(np.uint32),
        "payload_high_low": (payload_high & np.uint64(mask)).astype(np.uint32),
        "mantissa_high_hash": fold_hash(mantissa_high, tail_width),
        "ordered_high_hash": fold_hash(ordered_high, tail_width),
        "payload_high_hash": fold_hash(payload_high, tail_width),
        "exp_mantissa_hash": fold_hash(
            mantissa_high ^ (exponent << np.uint64(19)) ^ (channel << np.uint64(27)),
            tail_width,
        ),
        "exp_payload_hash": fold_hash(
            payload_high ^ (exponent << np.uint64(19)) ^ (channel << np.uint64(27)),
            tail_width,
        ),
        "xy_exp_payload_hash": fold_hash(
            payload_high
            ^ (exponent << np.uint64(19))
            ^ (channel << np.uint64(27))
            ^ ((x & np.uint64(15)) << np.uint64(31))
            ^ ((y & np.uint64(15)) << np.uint64(35)),
            tail_width,
        ),
    }

    west_payload = shift_zero(payload_high, 0, -1)
    north_payload = shift_zero(payload_high, -1, 0)
    predictors["west_payload_high_low"] = (west_payload & np.uint64(mask)).astype(
        np.uint32,
    )
    predictors["north_payload_high_low"] = (north_payload & np.uint64(mask)).astype(
        np.uint32,
    )
    predictors["avg_payload_high_low"] = (
        ((west_payload + north_payload) // np.uint64(2)) & np.uint64(mask)
    ).astype(np.uint32)

    previous_channel = np.zeros(bits.shape, dtype=np.uint64)
    if channels > 1:
        previous_channel[..., 1:] = payload_high[..., :-1]
    predictors["previous_channel_payload_high_low"] = (
        previous_channel & np.uint64(mask)
    ).astype(np.uint32)

    green = np.zeros(bits.shape, dtype=np.uint64)
    if channels >= 3:
        green[..., 0] = payload_high[..., 1]
        green[..., 1] = payload_high[..., 1]
        green[..., 2] = payload_high[..., 1]
        if channels > 3:
            green[..., 3:] = payload_high[..., 3:]
    predictors["green_payload_high_low"] = (green & np.uint64(mask)).astype(np.uint32)
    return predictors


def residual_payload(
    tail: np.ndarray,
    prediction: np.ndarray,
    tail_width: int,
    residual_kind: str,
) -> np.ndarray:
    mask = np.uint32((1 << tail_width) - 1)
    if residual_kind == "sub":
        return ((tail - prediction) & mask).astype(np.uint32)
    if residual_kind == "xor":
        return (tail ^ prediction).astype(np.uint32)
    raise ValueError(f"unknown residual kind: {residual_kind}")


def summarize_image(
    path: Path,
    crop_size: int,
    tile_size: int,
    previous_bits: int,
    tail_widths: tuple[int, ...],
    tail_source: str,
    residual_kinds: tuple[str, ...],
    route_header_bits: int,
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
        totals = Counter()
        selected = Counter()
        candidate_totals = Counter()
        for y in range(0, bits.shape[0], tile_size):
            for x in range(0, bits.shape[1], tile_size):
                sl = np.s_[y:y + tile_size, x:x + tile_size]
                body_tile = payloads["body"][sl]
                sign_tile = payloads["sign"][sl]
                sign = best_family_cost(sign_tile, (31,))
                high = best_family_cost(body_tile, range(tail_width, 31))
                current_low = best_family_cost(body_tile, range(tail_width))
                current_total = sign + high + current_low
                tail = tail_from_source(
                    bits[sl],
                    ordered[sl],
                    body_tile,
                    tail_width,
                    tail_source,
                )

                best_name = "current_low"
                best_low = current_low
                for predictor_name, prediction in visible_predictors(
                    bits[sl],
                    ordered[sl],
                    body_tile,
                    tail_width,
                ).items():
                    for residual_kind in residual_kinds:
                        if predictor_name == "zero" and residual_kind == "xor":
                            continue
                        residual = residual_payload(
                            tail,
                            prediction,
                            tail_width,
                            residual_kind,
                        )
                        route_bits = (
                            best_family_cost(residual, range(tail_width))
                            + route_header_bits
                        )
                        name = f"{predictor_name}_{residual_kind}"
                        candidate_totals[name] += sign + high + route_bits
                        if route_bits < best_low:
                            best_name = name
                            best_low = route_bits

                selected[best_name] += 1
                totals["current_total"] += current_total
                totals["hybrid_total"] += sign + high + best_low
                totals["sign"] += sign
                totals["high"] += high
                totals["current_low"] += current_low

        current_bits = float(totals["current_total"])
        hybrid_bits = float(totals["hybrid_total"])
        rows.append(
            {
                "tail_width": tail_width,
                "current_bits": current_bits,
                "hybrid_bits": hybrid_bits,
                "current_ratio": raw_bits / current_bits if current_bits else 1.0,
                "hybrid_ratio": raw_bits / hybrid_bits if hybrid_bits else 1.0,
                "gain": current_bits / hybrid_bits if hybrid_bits else 1.0,
                "selected": dict(selected),
                "totals": dict(totals),
                "candidate_ratios": {
                    name: raw_bits / value for name, value in candidate_totals.items()
                },
            },
        )
    return {
        "image": path.name,
        "shape": list(bits.shape),
        "crop_size": crop_size,
        "tile_size": tile_size,
        "previous_bits": previous_bits,
        "tail_source": tail_source,
        "residual_kinds": list(residual_kinds),
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
    parser.add_argument(
        "--tail-source",
        choices=("raw_mantissa", "body_payload", "ordered_low"),
        default="body_payload",
    )
    parser.add_argument("--residual-kinds", default="sub,xor")
    parser.add_argument("--route-header-bits", type=int, default=8)
    parser.add_argument("--no-save", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tail_widths = parse_ints(args.tail_widths)
    residual_kinds = parse_words(args.residual_kinds)
    rows = []
    for path in sorted(DATA_DIR.glob(args.glob)):
        row = summarize_image(
            path,
            args.crop_size,
            args.tile_size,
            args.previous_bits,
            tail_widths,
            args.tail_source,
            residual_kinds,
            args.route_header_bits,
        )
        rows.append(row)
        print(path.stem)
        for item in row["rows"]:
            best_candidates = sorted(
                item["candidate_ratios"].items(),
                key=lambda entry: entry[1],
                reverse=True,
            )[:3]
            best_text = " ".join(
                f"{name}={ratio:.3f}x" for name, ratio in best_candidates
            )
            print(
                f"  low{item['tail_width']:02d}: "
                f"current={item['current_ratio']:.3f}x "
                f"hybrid={item['hybrid_ratio']:.3f}x "
                f"gain={item['gain']:.4f} selected={item['selected']} "
                f"{best_text}"
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
            f"tail_visible_predictor_residual_{safe_glob}_tile{args.tile_size}"
            f"_prev{args.previous_bits}_crop{args.crop_size}"
            f"_tail{args.tail_widths.replace(',', '-')}_{args.tail_source}.json"
        )
        output.write_text(json.dumps(rows, indent=2))
        print(f"\nSaved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
