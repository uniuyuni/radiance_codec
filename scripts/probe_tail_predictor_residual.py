"""Probe exact low-tail predictor residual routes.

The context/table probes showed that low-tail values have conditional structure,
but transmitting per-context tables loses most of the gain.  This probe tries a
cheaper exact shape: predict the low ordered-body tail from decoder-visible
causal tail values, then code only the residual with the existing bitplane
context-family estimate.
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


def shift_tail(tail: np.ndarray, dy: int, dx: int) -> np.ndarray:
    out = np.zeros_like(tail)
    height, width, _channels = tail.shape
    dst_y0 = max(0, -dy)
    dst_y1 = min(height, height - dy)
    dst_x0 = max(0, -dx)
    dst_x1 = min(width, width - dx)
    src_y0 = dst_y0 + dy
    src_y1 = dst_y1 + dy
    src_x0 = dst_x0 + dx
    src_x1 = dst_x1 + dx
    out[dst_y0:dst_y1, dst_x0:dst_x1] = tail[src_y0:src_y1, src_x0:src_x1]
    return out


def paeth(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    p = a.astype(np.int64) + b.astype(np.int64) - c.astype(np.int64)
    pa = np.abs(p - a.astype(np.int64))
    pb = np.abs(p - b.astype(np.int64))
    pc = np.abs(p - c.astype(np.int64))
    return np.where((pa <= pb) & (pa <= pc), a, np.where(pb <= pc, b, c)).astype(
        np.uint32,
    )


def tail_prediction(tail: np.ndarray, mode: str, mask: np.uint32) -> np.ndarray:
    if mode == "zero":
        return np.zeros_like(tail, dtype=np.uint32)

    west = shift_tail(tail, 0, -1)
    north = shift_tail(tail, -1, 0)
    northwest = shift_tail(tail, -1, -1)
    if mode == "west":
        return west
    if mode == "north":
        return north
    if mode == "average":
        return (((west.astype(np.uint64) + north.astype(np.uint64)) // 2) & mask).astype(
            np.uint32,
        )
    if mode == "gradient":
        pred = west.astype(np.int64) + north.astype(np.int64) - northwest.astype(np.int64)
        return (pred & int(mask)).astype(np.uint32)
    if mode == "paeth":
        return paeth(west, north, northwest)
    if mode == "median":
        pred = west.astype(np.int64) + north.astype(np.int64) - northwest.astype(np.int64)
        lo = np.minimum(west, north).astype(np.int64)
        hi = np.maximum(west, north).astype(np.int64)
        return np.clip(pred, lo, hi).astype(np.uint32)
    if mode == "previous_channel":
        pred = np.zeros_like(tail, dtype=np.uint32)
        if tail.shape[2] > 1:
            pred[..., 1:] = tail[..., :-1]
        return pred
    if mode == "green":
        pred = np.zeros_like(tail, dtype=np.uint32)
        if tail.shape[2] >= 3:
            pred[..., 0] = tail[..., 1]
            pred[..., 2] = tail[..., 1]
        return pred
    raise ValueError(f"unknown predictor: {mode}")


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


def candidate_payloads(
    ordered_tail: np.ndarray,
    tail_width: int,
    modes: tuple[str, ...],
    residual_kinds: tuple[str, ...],
) -> dict[str, np.ndarray]:
    mask = np.uint32((1 << tail_width) - 1)
    out: dict[str, np.ndarray] = {}
    for mode in modes:
        prediction = tail_prediction(ordered_tail, mode, mask)
        for residual_kind in residual_kinds:
            if mode == "zero" and residual_kind == "xor":
                continue
            out[f"{mode}_{residual_kind}"] = residual_payload(
                ordered_tail,
                prediction,
                tail_width,
                residual_kind,
            )
    return out


def summarize_image(
    path: Path,
    crop_size: int,
    tile_size: int,
    previous_bits: int,
    tail_widths: tuple[int, ...],
    predictor_modes: tuple[str, ...],
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
        low_mask = np.uint32((1 << tail_width) - 1)
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

                best_name = "current_low"
                best_low = current_low
                ordered_tail = (ordered[sl] & low_mask).astype(np.uint32)
                for name, payload in candidate_payloads(
                    ordered_tail,
                    tail_width,
                    predictor_modes,
                    residual_kinds,
                ).items():
                    route_bits = (
                        best_family_cost(payload, range(tail_width))
                        + route_header_bits
                    )
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
        "predictor_modes": list(predictor_modes),
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
        "--predictors",
        default="zero,west,north,average,gradient,median,paeth,previous_channel,green",
    )
    parser.add_argument("--residual-kinds", default="sub,xor")
    parser.add_argument("--route-header-bits", type=int, default=8)
    parser.add_argument("--no-save", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tail_widths = parse_ints(args.tail_widths)
    predictor_modes = parse_words(args.predictors)
    residual_kinds = parse_words(args.residual_kinds)
    rows = []
    for path in sorted(DATA_DIR.glob(args.glob)):
        row = summarize_image(
            path,
            args.crop_size,
            args.tile_size,
            args.previous_bits,
            tail_widths,
            predictor_modes,
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
            f"tail_predictor_residual_{safe_glob}_tile{args.tile_size}"
            f"_prev{args.previous_bits}_crop{args.crop_size}"
            f"_tail{args.tail_widths.replace(',', '-')}.json"
        )
        output.write_text(json.dumps(rows, indent=2))
        print(f"\nSaved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
