"""Probe exact low-tail residuals from tile surface predictions.

The full-body surface probe lost: fitting an affine/quadratic surface to the
whole ordered float body did not beat the current GDX route once residuals and
coefficients were coded.  This narrower probe keeps the current high-body route
and only replaces the raw low mantissa/body tail with a modular residual against
the low bits of a transmitted tile surface.  It targets smooth puresky tiles
where the high body already carries the coarse gradient but the low tail may
still follow the same smooth surface.
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
from probe_tail_zero_mixture import trailing_zero_route_cost, zero_mask_route_cost  # noqa: E402
from probe_tile_surface_predictors import (  # noqa: E402
    BODY_MASK,
    coeff_overhead_bits,
    float_surface_prediction,
    ordered_surface_prediction,
    positive_log_surface_prediction,
)
from structural_delta_grouped_roundtrip import choose_group_payloads  # noqa: E402

RESULTS_DIR = ROOT / "results"


def geomean(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def best_bitplane_cost(payload: np.ndarray, bit_indices: range | tuple[int, ...]) -> float:
    total = 0.0
    for bit in bit_indices:
        total += min(
            context_family_cost(payload, bit, family)
            for family in CONTEXT_FAMILIES
        )
    return total


def low_tail_residual_payload(
    ordered_tile: np.ndarray,
    prediction: np.ndarray,
    tail_width: int,
) -> np.ndarray:
    mask = np.uint32((1 << tail_width) - 1)
    actual_tail = (ordered_tile & BODY_MASK) & mask
    pred_tail = (prediction & BODY_MASK) & mask
    return ((actual_tail - pred_tail) & mask).astype(np.uint32)


def high_corner_affine_prediction(high_ordered: np.ndarray) -> np.ndarray:
    height, width, channels = high_ordered.shape
    yy, xx = np.mgrid[0:height, 0:width]
    x = xx.astype(np.float64) / max(1, width - 1)
    y = yy.astype(np.float64) / max(1, height - 1)
    pred = np.zeros_like(high_ordered, dtype=np.uint32)
    for channel in range(channels):
        plane = high_ordered[:, :, channel].astype(np.float64)
        p00 = plane[0, 0]
        p10 = plane[0, width - 1]
        p01 = plane[height - 1, 0]
        values = p00 + (p10 - p00) * x + (p01 - p00) * y
        pred[:, :, channel] = np.clip(np.rint(values), 0, 0xffffffff).astype(np.uint32)
    return pred


def high_corner_bilinear_prediction(high_ordered: np.ndarray) -> np.ndarray:
    height, width, channels = high_ordered.shape
    yy, xx = np.mgrid[0:height, 0:width]
    x = xx.astype(np.float64) / max(1, width - 1)
    y = yy.astype(np.float64) / max(1, height - 1)
    pred = np.zeros_like(high_ordered, dtype=np.uint32)
    for channel in range(channels):
        plane = high_ordered[:, :, channel].astype(np.float64)
        p00 = plane[0, 0]
        p10 = plane[0, width - 1]
        p01 = plane[height - 1, 0]
        p11 = plane[height - 1, width - 1]
        values = (
            p00 * (1.0 - x) * (1.0 - y)
            + p10 * x * (1.0 - y)
            + p01 * (1.0 - x) * y
            + p11 * x * y
        )
        pred[:, :, channel] = np.clip(np.rint(values), 0, 0xffffffff).astype(np.uint32)
    return pred


def high_row_affine_prediction(high_ordered: np.ndarray) -> np.ndarray:
    height, width, channels = high_ordered.shape
    x = np.linspace(-1.0, 1.0, width, dtype=np.float64)
    design = np.stack([np.ones_like(x), x], axis=1)
    pred = np.zeros_like(high_ordered, dtype=np.uint32)
    for channel in range(channels):
        for y in range(height):
            values = high_ordered[y, :, channel].astype(np.float64)
            coeffs, *_ = np.linalg.lstsq(design, values, rcond=None)
            fitted = design @ coeffs
            pred[y, :, channel] = np.clip(
                np.rint(fitted),
                0,
                0xffffffff,
            ).astype(np.uint32)
    return pred


def high_column_affine_prediction(high_ordered: np.ndarray) -> np.ndarray:
    height, width, channels = high_ordered.shape
    y_coords = np.linspace(-1.0, 1.0, height, dtype=np.float64)
    design = np.stack([np.ones_like(y_coords), y_coords], axis=1)
    pred = np.zeros_like(high_ordered, dtype=np.uint32)
    for channel in range(channels):
        for x in range(width):
            values = high_ordered[:, x, channel].astype(np.float64)
            coeffs, *_ = np.linalg.lstsq(design, values, rcond=None)
            fitted = design @ coeffs
            pred[:, x, channel] = np.clip(
                np.rint(fitted),
                0,
                0xffffffff,
            ).astype(np.uint32)
    return pred


def progressive_high_surface_residual_payload(
    ordered_tile: np.ndarray,
    tail_width: int,
    kind: str,
) -> np.ndarray:
    payload = np.zeros_like(ordered_tile, dtype=np.uint32)
    for bit in range(tail_width - 1, -1, -1):
        low_mask = np.uint32((1 << (bit + 1)) - 1)
        known_ordered = (
            ordered_tile & np.uint32(0xffffffff ^ int(low_mask))
        ).astype(np.uint32)
        prediction = ordered_surface_prediction(known_ordered, kind)
        residual_bit = ((ordered_tile ^ prediction) >> np.uint32(bit)) & np.uint32(1)
        payload |= residual_bit.astype(np.uint32) << np.uint32(bit)
    return payload


def tile_candidates(
    ordered_tile: np.ndarray,
    pixels_tile: np.ndarray,
    tail_width: int,
) -> dict[str, np.ndarray]:
    candidates: dict[str, np.ndarray] = {}
    low_mask = np.uint32((1 << tail_width) - 1)
    high_ordered = (ordered_tile & np.uint32(0xffffffff ^ int(low_mask))).astype(
        np.uint32
    )
    candidates["high_corner_affine"] = low_tail_residual_payload(
        ordered_tile,
        high_corner_affine_prediction(high_ordered),
        tail_width,
    )
    candidates["high_corner_bilinear"] = low_tail_residual_payload(
        ordered_tile,
        high_corner_bilinear_prediction(high_ordered),
        tail_width,
    )
    candidates["high_row_affine"] = low_tail_residual_payload(
        ordered_tile,
        high_row_affine_prediction(high_ordered),
        tail_width,
    )
    candidates["high_column_affine"] = low_tail_residual_payload(
        ordered_tile,
        high_column_affine_prediction(high_ordered),
        tail_width,
    )
    for kind in ("affine", "quadratic"):
        candidates[f"progressive_high_{kind}"] = (
            progressive_high_surface_residual_payload(
                ordered_tile,
                tail_width,
                kind,
            )
        )
    for kind in ("affine", "quadratic"):
        candidates[f"high_ordered_{kind}"] = low_tail_residual_payload(
            ordered_tile,
            ordered_surface_prediction(high_ordered, kind),
            tail_width,
        )
        candidates[f"ordered_{kind}"] = low_tail_residual_payload(
            ordered_tile,
            ordered_surface_prediction(ordered_tile, kind),
            tail_width,
        )
        candidates[f"float_{kind}"] = low_tail_residual_payload(
            ordered_tile,
            float_surface_prediction(pixels_tile, kind),
            tail_width,
        )
        log_pred = positive_log_surface_prediction(pixels_tile, kind)
        if log_pred is not None:
            candidates[f"log_{kind}"] = low_tail_residual_payload(
                ordered_tile,
                log_pred,
                tail_width,
            )
    return candidates


def candidate_overhead_bits(channels: int, mode: str) -> int:
    if (
        mode.startswith("high_ordered_")
        or mode.startswith("high_corner_")
        or mode.startswith("high_row_")
        or mode.startswith("high_column_")
        or mode.startswith("progressive_high_")
    ):
        return 8
    return coeff_overhead_bits(channels, mode)


def summarize_image(
    path: Path,
    crop_size: int,
    tile_size: int,
    previous_bits: int,
    tail_widths: tuple[int, ...],
) -> dict:
    pixels = read_exr(path)
    if crop_size:
        pixels = pixels[:crop_size, :crop_size]
    pixels = np.ascontiguousarray(pixels, dtype=np.float32)
    bits = float_to_bits(pixels).copy()
    ordered = bits_to_ordered(bits)
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
        totals = Counter()
        selected = Counter()
        candidate_totals = Counter()
        for y in range(0, bits.shape[0], tile_size):
            for x in range(0, bits.shape[1], tile_size):
                sl = np.s_[y:y + tile_size, x:x + tile_size]
                body_tile = payloads["body"][sl]
                sign_tile = payloads["sign"][sl]
                sign = best_bitplane_cost(sign_tile, (31,))
                high = best_bitplane_cost(body_tile, range(tail_width, 31))
                low = best_bitplane_cost(body_tile, range(tail_width))
                current_tile = sign + high + low
                best_name = "current"
                best_low = low
                raw_tail = raw_mantissa[sl] & np.uint32((1 << tail_width) - 1)
                zero_mask, _zero_details = zero_mask_route_cost(
                    raw_tail,
                    tail_width,
                    8,
                )
                trailing_zero, _trailing_details = trailing_zero_route_cost(
                    raw_tail,
                    tail_width,
                    8,
                )
                for name, route_bits in (
                    ("zero_mask_raw_tail", zero_mask),
                    ("trailing_zero_raw_tail", trailing_zero),
                ):
                    candidate_totals[name] += sign + high + route_bits
                    if route_bits < best_low:
                        best_name = name
                        best_low = route_bits
                for name, residual in tile_candidates(
                    ordered[sl],
                    pixels[sl],
                    tail_width,
                ).items():
                    residual_bits = best_bitplane_cost(residual, range(tail_width))
                    route_bits = (
                        residual_bits
                        + candidate_overhead_bits(bits.shape[2], name)
                    )
                    candidate_totals[name] += sign + high + route_bits
                    if route_bits < best_low:
                        best_name = name
                        best_low = route_bits
                selected[best_name] += 1
                totals["current_total"] += current_tile
                totals["hybrid_total"] += sign + high + best_low
                totals["sign"] += sign
                totals["high"] += high
                totals["current_low"] += low
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
            }
        )
    return {
        "image": path.name,
        "shape": list(bits.shape),
        "crop_size": crop_size,
        "tile_size": tile_size,
        "previous_bits": previous_bits,
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
                f"  low{item['tail_width']:02d}: current={item['current_ratio']:.3f}x "
                f"hybrid={item['hybrid_ratio']:.3f}x gain={item['gain']:.4f} "
                f"selected={item['selected']} {best_text}"
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
        f"low_tail_surface_residual_{safe_glob}_tile{args.tile_size}"
        f"_prev{args.previous_bits}_crop{args.crop_size}"
        f"_tail{args.tail_widths.replace(',', '-')}.json"
    )
    output.write_text(json.dumps(rows, indent=2))
    print(f"\nSaved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
