"""Estimate spatial-context coding with ordered-float delta predictors.

The previous structural codec uses XOR residuals between float32 bit patterns.
For smooth HDR regions, the numerical prediction may be very close in ULPs
while XOR still leaves noisy lower mantissa planes. This probe maps float bits
to an order-preserving uint32 domain and codes modular prediction deltas:

    residual = ordered(actual) - ordered(prediction) mod 2^32

The residual is still lossless: decoder reconstructs ordered(actual) by adding
the residual back to the decoder-reproducible prediction.
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

from analyze_structural_predictors import (  # noqa: E402
    DATA_DIR,
    FIELD_GROUPS,
    float_med_prediction,
    float_planar_prediction,
    float_to_bits,
    load_blosc_ratios,
    predictors,
    read_exr,
    shifted,
)
from evaluate_structural_spatial_context import spatial_contexts  # noqa: E402

RESULTS_DIR = ROOT / "results"


def bits_to_ordered(bits: np.ndarray) -> np.ndarray:
    sign = (bits >> 31) != 0
    return np.where(sign, ~bits, bits ^ np.uint32(0x80000000)).astype(np.uint32)


def causal_shifted(values: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Return out[y, x] = values[y + dy, x + dx] when that source exists."""

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


def ordered_delta(actual: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    return (actual - prediction).astype(np.uint32)


def ordered_field_delta(
    actual: np.ndarray,
    prediction: np.ndarray,
    bit_indices: tuple[int, ...],
) -> np.ndarray:
    low_bit = min(bit_indices)
    width = len(bit_indices)
    mask = np.uint32((1 << width) - 1)
    actual_field = (actual >> np.uint32(low_bit)) & mask
    prediction_field = (prediction >> np.uint32(low_bit)) & mask
    residual = (actual_field - prediction_field) & mask
    return (residual << np.uint32(low_bit)).astype(np.uint32)


def ordered_planar_prediction(ordered: np.ndarray) -> np.ndarray:
    west = shifted(ordered, 0, -1).astype(np.int64)
    north = shifted(ordered, -1, 0).astype(np.int64)
    northwest = shifted(ordered, -1, -1).astype(np.int64)
    pred = west + north - northwest
    return np.clip(pred, 0, 0xFFFFFFFF).astype(np.uint32)


def ordered_planar_prediction_causal(ordered: np.ndarray) -> np.ndarray:
    west = causal_shifted(ordered, 0, -1).astype(np.int64)
    north = causal_shifted(ordered, -1, 0).astype(np.int64)
    northwest = causal_shifted(ordered, -1, -1).astype(np.int64)
    pred = west + north - northwest
    return np.clip(pred, 0, 0xFFFFFFFF).astype(np.uint32)


def ordered_med_prediction(ordered: np.ndarray) -> np.ndarray:
    west = shifted(ordered, 0, -1)
    north = shifted(ordered, -1, 0)
    northwest = shifted(ordered, -1, -1)
    mx = np.maximum(north, west)
    mn = np.minimum(north, west)
    planar = ordered_planar_prediction(ordered)
    return np.where(
        northwest >= mx,
        mn,
        np.where(northwest <= mn, mx, planar),
    ).astype(np.uint32)


def ordered_med_prediction_causal(ordered: np.ndarray) -> np.ndarray:
    west = causal_shifted(ordered, 0, -1)
    north = causal_shifted(ordered, -1, 0)
    northwest = causal_shifted(ordered, -1, -1)
    mx = np.maximum(north, west)
    mn = np.minimum(north, west)
    planar = ordered_planar_prediction_causal(ordered)
    return np.where(
        northwest >= mx,
        mn,
        np.where(northwest <= mn, mx, planar),
    ).astype(np.uint32)


def ordered_select_prediction(ordered: np.ndarray) -> np.ndarray:
    west = shifted(ordered, 0, -1).astype(np.int64)
    north = shifted(ordered, -1, 0).astype(np.int64)
    northwest = shifted(ordered, -1, -1).astype(np.int64)
    return np.where(
        np.abs(north - northwest) < np.abs(west - northwest),
        west,
        north,
    ).astype(np.uint32)


def ordered_paeth_prediction(ordered: np.ndarray) -> np.ndarray:
    west = shifted(ordered, 0, -1).astype(np.int64)
    north = shifted(ordered, -1, 0).astype(np.int64)
    northwest = shifted(ordered, -1, -1).astype(np.int64)
    planar = west + north - northwest
    dist_w = np.abs(planar - west)
    dist_n = np.abs(planar - north)
    dist_nw = np.abs(planar - northwest)
    return np.where(
        (dist_w <= dist_n) & (dist_w <= dist_nw),
        west,
        np.where(dist_n <= dist_nw, north, northwest),
    ).astype(np.uint32)


def ordered_select_prediction_causal(ordered: np.ndarray) -> np.ndarray:
    west = causal_shifted(ordered, 0, -1).astype(np.int64)
    north = causal_shifted(ordered, -1, 0).astype(np.int64)
    northwest = causal_shifted(ordered, -1, -1).astype(np.int64)
    return np.where(
        np.abs(north - northwest) < np.abs(west - northwest),
        west,
        north,
    ).astype(np.uint32)


def ordered_paeth_prediction_causal(ordered: np.ndarray) -> np.ndarray:
    west = causal_shifted(ordered, 0, -1).astype(np.int64)
    north = causal_shifted(ordered, -1, 0).astype(np.int64)
    northwest = causal_shifted(ordered, -1, -1).astype(np.int64)
    planar = west + north - northwest
    dist_w = np.abs(planar - west)
    dist_n = np.abs(planar - north)
    dist_nw = np.abs(planar - northwest)
    return np.where(
        (dist_w <= dist_n) & (dist_w <= dist_nw),
        west,
        np.where(dist_n <= dist_nw, north, northwest),
    ).astype(np.uint32)


def ordered_channel_green_prediction(ordered: np.ndarray) -> np.ndarray:
    pred = np.zeros_like(ordered)
    if ordered.shape[2] >= 3:
        pred[..., 0] = ordered[..., 1]
        pred[..., 2] = ordered[..., 1]
    return pred


def ordered_xor_predictors(
    bits: np.ndarray,
    pixels: np.ndarray | None = None,
    causal: bool = False,
) -> dict[str, np.ndarray]:
    ordered = bits_to_ordered(bits)
    shift_fn = causal_shifted if causal else shifted
    if causal:
        west = shift_fn(ordered, 0, -1)
        north = shift_fn(ordered, -1, 0)
        northwest = shift_fn(ordered, -1, -1)
        northeast = shift_fn(ordered, -1, 1)
    else:
        west = shift_fn(ordered, 0, -1)
        north = shift_fn(ordered, -1, 0)
        northwest = shift_fn(ordered, -1, -1)
        northeast = shift_fn(ordered, -1, 1)
    channel_previous = np.zeros_like(ordered)
    if ordered.shape[2] >= 2:
        channel_previous[..., 1:] = ordered[..., :-1]
    result = {
        "zero": np.zeros_like(ordered),
        "west": west,
        "north": north,
        "northwest": northwest,
        "northeast": northeast,
        "channel_green": ordered_channel_green_prediction(ordered),
        "channel_previous": channel_previous,
        "bit_xor_planar": west ^ north ^ northwest,
        "bit_majority": (west & north) | (west & northwest) | (north & northwest),
    }
    if pixels is not None:
        result["float_med"] = bits_to_ordered(float_med_prediction(pixels))
        result["float_planar"] = bits_to_ordered(float_planar_prediction(pixels))
    return result


def delta_predictors(
    bits: np.ndarray,
    pixels: np.ndarray | None = None,
    causal: bool = False,
) -> dict[str, np.ndarray]:
    ordered = bits_to_ordered(bits)
    shift_fn = causal_shifted if causal else shifted
    if causal:
        med = ordered_med_prediction_causal(ordered)
        planar = ordered_planar_prediction_causal(ordered)
        select = ordered_select_prediction_causal(ordered)
        paeth = ordered_paeth_prediction_causal(ordered)
        west = shift_fn(ordered, 0, -1)
        north = shift_fn(ordered, -1, 0)
        northwest = shift_fn(ordered, -1, -1)
        northeast = shift_fn(ordered, -1, 1)
    else:
        med = ordered_med_prediction(ordered)
        planar = ordered_planar_prediction(ordered)
        select = ordered_select_prediction(ordered)
        paeth = ordered_paeth_prediction(ordered)
        west = shift_fn(ordered, 0, -1)
        north = shift_fn(ordered, -1, 0)
        northwest = shift_fn(ordered, -1, -1)
        northeast = shift_fn(ordered, -1, 1)
    channel_previous = np.zeros_like(ordered)
    if ordered.shape[2] >= 2:
        channel_previous[..., 1:] = ordered[..., :-1]
    result = {
        "delta_west": west,
        "delta_north": north,
        "delta_northwest": northwest,
        "delta_northeast": northeast,
        "delta_channel_green": ordered_channel_green_prediction(ordered),
        "delta_channel_previous": channel_previous,
        "delta_med": med,
        "delta_planar": planar,
        "delta_select": select,
        "delta_paeth": paeth,
    }
    if pixels is not None:
        result["delta_float_med"] = bits_to_ordered(float_med_prediction(pixels))
        result["delta_float_planar"] = bits_to_ordered(
            float_planar_prediction(pixels)
        )
    return result


def context_bitplane_cost(
    bits: np.ndarray,
    bit_indices: tuple[int, ...],
    previous_bits: int,
    channel_context: bool,
) -> float:
    cost = 0.0
    for bit_index in bit_indices:
        plane = ((bits >> bit_index) & 1).astype(np.uint8, copy=False)
        context = spatial_contexts(plane)
        for offset in range(1, previous_bits + 1):
            previous_bit = bit_index + offset
            if previous_bit > 31:
                continue
            previous_plane = ((bits >> previous_bit) & 1).astype(np.uint8, copy=False)
            context |= previous_plane << (3 + offset)
        if channel_context:
            channel_plane = np.zeros_like(plane)
            channel_plane[..., 1:] = plane[..., :-1]
            context |= channel_plane << (4 + previous_bits)
        context_flat = context.reshape(-1)
        plane_flat = plane.reshape(-1)
        context_count = 16 << previous_bits
        if channel_context:
            context_count <<= 1
        totals = np.bincount(context_flat, minlength=context_count)
        ones = np.bincount(context_flat, weights=plane_flat, minlength=context_count)
        for context_id in range(context_count):
            total = int(totals[context_id])
            if total == 0:
                continue
            one_count = int(ones[context_id])
            zero_count = total - one_count
            log_probability = (
                math.lgamma(one_count + 0.5)
                + math.lgamma(zero_count + 0.5)
                - math.lgamma(total + 1.0)
                - 2.0 * math.lgamma(0.5)
            )
            cost += -log_probability / math.log(2.0)
    return cost


def estimate_image(
    path: Path,
    tile_size: int,
    mode_overhead_bits: int,
    previous_bits: int,
    include_xor: bool,
    include_delta: bool,
    channel_context: bool,
    field_delta: bool,
    safe_pixel_modes: bool,
    ordered_xor: bool,
    causal_predictors: bool,
) -> dict:
    pixels = read_exr(path)
    bits = float_to_bits(pixels)
    ordered = bits_to_ordered(bits)
    if ordered_xor:
        xor_source = ordered
        xor_predictor_bits = ordered_xor_predictors(
            bits,
            pixels,
            causal=causal_predictors,
        )
    else:
        xor_source = bits
        xor_predictor_bits = predictors(pixels)
    delta_prediction_bits = delta_predictors(
        bits,
        pixels,
        causal=causal_predictors,
    )
    if safe_pixel_modes:
        xor_predictor_bits.pop("channel_green", None)
        delta_prediction_bits.pop("delta_channel_green", None)
    height, width, channels = bits.shape
    raw_bits = bits.size * 32
    total_bits = 0.0
    field_bits_total = Counter()
    mode_histogram = Counter()
    field_mode_histogram = {field: Counter() for field in FIELD_GROUPS}

    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            sl = np.s_[y:y + tile_size, x:x + tile_size]
            tile = bits[sl]
            ordered_tile = ordered[sl]
            tile_values = tile.size
            for field, bit_indices in FIELD_GROUPS.items():
                candidates = {"raw": tile_values * len(bit_indices)}
                if include_xor:
                    for name, pred in xor_predictor_bits.items():
                        residual = xor_source[sl] ^ pred[sl]
                        candidates[f"xor_{name}"] = context_bitplane_cost(
                            residual,
                            bit_indices,
                            previous_bits,
                            channel_context,
                        )
                if include_delta:
                    for name, pred in delta_prediction_bits.items():
                        if field_delta:
                            residual = ordered_field_delta(
                                ordered_tile,
                                pred[sl],
                                bit_indices,
                            )
                        else:
                            residual = ordered_delta(ordered_tile, pred[sl])
                        candidates[name] = context_bitplane_cost(
                            residual,
                            bit_indices,
                            previous_bits,
                            channel_context,
                        )
                best_mode = min(candidates, key=candidates.__getitem__)
                best_bits = candidates[best_mode] + mode_overhead_bits
                total_bits += best_bits
                field_bits_total[field] += best_bits
                mode_histogram[best_mode] += 1
                field_mode_histogram[field][best_mode] += 1

    return {
        "image": path.name,
        "shape": [height, width, channels],
        "tile_size": tile_size,
        "previous_bits": previous_bits,
        "raw_bits": raw_bits,
        "estimated_bits": total_bits,
        "ratio": raw_bits / total_bits,
        "field_bits": dict(field_bits_total),
        "field_bits_per_value": {
            field: bits_value / bits.size
            for field, bits_value in field_bits_total.items()
        },
        "mode_histogram": dict(mode_histogram),
        "field_mode_histogram": {
            field: dict(histogram)
            for field, histogram in field_mode_histogram.items()
        },
    }


def geomean(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", default="*.exr")
    parser.add_argument("--tile-size", type=int, default=128)
    parser.add_argument("--mode-overhead-bits", type=int, default=5)
    parser.add_argument("--previous-bits", type=int, default=3)
    parser.add_argument("--no-xor", action="store_true")
    parser.add_argument("--no-delta", action="store_true")
    parser.add_argument("--channel-context", action="store_true")
    parser.add_argument("--field-delta", action="store_true")
    parser.add_argument("--safe-pixel-modes", action="store_true")
    parser.add_argument("--ordered-xor", action="store_true")
    parser.add_argument("--causal-predictors", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = sorted(DATA_DIR.glob(args.glob))
    if not paths:
        raise RuntimeError(f"no images matched {DATA_DIR / args.glob}")
    blosc = load_blosc_ratios()
    results = []
    print(
        f"{'image':43s} {'delta':>9s} {'Blosc':>9s} "
        f"{'hybrid':>9s} {'pick':>8s}"
    )
    print("-" * 86)
    for path in paths:
        result = estimate_image(
            path,
            args.tile_size,
            args.mode_overhead_bits,
            args.previous_bits,
            include_xor=not args.no_xor,
            include_delta=not args.no_delta,
            channel_context=args.channel_context,
            field_delta=args.field_delta,
            safe_pixel_modes=args.safe_pixel_modes,
            ordered_xor=args.ordered_xor,
            causal_predictors=args.causal_predictors,
        )
        blosc_ratio = blosc.get(path.stem)
        hybrid_ratio = max(result["ratio"], blosc_ratio or 0.0)
        pick = "delta" if result["ratio"] >= (blosc_ratio or 0.0) else "Blosc"
        result["blosc_ratio"] = blosc_ratio
        result["hybrid_ratio"] = hybrid_ratio
        result["hybrid_pick"] = pick
        results.append(result)
        blosc_text = f"{blosc_ratio:8.2f}x" if blosc_ratio is not None else "       -"
        print(
            f"{path.stem:43s} {result['ratio']:8.2f}x "
            f"{blosc_text} {hybrid_ratio:8.2f}x {pick:>8s}"
        )
    print("-" * 86)
    for key in ["ratio", "blosc_ratio", "hybrid_ratio"]:
        values = [result[key] for result in results if result.get(key) is not None]
        print(f"geomean {key:12s}: {geomean(values):.3f}x")

    output = RESULTS_DIR / (
        f"structural_delta_context_tile{args.tile_size}_prev{args.previous_bits}"
        f"{'_safepixel' if args.safe_pixel_modes else ''}"
        f"{'_ordxor' if args.ordered_xor else ''}"
        f"{'_causal' if args.causal_predictors else ''}"
        f"{'_field' if args.field_delta else ''}.json"
    )
    output.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
