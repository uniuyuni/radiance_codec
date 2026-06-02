"""Round-trip check for field-wise ordered-float delta reconstruction.

This is not an entropy coder.  It verifies the key representation claim:
different float fields may choose different ordered-delta predictors if the
decoder reconstructs each pixel's fields from low to high and computes the
incoming carry from already reconstructed lower bits.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_structural_predictors import (  # noqa: E402
    DATA_DIR,
    FIELD_GROUPS,
    float_to_bits,
    read_exr,
)
from evaluate_structural_delta_context import (  # noqa: E402
    bits_to_ordered,
    context_bitplane_cost,
    delta_predictors,
    ordered_delta,
    ordered_xor_predictors,
)

LOW_TO_HIGH_FIELDS = tuple(
    sorted(FIELD_GROUPS.items(), key=lambda item: min(item[1]))
)
FLOAT_ZERO_ORDERED = 0x80000000


def ordered_to_bits_scalar(value: int) -> int:
    value &= 0xFFFFFFFF
    if value & 0x80000000:
        return value ^ 0x80000000
    return (~value) & 0xFFFFFFFF


def bits_to_ordered_scalar(value: int) -> int:
    value &= 0xFFFFFFFF
    if value & 0x80000000:
        return (~value) & 0xFFFFFFFF
    return value ^ 0x80000000


def ordered_to_float32(value: int) -> np.float32:
    bits = np.array([ordered_to_bits_scalar(value)], dtype=np.uint32)
    return bits.view(np.float32)[0]


def float32_to_ordered(value: np.float32) -> int:
    bits = np.array([value], dtype=np.float32).view(np.uint32)[0]
    return bits_to_ordered_scalar(int(bits))


def field_layout(bit_indices: tuple[int, ...]) -> tuple[int, int, int]:
    low = min(bit_indices)
    high = max(bit_indices)
    width = high - low + 1
    mask = (1 << width) - 1
    return low, width, mask


def extract_digit(value: int, bit_indices: tuple[int, ...]) -> int:
    low, _width, mask = field_layout(bit_indices)
    return (value >> low) & mask


def ordered_med_scalar(values: np.ndarray, y: int, x: int, c: int) -> int:
    west = int(values[y, x - 1, c]) if x > 0 else 0
    north = int(values[y - 1, x, c]) if y > 0 else 0
    northwest = int(values[y - 1, x - 1, c]) if y > 0 and x > 0 else 0
    mx = max(north, west)
    mn = min(north, west)
    planar = max(0, min(0xFFFFFFFF, west + north - northwest))
    if northwest >= mx:
        return mn
    if northwest <= mn:
        return mx
    return planar


def float_med_scalar(values: np.ndarray, y: int, x: int, c: int) -> int:
    if y == 0:
        return int(values[y, x - 1, c]) if x > 0 else FLOAT_ZERO_ORDERED
    if x == 0:
        return int(values[y - 1, x, c])
    west = ordered_to_float32(int(values[y, x - 1, c]))
    north = ordered_to_float32(int(values[y - 1, x, c]))
    northwest = ordered_to_float32(int(values[y - 1, x - 1, c]))
    mx = np.maximum(north, west)
    mn = np.minimum(north, west)
    planar = np.float32(np.float32(north) + np.float32(west) - np.float32(northwest))
    pred = np.where(northwest >= mx, mn, np.where(northwest <= mn, mx, planar))
    return float32_to_ordered(np.float32(pred))


def float_planar_scalar(values: np.ndarray, y: int, x: int, c: int) -> int:
    if y == 0:
        return int(values[y, x - 1, c]) if x > 0 else FLOAT_ZERO_ORDERED
    if x == 0:
        return int(values[y - 1, x, c])
    west = ordered_to_float32(int(values[y, x - 1, c]))
    north = ordered_to_float32(int(values[y - 1, x, c]))
    northwest = ordered_to_float32(int(values[y - 1, x - 1, c]))
    pred = np.float32(np.float32(west) + np.float32(north) - np.float32(northwest))
    return float32_to_ordered(pred)


def predictor_scalar(mode: str, values: np.ndarray, y: int, x: int, c: int) -> int:
    if mode in {"delta_zero", "zero"}:
        return 0
    if mode in {"delta_west", "west"}:
        return int(values[y, x - 1, c]) if x > 0 else 0
    if mode in {"delta_north", "north"}:
        return int(values[y - 1, x, c]) if y > 0 else 0
    if mode in {"delta_northwest", "northwest"}:
        return int(values[y - 1, x - 1, c]) if y > 0 and x > 0 else 0
    if mode in {"delta_northeast", "northeast"}:
        return int(values[y - 1, x + 1, c]) if y > 0 and x + 1 < values.shape[1] else 0
    if mode in {"delta_channel_previous", "channel_previous"}:
        return int(values[y, x, c - 1]) if c > 0 else 0
    if mode in {"delta_channel_green", "channel_green"}:
        return (
            int(values[y, x, 1])
            if values.shape[2] >= 3 and c in {0, 2}
            else 0
        )
    if mode in {"delta_med", "med"}:
        return ordered_med_scalar(values, y, x, c)
    if mode in {"delta_planar", "planar"}:
        west = int(values[y, x - 1, c]) if x > 0 else 0
        north = int(values[y - 1, x, c]) if y > 0 else 0
        northwest = int(values[y - 1, x - 1, c]) if y > 0 and x > 0 else 0
        return max(0, min(0xFFFFFFFF, west + north - northwest))
    if mode in {"delta_select", "select"}:
        west = int(values[y, x - 1, c]) if x > 0 else 0
        north = int(values[y - 1, x, c]) if y > 0 else 0
        northwest = int(values[y - 1, x - 1, c]) if y > 0 and x > 0 else 0
        return west if abs(north - northwest) < abs(west - northwest) else north
    if mode in {"delta_paeth", "paeth"}:
        west = int(values[y, x - 1, c]) if x > 0 else 0
        north = int(values[y - 1, x, c]) if y > 0 else 0
        northwest = int(values[y - 1, x - 1, c]) if y > 0 and x > 0 else 0
        planar = west + north - northwest
        dist_w = abs(planar - west)
        dist_n = abs(planar - north)
        dist_nw = abs(planar - northwest)
        if dist_w <= dist_n and dist_w <= dist_nw:
            return west
        return north if dist_n <= dist_nw else northwest
    if mode in {"delta_float_med", "float_med"}:
        return float_med_scalar(values, y, x, c)
    if mode in {"delta_float_planar", "float_planar"}:
        return float_planar_scalar(values, y, x, c)
    if mode == "bit_xor_planar":
        west = int(values[y, x - 1, c]) if x > 0 else 0
        north = int(values[y - 1, x, c]) if y > 0 else 0
        northwest = int(values[y - 1, x - 1, c]) if y > 0 and x > 0 else 0
        return west ^ north ^ northwest
    if mode == "bit_majority":
        west = int(values[y, x - 1, c]) if x > 0 else 0
        north = int(values[y - 1, x, c]) if y > 0 else 0
        northwest = int(values[y - 1, x - 1, c]) if y > 0 and x > 0 else 0
        return (west & north) | (west & northwest) | (north & northwest)
    raise ValueError(f"unsupported predictor mode: {mode}")


def choose_modes_and_payloads(
    bits: np.ndarray,
    pixels: np.ndarray,
    tile_size: int,
    previous_bits: int,
) -> tuple[dict[tuple[int, int, str], str], dict[str, np.ndarray]]:
    ordered = bits_to_ordered(bits)
    xor_predictor_bits = ordered_xor_predictors(bits, pixels, causal=True)
    delta_prediction_bits = delta_predictors(bits, pixels, causal=True)
    xor_predictor_bits.pop("channel_green", None)
    delta_prediction_bits.pop("delta_channel_green", None)
    height, width, _channels = bits.shape
    modes: dict[tuple[int, int, str], str] = {}
    payloads = {
        field: np.zeros_like(ordered, dtype=np.uint32)
        for field in FIELD_GROUPS
    }

    for tile_y in range(0, height, tile_size):
        for tile_x in range(0, width, tile_size):
            sl = np.s_[tile_y:tile_y + tile_size, tile_x:tile_x + tile_size]
            ordered_tile = ordered[sl]
            for field, bit_indices in FIELD_GROUPS.items():
                candidates: dict[str, tuple[float, np.ndarray]] = {
                    "raw": (
                        float(ordered_tile.size * len(bit_indices)),
                        ordered_tile,
                    ),
                }
                for name, prediction in xor_predictor_bits.items():
                    residual = ordered_tile ^ prediction[sl]
                    candidates[f"xor_{name}"] = (
                        context_bitplane_cost(
                            residual,
                            bit_indices,
                            previous_bits,
                            channel_context=False,
                        ),
                        residual,
                    )
                for name, prediction in delta_prediction_bits.items():
                    residual = ordered_delta(ordered_tile, prediction[sl])
                    candidates[name] = (
                        context_bitplane_cost(
                            residual,
                            bit_indices,
                            previous_bits,
                            channel_context=False,
                        ),
                        residual,
                    )
                best_mode, (_cost, residual) = min(
                    candidates.items(),
                    key=lambda item: item[1][0],
                )
                modes[(tile_y, tile_x, field)] = best_mode
                low, _width, mask = field_layout(bit_indices)
                payloads[field][sl] = (
                    residual & np.uint32(mask << low)
                ).astype(np.uint32)
    return modes, payloads


def decode(
    modes: dict[tuple[int, int, str], str],
    payloads: dict[str, np.ndarray],
    shape: tuple[int, int, int],
    tile_size: int,
) -> np.ndarray:
    height, width, channels = shape
    decoded = np.zeros(shape, dtype=np.uint32)
    for y in range(height):
        tile_y = (y // tile_size) * tile_size
        for x in range(width):
            tile_x = (x // tile_size) * tile_size
            for c in range(channels):
                value = 0
                for field, bit_indices in LOW_TO_HIGH_FIELDS:
                    mode = modes[(tile_y, tile_x, field)]
                    low, _width, mask = field_layout(bit_indices)
                    payload_digit = int(payloads[field][y, x, c] >> low) & mask
                    if mode == "raw":
                        digit = payload_digit
                    elif mode.startswith("xor_"):
                        pred = predictor_scalar(mode[4:], decoded, y, x, c)
                        digit = extract_digit(pred, bit_indices) ^ payload_digit
                    else:
                        pred = predictor_scalar(mode, decoded, y, x, c)
                        pred_digit = extract_digit(pred, bit_indices)
                        if low == 0:
                            carry = 0
                        else:
                            lower_mask = (1 << low) - 1
                            carry = 1 if (pred & lower_mask) > (value & lower_mask) else 0
                        digit = (pred_digit + payload_digit + carry) & mask
                    value |= digit << low
                decoded[y, x, c] = np.uint32(value)
    return decoded


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", default="ph_studio_small_03_1k.exr")
    parser.add_argument("--tile-size", type=int, default=128)
    parser.add_argument("--previous-bits", type=int, default=4)
    parser.add_argument("--crop-size", type=int, default=128)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = sorted(DATA_DIR.glob(args.glob))
    if not paths:
        raise RuntimeError(f"no images matched {DATA_DIR / args.glob}")
    for path in paths:
        pixels = read_exr(path)
        if args.crop_size:
            pixels = pixels[: args.crop_size, : args.crop_size]
        bits = float_to_bits(pixels)
        ordered = bits_to_ordered(bits)
        modes, payloads = choose_modes_and_payloads(
            bits,
            pixels,
            args.tile_size,
            args.previous_bits,
        )
        decoded = decode(modes, payloads, ordered.shape, args.tile_size)
        if not np.array_equal(decoded, ordered):
            mismatch = np.argwhere(decoded != ordered)[0]
            y, x, c = (int(v) for v in mismatch)
            raise RuntimeError(
                f"{path.name}: mismatch at y={y} x={x} c={c}: "
                f"decoded=0x{int(decoded[y, x, c]):08x} "
                f"expected=0x{int(ordered[y, x, c]):08x}"
            )
        print(
            f"{path.name}: roundtrip ok "
            f"shape={ordered.shape} tile={args.tile_size} prev={args.previous_bits}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
