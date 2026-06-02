"""rANS roundtrip for field-wise ordered-float delta payloads.

This validates the decoder schedule needed by the ordered-delta representation:

* modes are selected per tile and float field;
* residual payload bits are decoded high-to-low for each pixel/channel, so
  higher residual bits and neighboring residual bits are available as context;
* after one pixel/channel's payload bits are decoded, ordered float fields are
  reconstructed low-to-high, computing delta carries from lower actual bits.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import constriction
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_structural_predictors import DATA_DIR, FIELD_GROUPS, float_to_bits, read_exr  # noqa: E402
from structural_delta_field_roundtrip import (  # noqa: E402
    LOW_TO_HIGH_FIELDS,
    bits_to_ordered,
    choose_modes_and_payloads,
    decode as decode_payloads,
    field_layout,
    predictor_scalar,
)

BIT_TO_FIELD = {
    bit: field
    for field, bit_indices in FIELD_GROUPS.items()
    for bit in bit_indices
}


def bit_at(values: np.ndarray, y: int, x: int, c: int, bit: int) -> int:
    return int((int(values[y, x, c]) >> bit) & 1)


def set_bit(values: np.ndarray, y: int, x: int, c: int, bit: int, value: int) -> None:
    mask = np.uint32(1 << bit)
    if value:
        values[y, x, c] |= mask
    else:
        values[y, x, c] &= ~mask


def context_id(
    residuals: np.ndarray,
    y: int,
    x: int,
    c: int,
    bit: int,
    tile_y: int,
    tile_x: int,
    tile_height: int,
    tile_width: int,
    previous_bits: int,
) -> int:
    local_y = y - tile_y
    local_x = x - tile_x
    context = 0
    if local_x > 0:
        context |= bit_at(residuals, y, x - 1, c, bit)
    if local_y > 0:
        context |= bit_at(residuals, y - 1, x, c, bit) << 1
    if local_y > 0 and local_x > 0:
        context |= bit_at(residuals, y - 1, x - 1, c, bit) << 2
    if local_y > 0 and local_x + 1 < tile_width:
        context |= bit_at(residuals, y - 1, x + 1, c, bit) << 3
    for offset in range(1, previous_bits + 1):
        previous_bit = bit + offset
        if previous_bit <= 31:
            context |= bit_at(residuals, y, x, c, previous_bit) << (3 + offset)
    return context


def make_counts(previous_bits: int, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    context_count = 16 << previous_bits
    ones = np.full(context_count, alpha, dtype=np.float64)
    totals = np.full(context_count, 2.0 * alpha, dtype=np.float64)
    return ones, totals


def iter_decode_order(shape: tuple[int, int, int], tile_size: int):
    height, width, channels = shape
    for y in range(height):
        tile_y = (y // tile_size) * tile_size
        tile_height = min(tile_size, height - tile_y)
        for x in range(width):
            tile_x = (x // tile_size) * tile_size
            tile_width = min(tile_size, width - tile_x)
            for c in range(channels):
                for field, bit_indices in FIELD_GROUPS.items():
                    yield y, x, c, tile_y, tile_x, tile_height, tile_width, field, bit_indices


def build_symbols_and_probs(
    payloads: dict[str, np.ndarray],
    modes: dict[tuple[int, int, str], str],
    shape: tuple[int, int, int],
    tile_size: int,
    previous_bits: int,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    residuals = np.zeros(shape, dtype=np.uint32)
    counts = defaultdict(lambda: make_counts(previous_bits, alpha))
    symbols = []
    probs = []

    for y, x, c, tile_y, tile_x, tile_height, tile_width, field, bit_indices in (
        iter_decode_order(shape, tile_size)
    ):
        mode = modes[(tile_y, tile_x, field)]
        for bit in bit_indices:
            symbol = bit_at(payloads[field], y, x, c, bit)
            if mode == "raw":
                prob = 0.5
            else:
                key = (tile_y, tile_x, field, bit)
                ones, totals = counts[key]
                context = context_id(
                    residuals,
                    y,
                    x,
                    c,
                    bit,
                    tile_y,
                    tile_x,
                    tile_height,
                    tile_width,
                    previous_bits,
                )
                prob = float(ones[context] / totals[context])
                ones[context] += symbol
                totals[context] += 1.0
            symbols.append(symbol)
            probs.append(prob)
            set_bit(residuals, y, x, c, bit, symbol)

    return (
        np.asarray(symbols, dtype=np.int32),
        np.asarray(probs, dtype=np.float32),
    )


def reconstruct_pixel(
    decoded: np.ndarray,
    residuals: np.ndarray,
    modes: dict[tuple[int, int, str], str],
    y: int,
    x: int,
    c: int,
    tile_y: int,
    tile_x: int,
) -> None:
    value = 0
    for field, bit_indices in LOW_TO_HIGH_FIELDS:
        mode = modes[(tile_y, tile_x, field)]
        low, _width, mask = field_layout(bit_indices)
        payload_digit = int(residuals[y, x, c] >> low) & mask
        if mode == "raw":
            digit = payload_digit
        elif mode.startswith("xor_"):
            pred = predictor_scalar(mode[4:], decoded, y, x, c)
            digit = ((pred >> low) & mask) ^ payload_digit
        else:
            pred = predictor_scalar(mode, decoded, y, x, c)
            pred_digit = (pred >> low) & mask
            if low == 0:
                carry = 0
            else:
                lower_mask = (1 << low) - 1
                carry = 1 if (pred & lower_mask) > (value & lower_mask) else 0
            digit = (pred_digit + payload_digit + carry) & mask
        value |= digit << low
    decoded[y, x, c] = np.uint32(value)


def decode_rans(
    compressed: np.ndarray,
    modes: dict[tuple[int, int, str], str],
    shape: tuple[int, int, int],
    tile_size: int,
    previous_bits: int,
    alpha: float,
) -> np.ndarray:
    coder = constriction.stream.stack.AnsCoder(compressed)
    model = constriction.stream.model.Bernoulli(perfect=False)
    residuals = np.zeros(shape, dtype=np.uint32)
    decoded = np.zeros(shape, dtype=np.uint32)
    counts = defaultdict(lambda: make_counts(previous_bits, alpha))

    current_pixel = None
    for y, x, c, tile_y, tile_x, tile_height, tile_width, field, bit_indices in (
        iter_decode_order(shape, tile_size)
    ):
        if current_pixel is None:
            current_pixel = (y, x, c, tile_y, tile_x)
        elif current_pixel[:3] != (y, x, c):
            py, px, pc, ptile_y, ptile_x = current_pixel
            reconstruct_pixel(decoded, residuals, modes, py, px, pc, ptile_y, ptile_x)
            current_pixel = (y, x, c, tile_y, tile_x)

        mode = modes[(tile_y, tile_x, field)]
        for bit in bit_indices:
            if mode == "raw":
                prob = np.asarray([0.5], dtype=np.float32)
                symbol = int(coder.decode(model, prob)[0])
            else:
                key = (tile_y, tile_x, field, bit)
                ones, totals = counts[key]
                context = context_id(
                    residuals,
                    y,
                    x,
                    c,
                    bit,
                    tile_y,
                    tile_x,
                    tile_height,
                    tile_width,
                    previous_bits,
                )
                prob = np.asarray([ones[context] / totals[context]], dtype=np.float32)
                symbol = int(coder.decode(model, prob)[0])
                ones[context] += symbol
                totals[context] += 1.0
            set_bit(residuals, y, x, c, bit, symbol)

    if current_pixel is not None:
        py, px, pc, ptile_y, ptile_x = current_pixel
        reconstruct_pixel(decoded, residuals, modes, py, px, pc, ptile_y, ptile_x)
    return decoded


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="ph_studio_small_03_1k.exr")
    parser.add_argument("--tile-size", type=int, default=128)
    parser.add_argument("--previous-bits", type=int, default=4)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--mode-overhead-bits", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pixels = read_exr(DATA_DIR / args.image)
    if args.crop_size:
        pixels = pixels[: args.crop_size, : args.crop_size]
    bits = float_to_bits(np.ascontiguousarray(pixels, dtype=np.float32)).copy()
    raw_bits = bits.size * 32

    t0 = time.perf_counter()
    modes, payloads = choose_modes_and_payloads(
        bits,
        pixels,
        args.tile_size,
        args.previous_bits,
    )
    selected = decode_payloads(modes, payloads, bits.shape, args.tile_size)
    if not np.array_equal(selected, bits_to_ordered(bits)):
        raise RuntimeError("payload representation roundtrip failed before rANS")
    t1 = time.perf_counter()
    symbols, probs = build_symbols_and_probs(
        payloads,
        modes,
        bits.shape,
        args.tile_size,
        args.previous_bits,
        args.alpha,
    )
    t2 = time.perf_counter()
    coder = constriction.stream.stack.AnsCoder()
    model = constriction.stream.model.Bernoulli(perfect=False)
    coder.encode_reverse(symbols, model, probs)
    compressed = coder.get_compressed()
    t3 = time.perf_counter()
    decoded = decode_rans(
        compressed,
        modes,
        bits.shape,
        args.tile_size,
        args.previous_bits,
        args.alpha,
    )
    t4 = time.perf_counter()
    ok = bool(np.array_equal(decoded, selected))
    side_bits = len(modes) * args.mode_overhead_bits
    payload_bytes = compressed.nbytes + math.ceil(side_bits / 8)

    print(f"image:        {args.image}")
    print(f"shape:        {bits.shape}")
    print(f"symbols:      {len(symbols)}")
    print(f"raw:          {raw_bits / 8:.0f} bytes")
    print(
        f"rANS payload: {payload_bytes:.0f} bytes "
        f"({(raw_bits / 8) / payload_bytes:.3f}x)"
    )
    print(f"valid bits:   {coder.num_valid_bits() + side_bits}")
    print(f"select:       {t1 - t0:.2f}s")
    print(f"prob pass:    {t2 - t1:.2f}s")
    print(f"encode:       {t3 - t2:.2f}s")
    print(f"decode:       {t4 - t3:.2f}s")
    print(f"roundtrip:    {ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
