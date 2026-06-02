"""Bit-exact rANS roundtrip for the decoder-safe spatial context model.

This is intentionally a correctness probe, not a fast production encoder. It
uses the same PIC-like context idea as evaluate_structural_spatial_context.py:

* choose one reversible predictor mode per tile/float field;
* encode residual bit-planes from high bits to low bits;
* predict each residual bit with adaptive contexts from decoded neighbors and
  already decoded higher residual bits;
* decode with only the stored mode map and the compressed rANS stream.

The default mode set excludes float_med/float_planar because those require a
carefully defined full-pixel decode schedule. The remaining modes are directly
decoder-safe under tile-raster, field-major, bit-plane-major decoding.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import constriction
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_structural_predictors import (  # noqa: E402
    DATA_DIR,
    FIELD_GROUPS,
    float_to_bits,
    predictors,
    read_exr,
)
from evaluate_structural_spatial_context import context_bitplane_cost  # noqa: E402

SAFE_MODES = [
    "zero",
    "west",
    "north",
    "northwest",
    "northeast",
    "bit_xor_planar",
    "bit_majority",
]


@dataclass(frozen=True)
class Record:
    y: int
    x: int
    height: int
    width: int
    field: str
    mode: str


def bit_at(values: np.ndarray, y: int, x: int, c: int, bit_index: int) -> int:
    return int((int(values[y, x, c]) >> bit_index) & 1)


def set_bit(values: np.ndarray, y: int, x: int, c: int, bit_index: int, bit: int) -> None:
    mask = np.uint32(1 << bit_index)
    if bit:
        values[y, x, c] |= mask
    else:
        values[y, x, c] &= ~mask


def predictor_bit(
    values: np.ndarray,
    y: int,
    x: int,
    c: int,
    bit_index: int,
    mode: str,
) -> int:
    height, width, _channels = values.shape
    if mode == "zero" or mode == "raw":
        return 0
    if mode == "west":
        return bit_at(values, y, x - 1, c, bit_index) if x > 0 else 0
    if mode == "north":
        return bit_at(values, y - 1, x, c, bit_index) if y > 0 else 0
    if mode == "northwest":
        return bit_at(values, y - 1, x - 1, c, bit_index) if y > 0 and x > 0 else 0
    if mode == "northeast":
        return (
            bit_at(values, y - 1, x + 1, c, bit_index)
            if y > 0 and x + 1 < width
            else 0
        )
    if mode == "bit_xor_planar":
        west = bit_at(values, y, x - 1, c, bit_index) if x > 0 else 0
        north = bit_at(values, y - 1, x, c, bit_index) if y > 0 else 0
        northwest = bit_at(values, y - 1, x - 1, c, bit_index) if y > 0 and x > 0 else 0
        return west ^ north ^ northwest
    if mode == "bit_majority":
        west = bit_at(values, y, x - 1, c, bit_index) if x > 0 else 0
        north = bit_at(values, y - 1, x, c, bit_index) if y > 0 else 0
        northwest = bit_at(values, y - 1, x - 1, c, bit_index) if y > 0 and x > 0 else 0
        return (west & north) | (west & northwest) | (north & northwest)
    raise ValueError(f"unsupported decoder-safe mode: {mode}")


def residual_bit(
    values: np.ndarray,
    y: int,
    x: int,
    c: int,
    bit_index: int,
    mode: str,
) -> int:
    return bit_at(values, y, x, c, bit_index) ^ predictor_bit(
        values, y, x, c, bit_index, mode
    )


def context_id(
    values: np.ndarray,
    y: int,
    x: int,
    c: int,
    bit_index: int,
    record: Record,
    previous_bits: int,
) -> int:
    local_y = y - record.y
    local_x = x - record.x
    context = 0
    if local_x > 0:
        context |= residual_bit(values, y, x - 1, c, bit_index, record.mode)
    if local_y > 0:
        context |= residual_bit(values, y - 1, x, c, bit_index, record.mode) << 1
    if local_y > 0 and local_x > 0:
        context |= residual_bit(values, y - 1, x - 1, c, bit_index, record.mode) << 2
    if local_y > 0 and local_x + 1 < record.width:
        context |= residual_bit(values, y - 1, x + 1, c, bit_index, record.mode) << 3
    for offset in range(1, previous_bits + 1):
        previous_bit = bit_index + offset
        if previous_bit <= 31:
            context |= (
                residual_bit(values, y, x, c, previous_bit, record.mode)
                << (3 + offset)
            )
    return context


def iter_positions(record: Record, channels: int):
    for yy in range(record.y, record.y + record.height):
        for xx in range(record.x, record.x + record.width):
            for channel in range(channels):
                yield yy, xx, channel


def adaptive_symbols_and_probs(
    values: np.ndarray,
    record: Record,
    bit_index: int,
    previous_bits: int,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    channels = values.shape[2]
    context_count = 16 << previous_bits
    ones = np.full(context_count, alpha, dtype=np.float64)
    totals = np.full(context_count, 2.0 * alpha, dtype=np.float64)
    symbols = []
    probs = []
    for y, x, c in iter_positions(record, channels):
        context = context_id(values, y, x, c, bit_index, record, previous_bits)
        symbol = residual_bit(values, y, x, c, bit_index, record.mode)
        symbols.append(symbol)
        probs.append(ones[context] / totals[context])
        ones[context] += symbol
        totals[context] += 1.0
    return (
        np.asarray(symbols, dtype=np.int32),
        np.asarray(probs, dtype=np.float32),
    )


def raw_symbols_and_probs(
    values: np.ndarray,
    record: Record,
    bit_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    channels = values.shape[2]
    symbols = [
        bit_at(values, y, x, c, bit_index)
        for y, x, c in iter_positions(record, channels)
    ]
    return (
        np.asarray(symbols, dtype=np.int32),
        np.full(len(symbols), 0.5, dtype=np.float32),
    )


def choose_records(
    bits: np.ndarray,
    tile_size: int,
    previous_bits: int,
    mode_overhead_bits: int,
    modes: list[str],
) -> tuple[list[Record], float]:
    predictor_bits = {
        mode: pred
        for mode, pred in predictors(bits.view(np.float32)).items()
        if mode in modes
    }
    height, width, _channels = bits.shape
    records = []
    estimated_bits = 0.0
    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            tile = bits[y:y + tile_size, x:x + tile_size]
            tile_height, tile_width, _ = tile.shape
            sl = np.s_[y:y + tile_height, x:x + tile_width]
            tile_values = tile.size
            for field, bit_indices in FIELD_GROUPS.items():
                candidates = {"raw": tile_values * len(bit_indices)}
                for mode, pred in predictor_bits.items():
                    residual = tile ^ pred[sl]
                    candidates[mode] = context_bitplane_cost(
                        residual,
                        bit_indices,
                        previous_bits,
                    )
                mode = min(candidates, key=candidates.__getitem__)
                records.append(Record(y, x, tile_height, tile_width, field, mode))
                estimated_bits += candidates[mode] + mode_overhead_bits
    return records, estimated_bits


def encode(
    bits: np.ndarray,
    records: list[Record],
    previous_bits: int,
    alpha: float,
) -> constriction.stream.stack.AnsCoder:
    coder = constriction.stream.stack.AnsCoder()
    model = constriction.stream.model.Bernoulli(perfect=False)
    for record in reversed(records):
        for bit_index in reversed(FIELD_GROUPS[record.field]):
            if record.mode == "raw":
                symbols, probs = raw_symbols_and_probs(bits, record, bit_index)
            else:
                symbols, probs = adaptive_symbols_and_probs(
                    bits,
                    record,
                    bit_index,
                    previous_bits,
                    alpha,
                )
            coder.encode_reverse(symbols, model, probs)
    return coder


def decode(
    compressed: np.ndarray,
    shape: tuple[int, int, int],
    records: list[Record],
    previous_bits: int,
    alpha: float,
) -> np.ndarray:
    coder = constriction.stream.stack.AnsCoder(compressed)
    model = constriction.stream.model.Bernoulli(perfect=False)
    out = np.zeros(shape, dtype=np.uint32)
    for record in records:
        channels = shape[2]
        for bit_index in FIELD_GROUPS[record.field]:
            if record.mode == "raw":
                symbol_count = record.height * record.width * channels
                probs = np.full(symbol_count, 0.5, dtype=np.float32)
                symbols = coder.decode(model, probs)
                for symbol, (y, x, c) in zip(symbols, iter_positions(record, channels)):
                    set_bit(out, y, x, c, bit_index, int(symbol))
                continue

            context_count = 16 << previous_bits
            ones = np.full(context_count, alpha, dtype=np.float64)
            totals = np.full(context_count, 2.0 * alpha, dtype=np.float64)
            for y, x, c in iter_positions(record, channels):
                context = context_id(out, y, x, c, bit_index, record, previous_bits)
                probability = np.asarray([ones[context] / totals[context]], dtype=np.float32)
                residual = int(coder.decode(model, probability)[0])
                value = residual ^ predictor_bit(out, y, x, c, bit_index, record.mode)
                set_bit(out, y, x, c, bit_index, value)
                ones[context] += residual
                totals[context] += 1.0
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="ph_studio_small_03_1k.exr")
    parser.add_argument("--tile-size", type=int, default=128)
    parser.add_argument("--previous-bits", type=int, default=3)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--mode-overhead-bits", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--modes", nargs="+", default=SAFE_MODES)
    parser.add_argument("--no-roundtrip", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pixels = read_exr(DATA_DIR / args.image)
    if args.crop_size:
        pixels = pixels[:args.crop_size, :args.crop_size]
    bits = float_to_bits(np.ascontiguousarray(pixels, dtype=np.float32)).copy()
    raw_bits = bits.size * 32

    t0 = time.perf_counter()
    records, estimated_bits = choose_records(
        bits,
        args.tile_size,
        args.previous_bits,
        args.mode_overhead_bits,
        args.modes,
    )
    t1 = time.perf_counter()
    coder = encode(bits, records, args.previous_bits, args.alpha)
    compressed = coder.get_compressed()
    t2 = time.perf_counter()
    side_bits = len(records) * args.mode_overhead_bits
    payload_bytes = compressed.nbytes + math.ceil(side_bits / 8)
    ok = None
    decode_sec = None
    if not args.no_roundtrip:
        decoded = decode(compressed, bits.shape, records, args.previous_bits, args.alpha)
        decode_sec = time.perf_counter() - t2
        ok = bool(np.array_equal(decoded, bits))

    print(f"image:        {args.image}")
    print(f"shape:        {bits.shape}")
    print(f"modes:        {', '.join(args.modes)}")
    print(f"records:      {len(records)}")
    print(f"raw:          {raw_bits / 8:.0f} bytes")
    print(
        f"estimate:     {estimated_bits / 8:.0f} bytes "
        f"({raw_bits / estimated_bits:.3f}x)"
    )
    print(
        f"rANS payload: {payload_bytes:.0f} bytes "
        f"({(raw_bits / 8) / payload_bytes:.3f}x)"
    )
    print(f"valid bits:   {coder.num_valid_bits() + side_bits}")
    print(f"select:       {t1 - t0:.2f}s")
    print(f"encode:       {t2 - t1:.2f}s")
    if ok is not None:
        print(f"decode:       {decode_sec:.2f}s")
        print(f"roundtrip:    {ok}")
    return 0 if ok is not False else 1


if __name__ == "__main__":
    raise SystemExit(main())
