"""rANS roundtrip for grouped ordered-float delta payloads."""
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

from analyze_structural_predictors import DATA_DIR, float_to_bits, read_exr  # noqa: E402
from evaluate_structural_delta_grouped import (  # noqa: E402
    GROUP_PRESETS,
    group_bit_indices,
)
from structural_delta_grouped_roundtrip import (  # noqa: E402
    choose_group_payloads,
    decode_grouped,
)
from evaluate_structural_delta_context import bits_to_ordered  # noqa: E402


@dataclass(frozen=True)
class Record:
    y: int
    x: int
    height: int
    width: int
    group_name: str
    fields: tuple[str, ...]
    mode: str


def bit_at(values: np.ndarray, y: int, x: int, c: int, bit: int) -> int:
    return int((int(values[y, x, c]) >> bit) & 1)


def set_bit(values: np.ndarray, y: int, x: int, c: int, bit: int, value: int) -> None:
    mask = np.uint32(1 << bit)
    if value:
        values[y, x, c] |= mask
    else:
        values[y, x, c] &= ~mask


def context_id(
    payloads: dict[str, np.ndarray],
    record: Record,
    y: int,
    x: int,
    c: int,
    bit: int,
    previous_bits: int,
) -> int:
    values = payloads[record.group_name]
    local_y = y - record.y
    local_x = x - record.x
    context = 0
    if local_x > 0:
        context |= bit_at(values, y, x - 1, c, bit)
    if local_y > 0:
        context |= bit_at(values, y - 1, x, c, bit) << 1
    if local_y > 0 and local_x > 0:
        context |= bit_at(values, y - 1, x - 1, c, bit) << 2
    if local_y > 0 and local_x + 1 < record.width:
        context |= bit_at(values, y - 1, x + 1, c, bit) << 3
    for offset in range(1, previous_bits + 1):
        previous_bit = bit + offset
        if previous_bit <= 31:
            context |= bit_at(values, y, x, c, previous_bit) << (3 + offset)
    return context


def iter_positions(record: Record, channels: int):
    for y in range(record.y, record.y + record.height):
        for x in range(record.x, record.x + record.width):
            for c in range(channels):
                yield y, x, c


def make_records(
    modes: dict[tuple[int, int, str], str],
    shape: tuple[int, int, int],
    tile_size: int,
    preset: str,
) -> list[Record]:
    height, width, _channels = shape
    records = []
    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            tile_height = min(tile_size, height - y)
            tile_width = min(tile_size, width - x)
            for group_name, fields in GROUP_PRESETS[preset]:
                records.append(
                    Record(
                        y=y,
                        x=x,
                        height=tile_height,
                        width=tile_width,
                        group_name=group_name,
                        fields=fields,
                        mode=modes[(y, x, group_name)],
                    )
                )
    return records


def adaptive_symbols_and_probs(
    payloads: dict[str, np.ndarray],
    record: Record,
    bit: int,
    previous_bits: int,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    channels = next(iter(payloads.values())).shape[2]
    context_count = 16 << previous_bits
    ones = np.full(context_count, alpha, dtype=np.float64)
    totals = np.full(context_count, 2.0 * alpha, dtype=np.float64)
    symbols = []
    probs = []
    values = payloads[record.group_name]
    for y, x, c in iter_positions(record, channels):
        context = context_id(payloads, record, y, x, c, bit, previous_bits)
        symbol = bit_at(values, y, x, c, bit)
        symbols.append(symbol)
        probs.append(ones[context] / totals[context])
        ones[context] += symbol
        totals[context] += 1.0
    return np.asarray(symbols, dtype=np.int32), np.asarray(probs, dtype=np.float32)


def raw_symbols_and_probs(
    payloads: dict[str, np.ndarray],
    record: Record,
    bit: int,
) -> tuple[np.ndarray, np.ndarray]:
    channels = next(iter(payloads.values())).shape[2]
    values = payloads[record.group_name]
    symbols = [
        bit_at(values, y, x, c, bit)
        for y, x, c in iter_positions(record, channels)
    ]
    return (
        np.asarray(symbols, dtype=np.int32),
        np.full(len(symbols), 0.5, dtype=np.float32),
    )


def encode(
    payloads: dict[str, np.ndarray],
    records: list[Record],
    previous_bits: int,
    alpha: float,
) -> constriction.stream.stack.AnsCoder:
    coder = constriction.stream.stack.AnsCoder()
    model = constriction.stream.model.Bernoulli(perfect=False)
    for record in reversed(records):
        for bit in reversed(group_bit_indices(record.fields)):
            if record.mode == "raw":
                symbols, probs = raw_symbols_and_probs(payloads, record, bit)
            else:
                symbols, probs = adaptive_symbols_and_probs(
                    payloads,
                    record,
                    bit,
                    previous_bits,
                    alpha,
                )
            coder.encode_reverse(symbols, model, probs)
    return coder


def decode(
    compressed: np.ndarray,
    modes: dict[tuple[int, int, str], str],
    carries: dict[str, np.ndarray],
    shape: tuple[int, int, int],
    records: list[Record],
    previous_bits: int,
    alpha: float,
    preset: str,
    tile_size: int,
) -> np.ndarray:
    coder = constriction.stream.stack.AnsCoder(compressed)
    model = constriction.stream.model.Bernoulli(perfect=False)
    payloads = {
        group_name: np.zeros(shape, dtype=np.uint32)
        for group_name, _fields in GROUP_PRESETS[preset]
    }
    channels = shape[2]
    for record in records:
        for bit in group_bit_indices(record.fields):
            if record.mode == "raw":
                count = record.height * record.width * channels
                probs = np.full(count, 0.5, dtype=np.float32)
                symbols = coder.decode(model, probs)
                for symbol, (y, x, c) in zip(symbols, iter_positions(record, channels)):
                    set_bit(payloads[record.group_name], y, x, c, bit, int(symbol))
                continue

            context_count = 16 << previous_bits
            ones = np.full(context_count, alpha, dtype=np.float64)
            totals = np.full(context_count, 2.0 * alpha, dtype=np.float64)
            for y, x, c in iter_positions(record, channels):
                context = context_id(payloads, record, y, x, c, bit, previous_bits)
                prob = np.asarray([ones[context] / totals[context]], dtype=np.float32)
                symbol = int(coder.decode(model, prob)[0])
                set_bit(payloads[record.group_name], y, x, c, bit, symbol)
                ones[context] += symbol
                totals[context] += 1.0

    return decode_grouped(modes, payloads, carries, shape, tile_size, preset)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="ph_studio_small_03_1k.exr")
    parser.add_argument("--tile-size", type=int, default=128)
    parser.add_argument("--previous-bits", type=int, default=3)
    parser.add_argument("--preset", choices=sorted(GROUP_PRESETS), default="body")
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
    ordered = bits_to_ordered(bits)
    raw_bits = bits.size * 32

    t0 = time.perf_counter()
    modes, payloads, carries = choose_group_payloads(
        bits,
        pixels,
        args.tile_size,
        args.previous_bits,
        args.preset,
    )
    records = make_records(modes, ordered.shape, args.tile_size, args.preset)
    t1 = time.perf_counter()
    coder = encode(payloads, records, args.previous_bits, args.alpha)
    compressed = coder.get_compressed()
    t2 = time.perf_counter()
    decoded = decode(
        compressed,
        modes,
        carries,
        ordered.shape,
        records,
        args.previous_bits,
        args.alpha,
        args.preset,
        args.tile_size,
    )
    t3 = time.perf_counter()
    ok = bool(np.array_equal(decoded, ordered))
    side_bits = len(records) * args.mode_overhead_bits
    payload_bytes = compressed.nbytes + math.ceil(side_bits / 8)

    print(f"image:        {args.image}")
    print(f"shape:        {ordered.shape}")
    print(f"preset:       {args.preset}")
    print(f"records:      {len(records)}")
    print(f"raw:          {raw_bits / 8:.0f} bytes")
    print(
        f"rANS payload: {payload_bytes:.0f} bytes "
        f"({(raw_bits / 8) / payload_bytes:.3f}x)"
    )
    print(f"valid bits:   {coder.num_valid_bits() + side_bits}")
    print(f"select:       {t1 - t0:.2f}s")
    print(f"encode:       {t2 - t1:.2f}s")
    print(f"decode:       {t3 - t2:.2f}s")
    print(f"roundtrip:    {ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
