"""Audit float32 mantissa-tail structure before and after grouped-delta.

This looks for reversible structure that bitplane contexts may hide:

* raw mantissa trailing-zero / low-k-zero rates, which reveal half/quantized or
  procedurally rounded sources;
* exponent-bucket low-tail entropy, which tests whether exponent explains the
  apparent random tail;
* grouped-delta payload tail entropy for the current production-shaped
  representation.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_structural_predictors import DATA_DIR, float_to_bits, read_exr  # noqa: E402
from evaluate_structural_delta_context import bits_to_ordered  # noqa: E402
from structural_delta_grouped_roundtrip import choose_group_payloads  # noqa: E402

RESULTS_DIR = ROOT / "results"


def entropy_bits(values: np.ndarray, width: int) -> float:
    if values.size == 0:
        return 0.0
    if width <= 12:
        counts = np.bincount(values.astype(np.int64), minlength=1 << width)
        probs = counts[counts > 0].astype(np.float64) / float(values.size)
        return float(-(probs * np.log2(probs)).sum())
    # Large alphabets are summarized bytewise to avoid huge arrays.
    total = 0.0
    for shift in range(0, width, 8):
        chunk_width = min(8, width - shift)
        chunk = (values >> np.uint32(shift)) & np.uint32((1 << chunk_width) - 1)
        total += entropy_bits(chunk, chunk_width)
    return total


def trailing_zero_count(values: np.ndarray, max_bits: int) -> np.ndarray:
    counts = np.zeros(values.shape, dtype=np.uint8)
    active = np.ones(values.shape, dtype=bool)
    for bit in range(max_bits):
        zero = ((values >> np.uint32(bit)) & np.uint32(1)) == 0
        counts += (active & zero).astype(np.uint8)
        active &= zero
    return counts


def low_zero_rates(values: np.ndarray, max_bits: int) -> dict[str, float]:
    flat = values.reshape(-1)
    rates = {}
    for bits in range(1, max_bits + 1):
        mask = np.uint32((1 << bits) - 1)
        rates[str(bits)] = float(np.mean((flat & mask) == 0))
    return rates


def exponent_bucket_stats(bits: np.ndarray, tail_bits: int) -> list[dict[str, float]]:
    exponent = ((bits >> np.uint32(23)) & np.uint32(0xff)).reshape(-1)
    mantissa = (bits & np.uint32(0x7fffff)).reshape(-1)
    tail = mantissa & np.uint32((1 << tail_bits) - 1)
    rows = []
    for exp, count in Counter(int(v) for v in exponent).most_common():
        if count < 128:
            continue
        selected = tail[exponent == exp]
        rows.append(
            {
                "exponent": exp,
                "count": int(count),
                "tail_entropy_bits": entropy_bits(selected, tail_bits),
                "tail_bits_per_bit": entropy_bits(selected, tail_bits) / tail_bits,
                "low8_zero_rate": float(np.mean((selected & np.uint32(0xff)) == 0)),
            }
        )
    return rows[:12]


def bitplane_entropy(values: np.ndarray, bits: range) -> dict[str, float]:
    flat = values.reshape(-1)
    out = {}
    for bit in bits:
        plane = ((flat >> np.uint32(bit)) & np.uint32(1)).astype(np.uint8)
        p = float(plane.mean())
        if p <= 0.0 or p >= 1.0:
            h = 0.0
        else:
            h = -(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p))
        out[str(bit)] = h
    return out


def summarize_image(path: Path, crop_size: int, tile_size: int, previous_bits: int) -> dict:
    pixels = read_exr(path)
    if crop_size:
        pixels = pixels[:crop_size, :crop_size]
    pixels = np.ascontiguousarray(pixels, dtype=np.float32)
    bits = float_to_bits(pixels).copy()
    mantissa = bits & np.uint32(0x7fffff)
    ordered = bits_to_ordered(bits)

    modes, payloads, _carries = choose_group_payloads(
        bits,
        pixels,
        tile_size,
        previous_bits,
        "body",
    )
    payload = payloads["body"]
    mode_histogram = Counter(modes.values())
    tz = trailing_zero_count(mantissa, 23)
    raw_low15 = mantissa & np.uint32((1 << 15) - 1)
    payload_low15 = payload & np.uint32((1 << 15) - 1)
    payload_mid = (payload >> np.uint32(15)) & np.uint32((1 << 6) - 1)

    return {
        "image": path.name,
        "shape": list(bits.shape),
        "crop_size": crop_size,
        "tile_size": tile_size,
        "previous_bits": previous_bits,
        "mode_histogram": dict(mode_histogram),
        "raw_low15_entropy_bits": entropy_bits(raw_low15.reshape(-1), 15),
        "payload_low15_entropy_bits": entropy_bits(payload_low15.reshape(-1), 15),
        "payload_bits15_20_entropy_bits": entropy_bits(payload_mid.reshape(-1), 6),
        "raw_low15_bits_per_bit": entropy_bits(raw_low15.reshape(-1), 15) / 15.0,
        "payload_low15_bits_per_bit": entropy_bits(payload_low15.reshape(-1), 15) / 15.0,
        "payload_bits15_20_bits_per_bit": entropy_bits(payload_mid.reshape(-1), 6) / 6.0,
        "raw_trailing_zero_histogram": {
            str(k): int(v) for k, v in sorted(Counter(int(v) for v in tz.reshape(-1)).items())
        },
        "raw_low_zero_rates": low_zero_rates(mantissa, 16),
        "payload_low_zero_rates": low_zero_rates(payload, 16),
        "raw_exponent_tail15_top": exponent_bucket_stats(bits, 15),
        "raw_bitplane_entropy_0_22": bitplane_entropy(mantissa, range(0, 23)),
        "payload_bitplane_entropy_0_30": bitplane_entropy(payload, range(0, 31)),
        "ordered_low_zero_rates": low_zero_rates(ordered, 16),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", default="ph_*_1k.exr")
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--tile-size", type=int, default=128)
    parser.add_argument("--previous-bits", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = []
    for path in sorted(DATA_DIR.glob(args.glob)):
        row = summarize_image(path, args.crop_size, args.tile_size, args.previous_bits)
        rows.append(row)
        print(
            f"{path.stem:43s} "
            f"raw_low15={row['raw_low15_bits_per_bit']:.3f} "
            f"payload_low15={row['payload_low15_bits_per_bit']:.3f} "
            f"payload_15_20={row['payload_bits15_20_bits_per_bit']:.3f} "
            f"low8zero={row['raw_low_zero_rates']['8']:.3f}"
        )
        if args.limit and len(rows) >= args.limit:
            break
    output = RESULTS_DIR / (
        f"float_tail_structure_{args.glob.replace('*', 'star').replace('.', '_')}"
        f"_crop{args.crop_size}.json"
    )
    output.write_text(json.dumps(rows, indent=2))
    print(f"\nSaved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
