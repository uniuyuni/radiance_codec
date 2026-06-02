"""rANS probe for the structural float-field predictor.

Encodes selected tile/field bit planes with constriction to check whether the
entropy estimates from evaluate_structural_hybrid.py are realistic.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from collections import defaultdict

import constriction
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_structural_predictors import (  # noqa: E402
    DATA_DIR,
    FIELD_GROUPS,
    binary_entropy,
    float_to_bits,
    predictors,
    read_exr,
)


def field_bitstreams(values: np.ndarray, bit_indices: tuple[int, ...]) -> list[np.ndarray]:
    flat = values.reshape(-1)
    return [((flat >> bit_index) & 1).astype(np.int32) for bit_index in bit_indices]


def entropy_bits(symbols: np.ndarray) -> float:
    probability = float(symbols.mean()) if symbols.size else 0.0
    return binary_entropy(probability) * symbols.size


def entropy_bits_per_plane(symbols_by_plane: list[np.ndarray]) -> float:
    return sum(entropy_bits(symbols) for symbols in symbols_by_plane)


def encode_binary_stream(
    coder: constriction.stream.stack.AnsCoder,
    symbols: np.ndarray,
) -> int:
    ones = int(symbols.sum())
    zeros = int(symbols.size - ones)
    if symbols.size == 0:
        return 0
    # Smoothed binary probability table. The count header is accounted for
    # separately by callers; this PMF must simply be decodable.
    probs = np.array([zeros + 1, ones + 1], dtype=np.float32)
    probs /= probs.sum()
    model = constriction.stream.model.Categorical(probs, perfect=False)
    before = coder.num_valid_bits()
    coder.encode_reverse(symbols.astype(np.int32), model)
    return coder.num_valid_bits() - before


def encode_binary_planes(
    coder: constriction.stream.stack.AnsCoder,
    symbols_by_plane: list[np.ndarray],
) -> int:
    valid_bits = 0
    for symbols in symbols_by_plane:
        valid_bits += encode_binary_stream(coder, symbols)
    return valid_bits


def probe(
    image: Path,
    tile_size: int,
    fields: list[str],
    count_header_bits: int,
) -> dict:
    pixels = read_exr(image)
    bits = float_to_bits(pixels)
    predictor_bits = predictors(pixels)
    height, width, channels = bits.shape
    total_raw_bits = 0
    total_entropy_bits = 0.0
    total_valid_bits = 0
    raw_payload_bits = 0
    header_bits_total = 0
    coder = constriction.stream.stack.AnsCoder()
    mode_counts = {}
    t0 = time.perf_counter()

    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            sl = np.s_[y:y + tile_size, x:x + tile_size]
            tile_values = bits[sl].size
            for field in fields:
                bit_indices = FIELD_GROUPS[field]
                best_mode = "raw"
                best_symbols = None
                best_entropy = tile_values * len(bit_indices)
                for name, pred in predictor_bits.items():
                    residual = bits[sl] ^ pred[sl]
                    symbols = field_bitstreams(residual, bit_indices)
                    cost = entropy_bits_per_plane(symbols)
                    if cost < best_entropy:
                        best_entropy = cost
                        best_symbols = symbols
                        best_mode = name
                total_raw_bits += tile_values * len(bit_indices)
                mode_counts[best_mode] = mode_counts.get(best_mode, 0) + 1
                if best_mode == "raw":
                    total_entropy_bits += tile_values * len(bit_indices)
                    raw_payload_bits += tile_values * len(bit_indices)
                    total_valid_bits += tile_values * len(bit_indices)
                    continue
                valid_bits = encode_binary_planes(coder, best_symbols)
                header_bits = count_header_bits * len(bit_indices)
                total_entropy_bits += best_entropy + header_bits
                header_bits_total += header_bits
                total_valid_bits += valid_bits + header_bits

    elapsed = time.perf_counter() - t0
    total_payload_bytes = (
        coder.get_compressed().nbytes
        + math.ceil(raw_payload_bits / 8)
        + math.ceil(header_bits_total / 8)
    )
    return {
        "image": image.name,
        "shape": [height, width, channels],
        "tile_size": tile_size,
        "fields": fields,
        "raw_bits": total_raw_bits,
        "entropy_estimated_bits": total_entropy_bits,
        "rans_valid_bits": total_valid_bits,
        "payload_bytes": total_payload_bytes,
        "entropy_ratio": total_raw_bits / total_entropy_bits,
        "payload_ratio": (total_raw_bits / 8) / total_payload_bytes,
        "mode_counts": mode_counts,
        "elapsed_sec": elapsed,
    }


def probe_grouped(
    image: Path,
    tile_size: int,
    fields: list[str],
    count_header_bits: int,
    mode_overhead_bits: int,
) -> dict:
    pixels = read_exr(image)
    bits = float_to_bits(pixels)
    predictor_bits = predictors(pixels)
    height, width, channels = bits.shape
    total_raw_bits = 0
    total_entropy_bits = 0.0
    raw_payload_bits = 0
    mode_bits = 0
    mode_counts = {}
    groups: dict[tuple[str, int, str], list[np.ndarray]] = defaultdict(list)
    t0 = time.perf_counter()

    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            sl = np.s_[y:y + tile_size, x:x + tile_size]
            tile_values = bits[sl].size
            for field in fields:
                bit_indices = FIELD_GROUPS[field]
                best_mode = "raw"
                best_symbols = None
                best_entropy = tile_values * len(bit_indices)
                for name, pred in predictor_bits.items():
                    residual = bits[sl] ^ pred[sl]
                    symbols = field_bitstreams(residual, bit_indices)
                    cost = entropy_bits_per_plane(symbols)
                    if cost < best_entropy:
                        best_entropy = cost
                        best_symbols = symbols
                        best_mode = name
                total_raw_bits += tile_values * len(bit_indices)
                mode_bits += mode_overhead_bits
                mode_counts[best_mode] = mode_counts.get(best_mode, 0) + 1
                if best_mode == "raw":
                    raw_payload_bits += tile_values * len(bit_indices)
                    total_entropy_bits += tile_values * len(bit_indices)
                    continue
                total_entropy_bits += best_entropy
                for bit_index, symbols in zip(bit_indices, best_symbols):
                    groups[(field, bit_index, best_mode)].append(symbols)

    coder = constriction.stream.stack.AnsCoder()
    header_bits_total = 0
    total_valid_bits = 0
    for _key, chunks in groups.items():
        symbols = np.concatenate(chunks)
        header_bits_total += count_header_bits
        total_valid_bits += encode_binary_stream(coder, symbols)
    total_payload_bytes = (
        coder.get_compressed().nbytes
        + math.ceil(raw_payload_bits / 8)
        + math.ceil((header_bits_total + mode_bits) / 8)
    )
    total_entropy_bits += header_bits_total + mode_bits
    elapsed = time.perf_counter() - t0
    return {
        "image": image.name,
        "shape": [height, width, channels],
        "tile_size": tile_size,
        "fields": fields,
        "raw_bits": total_raw_bits,
        "entropy_estimated_bits": total_entropy_bits,
        "rans_valid_bits": total_valid_bits + header_bits_total + mode_bits + raw_payload_bits,
        "payload_bytes": total_payload_bytes,
        "entropy_ratio": total_raw_bits / total_entropy_bits,
        "payload_ratio": (total_raw_bits / 8) / total_payload_bytes,
        "mode_counts": mode_counts,
        "groups": len(groups),
        "elapsed_sec": elapsed,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="ph_studio_small_03_1k.exr")
    parser.add_argument("--tile-size", type=int, default=16)
    parser.add_argument(
        "--fields",
        nargs="+",
        default=["exponent", "mantissa_hi"],
        choices=sorted(FIELD_GROUPS),
    )
    parser.add_argument("--count-header-bits", type=int, default=16)
    parser.add_argument("--mode-overhead-bits", type=int, default=4)
    parser.add_argument("--per-tile-streams", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.per_tile_streams:
        result = probe(DATA_DIR / args.image, args.tile_size, args.fields, args.count_header_bits)
    else:
        result = probe_grouped(
            DATA_DIR / args.image,
            args.tile_size,
            args.fields,
            args.count_header_bits,
            args.mode_overhead_bits,
        )
    print(f"image:       {result['image']}")
    print(f"fields:      {', '.join(result['fields'])}")
    print(f"raw:         {result['raw_bits'] / 8:.0f} bytes")
    print(f"entropy est: {result['entropy_estimated_bits'] / 8:.0f} bytes "
          f"({result['entropy_ratio']:.3f}x)")
    print(f"rANS payload:{result['payload_bytes']:.0f} bytes "
          f"({result['payload_ratio']:.3f}x)")
    print(f"valid bits:  {result['rans_valid_bits']}")
    print(f"modes:       {result['mode_counts']}")
    if "groups" in result:
        print(f"groups:      {result['groups']}")
    print(f"elapsed:     {result['elapsed_sec']:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
