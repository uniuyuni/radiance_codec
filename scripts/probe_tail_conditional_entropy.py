"""Probe conditional entropy of raw low mantissa tails as whole symbols.

Bitplane coding can miss structure where the entire low-tail value is predictable
from already-decoded high float fields.  This probe computes lower bounds for
coding the low raw mantissa tail as a symbol conditioned on exponent/high
mantissa/channel/spatial features, then compares those bounds with the current
GDX low-bit family estimate.
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
from probe_grouped_tail_mlp import CONTEXT_FAMILIES, context_family_cost  # noqa: E402
from structural_delta_grouped_roundtrip import choose_group_payloads  # noqa: E402

RESULTS_DIR = ROOT / "results"


def entropy_from_counts(counts: np.ndarray) -> float:
    total = float(np.sum(counts))
    if total == 0.0:
        return 0.0
    probs = counts.astype(np.float64) / total
    return float(-np.sum(probs * np.log2(probs)) * total)


def conditional_symbol_entropy(
    tail: np.ndarray,
    context: np.ndarray,
    width: int,
) -> float:
    flat_tail = tail.reshape(-1).astype(np.uint64)
    flat_context = context.reshape(-1).astype(np.uint64)
    combined = (flat_context << np.uint64(width)) | flat_tail
    unique, counts = np.unique(combined, return_counts=True)
    contexts = unique >> np.uint64(width)
    total = 0.0
    start = 0
    while start < unique.size:
        end = start + 1
        while end < unique.size and contexts[end] == contexts[start]:
            end += 1
        total += entropy_from_counts(counts[start:end])
        start = end
    return total


def fold_hash(values: np.ndarray, bits: int) -> np.ndarray:
    mask = np.uint64((1 << bits) - 1)
    mixed = values.astype(np.uint64)
    mixed ^= mixed >> np.uint64(bits)
    mixed ^= mixed >> np.uint64(bits * 2)
    mixed *= np.uint64(0x9E3779B1)
    mixed ^= mixed >> np.uint64(16)
    return (mixed & mask).astype(np.uint64)


def context_variants(bits: np.ndarray, tail_width: int, hash_bits: int) -> dict[str, np.ndarray]:
    height, width, channels = bits.shape
    exponent = ((bits >> np.uint32(23)) & np.uint32(0xff)).astype(np.uint64)
    mantissa = (bits & np.uint32(0x7fffff)).astype(np.uint64)
    high = mantissa >> np.uint64(tail_width)
    c = np.arange(channels, dtype=np.uint64)[None, None, :]
    channel = np.broadcast_to(c, bits.shape)
    yy, xx = np.mgrid[0:height, 0:width]
    x = np.broadcast_to(xx[:, :, None].astype(np.uint64), bits.shape)
    y = np.broadcast_to(yy[:, :, None].astype(np.uint64), bits.shape)
    high_hash = fold_hash(
        high
        ^ (exponent << np.uint64(17))
        ^ (channel << np.uint64(25)),
        hash_bits,
    )
    high_xy_hash = fold_hash(
        high
        ^ (exponent << np.uint64(17))
        ^ (channel << np.uint64(25))
        ^ ((x & np.uint64(7)) << np.uint64(27))
        ^ ((y & np.uint64(7)) << np.uint64(30)),
        hash_bits,
    )
    return {
        "none": np.zeros(bits.shape, dtype=np.uint64),
        "channel": channel,
        "exponent": exponent,
        "exponent_channel": (exponent << np.uint64(2)) | channel,
        "high4_channel": ((high & np.uint64(0x0f)) << np.uint64(2)) | channel,
        "high8_channel": ((high & np.uint64(0xff)) << np.uint64(2)) | channel,
        "exp_high4_channel": (
            (exponent << np.uint64(6))
            | ((high & np.uint64(0x0f)) << np.uint64(2))
            | channel
        ),
        "exp_high8_channel": (
            (exponent << np.uint64(10))
            | ((high & np.uint64(0xff)) << np.uint64(2))
            | channel
        ),
        f"hash{hash_bits}_exp_high_channel": high_hash,
        f"hash{hash_bits}_exp_high_xy_channel": high_xy_hash,
    }


def best_gdx_low_cost(payload: np.ndarray, width: int) -> float:
    total = 0.0
    for bit in range(width):
        total += min(
            context_family_cost(payload, bit, family)
            for family in CONTEXT_FAMILIES
        )
    return total


def summarize_image(
    path: Path,
    crop_size: int,
    tile_size: int,
    previous_bits: int,
    tail_widths: tuple[int, ...],
    hash_bits: int,
) -> dict:
    pixels = read_exr(path)
    if crop_size:
        pixels = pixels[:crop_size, :crop_size]
    pixels = np.ascontiguousarray(pixels, dtype=np.float32)
    bits = float_to_bits(pixels).copy()
    mantissa = bits & np.uint32(0x7fffff)
    _modes, payloads, _carries = choose_group_payloads(
        bits,
        pixels,
        tile_size,
        previous_bits,
        "body",
    )
    rows = []
    for tail_width in tail_widths:
        mask = np.uint32((1 << tail_width) - 1)
        tail = mantissa & mask
        gdx_cost = 0.0
        conditional_costs = Counter()
        for y in range(0, bits.shape[0], tile_size):
            for x in range(0, bits.shape[1], tile_size):
                sl = np.s_[y:y + tile_size, x:x + tile_size]
                gdx_cost += best_gdx_low_cost(payloads["body"][sl], tail_width)
                variants = context_variants(bits[sl], tail_width, hash_bits)
                for name, context in variants.items():
                    conditional_costs[name] += conditional_symbol_entropy(
                        tail[sl],
                        context,
                        tail_width,
                    )
        samples = int(tail.size)
        best_name, best_cost = min(conditional_costs.items(), key=lambda item: item[1])
        rows.append(
            {
                "tail_width": tail_width,
                "samples": samples,
                "gdx_low_bits": gdx_cost,
                "gdx_bits_per_sample": gdx_cost / samples,
                "best_context": best_name,
                "best_conditional_bits": float(best_cost),
                "best_conditional_bits_per_sample": float(best_cost) / samples,
                "conditional_bits_per_sample": {
                    name: float(cost) / samples
                    for name, cost in sorted(conditional_costs.items())
                },
            }
        )
    return {
        "image": path.name,
        "shape": list(bits.shape),
        "crop_size": crop_size,
        "tile_size": tile_size,
        "previous_bits": previous_bits,
        "hash_bits": hash_bits,
        "rows": rows,
    }


def parse_widths(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split(",") if part)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", default="ph_*puresky*.exr")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--tile-size", type=int, default=128)
    parser.add_argument("--previous-bits", type=int, default=4)
    parser.add_argument("--tail-widths", default="8,10,12,15")
    parser.add_argument("--hash-bits", type=int, default=12)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tail_widths = parse_widths(args.tail_widths)
    rows = []
    for path in sorted(DATA_DIR.glob(args.glob)):
        row = summarize_image(
            path,
            args.crop_size,
            args.tile_size,
            args.previous_bits,
            tail_widths,
            args.hash_bits,
        )
        rows.append(row)
        print(path.stem)
        for item in row["rows"]:
            print(
                f"  low{item['tail_width']:02d}: "
                f"gdx={item['gdx_bits_per_sample']:.3f} bps "
                f"best={item['best_conditional_bits_per_sample']:.3f} bps "
                f"{item['best_context']}"
            )
        if args.limit and len(rows) >= args.limit:
            break
    safe_glob = args.glob.replace("*", "star").replace(".", "_")
    output = RESULTS_DIR / (
        f"tail_conditional_entropy_{safe_glob}_tile{args.tile_size}"
        f"_prev{args.previous_bits}_crop{args.crop_size}"
        f"_tail{args.tail_widths.replace(',', '-')}_hash{args.hash_bits}.json"
    )
    output.write_text(json.dumps(rows, indent=2))
    print(f"\nSaved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
