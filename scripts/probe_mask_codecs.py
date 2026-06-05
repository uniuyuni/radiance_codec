"""Probe binary route-mask coding alternatives.

The current route estimate uses order-0 or west/north binary entropy.  This
script keeps the mask fixed and compares simple decodable representations:
context entropy, packed bits with zlib, row/column run lengths, and tile split
modes.  It is useful for both exact and near-lossless routes because mask
coding is independent of the reconstructed values.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import zlib
from pathlib import Path

import numpy as np
import OpenImageIO as oiio

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
sys.path.insert(0, str(ROOT / "scripts"))

from probe_adaptive_ycocg_router import entropy_bits_from_counts  # noqa: E402
from probe_darkbits_router_payload import summarize_binary_mask  # noqa: E402
from probe_vst_chroma_nr import safe_name  # noqa: E402


def read_mask_png(path: Path) -> np.ndarray:
    image_input = oiio.ImageInput.open(str(path))
    if image_input is None:
        raise RuntimeError(f"can't open {path}: {oiio.geterror()}")
    spec = image_input.spec()
    pixels = image_input.read_image(format=oiio.UINT16)
    image_input.close()
    if pixels is None:
        raise RuntimeError(f"can't read {path}: {oiio.geterror()}")
    image = np.asarray(pixels, dtype=np.uint16).reshape(
        spec.height,
        spec.width,
        spec.nchannels,
    )
    return np.any(image[:, :, : min(3, spec.nchannels)] > 0, axis=2)


def packbits_bytes(mask: np.ndarray, order: str) -> bytes:
    if order == "row":
        flat = mask.reshape(-1)
    elif order == "column":
        flat = mask.T.reshape(-1)
    else:
        raise ValueError(order)
    return np.packbits(flat.astype(np.uint8), bitorder="little").tobytes()


def varint_size(value: int) -> int:
    value = int(value)
    size = 1
    while value >= 128:
        value >>= 7
        size += 1
    return size


def run_lengths_1d(bits: np.ndarray) -> list[int]:
    if bits.size == 0:
        return []
    changes = np.flatnonzero(bits[1:] != bits[:-1]) + 1
    edges = np.concatenate(([0], changes, [bits.size]))
    return [int(edges[i + 1] - edges[i]) for i in range(len(edges) - 1)]


def rle_bytes(mask: np.ndarray, axis: str) -> dict:
    total = 0
    run_count = 0
    length_counts: dict[int, int] = {}
    lines = mask if axis == "row" else mask.T
    for line in lines:
        total += 1  # start bit per row/column, rounded up as a byte in this probe.
        runs = run_lengths_1d(line.astype(np.uint8))
        run_count += len(runs)
        for length in runs:
            total += varint_size(length)
            length_counts[length] = length_counts.get(length, 0) + 1
    counts = np.asarray(list(length_counts.values()), dtype=np.int64)
    entropy_bytes = int(math.ceil(entropy_bits_from_counts(counts) / 8.0) + 256)
    return {
        "axis": axis,
        "estimated_bytes_varint": total,
        "run_count": run_count,
        "unique_lengths": int(len(length_counts)),
        "length_entropy_bytes_plus_model": entropy_bytes,
    }


def context_entropy(mask: np.ndarray) -> list[dict]:
    h, w = mask.shape
    yy, xx = np.indices((h, w), dtype=np.int32)
    m = mask.astype(np.int32)
    west = np.zeros(mask.shape, dtype=np.int32)
    north = np.zeros(mask.shape, dtype=np.int32)
    northwest = np.zeros(mask.shape, dtype=np.int32)
    west[:, 1:] = m[:, :-1]
    north[1:, :] = m[:-1, :]
    northwest[1:, 1:] = m[:-1, :-1]
    contexts = {
        "order0": np.zeros(mask.shape, dtype=np.int32),
        "west": west,
        "north": north,
        "west_north": west + 2 * north,
        "west_north_nw": west + 2 * north + 4 * northwest,
        "phase2_west_north": (xx & 1) + 2 * (yy & 1) + 4 * (west + 2 * north),
        "xtrans6_west_north": (xx % 6) + 6 * (yy % 6) + 36 * (west + 2 * north),
    }
    rows = []
    symbols = m.reshape(-1)
    for name, context in contexts.items():
        flat_context = context.reshape(-1).astype(np.int64)
        context_count = int(flat_context.max()) + 1 if flat_context.size else 1
        joint = np.bincount(
            flat_context * 2 + symbols.astype(np.int64),
            minlength=context_count * 2,
        ).reshape(context_count, 2)
        bits = 0.0
        for row in joint:
            bits += entropy_bits_from_counts(row)
        model_bytes = context_count * 2 * 2
        rows.append({
            "name": name,
            "context_count": context_count,
            "estimated_bits": bits,
            "estimated_bytes_plus_model": int(math.ceil(bits / 8.0) + model_bytes + 16),
            "bits_per_sample": bits / float(mask.size),
        })
    rows.sort(key=lambda row: row["estimated_bytes_plus_model"])
    return rows


def tile_split(mask: np.ndarray, tile: int) -> dict:
    h, w = mask.shape
    mode_counts = np.zeros(3, dtype=np.int64)
    raw_mixed_bits = 0
    for y in range(0, h, tile):
        for x in range(0, w, tile):
            block = mask[y:min(y + tile, h), x:min(x + tile, w)]
            count = int(np.count_nonzero(block))
            if count == 0:
                mode_counts[0] += 1
            elif count == block.size:
                mode_counts[1] += 1
            else:
                mode_counts[2] += 1
                raw_mixed_bits += int(block.size)
    mode_bits = entropy_bits_from_counts(mode_counts)
    total_bytes = int(math.ceil((mode_bits + raw_mixed_bits) / 8.0) + 64)
    return {
        "tile": tile,
        "mode_counts": {
            "zero": int(mode_counts[0]),
            "one": int(mode_counts[1]),
            "mixed": int(mode_counts[2]),
        },
        "mode_bits": mode_bits,
        "raw_mixed_bits": raw_mixed_bits,
        "estimated_bytes": total_bytes,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask-png", type=Path, action="append", required=True)
    parser.add_argument("--label", default="mask_codecs")
    args = parser.parse_args()

    masks = [read_mask_png(path) for path in args.mask_png]
    shape = masks[0].shape
    for path, mask in zip(args.mask_png, masks):
        if mask.shape != shape:
            raise RuntimeError(f"shape mismatch for {path}: {mask.shape} != {shape}")
    combined = np.zeros(shape, dtype=bool)
    for mask in masks:
        combined |= mask

    existing = summarize_binary_mask(combined)
    packed = []
    for order in ("row", "column"):
        raw = packbits_bytes(combined, order)
        for level in (1, 6, 9):
            packed.append({
                "order": order,
                "zlib_level": level,
                "raw_packed_bytes": len(raw),
                "compressed_bytes": len(zlib.compress(raw, level)),
            })
    rows = {
        "image_shape": list(shape),
        "mask_pixel_rate": float(np.mean(combined)),
        "input_masks": [str(path) for path in args.mask_png],
        "existing_summary": existing,
        "context_entropy": context_entropy(combined),
        "packed_zlib": packed,
        "rle": [
            rle_bytes(combined, "row"),
            rle_bytes(combined, "column"),
        ],
        "tile_split": [tile_split(combined, tile) for tile in (8, 16, 32, 64, 128)],
        "note": (
            "Mask-only coding probe. All rows are lossless for the mask; zlib "
            "numbers are actual compressed sizes for packed bit streams."
        ),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = RESULTS_DIR / f"{safe_name(args.label)}.json"
    output.write_text(json.dumps(rows, indent=2))
    best_context = rows["context_entropy"][0]
    best_zlib = min(rows["packed_zlib"], key=lambda row: row["compressed_bytes"])
    best_tile = min(rows["tile_split"], key=lambda row: row["estimated_bytes"])
    print(f"mask={rows['mask_pixel_rate']:.2%}")
    print(
        f"existing order0={int(existing['order0']['total_bytes']):,} "
        f"west_north={int(existing['west_north']['total_bytes']):,}"
    )
    print(
        f"best_context={best_context['name']} "
        f"{best_context['estimated_bytes_plus_model']:,} bytes"
    )
    print(
        f"best_zlib={best_zlib['order']} level{best_zlib['zlib_level']} "
        f"{best_zlib['compressed_bytes']:,} bytes"
    )
    print(f"best_tile={best_tile['tile']} {best_tile['estimated_bytes']:,} bytes")
    print(f"summary={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
