"""Estimate a dedicated linear-range index route for 32x/64x near-lossless.

The current product path quantizes values and then sends the quantized float32
image to GroupedDelta.  This probe instead measures the entropy of the actual
N-bit index image produced by linear_range quantization.  It is a lower-bound
probe for a future dedicated index codec, not a bitstream implementation.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import OpenImageIO as oiio

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"


def read_exr(path: Path, crop_size: int) -> np.ndarray:
    image_input = oiio.ImageInput.open(str(path))
    if image_input is None:
        raise RuntimeError(f"can't open {path}: {oiio.geterror()}")
    spec = image_input.spec()
    height = spec.height if crop_size == 0 else min(crop_size, spec.height)
    width = spec.width if crop_size == 0 else min(crop_size, spec.width)
    if crop_size:
        pixels = image_input.read_scanlines(
            0, height, 0, 0, spec.nchannels, oiio.FLOAT)
        image_input.close()
        if pixels is None:
            raise RuntimeError(f"can't read {path}: {oiio.geterror()}")
        return np.asarray(pixels, dtype=np.float32).reshape(
            height, spec.width, spec.nchannels)[:, :width, :]
    pixels = image_input.read_image(format=oiio.FLOAT)
    image_input.close()
    if pixels is None:
        raise RuntimeError(f"can't read {path}: {oiio.geterror()}")
    return np.asarray(pixels, dtype=np.float32).reshape(
        spec.height, spec.width, spec.nchannels)


def entropy(values: np.ndarray, alphabet: int | None = None) -> float:
    flat = values.reshape(-1).astype(np.int64)
    counts = np.bincount(flat, minlength=alphabet or 0)
    counts = counts[counts > 0]
    if counts.size == 0:
        return 0.0
    probs = counts / counts.sum()
    return float(-(probs * np.log2(probs)).sum())


def linear_indices(pixels: np.ndarray, bits: int) -> np.ndarray:
    levels = (1 << bits) - 1
    out = np.zeros(pixels.shape, dtype=np.uint16)
    for channel in range(pixels.shape[2]):
        values = pixels[:, :, channel].astype(np.float64)
        finite = np.isfinite(values)
        if not bool(np.any(finite)):
            continue
        lo = float(values[finite].min())
        hi = float(values[finite].max())
        if hi <= lo:
            continue
        quantized = np.floor((values - lo) / (hi - lo) * levels + 0.5)
        out[:, :, channel] = np.clip(quantized, 0, levels).astype(np.uint16)
    return out


def avg_predictor(indices: np.ndarray) -> np.ndarray:
    west = np.zeros_like(indices)
    north = np.zeros_like(indices)
    west[:, 1:] = indices[:, :-1]
    north[1:] = indices[:-1]
    return ((west.astype(np.uint32) + north.astype(np.uint32)) // 2).astype(
        np.uint16)


def packbits_little(mask: np.ndarray) -> bytes:
    return np.packbits(mask.reshape(-1).astype(np.uint8), bitorder="little").tobytes()


def zstd_size(data: bytes, level: int = 19) -> int | None:
    if shutil.which("zstd") is None:
        return None
    with tempfile.TemporaryDirectory() as directory:
        raw_path = os.path.join(directory, "raw.bin")
        zst_path = os.path.join(directory, "raw.bin.zst")
        with open(raw_path, "wb") as handle:
            handle.write(data)
        subprocess.run(
            ["zstd", f"-{level}", "--ultra", "-q", "-f", raw_path, "-o", zst_path],
            check=True,
        )
        return os.path.getsize(zst_path)


def kt_binary_context_cost(mask: np.ndarray) -> float:
    counts = np.full((8, 2), 0.5, dtype=np.float64)
    cost = 0.0
    height, width = mask.shape
    for y in range(height):
        for x in range(width):
            left = int(mask[y, x - 1]) if x else 0
            up = int(mask[y - 1, x]) if y else 0
            up_left = int(mask[y - 1, x - 1]) if y and x else 0
            context = left | (up << 1) | (up_left << 2)
            bit = int(mask[y, x])
            total = counts[context].sum()
            cost -= math.log2(counts[context, bit] / total)
            counts[context, bit] += 1.0
    return cost


def tiled_context_mask_cost(mask: np.ndarray, tile_size: int) -> float:
    cost = 0.0
    for y in range(0, mask.shape[0], tile_size):
        for x in range(0, mask.shape[1], tile_size):
            cost += kt_binary_context_cost(mask[y:y + tile_size, x:x + tile_size])
    return cost


def tile_entropy_bps(
    residual: np.ndarray,
    alphabet: int,
    tile_size: int,
) -> float:
    total_bits = 0.0
    total_samples = residual.size
    for y in range(0, residual.shape[0], tile_size):
        for x in range(0, residual.shape[1], tile_size):
            tile = residual[y:y + tile_size, x:x + tile_size]
            total_bits += entropy(tile, alphabet) * tile.size
    return total_bits / total_samples if total_samples else 0.0


def summarize(path: Path, crop_size: int, bits_values: tuple[int, ...],
              tile_sizes: tuple[int, ...]) -> dict:
    pixels = np.ascontiguousarray(read_exr(path, crop_size), dtype=np.float32)
    rows = []
    for bits in bits_values:
        alphabet = 1 << bits
        indices = linear_indices(pixels, bits)
        predictor = avg_predictor(indices)
        residual = ((indices.astype(np.int32) - predictor.astype(np.int32))
                    & (alphabet - 1)).astype(np.uint16)
        raw_h = entropy(indices, alphabet)
        residual_h = entropy(residual, alphabet)
        residual_stream = b"".join(
            residual[:, :, channel].astype(np.uint8).tobytes()
            for channel in range(indices.shape[2])
        )
        residual_zstd = zstd_size(residual_stream)
        split_masks = []
        split_values = []
        for channel in range(indices.shape[2]):
            channel_residual = residual[:, :, channel].astype(np.uint8)
            nonzero = channel_residual != 0
            split_masks.append(packbits_little(nonzero))
            split_values.append(
                (channel_residual[nonzero] - 1).astype(np.uint8).tobytes())
        split_mask_stream = b"".join(split_masks)
        split_value_stream = b"".join(split_values)
        split_mask_zstd = zstd_size(split_mask_stream)
        split_value_zstd = zstd_size(split_value_stream)
        header_bits = 512.0
        zstd_rows = {}
        if residual_zstd is not None:
            residual_zstd_bps = (
                residual_zstd * 8.0 + header_bits) / indices.size
            zstd_rows["residual_zstd_bytes"] = residual_zstd
            zstd_rows["residual_zstd_bps"] = residual_zstd_bps
            zstd_rows["residual_zstd_ratio_vs_float32"] = 32.0 / residual_zstd_bps
        if split_mask_zstd is not None and split_value_zstd is not None:
            split_zstd_bps = (
                (split_mask_zstd + split_value_zstd) * 8.0 + header_bits
            ) / indices.size
            zstd_rows["split_mask_zstd_bytes"] = split_mask_zstd
            zstd_rows["split_value_zstd_bytes"] = split_value_zstd
            zstd_rows["split_zstd_bps"] = split_zstd_bps
            zstd_rows["split_zstd_ratio_vs_float32"] = 32.0 / split_zstd_bps
        tile_rows = []
        for tile_size in tile_sizes:
            all_bps = 0.0
            context_mask_bits = 0.0
            nz_entropy_bits = 0.0
            channel_rows = []
            for channel in range(indices.shape[2]):
                channel_residual = residual[:, :, channel]
                channel_bps = tile_entropy_bps(channel_residual, alphabet, tile_size)
                channel_rows.append(channel_bps)
                all_bps += channel_bps * indices[:, :, channel].size
                nonzero = channel_residual != 0
                context_mask_bits += tiled_context_mask_cost(nonzero, tile_size)
                nonzero_values = (channel_residual[nonzero] - 1).astype(np.int64)
                nz_entropy_bits += (
                    entropy(nonzero_values, alphabet - 1) * nonzero_values.size
                    if nonzero_values.size else 0.0
                )
            all_bps /= indices.size
            context_nz_entropy_bps = (
                context_mask_bits + nz_entropy_bits + header_bits
            ) / indices.size
            context_nz_zstd_bps = None
            if split_value_zstd is not None:
                context_nz_zstd_bps = (
                    context_mask_bits + split_value_zstd * 8.0 + header_bits
                ) / indices.size
            tile_rows.append({
                "tile_size": tile_size,
                "oracle_bps": all_bps,
                "oracle_ratio_vs_float32": 32.0 / all_bps if all_bps else math.inf,
                "channel_bps": channel_rows,
                "context_mask_plus_nz_entropy_bps": context_nz_entropy_bps,
                "context_mask_plus_nz_entropy_ratio_vs_float32": (
                    32.0 / context_nz_entropy_bps
                    if context_nz_entropy_bps else math.inf
                ),
                "context_mask_plus_nz_zstd_bps": context_nz_zstd_bps,
                "context_mask_plus_nz_zstd_ratio_vs_float32": (
                    32.0 / context_nz_zstd_bps
                    if context_nz_zstd_bps else None
                ),
            })
        rows.append({
            "bits": bits,
            "target_bps_for_64x": 0.5,
            "raw_index_entropy_bps": raw_h,
            "avg_residual_entropy_bps": residual_h,
            "avg_residual_ratio_vs_float32": (
                32.0 / residual_h if residual_h else math.inf
            ),
            **zstd_rows,
            "tile_oracle": tile_rows,
        })
    return {
        "image": path.name,
        "shape": list(pixels.shape),
        "crop_size": crop_size,
        "rows": rows,
    }


def parse_ints(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split(",") if part)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", default="sample_DSCF0009.EXR")
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--bits", default="7")
    parser.add_argument("--tile-sizes", default="16,32,64,128")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    bits_values = parse_ints(args.bits)
    tile_sizes = parse_ints(args.tile_sizes)
    results = []
    for path in sorted(DATA_DIR.glob(args.glob)):
        row = summarize(path, args.crop_size, bits_values, tile_sizes)
        results.append(row)
        print(path.stem)
        for item in row["rows"]:
            print(
                f"  bits{item['bits']:02d}: "
                f"rawH={item['raw_index_entropy_bps']:.4f} "
                f"avgResH={item['avg_residual_entropy_bps']:.4f} "
                f"avgResRatio={item['avg_residual_ratio_vs_float32']:.2f}x")
            for tile in item["tile_oracle"]:
                print(
                    f"    tile{tile['tile_size']:03d}: "
                    f"oracle={tile['oracle_bps']:.4f}bps "
                    f"ratio={tile['oracle_ratio_vs_float32']:.2f}x "
                    f"ctx+nzH={tile['context_mask_plus_nz_entropy_bps']:.4f}bps "
                    f"ratio={tile['context_mask_plus_nz_entropy_ratio_vs_float32']:.2f}x "
                    f"ctx+nzZ={tile['context_mask_plus_nz_zstd_bps']:.4f}bps "
                    f"ratio={tile['context_mask_plus_nz_zstd_ratio_vs_float32']:.2f}x "
                    f"ch={','.join(f'{v:.3f}' for v in tile['channel_bps'])}")
            if "split_zstd_bps" in item:
                print(
                    f"    zstd split: {item['split_zstd_bps']:.4f}bps "
                    f"ratio={item['split_zstd_ratio_vs_float32']:.2f}x "
                    f"mask={item['split_mask_zstd_bytes']}B "
                    f"values={item['split_value_zstd_bytes']}B")
    if not args.no_save:
        safe_glob = args.glob.replace("*", "star").replace(".", "_")
        output = RESULTS_DIR / (
            f"linear_index_route_{safe_glob}_crop{args.crop_size}"
            f"_bits{args.bits.replace(',', '-')}"
            f"_tile{args.tile_sizes.replace(',', '-')}.json"
        )
        output.write_text(json.dumps(results, indent=2))
        print(f"\nSaved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
