"""Audit the implemented StageLinearIndex payload split.

This script does not change codec behavior.  It encodes an image with the
current StageLinearIndex route, parses the outer HDR0 and inner LIDX headers,
and reproduces the transform-index residual stream in Python to report the
mask/value split that is hiding behind a single encoded byte count.
"""
from __future__ import annotations

import argparse
import json
import math
import struct
import sys
import time
from pathlib import Path

import numpy as np
import OpenImageIO as oiio

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
sys.path.insert(0, str(ROOT / "codec" / "python"))

import radiance_codec  # noqa: E402
from benchmark_linear_index_codec import error_stats  # noqa: E402


VALUE_MODES = {
    0: "byte-rans",
    1: "bitplane-rans",
    2: "symbol-rans",
    3: "small-escape-rans",
    4: "small-escape-channel-split-rans",
}

PREDICTOR_MODES = {
    0: "avg",
    1: "med",
}

TRANSFORM_MODES = {
    0: "linear",
    1: "signed-log",
    2: "sqrt",
    3: "gamma075",
    4: "gamma025",
    5: "asinh",
}


def read_exr(path: Path, crop_size: int) -> np.ndarray:
    image_input = oiio.ImageInput.open(str(path))
    if image_input is None:
        raise RuntimeError(f"can't open {path}: {oiio.geterror()}")
    spec = image_input.spec()
    height = spec.height if crop_size == 0 else min(crop_size, spec.height)
    width = spec.width if crop_size == 0 else min(crop_size, spec.width)
    if crop_size:
        pixels = image_input.read_scanlines(
            0,
            height,
            0,
            0,
            spec.nchannels,
            oiio.FLOAT,
        )
    else:
        pixels = image_input.read_image(format=oiio.FLOAT)
    image_input.close()
    if pixels is None:
        raise RuntimeError(f"can't read {path}: {oiio.geterror()}")
    return np.ascontiguousarray(
        np.asarray(pixels, dtype=np.float32).reshape(
            height,
            spec.width,
            spec.nchannels,
        )[:, :width, :]
    )


def parse_hdr0(encoded: bytes) -> dict:
    if len(encoded) < 24 or encoded[:4] != b"HDR0":
        raise RuntimeError("not a HDR0 stream")
    version = encoded[4]
    if version == 1:
        header_size = 21
    elif version == 2:
        header_size = 22
    elif version == 3:
        header_size = 24
    else:
        raise RuntimeError(f"unsupported HDR0 version {version}")
    if len(encoded) < header_size:
        raise RuntimeError("truncated HDR0 header")
    width, height = struct.unpack_from("<II", encoded, 5)
    channels = encoded[13]
    pixel_format = encoded[14]
    stages = struct.unpack_from("<I", encoded, 15)[0]
    rans_mode = encoded[19]
    effort = encoded[20]
    near_lossless_bits = encoded[21] if version >= 2 else 0
    near_lossless_policy = encoded[22] if version >= 3 else 0
    sign_class = encoded[23] if version >= 3 else None
    return {
        "version": version,
        "header_bytes": header_size,
        "width": width,
        "height": height,
        "channels": channels,
        "pixel_format": pixel_format,
        "stages": stages,
        "rans_mode": rans_mode,
        "effort": effort,
        "near_lossless_bits": near_lossless_bits,
        "near_lossless_policy": near_lossless_policy,
        "sign_class": sign_class,
        "payload_bytes": len(encoded) - header_size,
    }


def parse_lidx(payload: bytes) -> dict:
    header_size = 25
    if len(payload) < header_size or payload[:4] != b"LIDX":
        raise RuntimeError("HDR0 payload is not an LIDX stream")
    version = payload[4]
    bits = payload[5]
    tile_size = struct.unpack_from("<H", payload, 6)[0]
    channels = payload[8]
    effort = payload[9]
    value_mode = payload[10]
    predictor_mode = payload[11]
    transform_mode = payload[12]
    range_count = struct.unpack_from("<I", payload, 13)[0]
    mask_payload_bytes = struct.unpack_from("<I", payload, 17)[0]
    value_payload_bytes = struct.unpack_from("<I", payload, 21)[0]
    ranges_bytes = range_count * 8
    expected = header_size + ranges_bytes + mask_payload_bytes + value_payload_bytes
    if len(payload) < expected:
        raise RuntimeError("truncated LIDX payload")
    ranges = []
    p = header_size
    for _ in range(range_count):
        lo_bits, hi_bits = struct.unpack_from("<II", payload, p)
        ranges.append({
            "lo_bits": lo_bits,
            "hi_bits": hi_bits,
            "lo": struct.unpack("<f", struct.pack("<I", lo_bits))[0],
            "hi": struct.unpack("<f", struct.pack("<I", hi_bits))[0],
        })
        p += 8
    return {
        "version": version,
        "bits": bits,
        "tile_size": tile_size,
        "channels": channels,
        "effort": effort,
        "value_mode": VALUE_MODES.get(value_mode, f"unknown-{value_mode}"),
        "predictor_mode": PREDICTOR_MODES.get(
            predictor_mode,
            f"unknown-{predictor_mode}",
        ),
        "transform_mode": TRANSFORM_MODES.get(transform_mode, f"unknown-{transform_mode}"),
        "range_count": range_count,
        "header_bytes": header_size,
        "ranges_bytes": ranges_bytes,
        "mask_payload_bytes": mask_payload_bytes,
        "value_payload_bytes": value_payload_bytes,
        "trailing_bytes": len(payload) - expected,
        "ranges": ranges,
    }


def transform_values(values: np.ndarray, transform: str) -> np.ndarray:
    if transform == "linear":
        return values
    if transform == "signed-log":
        return np.sign(values) * np.log2(1.0 + np.abs(values))
    if transform == "sqrt":
        return np.sign(values) * np.sqrt(np.abs(values))
    if transform == "gamma075":
        return np.sign(values) * np.power(np.abs(values), 0.75)
    if transform == "gamma025":
        return np.sign(values) * np.power(np.abs(values), 0.25)
    if transform == "asinh":
        return np.arcsinh(values)
    raise RuntimeError(f"unknown transform {transform}")


def quantize_indices(pixels: np.ndarray, bits: int, transform: str) -> np.ndarray:
    levels = (1 << bits) - 1
    indices = np.zeros(pixels.shape, dtype=np.uint16)
    for channel in range(pixels.shape[2]):
        values = pixels[:, :, channel].astype(np.float64)
        transformed = transform_values(values, transform)
        range_values = transformed.astype(np.float32).astype(np.float64)
        lo = float(np.min(range_values))
        hi = float(np.max(range_values))
        if not hi > lo:
            continue
        q = np.floor((transformed - lo) / (hi - lo) * levels + 0.5)
        indices[:, :, channel] = np.clip(q, 0, levels).astype(np.uint16)
    return indices


def predict_spatial(channel_indices: np.ndarray, predictor_mode: str) -> np.ndarray:
    west = np.zeros(channel_indices.shape, dtype=np.int32)
    north = np.zeros(channel_indices.shape, dtype=np.int32)
    west[:, 1:] = channel_indices[:, :-1]
    north[1:, :] = channel_indices[:-1, :]
    if predictor_mode == "west":
        return west
    if predictor_mode == "north":
        return north
    if predictor_mode == "avg":
        return (west + north) // 2

    northwest = np.zeros(channel_indices.shape, dtype=np.int32)
    northwest[1:, 1:] = channel_indices[:-1, :-1]
    hi = np.maximum(west, north)
    lo = np.minimum(west, north)
    gradient = west + north - northwest
    return np.where(northwest >= hi, lo, np.where(northwest <= lo, hi, gradient))


def predict_for_channel(indices: np.ndarray, channel: int, predictor_mode: str) -> np.ndarray:
    channel_indices = indices[:, :, channel]
    if predictor_mode in {"west", "north", "avg", "med"}:
        return predict_spatial(channel_indices, predictor_mode)
    if predictor_mode == "zero":
        return np.zeros(channel_indices.shape, dtype=np.int32)
    if predictor_mode == "prev-channel-med0":
        if channel == 0:
            return predict_spatial(channel_indices, "med")
        return indices[:, :, channel - 1].astype(np.int32)
    if predictor_mode == "prev-channel-avg0":
        if channel == 0:
            return predict_spatial(channel_indices, "avg")
        return indices[:, :, channel - 1].astype(np.int32)
    if predictor_mode == "med-prev-avg":
        spatial = predict_spatial(channel_indices, "med")
        if channel == 0:
            return spatial
        previous = indices[:, :, channel - 1].astype(np.int32)
        return (spatial + previous) // 2
    raise RuntimeError(f"unknown predictor mode {predictor_mode}")


def residual_stats(indices: np.ndarray, bits: int, predictor_mode: str) -> dict:
    height, width, channels = indices.shape
    alphabet_mask = (1 << bits) - 1
    residual_counts = np.zeros(alphabet_mask + 1, dtype=np.int64)
    value_counts = np.zeros(alphabet_mask, dtype=np.int64)
    nonzero_by_channel = []
    total_by_channel = [height * width] * channels
    total = height * width * channels
    zero_run_hist = np.zeros(1025, dtype=np.int64)
    previous_nonzero = -1
    sequence_offset = 0
    nonzero_count = 0

    for channel in range(channels):
        channel_indices = indices[:, :, channel]
        pred = predict_for_channel(indices, channel, predictor_mode)
        residual = (
            channel_indices.astype(np.int32) + alphabet_mask + 1 - pred
        ) & alphabet_mask
        flat = residual.reshape(-1)
        residual_counts += np.bincount(
            flat.astype(np.int64),
            minlength=alphabet_mask + 1,
        )
        nonzero = flat != 0
        channel_nonzero = int(nonzero.sum())
        nonzero_by_channel.append(channel_nonzero)
        nonzero_count += channel_nonzero
        if channel_nonzero:
            value_counts += np.bincount(
                (flat[nonzero] - 1).astype(np.int64),
                minlength=alphabet_mask,
            )
            positions = np.flatnonzero(nonzero) + sequence_offset
            runs = np.diff(
                np.concatenate((np.array([previous_nonzero]), positions))
            ) - 1
            zero_run_hist += np.bincount(
                np.minimum(runs, 1024).astype(np.int64),
                minlength=1025,
            )
            previous_nonzero = int(positions[-1])
        sequence_offset += height * width

    final_run = total - previous_nonzero - 1
    zero_run_hist[min(final_run, 1024)] += 1
    zero_run_count = int(zero_run_hist.sum())
    zero_run_sum = total - nonzero_count
    zero_run_p95 = percentile_from_histogram(zero_run_hist, 0.95)
    entropy = entropy_bits_from_array_counts(residual_counts, total)
    value_entropy = entropy_bits_from_array_counts(value_counts, nonzero_count)
    return {
        "samples": total,
        "nonzero_count": nonzero_count,
        "nonzero_rate": nonzero_count / total if total else 0.0,
        "nonzero_by_channel": nonzero_by_channel,
        "nonzero_rate_by_channel": [
            n / total_by_channel[i] if total_by_channel[i] else 0.0
            for i, n in enumerate(nonzero_by_channel)
        ],
        "zero_run_mean": zero_run_sum / zero_run_count if zero_run_count else 0.0,
        "zero_run_p95": zero_run_p95,
        "residual_order0_entropy_bits_per_sample": entropy,
        "value_order0_entropy_bits_per_nonzero": value_entropy,
        "top_residuals": top_counts(residual_counts, 12),
        "top_values": top_counts(value_counts, 12),
    }


def entropy_bits_from_array_counts(counts: np.ndarray, total: int) -> float:
    if total <= 0:
        return 0.0
    entropy = 0.0
    nz = counts[counts > 0].astype(np.float64)
    if nz.size:
        probs = nz / float(total)
        entropy = float(-(probs * np.log2(probs)).sum())
    return float(entropy)


def top_counts(counts: np.ndarray, limit: int) -> list[dict[str, int]]:
    if counts.size == 0:
        return []
    order = np.argsort(counts)[::-1]
    rows = []
    for symbol in order[:limit]:
        count = int(counts[symbol])
        if count == 0:
            break
        rows.append({"symbol": int(symbol), "count": count})
    return rows


def percentile_from_histogram(counts: np.ndarray, quantile: float) -> float:
    total = int(counts.sum())
    if total <= 0:
        return 0.0
    target = int(math.ceil(total * quantile))
    cumulative = np.cumsum(counts)
    index = int(np.searchsorted(cumulative, target, side="left"))
    return float(index)


def summarize(
    path: Path,
    crop_size: int,
    bits: int,
    transform: str,
    skip_quality: bool,
    candidate_predictors: tuple[str, ...],
) -> dict:
    pixels = read_exr(path, crop_size)
    t0 = time.perf_counter()
    encoded = radiance_codec.encode_linear_index_near_lossless(
        pixels,
        bits=bits,
        effort=9,
        transform=transform,
    )
    t1 = time.perf_counter()
    quality = None
    decode_seconds = None
    if skip_quality:
        t2 = t1
    else:
        decoded = radiance_codec.decode(encoded)
        t2 = time.perf_counter()
        quality = error_stats(pixels, decoded)
        decode_seconds = t2 - t1
    hdr0 = parse_hdr0(encoded)
    payload_offset = hdr0["header_bytes"]
    lidx = parse_lidx(encoded[payload_offset:])
    indices = quantize_indices(pixels, bits, transform)
    residual = residual_stats(indices, bits, lidx["predictor_mode"])
    candidate_rows = []
    for predictor in candidate_predictors:
        stats = residual_stats(indices, bits, predictor)
        candidate_rows.append({
            "predictor_mode": predictor,
            "nonzero_rate": stats["nonzero_rate"],
            "nonzero_rate_by_channel": stats["nonzero_rate_by_channel"],
            "residual_order0_entropy_bits_per_sample":
                stats["residual_order0_entropy_bits_per_sample"],
            "value_order0_entropy_bits_per_nonzero":
                stats["value_order0_entropy_bits_per_nonzero"],
            "top_residuals": stats["top_residuals"][:6],
        })
    encoded_bytes = len(encoded)
    raw_bytes = pixels.nbytes
    overhead_bytes = (
        hdr0["header_bytes"]
        + lidx["header_bytes"]
        + lidx["ranges_bytes"]
        + lidx["trailing_bytes"]
    )
    return {
        "image": path.name,
        "shape": list(pixels.shape),
        "crop_size": crop_size,
        "bits": bits,
        "transform": transform,
        "encoded_bytes": encoded_bytes,
        "raw_bytes": raw_bytes,
        "ratio_vs_original": raw_bytes / encoded_bytes,
        "encode_seconds": t1 - t0,
        "decode_seconds": decode_seconds,
        "quality": quality,
        "hdr0": hdr0,
        "lidx": lidx,
        "payload_budget": {
            "outer_header_bytes": hdr0["header_bytes"],
            "lidx_header_bytes": lidx["header_bytes"],
            "range_metadata_bytes": lidx["ranges_bytes"],
            "mask_payload_bytes": lidx["mask_payload_bytes"],
            "value_payload_bytes": lidx["value_payload_bytes"],
            "trailing_bytes": lidx["trailing_bytes"],
            "fixed_overhead_bytes": overhead_bytes,
            "mask_share": lidx["mask_payload_bytes"] / encoded_bytes,
            "value_share": lidx["value_payload_bytes"] / encoded_bytes,
            "fixed_overhead_share": overhead_bytes / encoded_bytes,
            "bytes_per_sample": encoded_bytes / residual["samples"],
            "mask_bits_per_sample": lidx["mask_payload_bytes"] * 8 / residual["samples"],
            "value_bits_per_sample": lidx["value_payload_bytes"] * 8 / residual["samples"],
            "total_bits_per_sample": encoded_bytes * 8 / residual["samples"],
        },
        "residual": residual,
        "candidate_predictors": candidate_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", default="sample_DSCF0009.EXR")
    parser.add_argument("--crop-size", type=int, default=1024)
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--transform", default="gamma075")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--candidate-predictors",
        default="",
        help=(
            "Comma-separated predictor probes. Supported: zero, west, north, "
            "avg, med, prev-channel-med0, prev-channel-avg0, med-prev-avg."
        ),
    )
    parser.add_argument("--skip-quality", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    rows = []
    for path in sorted(DATA_DIR.glob(args.glob)):
        row = summarize(
            path,
            args.crop_size,
            args.bits,
            args.transform,
            args.skip_quality,
            tuple(
                part.strip()
                for part in args.candidate_predictors.split(",")
                if part.strip()
            ),
        )
        rows.append(row)
        budget = row["payload_budget"]
        residual = row["residual"]
        decode_text = (
            "skipped"
            if row["decode_seconds"] is None
            else f"{row['decode_seconds']:.3f}s"
        )
        print(path.name)
        print(
            f"  {args.transform} bits{args.bits}: "
            f"{row['ratio_vs_original']:.2f}x "
            f"bytes={row['encoded_bytes']} "
            f"enc={row['encode_seconds']:.3f}s "
            f"dec={decode_text}"
        )
        print(
            f"  split: mask={budget['mask_payload_bytes']} "
            f"({budget['mask_bits_per_sample']:.3f} b/smp), "
            f"value={budget['value_payload_bytes']} "
            f"({budget['value_bits_per_sample']:.3f} b/smp), "
            f"overhead={budget['fixed_overhead_bytes']}"
        )
        print(
            f"  residual: predictor={row['lidx']['predictor_mode']} "
            f"value_mode={row['lidx']['value_mode']} "
            f"nonzero={residual['nonzero_rate']:.2%} "
            f"value_entropy={residual['value_order0_entropy_bits_per_nonzero']:.3f}"
        )
        q = row["quality"]
        if q is None:
            print("  quality: skipped")
        else:
            print(
                f"  quality: log_rmse={q['signed_log2_rmse']:.3e} "
                f"p99={q['signed_log2_p99']:.3e} "
                f"grad={q['gradient_signed_log2_nrmse']:.3e}"
            )
        for candidate in row["candidate_predictors"]:
            print(
                f"  candidate {candidate['predictor_mode']}: "
                f"nonzero={candidate['nonzero_rate']:.2%} "
                f"resH={candidate['residual_order0_entropy_bits_per_sample']:.3f} "
                f"valueH={candidate['value_order0_entropy_bits_per_nonzero']:.3f}"
            )
        if args.limit and len(rows) >= args.limit:
            break

    if not args.no_save:
        safe_glob = args.glob.replace("*", "star").replace(".", "_")
        output = RESULTS_DIR / (
            f"linear_index_payload_audit_{safe_glob}"
            f"_crop{args.crop_size}_{args.transform}_bits{args.bits}.json"
        )
        output.write_text(json.dumps(rows, indent=2))
        print(f"\nSaved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
