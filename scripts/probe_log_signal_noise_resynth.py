"""Probe log-domain signal/noise separation with parametric noise resynthesis.

This is a deliberately visual near-lossless experiment, not a bit-exact route.
The hypothesis:

* convert linear RGB to log2(rgb + eps);
* low-pass in log space to separate smooth signal from high-frequency noise;
* quantize/compress only the signal;
* store a low-resolution noise sigma map and a seed;
* synthesize fresh noise on decode.

It is meant to test whether replacing high-entropy sensor/demosaic noise with
statistically similar noise can preserve exposure-lift appearance while
removing the pressure to encode every noisy mantissa bit.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
OUTPUT_DIR = ROOT / "outputs" / "previews" / "log_signal_noise"
sys.path.insert(0, str(ROOT / "codec" / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

import radiance_codec  # noqa: E402
from audit_visual_gate import (  # noqa: E402
    candidate_metrics,
    default_gate_args,
    evaluate_gate,
    summarize_key_metrics,
)
from export_linear_index_preview_png import to_preview_u16, write_png  # noqa: E402
from probe_darkbits_router_payload import read_exr  # noqa: E402
from probe_ycocg_tile_range import residual_entropy_bytes  # noqa: E402


def parse_ints(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split(",") if part.strip())


def parse_floats(text: str) -> tuple[float, ...]:
    return tuple(float(part) for part in text.split(",") if part.strip())


def parse_thresholds(text: str) -> tuple[tuple[float, float] | None, ...]:
    values: list[tuple[float, float] | None] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if part == "none":
            values.append(None)
            continue
        lo_text, hi_text = part.split(":", 1)
        values.append((float(lo_text), float(hi_text)))
    return tuple(values)


def safe_name(text: str) -> str:
    return (
        text.replace("*", "star")
        .replace(".", "_")
        .replace(",", "-")
        .replace("/", "_")
        .replace(":", "_")
    )


def box_mean_2d(values: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return values.astype(np.float64, copy=True)
    padded = np.pad(values.astype(np.float64, copy=False), radius, mode="reflect")
    integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant").cumsum(axis=0).cumsum(axis=1)
    size = 2 * radius + 1
    total = (
        integral[size:, size:]
        - integral[:-size, size:]
        - integral[size:, :-size]
        + integral[:-size, :-size]
    )
    return total / float(size * size)


def lowpass_log(log_rgb: np.ndarray, radius: int, passes: int) -> np.ndarray:
    out = log_rgb.astype(np.float64, copy=True)
    for _ in range(max(1, passes)):
        for channel in range(3):
            out[:, :, channel] = box_mean_2d(out[:, :, channel], radius)
    return out.astype(np.float32)


def local_std_2d(values: np.ndarray, radius: int) -> np.ndarray:
    mean = box_mean_2d(values, radius)
    mean_sq = box_mean_2d(values * values, radius)
    return np.sqrt(np.maximum(mean_sq - mean * mean, 0.0))


def smoothstep(edge0: float, edge1: float, values: np.ndarray) -> np.ndarray:
    if edge1 <= edge0:
        return (values >= edge1).astype(np.float64)
    t = np.clip((values - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def adaptive_signal(
    log_rgb: np.ndarray,
    radius: int,
    passes: int,
    texture_threshold: tuple[float, float] | None,
) -> np.ndarray:
    base = lowpass_log(log_rgb, radius, passes)
    if texture_threshold is None:
        return base
    log_luma = (
        0.2126 * log_rgb[:, :, 0].astype(np.float64)
        + 0.7152 * log_rgb[:, :, 1].astype(np.float64)
        + 0.0722 * log_rgb[:, :, 2].astype(np.float64)
    )
    std = local_std_2d(log_luma, radius)
    weight = smoothstep(texture_threshold[0], texture_threshold[1], std)[:, :, None]
    return (
        base.astype(np.float64) * (1.0 - weight)
        + log_rgb.astype(np.float64) * weight
    ).astype(np.float32)


def block_sigma(noise: np.ndarray, block_size: int, sigma_floor: float) -> tuple[np.ndarray, np.ndarray]:
    h, w, c = noise.shape
    by = math.ceil(h / block_size)
    bx = math.ceil(w / block_size)
    sigma_map = np.zeros((by, bx, c), dtype=np.float32)
    expanded = np.zeros(noise.shape, dtype=np.float32)
    for yb in range(by):
        for xb in range(bx):
            y0 = yb * block_size
            x0 = xb * block_size
            tile = noise[y0:min(y0 + block_size, h), x0:min(x0 + block_size, w), :]
            sigma = np.std(tile.reshape(-1, c), axis=0).astype(np.float32)
            sigma = np.maximum(sigma, np.float32(sigma_floor))
            sigma_map[yb, xb, :] = sigma
            expanded[y0:min(y0 + block_size, h), x0:min(x0 + block_size, w), :] = sigma
    return sigma_map, expanded


def quantize_signal(signal: np.ndarray, bits: int) -> tuple[np.ndarray, np.ndarray, list[list[float]]]:
    levels = (1 << bits) - 1
    indices = np.zeros(signal.shape, dtype=np.uint16)
    decoded = np.zeros(signal.shape, dtype=np.float32)
    ranges = []
    for channel in range(3):
        values = signal[:, :, channel].astype(np.float64)
        lo = float(np.min(values))
        hi = float(np.max(values))
        ranges.append([lo, hi])
        if not hi > lo:
            decoded[:, :, channel] = np.float32(lo)
            continue
        q = np.floor((values - lo) / (hi - lo) * levels + 0.5)
        q = np.clip(q, 0, levels).astype(np.uint16)
        rec = lo + q.astype(np.float64) * (hi - lo) / levels
        indices[:, :, channel] = q
        decoded[:, :, channel] = rec.astype(np.float32)
    return indices, decoded, ranges


def synth_noise(shape: tuple[int, int, int], seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    return rng.standard_normal(shape).astype(np.float32)


def decode_candidate(
    pixels: np.ndarray,
    signal_bits: int,
    radius: int,
    passes: int,
    block_size: int,
    noise_scale: float,
    eps: float,
    seed: int,
    sigma_bits: int,
    sigma_floor: float,
    texture_threshold: tuple[float, float] | None,
) -> tuple[np.ndarray, dict]:
    rgb = np.maximum(pixels[:, :, :3].astype(np.float64), 0.0)
    log_rgb = np.log2(rgb + float(eps)).astype(np.float32)
    signal = adaptive_signal(log_rgb, radius, passes, texture_threshold)
    noise = log_rgb - signal
    sigma_map, sigma_expanded = block_sigma(noise, block_size, sigma_floor)
    indices, decoded_signal, ranges = quantize_signal(signal, signal_bits)
    noise_random = synth_noise(log_rgb.shape, seed)
    decoded_log = decoded_signal + noise_random * sigma_expanded * float(noise_scale)
    decoded_rgb = np.maximum(np.exp2(decoded_log.astype(np.float64)) - float(eps), 0.0)
    decoded = np.array(pixels, copy=True)
    decoded[:, :, :3] = decoded_rgb.astype(np.float32)

    index_payload = residual_entropy_bytes(
        indices,
        (signal_bits, signal_bits, signal_bits),
    )
    by, bx, _ = sigma_map.shape
    sigma_bytes = int(math.ceil(by * bx * 3 * int(sigma_bits) / 8.0))
    range_bytes = 3 * 2 * 4
    seed_bytes = 8
    header_bytes = 128
    estimated_bytes = (
        int(index_payload["estimated_bytes"])
        + sigma_bytes
        + range_bytes
        + seed_bytes
        + header_bytes
    )
    return np.ascontiguousarray(decoded), {
        "signal_ranges": ranges,
        "index_payload": index_payload,
        "sigma_shape": list(sigma_map.shape),
        "sigma_bytes": sigma_bytes,
        "sigma_bits": sigma_bits,
        "sigma_mean": [float(v) for v in np.mean(sigma_map.reshape(-1, 3), axis=0)],
        "sigma_p99": [float(v) for v in np.percentile(sigma_map.reshape(-1, 3), 99.0, axis=0)],
        "range_bytes": range_bytes,
        "seed_bytes": seed_bytes,
        "estimated_bytes": estimated_bytes,
        "texture_threshold": None if texture_threshold is None else list(texture_threshold),
    }


def summarize_case(
    pixels: np.ndarray,
    anchor_regions: dict,
    anchor_regions_lift: dict,
    signal_bits: int,
    radius: int,
    passes: int,
    block_size: int,
    noise_scale: float,
    eps: float,
    seed: int,
    sigma_bits: int,
    sigma_floor: float,
    texture_threshold: tuple[float, float] | None,
    white: float,
    lift_white: float,
    gamma: float,
) -> dict:
    decoded, payload = decode_candidate(
        pixels,
        signal_bits,
        radius,
        passes,
        block_size,
        noise_scale,
        eps,
        seed,
        sigma_bits,
        sigma_floor,
        texture_threshold,
    )
    regions = candidate_metrics(pixels, decoded, white, gamma)
    regions_lift = candidate_metrics(pixels, decoded, lift_white, gamma)
    gate = evaluate_gate(regions, anchor_regions, default_gate_args())
    gate_lift = evaluate_gate(regions_lift, anchor_regions_lift, default_gate_args())
    key_metrics = summarize_key_metrics(regions, anchor_regions)
    key_metrics_lift = summarize_key_metrics(regions_lift, anchor_regions_lift)
    estimated_bytes = int(payload["estimated_bytes"])
    return {
        "label": (
            f"logsig_b{signal_bits}_r{radius}_p{passes}"
            f"_blk{block_size}_ns{noise_scale:g}_eps{eps:g}"
            f"_tx{'none' if texture_threshold is None else f'{texture_threshold[0]:g}-{texture_threshold[1]:g}'}"
        ),
        "signal_bits": signal_bits,
        "radius": radius,
        "passes": passes,
        "block_size": block_size,
        "noise_scale": noise_scale,
        "eps": eps,
        "seed": seed,
        "texture_threshold": None if texture_threshold is None else list(texture_threshold),
        "raw_bytes": int(pixels.nbytes),
        "estimated_bytes": estimated_bytes,
        "estimated_ratio": pixels.nbytes / float(estimated_bytes),
        "payload": payload,
        "gate": gate,
        "gate_lift": gate_lift,
        "key_metrics": key_metrics,
        "key_metrics_lift": key_metrics_lift,
        "decoded": decoded,
    }


def decision_rank(decision: str) -> int:
    return {"pass": 0, "maybe": 1, "reject": 2}.get(decision, 9)


def print_row(row: dict, prefix: str = "") -> None:
    dark = row["key_metrics"]["dark_0_0.25"]
    high = row["key_metrics"]["highlight_1_4"]
    lift_dark = row["key_metrics_lift"]["dark_0_0.25"]
    print(
        f"{prefix}{row['gate']['decision'].upper():6s}/"
        f"{row['gate_lift']['decision'].upper():6s} "
        f"{row['label']:42s} "
        f"est={row['estimated_bytes']:,} ratio={row['estimated_ratio']:.2f}x "
        f"darkD={dark['lost_detail_delta_vs_anchor']:.2%} "
        f"liftDarkD={lift_dark['lost_detail_delta_vs_anchor']:.2%} "
        f"hiD={high['lost_detail_delta_vs_anchor']:.2%}",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", default="sample_DSCF0009.EXR")
    parser.add_argument("--crop-size", type=int, default=1024)
    parser.add_argument("--signal-bits", default="8,9,10")
    parser.add_argument("--radii", default="2,4,8")
    parser.add_argument("--passes", type=int, default=2)
    parser.add_argument("--block-sizes", default="32,64")
    parser.add_argument("--noise-scales", default="0,0.7,1")
    parser.add_argument("--texture-thresholds", default="none")
    parser.add_argument("--eps", type=float, default=1.0e-4)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--sigma-bits", type=int, default=8)
    parser.add_argument("--sigma-floor", type=float, default=0.0)
    parser.add_argument("--white", type=float, default=4.0)
    parser.add_argument("--lift-white", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=2.2)
    parser.add_argument("--anchor-transform", default="signed-log")
    parser.add_argument("--anchor-bits", type=int, default=10)
    parser.add_argument("--export-labels", default="")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    export_labels = set(part.strip() for part in args.export_labels.split(",") if part.strip())
    summaries = []
    for path in sorted(DATA_DIR.glob(args.glob)):
        print(path.name, flush=True)
        pixels = read_exr(path, args.crop_size)
        anchor = radiance_codec.quantize_linear_index(
            pixels,
            bits=args.anchor_bits,
            transform=args.anchor_transform,
        )
        anchor_regions = candidate_metrics(pixels, anchor, args.white, args.gamma)
        anchor_regions_lift = candidate_metrics(pixels, anchor, args.lift_white, args.gamma)
        rows = []
        for signal_bits in parse_ints(args.signal_bits):
            for radius in parse_ints(args.radii):
                for block_size in parse_ints(args.block_sizes):
                    for noise_scale in parse_floats(args.noise_scales):
                        for texture_threshold in parse_thresholds(args.texture_thresholds):
                            rows.append(
                                summarize_case(
                                    pixels,
                                    anchor_regions,
                                    anchor_regions_lift,
                                    signal_bits,
                                    radius,
                                    args.passes,
                                    block_size,
                                    noise_scale,
                                    args.eps,
                                    args.seed,
                                    args.sigma_bits,
                                    args.sigma_floor,
                                    texture_threshold,
                                    args.white,
                                    args.lift_white,
                                    args.gamma,
                                )
                            )
        rows.sort(
            key=lambda row: (
                decision_rank(row["gate"]["decision"]),
                decision_rank(row["gate_lift"]["decision"]),
                row["estimated_bytes"],
            )
        )
        stem = safe_name(path.stem)
        crop_label = "full" if args.crop_size == 0 else f"crop{args.crop_size}"
        original_png = args.output_dir / f"{stem}_{crop_label}_original_w{args.white:g}_g{args.gamma:g}.png"
        if export_labels and not original_png.exists():
            write_png(original_png, to_preview_u16(pixels, args.white, args.gamma))
        for row in rows:
            if row["label"] in export_labels:
                decoded_png = (
                    args.output_dir
                    / f"{stem}_{crop_label}_{safe_name(row['label'])}_w{args.white:g}_g{args.gamma:g}_decoded.png"
                )
                write_png(decoded_png, to_preview_u16(row["decoded"], args.white, args.gamma))
                row["decoded_png"] = str(decoded_png)
        for row in rows[:args.top]:
            print_row(row, prefix="  ")
            if "decoded_png" in row:
                print(f"    decoded_png={row['decoded_png']}", flush=True)
        serializable = []
        for row in rows:
            copy = dict(row)
            copy.pop("decoded", None)
            serializable.append(copy)
        summaries.append({
            "image": path.name,
            "shape": list(pixels.shape),
            "crop_size": args.crop_size,
            "white": args.white,
            "lift_white": args.lift_white,
            "gamma": args.gamma,
            "rows": serializable,
        })
        if args.limit and len(summaries) >= args.limit:
            break
    if not summaries:
        print(f"no files matched: {args.glob}", flush=True)
        return 1
    if not args.no_save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        output = RESULTS_DIR / (
            f"log_signal_noise_{safe_name(args.glob)}"
            f"_crop{args.crop_size}_b{safe_name(args.signal_bits)}"
            f"_r{safe_name(args.radii)}_blk{safe_name(args.block_sizes)}.json"
        )
        output.write_text(json.dumps(summaries, indent=2))
        print(f"Saved: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
