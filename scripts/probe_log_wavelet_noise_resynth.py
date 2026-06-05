"""Probe log-domain Haar wavelet signal/noise separation.

This is the frequency-domain counterpart to probe_log_signal_noise_resynth.py.
It tests the "do not blur in spatial space" idea:

* log2(rgb + eps)
* multilevel Haar transform
* quantize only the final lowpass (LL) signal
* model high-frequency subbands with block sigma maps + deterministic noise
* optionally keep strong high-frequency coefficients as sparse edge/detail escapes

The output is a visual upper-bound probe, not a production bitstream.
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
OUTPUT_DIR = ROOT / "outputs" / "previews" / "log_wavelet_noise"
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


def safe_name(text: str) -> str:
    return (
        text.replace("*", "star")
        .replace(".", "_")
        .replace(",", "-")
        .replace("/", "_")
        .replace(":", "_")
    )


def haar_forward(log_rgb: np.ndarray, levels: int) -> tuple[np.ndarray, list[dict]]:
    coeff = log_rgb.astype(np.float32, copy=True)
    h, w, _ = coeff.shape
    subbands: list[dict] = []
    for level in range(levels):
        current_h = h >> level
        current_w = w >> level
        if current_h < 2 or current_w < 2 or current_h % 2 or current_w % 2:
            break
        region = coeff[:current_h, :current_w, :].astype(np.float64)
        a = region[0::2, 0::2, :]
        b = region[0::2, 1::2, :]
        c = region[1::2, 0::2, :]
        d = region[1::2, 1::2, :]
        ll = (a + b + c + d) * 0.25
        hcoef = (a - b + c - d) * 0.25
        vcoef = (a + b - c - d) * 0.25
        qcoef = (a - b - c + d) * 0.25
        h2 = current_h // 2
        w2 = current_w // 2
        coeff[:h2, :w2, :] = ll.astype(np.float32)
        coeff[:h2, w2:current_w, :] = hcoef.astype(np.float32)
        coeff[h2:current_h, :w2, :] = vcoef.astype(np.float32)
        coeff[h2:current_h, w2:current_w, :] = qcoef.astype(np.float32)
        subbands.append({
            "level": level + 1,
            "h": current_h,
            "w": current_w,
            "low_h": h2,
            "low_w": w2,
        })
    return coeff, subbands


def haar_inverse(coeff: np.ndarray, subbands: list[dict]) -> np.ndarray:
    out = coeff.astype(np.float32, copy=True)
    for band in reversed(subbands):
        current_h = int(band["h"])
        current_w = int(band["w"])
        h2 = int(band["low_h"])
        w2 = int(band["low_w"])
        ll = out[:h2, :w2, :].astype(np.float64)
        hcoef = out[:h2, w2:current_w, :].astype(np.float64)
        vcoef = out[h2:current_h, :w2, :].astype(np.float64)
        qcoef = out[h2:current_h, w2:current_w, :].astype(np.float64)
        region = np.empty((current_h, current_w, out.shape[2]), dtype=np.float32)
        region[0::2, 0::2, :] = (ll + hcoef + vcoef + qcoef).astype(np.float32)
        region[0::2, 1::2, :] = (ll - hcoef + vcoef - qcoef).astype(np.float32)
        region[1::2, 0::2, :] = (ll + hcoef - vcoef - qcoef).astype(np.float32)
        region[1::2, 1::2, :] = (ll - hcoef - vcoef + qcoef).astype(np.float32)
        out[:current_h, :current_w, :] = region
    return out


def quantize_array(values: np.ndarray, bits: int) -> tuple[np.ndarray, np.ndarray, list[list[float]]]:
    levels = (1 << bits) - 1
    indices = np.zeros(values.shape, dtype=np.uint16)
    decoded = np.zeros(values.shape, dtype=np.float32)
    ranges = []
    for channel in range(values.shape[2]):
        channel_values = values[:, :, channel].astype(np.float64)
        lo = float(np.min(channel_values))
        hi = float(np.max(channel_values))
        ranges.append([lo, hi])
        if not hi > lo:
            decoded[:, :, channel] = np.float32(lo)
            continue
        q = np.floor((channel_values - lo) / (hi - lo) * levels + 0.5)
        q = np.clip(q, 0, levels).astype(np.uint16)
        rec = lo + q.astype(np.float64) * (hi - lo) / levels
        indices[:, :, channel] = q
        decoded[:, :, channel] = rec.astype(np.float32)
    return indices, decoded, ranges


def block_sigma(values: np.ndarray, block_size: int, sigma_floor: float) -> tuple[np.ndarray, np.ndarray]:
    h, w, c = values.shape
    by = math.ceil(h / block_size)
    bx = math.ceil(w / block_size)
    sigma_map = np.zeros((by, bx, c), dtype=np.float32)
    expanded = np.zeros(values.shape, dtype=np.float32)
    for yb in range(by):
        for xb in range(bx):
            y0 = yb * block_size
            x0 = xb * block_size
            tile = values[y0:min(y0 + block_size, h), x0:min(x0 + block_size, w), :]
            sigma = np.std(tile.reshape(-1, c), axis=0).astype(np.float32)
            sigma = np.maximum(sigma, np.float32(sigma_floor))
            sigma_map[yb, xb, :] = sigma
            expanded[y0:min(y0 + block_size, h), x0:min(x0 + block_size, w), :] = sigma
    return sigma_map, expanded


def synth_subband(
    original: np.ndarray,
    rng: np.random.Generator,
    block_size: int,
    noise_scale: float,
    keep_sigma: float,
    escape_value_bits: int,
    sigma_floor: float,
) -> tuple[np.ndarray, dict]:
    sigma_map, sigma = block_sigma(original, block_size, sigma_floor)
    generated = rng.standard_normal(original.shape).astype(np.float32) * sigma * float(noise_scale)
    keep_mask = np.zeros(original.shape, dtype=bool)
    if keep_sigma > 0.0:
        keep_mask = np.abs(original) > (sigma * float(keep_sigma))
        generated[keep_mask] = original[keep_mask]
    keep_count = int(np.count_nonzero(keep_mask))
    sigma_bytes = int(math.ceil(np.prod(sigma_map.shape) * 8 / 8.0))
    escape_bytes = int(math.ceil(keep_count * int(escape_value_bits) / 8.0))
    total_samples = int(np.prod(original.shape))
    mask_bits = 0.0
    if 0 < keep_count < total_samples:
        p = keep_count / float(total_samples)
        mask_bits = -total_samples * (p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p))
    mask_bytes = int(math.ceil(mask_bits / 8.0))
    return generated.astype(np.float32), {
        "sigma_shape": list(sigma_map.shape),
        "sigma_bytes": sigma_bytes,
        "keep_count": keep_count,
        "keep_rate": keep_count / float(total_samples),
        "mask_bytes": mask_bytes,
        "escape_bytes": escape_bytes,
        "estimated_bytes": sigma_bytes + mask_bytes + escape_bytes,
        "sigma_mean": [float(v) for v in np.mean(sigma_map.reshape(-1, sigma_map.shape[2]), axis=0)],
    }


def decode_candidate(
    pixels: np.ndarray,
    levels: int,
    low_bits: int,
    block_size: int,
    noise_scale: float,
    keep_sigma: float,
    eps: float,
    seed: int,
    sigma_floor: float,
    escape_value_bits: int,
) -> tuple[np.ndarray, dict]:
    rgb = np.maximum(pixels[:, :, :3].astype(np.float64), 0.0)
    log_rgb = np.log2(rgb + float(eps)).astype(np.float32)
    coeff, subbands = haar_forward(log_rgb, levels)
    if not subbands:
        raise ValueError("image is too small or odd-sized for Haar levels")
    low_h = int(subbands[-1]["low_h"])
    low_w = int(subbands[-1]["low_w"])
    ll = coeff[:low_h, :low_w, :]
    low_indices, low_decoded, low_ranges = quantize_array(ll, low_bits)
    decoded_coeff = np.zeros_like(coeff)
    decoded_coeff[:low_h, :low_w, :] = low_decoded
    low_payload = residual_entropy_bytes(low_indices, (low_bits, low_bits, low_bits))

    rng = np.random.default_rng(int(seed))
    subband_payloads = []
    for band in subbands:
        current_h = int(band["h"])
        current_w = int(band["w"])
        h2 = int(band["low_h"])
        w2 = int(band["low_w"])
        slices = (
            np.s_[:h2, w2:current_w, :],
            np.s_[h2:current_h, :w2, :],
            np.s_[h2:current_h, w2:current_w, :],
        )
        names = ("h", "v", "q")
        for name, sl in zip(names, slices):
            synth, payload = synth_subband(
                coeff[sl],
                rng,
                block_size,
                noise_scale,
                keep_sigma,
                escape_value_bits,
                sigma_floor,
            )
            decoded_coeff[sl] = synth
            payload["level"] = int(band["level"])
            payload["subband"] = name
            subband_payloads.append(payload)
    decoded_log = haar_inverse(decoded_coeff, subbands)
    decoded_rgb = np.maximum(np.exp2(decoded_log.astype(np.float64)) - float(eps), 0.0)
    decoded = np.array(pixels, copy=True)
    decoded[:, :, :3] = decoded_rgb.astype(np.float32)
    high_bytes = sum(int(row["estimated_bytes"]) for row in subband_payloads)
    estimated_bytes = int(low_payload["estimated_bytes"] + high_bytes + 128 + 8 + 3 * 2 * 4)
    return np.ascontiguousarray(decoded), {
        "levels": len(subbands),
        "low_shape": [low_h, low_w, 3],
        "low_ranges": low_ranges,
        "low_payload": low_payload,
        "subband_payloads": subband_payloads,
        "high_bytes": high_bytes,
        "estimated_bytes": estimated_bytes,
    }


def summarize_case(
    pixels: np.ndarray,
    anchor_regions: dict,
    anchor_regions_lift: dict,
    levels: int,
    low_bits: int,
    block_size: int,
    noise_scale: float,
    keep_sigma: float,
    eps: float,
    seed: int,
    sigma_floor: float,
    escape_value_bits: int,
    white: float,
    lift_white: float,
    gamma: float,
) -> dict:
    decoded, payload = decode_candidate(
        pixels,
        levels,
        low_bits,
        block_size,
        noise_scale,
        keep_sigma,
        eps,
        seed,
        sigma_floor,
        escape_value_bits,
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
            f"loghaar_l{levels}_b{low_bits}_blk{block_size}"
            f"_ns{noise_scale:g}_keep{keep_sigma:g}_eps{eps:g}"
        ),
        "levels": levels,
        "low_bits": low_bits,
        "block_size": block_size,
        "noise_scale": noise_scale,
        "keep_sigma": keep_sigma,
        "eps": eps,
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
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--levels", default="1,2,3")
    parser.add_argument("--low-bits", default="8,9,10")
    parser.add_argument("--block-sizes", default="16,32")
    parser.add_argument("--noise-scales", default="0.7,1")
    parser.add_argument("--keep-sigmas", default="0,2,3")
    parser.add_argument("--eps", type=float, default=1.0e-4)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--sigma-floor", type=float, default=0.0)
    parser.add_argument("--escape-value-bits", type=int, default=12)
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
        for levels in parse_ints(args.levels):
            for low_bits in parse_ints(args.low_bits):
                for block_size in parse_ints(args.block_sizes):
                    for noise_scale in parse_floats(args.noise_scales):
                        for keep_sigma in parse_floats(args.keep_sigmas):
                            rows.append(
                                summarize_case(
                                    pixels,
                                    anchor_regions,
                                    anchor_regions_lift,
                                    levels,
                                    low_bits,
                                    block_size,
                                    noise_scale,
                                    keep_sigma,
                                    args.eps,
                                    args.seed,
                                    args.sigma_floor,
                                    args.escape_value_bits,
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
            row_copy = dict(row)
            row_copy.pop("decoded", None)
            serializable.append(row_copy)
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
            f"log_wavelet_noise_{safe_name(args.glob)}"
            f"_crop{args.crop_size}_l{safe_name(args.levels)}"
            f"_b{safe_name(args.low_bits)}.json"
        )
        output.write_text(json.dumps(summaries, indent=2))
        print(f"Saved: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
