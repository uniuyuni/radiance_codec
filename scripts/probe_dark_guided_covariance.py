"""Probe dark-only guided-filter signal plus correlated noise resynthesis.

This is a component probe, not a full codec route.  It replaces only dark
pixels with:

* guided-filtered log RGB signal;
* quantized signal;
* block-wise 3x3 covariance correlated synthetic noise.

Pixels outside the dark mask are left as the original image so the probe can
answer one narrow question: does this component improve the dark-region
appearance without dragging highlight/midtone failures into the test?
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
OUTPUT_DIR = ROOT / "outputs" / "previews" / "dark_guided_covariance"
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
from probe_adaptive_ycocg_router import entropy_bits_from_counts, luma_linear  # noqa: E402
from probe_darkbits_router_payload import read_exr, summarize_binary_mask  # noqa: E402
from probe_log_signal_noise_resynth import box_mean_2d, quantize_signal  # noqa: E402


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


def guided_filter_gray(guide: np.ndarray, values: np.ndarray, radius: int, eps: float) -> np.ndarray:
    mean_i = box_mean_2d(guide, radius)
    mean_p = box_mean_2d(values, radius)
    corr_i = box_mean_2d(guide * guide, radius)
    corr_ip = box_mean_2d(guide * values, radius)
    var_i = corr_i - mean_i * mean_i
    cov_ip = corr_ip - mean_i * mean_p
    a = cov_ip / (var_i + float(eps))
    b = mean_p - a * mean_i
    mean_a = box_mean_2d(a, radius)
    mean_b = box_mean_2d(b, radius)
    return (mean_a * guide + mean_b).astype(np.float32)


def guided_log_signal(log_rgb: np.ndarray, radius: int, eps: float) -> np.ndarray:
    guide = (
        0.2126 * log_rgb[:, :, 0].astype(np.float64)
        + 0.7152 * log_rgb[:, :, 1].astype(np.float64)
        + 0.0722 * log_rgb[:, :, 2].astype(np.float64)
    )
    out = np.zeros(log_rgb.shape, dtype=np.float32)
    for channel in range(3):
        out[:, :, channel] = guided_filter_gray(
            guide,
            log_rgb[:, :, channel].astype(np.float64),
            radius,
            eps,
        )
    return out


def binary_entropy_bits(ones: int, total: int) -> float:
    if total <= 0 or ones <= 0 or ones >= total:
        return 0.0
    p = ones / float(total)
    return -float(total) * (p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p))


def selected_index_payload(indices: np.ndarray, bits: int, mask: np.ndarray) -> dict:
    selected = int(np.count_nonzero(mask))
    rows = []
    total_bits = 0.0
    if selected == 0:
        return {"estimated_bits": 0.0, "estimated_bytes": 0, "channels": rows}
    alphabet = 1 << bits
    for channel in range(3):
        values = indices[:, :, channel][mask].astype(np.int64)
        counts = np.bincount(values, minlength=alphabet)
        cost = entropy_bits_from_counts(counts)
        total_bits += cost
        rows.append({
            "channel": channel,
            "bits": bits,
            "samples": selected,
            "estimated_bits": cost,
            "bits_per_sample": cost / float(selected),
        })
    return {
        "estimated_bits": total_bits,
        "estimated_bytes": int(math.ceil(total_bits / 8.0)),
        "channels": rows,
    }


def block_covariance_noise(
    noise: np.ndarray,
    mask: np.ndarray,
    block_size: int,
    seed: int,
    noise_scale: float,
    covariance_bits: int,
) -> tuple[np.ndarray, dict]:
    rng = np.random.default_rng(int(seed))
    h, w, c = noise.shape
    out = np.zeros(noise.shape, dtype=np.float32)
    by = math.ceil(h / block_size)
    bx = math.ceil(w / block_size)
    active_blocks = 0
    dark_samples = 0
    cov_rows = []
    jitter = 1.0e-6
    for yb in range(by):
        for xb in range(bx):
            y0 = yb * block_size
            x0 = xb * block_size
            y1 = min(y0 + block_size, h)
            x1 = min(x0 + block_size, w)
            tile_mask = mask[y0:y1, x0:x1]
            if not bool(np.any(tile_mask)):
                continue
            active_blocks += 1
            tile_noise = noise[y0:y1, x0:x1, :]
            selected = tile_noise[tile_mask].reshape(-1, c)
            dark_samples += int(selected.shape[0])
            if selected.shape[0] < 4:
                selected = tile_noise.reshape(-1, c)
            cov = np.cov(selected.astype(np.float64), rowvar=False)
            if cov.shape != (3, 3):
                cov = np.eye(3, dtype=np.float64) * float(np.var(selected))
            cov = np.asarray(cov, dtype=np.float64)
            cov = (cov + cov.T) * 0.5
            cov.flat[::4] += jitter
            try:
                chol = np.linalg.cholesky(cov)
            except np.linalg.LinAlgError:
                eigvals, eigvecs = np.linalg.eigh(cov)
                eigvals = np.maximum(eigvals, jitter)
                chol = eigvecs @ np.diag(np.sqrt(eigvals))
            z = rng.standard_normal((y1 - y0, x1 - x0, c)).astype(np.float64)
            synth = z @ chol.T
            out[y0:y1, x0:x1, :] = (synth * float(noise_scale)).astype(np.float32)
            cov_rows.append({
                "x": x0,
                "y": y0,
                "samples": int(selected.shape[0]),
                "diag": [float(cov[i, i]) for i in range(3)],
            })
    cov_values = active_blocks * 6
    cov_bytes = int(math.ceil(cov_values * int(covariance_bits) / 8.0))
    return out, {
        "block_size": block_size,
        "active_blocks": active_blocks,
        "dark_samples": dark_samples,
        "covariance_values": cov_values,
        "covariance_bits": covariance_bits,
        "covariance_bytes": cov_bytes,
        "covariance_preview": cov_rows[:16],
    }


def summarize_case(
    pixels: np.ndarray,
    anchor_regions: dict,
    anchor_regions_lift: dict,
    dark_mask: np.ndarray,
    signal_bits: int,
    radius: int,
    guide_eps: float,
    block_size: int,
    noise_scale: float,
    eps: float,
    seed: int,
    covariance_bits: int,
    white: float,
    lift_white: float,
    gamma: float,
) -> dict:
    rgb = np.maximum(pixels[:, :, :3].astype(np.float64), 0.0)
    log_rgb = np.log2(rgb + float(eps)).astype(np.float32)
    signal = guided_log_signal(log_rgb, radius, guide_eps)
    noise = log_rgb - signal
    indices, decoded_signal, ranges = quantize_signal(signal, signal_bits)
    synth_noise, cov_payload = block_covariance_noise(
        noise,
        dark_mask,
        block_size,
        seed,
        noise_scale,
        covariance_bits,
    )
    decoded_log = np.array(log_rgb, copy=True)
    decoded_log[dark_mask] = (decoded_signal + synth_noise)[dark_mask]
    decoded_rgb = np.maximum(np.exp2(decoded_log.astype(np.float64)) - float(eps), 0.0)
    decoded = np.array(pixels, copy=True)
    decoded[:, :, :3] = decoded_rgb.astype(np.float32)

    signal_payload = selected_index_payload(indices, signal_bits, dark_mask)
    mask_summary = summarize_binary_mask(dark_mask)
    mask_payload = min(
        mask_summary["order0"],
        mask_summary["west_north"],
        key=lambda row: row["total_bytes"],
    )
    estimated_bytes = (
        signal_payload["estimated_bytes"]
        + cov_payload["covariance_bytes"]
        + int(mask_payload["total_bytes"])
        + 128
    )

    regions = candidate_metrics(pixels, decoded, white, gamma)
    regions_lift = candidate_metrics(pixels, decoded, lift_white, gamma)
    gate = evaluate_gate(regions, anchor_regions, default_gate_args())
    gate_lift = evaluate_gate(regions_lift, anchor_regions_lift, default_gate_args())
    key_metrics = summarize_key_metrics(regions, anchor_regions)
    key_metrics_lift = summarize_key_metrics(regions_lift, anchor_regions_lift)
    return {
        "label": (
            f"darkguided_b{signal_bits}_r{radius}_ge{guide_eps:g}"
            f"_blk{block_size}_ns{noise_scale:g}_eps{eps:g}"
        ),
        "signal_bits": signal_bits,
        "radius": radius,
        "guide_eps": guide_eps,
        "block_size": block_size,
        "noise_scale": noise_scale,
        "eps": eps,
        "dark_pixel_rate": float(np.mean(dark_mask)),
        "raw_bytes": int(pixels.nbytes),
        "component_estimated_bytes": int(estimated_bytes),
        "component_ratio_vs_raw": pixels.nbytes / float(estimated_bytes),
        "payload": {
            "signal_ranges": ranges,
            "signal_payload": signal_payload,
            "covariance_payload": cov_payload,
            "mask_payload": mask_payload,
        },
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
    lift_dark = row["key_metrics_lift"]["dark_0_0.25"]
    high = row["key_metrics"]["highlight_1_4"]
    print(
        f"{prefix}{row['gate']['decision'].upper():6s}/"
        f"{row['gate_lift']['decision'].upper():6s} "
        f"{row['label']:44s} "
        f"component={row['component_estimated_bytes']:,} "
        f"dark={row['dark_pixel_rate']:.2%} "
        f"darkD={dark['lost_detail_delta_vs_anchor']:.2%} "
        f"liftDarkD={lift_dark['lost_detail_delta_vs_anchor']:.2%} "
        f"hiD={high['lost_detail_delta_vs_anchor']:.2%}",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", default="sample_DSCF0009.EXR")
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--dark-max", type=float, default=0.25)
    parser.add_argument("--signal-bits", default="8,9,10")
    parser.add_argument("--radii", default="2,4,8")
    parser.add_argument("--guide-eps", default="0.01,0.05,0.1")
    parser.add_argument("--block-sizes", default="16,32")
    parser.add_argument("--noise-scales", default="0.7,1")
    parser.add_argument("--eps", type=float, default=1.0e-4)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--covariance-bits", type=int, default=12)
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
        dark_mask = luma_linear(pixels) <= float(args.dark_max)
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
                for guide_eps in parse_floats(args.guide_eps):
                    for block_size in parse_ints(args.block_sizes):
                        for noise_scale in parse_floats(args.noise_scales):
                            rows.append(
                                summarize_case(
                                    pixels,
                                    anchor_regions,
                                    anchor_regions_lift,
                                    dark_mask,
                                    signal_bits,
                                    radius,
                                    guide_eps,
                                    block_size,
                                    noise_scale,
                                    args.eps,
                                    args.seed,
                                    args.covariance_bits,
                                    args.white,
                                    args.lift_white,
                                    args.gamma,
                                )
                            )
        rows.sort(
            key=lambda row: (
                decision_rank(row["gate"]["decision"]),
                decision_rank(row["gate_lift"]["decision"]),
                row["component_estimated_bytes"],
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
            "dark_max": args.dark_max,
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
            f"dark_guided_covariance_{safe_name(args.glob)}"
            f"_crop{args.crop_size}_dark{args.dark_max:g}.json"
        )
        output.write_text(json.dumps(summaries, indent=2))
        print(f"Saved: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
