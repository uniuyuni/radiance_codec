"""Audit display-space quality by tonal region.

The numerical HDR metrics used during early probes missed visible dark-band
loss and highlight detail loss.  This script compares candidates after the
same display mapping used for preview PNGs:

    display = clamp(value / white, 0, 1) ** (1 / gamma)

Regions are selected from the original linear max RGB value by default:
dark, mid, highlight, and extreme highlight.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import OpenImageIO as oiio

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
sys.path.insert(0, str(ROOT / "codec" / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

import radiance_codec  # noqa: E402
from probe_parametric_transform_index import quantize_power  # noqa: E402
from probe_power_recon_table import table_reconstruct  # noqa: E402


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


def display_map(pixels: np.ndarray, white: float, gamma: float) -> np.ndarray:
    rgb = pixels[:, :, :3].astype(np.float64)
    normalized = np.clip(rgb / float(white), 0.0, 1.0)
    return np.power(normalized, 1.0 / float(gamma))


def luminance(display_rgb: np.ndarray) -> np.ndarray:
    return (
        display_rgb[:, :, 0] * 0.2126
        + display_rgb[:, :, 1] * 0.7152
        + display_rgb[:, :, 2] * 0.0722
    )


def region_masks(pixels: np.ndarray) -> dict[str, np.ndarray]:
    value = np.max(pixels[:, :, :3].astype(np.float64), axis=2)
    return {
        "dark_0_0.25": value <= 0.25,
        "mid_0.25_1": (value > 0.25) & (value <= 1.0),
        "highlight_1_4": (value > 1.0) & (value <= 4.0),
        "extreme_gt4": value > 4.0,
        "all": np.ones(value.shape, dtype=bool),
    }


def scalar_error_stats(original: np.ndarray, decoded: np.ndarray, mask: np.ndarray) -> dict:
    if not bool(np.any(mask)):
        return {
            "samples": 0,
            "rmse": 0.0,
            "mae": 0.0,
            "p99_abs": 0.0,
            "max_abs": 0.0,
            "psnr_db": math.inf,
        }
    diff = decoded[mask] - original[mask]
    abs_diff = np.abs(diff)
    rmse = float(np.sqrt(np.mean(diff * diff)))
    psnr = math.inf if rmse == 0.0 else 20.0 * math.log10(1.0 / rmse)
    return {
        "samples": int(mask.sum()),
        "rmse": rmse,
        "mae": float(np.mean(abs_diff)),
        "p99_abs": float(np.percentile(abs_diff, 99.0)),
        "max_abs": float(np.max(abs_diff)),
        "psnr_db": psnr,
    }


def gradient_stats(original_luma: np.ndarray, decoded_luma: np.ndarray, mask: np.ndarray) -> dict:
    masks = [
        mask[:, 1:] & mask[:, :-1],
        mask[1:, :] & mask[:-1, :],
    ]
    orig_grads = [
        original_luma[:, 1:] - original_luma[:, :-1],
        original_luma[1:, :] - original_luma[:-1, :],
    ]
    dec_grads = [
        decoded_luma[:, 1:] - decoded_luma[:, :-1],
        decoded_luma[1:, :] - decoded_luma[:-1, :],
    ]
    selected_orig = []
    selected_dec = []
    for grad_mask, orig_grad, dec_grad in zip(masks, orig_grads, dec_grads):
        if bool(np.any(grad_mask)):
            selected_orig.append(orig_grad[grad_mask])
            selected_dec.append(dec_grad[grad_mask])
    if not selected_orig:
        return {
            "gradient_samples": 0,
            "grad_rmse": 0.0,
            "grad_p99_abs": 0.0,
            "grad_nrmse": 0.0,
            "grad_energy_ratio": 1.0,
            "grad_correlation": 1.0,
            "lost_detail_rate": 0.0,
        }
    orig = np.concatenate(selected_orig)
    dec = np.concatenate(selected_dec)
    diff = dec - orig
    signal = float(np.sqrt(np.mean(orig * orig)))
    dec_signal = float(np.sqrt(np.mean(dec * dec)))
    rmse = float(np.sqrt(np.mean(diff * diff)))
    denom = math.sqrt(float(np.sum(orig * orig)) * float(np.sum(dec * dec)))
    corr = 1.0 if denom == 0.0 else float(np.sum(orig * dec) / denom)
    # A display-luma gradient above 1/4096 is usually visible in smooth areas.
    visible = np.abs(orig) > (1.0 / 4096.0)
    lost = visible & (np.abs(dec) < np.abs(orig) * 0.5)
    return {
        "gradient_samples": int(orig.size),
        "grad_rmse": rmse,
        "grad_p99_abs": float(np.percentile(np.abs(diff), 99.0)),
        "grad_nrmse": 0.0 if signal == 0.0 else rmse / signal,
        "grad_energy_ratio": 1.0 if signal == 0.0 else dec_signal / signal,
        "grad_correlation": corr,
        "lost_detail_rate": float(np.mean(lost)) if bool(np.any(visible)) else 0.0,
    }


def decode_candidate(pixels: np.ndarray, spec: str) -> tuple[str, np.ndarray, dict]:
    parts = spec.split(":")
    if parts[0] == "codec":
        if len(parts) != 3:
            raise ValueError("codec candidate format: codec:transform:bits")
        transform = parts[1]
        bits = int(parts[2])
        t0 = time.perf_counter()
        encoded = radiance_codec.encode_linear_index_near_lossless(
            pixels,
            bits=bits,
            effort=9,
            transform=transform,
        )
        t1 = time.perf_counter()
        decoded = radiance_codec.decode(encoded, pixels.shape)
        t2 = time.perf_counter()
        return (
            f"codec_{transform}_bits{bits}",
            decoded,
            {
                "encoded_bytes": len(encoded),
                "ratio_vs_original": pixels.nbytes / len(encoded),
                "encode_seconds": t1 - t0,
                "decode_seconds": t2 - t1,
            },
        )
    if parts[0] == "power-table":
        if len(parts) != 4:
            raise ValueError("power-table candidate format: power-table:gamma:bits:table")
        power_gamma = float(parts[1])
        bits = int(parts[2])
        table_mode = parts[3]
        t0 = time.perf_counter()
        indices, decoded = quantize_power(pixels, bits, power_gamma)
        if table_mode != "center":
            decoded = table_reconstruct(pixels, indices, bits, table_mode)
        t1 = time.perf_counter()
        return (
            f"power{power_gamma:g}_bits{bits}_{table_mode}",
            decoded,
            {
                "encoded_bytes": None,
                "ratio_vs_original": None,
                "encode_seconds": None,
                "decode_seconds": t1 - t0,
            },
        )
    raise ValueError(f"unknown candidate: {spec}")


def summarize_candidate(
    pixels: np.ndarray,
    candidate: str,
    white: float,
    gamma: float,
) -> dict:
    label, decoded, payload = decode_candidate(pixels, candidate)
    original_display = display_map(pixels, white, gamma)
    decoded_display = display_map(decoded, white, gamma)
    original_luma = luminance(original_display)
    decoded_luma = luminance(decoded_display)
    masks = region_masks(pixels)
    regions = {}
    for name, mask in masks.items():
        regions[name] = {
            "display_rgb": scalar_error_stats(
                original_display,
                decoded_display,
                np.repeat(mask[:, :, None], 3, axis=2),
            ),
            "display_luma": scalar_error_stats(original_luma, decoded_luma, mask),
            "display_luma_gradient": gradient_stats(original_luma, decoded_luma, mask),
        }
    return {
        "candidate": candidate,
        "label": label,
        **payload,
        "regions": regions,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", default="sample_DSCF0009.EXR")
    parser.add_argument("--crop-size", type=int, default=0)
    parser.add_argument(
        "--candidates",
        default="codec:signed-log:10,power-table:1.17:10:signed-log-mean",
    )
    parser.add_argument("--white", type=float, default=4.0)
    parser.add_argument("--gamma", type=float, default=2.2)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    candidate_specs = [
        part.strip() for part in args.candidates.split(",") if part.strip()
    ]
    summaries = []
    for path in sorted(DATA_DIR.glob(args.glob)):
        pixels = read_exr(path, args.crop_size)
        rows = [
            summarize_candidate(pixels, candidate, args.white, args.gamma)
            for candidate in candidate_specs
        ]
        summary = {
            "image": path.name,
            "shape": list(pixels.shape),
            "crop_size": args.crop_size,
            "white": args.white,
            "gamma": args.gamma,
            "rows": rows,
        }
        summaries.append(summary)
        print(path.name)
        for row in rows:
            print(f"  {row['label']}")
            if row["encoded_bytes"] is not None:
                print(
                    f"    encoded={row['encoded_bytes']:,} "
                    f"ratio={row['ratio_vs_original']:.2f}x"
                )
            for region_name in ("dark_0_0.25", "mid_0.25_1", "highlight_1_4", "extreme_gt4"):
                region = row["regions"][region_name]
                luma = region["display_luma"]
                grad = region["display_luma_gradient"]
                print(
                    f"    {region_name:15s} "
                    f"luma_rmse={luma['rmse']:.3e} "
                    f"luma_p99={luma['p99_abs']:.3e} "
                    f"grad_nrmse={grad['grad_nrmse']:.3e} "
                    f"grad_ratio={grad['grad_energy_ratio']:.3f} "
                    f"lost={grad['lost_detail_rate']:.3%}"
                )
    if not args.no_save:
        safe_glob = args.glob.replace("*", "star").replace(".", "_")
        output = RESULTS_DIR / (
            f"display_quality_regions_{safe_glob}"
            f"_crop{args.crop_size}_w{args.white:g}_g{args.gamma:g}.json"
        )
        output.write_text(json.dumps(summaries, indent=2))
        print(f"\nSaved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
