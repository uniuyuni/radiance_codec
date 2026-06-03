"""Create visual previews for signed-log linear-index dithering experiments.

This is intentionally a probe, not codec behavior.  It quantizes signed-log
transformed values to a requested bit depth, optionally applies deterministic
Floyd-Steinberg error diffusion in index space, reconstructs float32 pixels,
and writes 16-bit gamma-corrected PNG previews into a phase subdirectory.
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
OUTPUT_DIR = ROOT / "outputs"
RESULTS_DIR = ROOT / "results"
DEFAULT_PREVIEW_DIR = OUTPUT_DIR / "previews" / "dither_breakthrough"
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_linear_index_codec import error_stats  # noqa: E402
from export_linear_index_preview_png import to_preview_u16, write_png  # noqa: E402
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


def signed_log(values: np.ndarray) -> np.ndarray:
    values64 = values.astype(np.float64)
    return np.sign(values64) * np.log2(1.0 + np.abs(values64))


def inverse_signed_log(values: np.ndarray) -> np.ndarray:
    values64 = values.astype(np.float64)
    return np.sign(values64) * (np.exp2(np.abs(values64)) - 1.0)


def box_mean(values: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return values.astype(np.float64, copy=False)
    padded = np.pad(values.astype(np.float64, copy=False), radius, mode="reflect")
    integral = np.pad(
        padded,
        ((1, 0), (1, 0)),
        mode="constant",
        constant_values=0.0,
    ).cumsum(axis=0).cumsum(axis=1)
    size = 2 * radius + 1
    total = (
        integral[size:, size:]
        - integral[:-size, size:]
        - integral[size:, :-size]
        + integral[:-size, :-size]
    )
    return total / float(size * size)


def local_std(values: np.ndarray, radius: int) -> np.ndarray:
    mean = box_mean(values, radius)
    mean_sq = box_mean(values * values, radius)
    variance = np.maximum(mean_sq - mean * mean, 0.0)
    return np.sqrt(variance)


def dark_luma_mask(pixels: np.ndarray, dark_max: float) -> np.ndarray:
    rgb = pixels[:, :, :3].astype(np.float64)
    luma = 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]
    return luma <= float(dark_max)


def adaptive_dither_mask(
    pixels: np.ndarray,
    transformed: np.ndarray,
    step: float,
    channel: int,
    radius: int,
    std_step_threshold: float,
    dark_max: float,
) -> np.ndarray:
    std = local_std(transformed, radius)
    banding_prone = std < float(std_step_threshold) * float(step)
    if dark_max >= 0.0 and pixels.shape[2] >= 3:
        banding_prone &= dark_luma_mask(pixels, dark_max)
    return banding_prone


def dark_router_mask(
    pixels: np.ndarray,
    bits: int,
    dark_max: float,
    radius: int,
    std_step_threshold: float,
    mode: str,
) -> np.ndarray:
    luma_mask = dark_luma_mask(pixels, dark_max)
    if mode == "luma":
        return luma_mask
    if mode != "banding":
        raise ValueError(f"unknown dark router mask mode: {mode}")

    levels = (1 << bits) - 1
    risk = np.zeros(pixels.shape[:2], dtype=bool)
    for channel in range(pixels.shape[2]):
        transformed = signed_log(pixels[:, :, channel])
        range_values = transformed.astype(np.float32).astype(np.float64)
        lo = float(np.min(range_values))
        hi = float(np.max(range_values))
        if not hi > lo:
            continue
        step = (hi - lo) / levels
        risk |= local_std(transformed, radius) < float(std_step_threshold) * float(step)
    return luma_mask & risk


def quantize_no_dither(values: np.ndarray, levels: int) -> np.ndarray:
    return np.rint(values).clip(0, levels).astype(np.uint16)


def hash_noise(shape: tuple[int, int], channel: int) -> np.ndarray:
    yy, xx = np.indices(shape, dtype=np.uint32)
    with np.errstate(over="ignore"):
        h = xx * np.uint32(374761393)
        h ^= yy * np.uint32(668265263)
        h ^= np.uint32(channel + 1) * np.uint32(2246822519)
        h ^= h >> np.uint32(13)
        h *= np.uint32(1274126177)
        h ^= h >> np.uint32(16)
    return (h.astype(np.float64) / float(2**32)) - 0.5


def quantize_ordered_dither(
    values: np.ndarray,
    levels: int,
    mask: np.ndarray | None,
    channel: int,
    amplitude: float,
) -> np.ndarray:
    work = values.astype(np.float64, copy=True)
    active = np.ones(values.shape, dtype=bool) if mask is None else mask.astype(bool, copy=False)
    noise = hash_noise(values.shape, channel) * float(amplitude)
    work[active] += noise[active]
    return np.rint(work).clip(0, levels).astype(np.uint16)


def quantize_block_average_dither(
    values: np.ndarray,
    levels: int,
    mask: np.ndarray | None,
    channel: int,
    block_size: int,
) -> np.ndarray:
    """Choose floor/ceil counts per block so local mean stays close to source.

    This is a probe for dark-region visual blending.  Unlike FS diffusion, it
    does not push quantization error as a visible high-frequency walk; each
    block gets roughly the right number of ceil samples for its fractional mass.
    """
    active = np.ones(values.shape, dtype=bool) if mask is None else mask.astype(bool, copy=False)
    out = quantize_no_dither(values, levels)
    size = max(int(block_size), 1)
    height, width = values.shape
    floor_values = np.floor(values).clip(0, levels)
    fractions = values - floor_values
    tie = hash_noise(values.shape, channel) * 1.0e-6

    for y0 in range(0, height, size):
        y1 = min(y0 + size, height)
        for x0 in range(0, width, size):
            x1 = min(x0 + size, width)
            block_active = active[y0:y1, x0:x1]
            if not np.any(block_active):
                continue

            block_floor = floor_values[y0:y1, x0:x1]
            block_frac = fractions[y0:y1, x0:x1]
            eligible = block_active & (block_floor < levels)
            if not np.any(eligible):
                out[y0:y1, x0:x1][block_active] = block_floor[block_active].astype(np.uint16)
                continue

            ceil_count = int(np.rint(float(np.sum(block_frac[block_active]))))
            ceil_count = min(max(ceil_count, 0), int(np.count_nonzero(eligible)))
            block_out = out[y0:y1, x0:x1]
            block_out[block_active] = block_floor[block_active].astype(np.uint16)
            if ceil_count > 0:
                score = block_frac + tie[y0:y1, x0:x1]
                flat_scores = score[eligible]
                threshold = np.partition(flat_scores, flat_scores.size - ceil_count)[
                    flat_scores.size - ceil_count
                ]
                ceil_mask = eligible & (score >= threshold)
                selected = np.flatnonzero(ceil_mask)
                if selected.size > ceil_count:
                    keep = selected[np.argsort(score.ravel()[selected])[-ceil_count:]]
                    ceil_mask = np.zeros(ceil_mask.shape, dtype=bool)
                    ceil_mask.ravel()[keep] = True
                block_out[ceil_mask] = (block_floor[ceil_mask] + 1.0).astype(np.uint16)
    return out


def quantize_fs_dither(
    values: np.ndarray,
    levels: int,
    mask: np.ndarray | None,
    serpentine: bool,
    diffusion_strength: float,
    error_clip: float,
) -> np.ndarray:
    height, width = values.shape
    work = values.astype(np.float64, copy=True)
    out = np.zeros(values.shape, dtype=np.uint16)
    if mask is None:
        active = np.ones(values.shape, dtype=bool)
    else:
        active = mask.astype(bool, copy=False)

    for y in range(height):
        if serpentine and (y & 1):
            x_iter = range(width - 1, -1, -1)
            direction = -1
        else:
            x_iter = range(width)
            direction = 1
        for x in x_iter:
            if not active[y, x]:
                q = round(float(values[y, x]))
                out[y, x] = np.uint16(min(max(q, 0), levels))
                continue

            old = float(work[y, x])
            q = round(old)
            q = min(max(q, 0), levels)
            out[y, x] = np.uint16(q)
            error = old - float(q)
            if error_clip > 0.0:
                error = min(max(error, -error_clip), error_clip)
            error *= diffusion_strength

            nx = x + direction
            px = x - direction
            if 0 <= nx < width and active[y, nx]:
                work[y, nx] += error * (7.0 / 16.0)
            if y + 1 < height:
                if 0 <= px < width and active[y + 1, px]:
                    work[y + 1, px] += error * (3.0 / 16.0)
                if active[y + 1, x]:
                    work[y + 1, x] += error * (5.0 / 16.0)
                if 0 <= nx < width and active[y + 1, nx]:
                    work[y + 1, nx] += error * (1.0 / 16.0)
    return out


def reconstruct_signed_log(
    pixels: np.ndarray,
    bits: int,
    dither: str,
    adaptive_radius: int,
    std_step_threshold: float,
    dark_max: float,
    serpentine: bool,
    diffusion_strength: float,
    error_clip: float,
    ordered_amplitude: float,
    block_size: int,
    recon_bias: float,
    recon_bias_dark_max: float,
    recon_table: str,
    dark_bits: int,
    dark_bits_max: float,
    dark_bits_mask: str,
) -> tuple[np.ndarray, dict]:
    levels = (1 << bits) - 1
    dark_levels = (1 << dark_bits) - 1 if dark_bits > 0 else 0
    decoded = np.zeros(pixels.shape, dtype=np.float32)
    indices = np.zeros(pixels.shape, dtype=np.uint16)
    masks = []
    ranges = []
    for channel in range(pixels.shape[2]):
        transformed = signed_log(pixels[:, :, channel])
        range_values = transformed.astype(np.float32).astype(np.float64)
        lo = float(np.min(range_values))
        hi = float(np.max(range_values))
        ranges.append({"channel": channel, "lo": lo, "hi": hi})
        if not hi > lo:
            decoded[:, :, channel] = inverse_signed_log(np.asarray(lo)).astype(np.float32)
            masks.append(np.zeros(transformed.shape, dtype=bool))
            continue

        scale = levels / (hi - lo)
        q_values = (transformed - lo) * scale
        step = (hi - lo) / levels
        mask = None
        if dither == "none":
            q = quantize_no_dither(q_values, levels)
            mask = np.zeros(transformed.shape, dtype=bool)
        elif dither == "fs":
            q = quantize_fs_dither(
                q_values,
                levels,
                None,
                serpentine,
                diffusion_strength,
                error_clip,
            )
            mask = np.ones(transformed.shape, dtype=bool)
        elif dither == "ordered":
            q = quantize_ordered_dither(
                q_values,
                levels,
                None,
                channel,
                ordered_amplitude,
            )
            mask = np.ones(transformed.shape, dtype=bool)
        elif dither == "block":
            q = quantize_block_average_dither(
                q_values,
                levels,
                None,
                channel,
                block_size,
            )
            mask = np.ones(transformed.shape, dtype=bool)
        elif dither == "adaptive-fs":
            mask = adaptive_dither_mask(
                pixels,
                transformed,
                step,
                channel,
                adaptive_radius,
                std_step_threshold,
                dark_max,
            )
            q = quantize_fs_dither(
                q_values,
                levels,
                mask,
                serpentine,
                diffusion_strength,
                error_clip,
            )
        elif dither == "adaptive-ordered":
            mask = adaptive_dither_mask(
                pixels,
                transformed,
                step,
                channel,
                adaptive_radius,
                std_step_threshold,
                dark_max,
            )
            q = quantize_ordered_dither(
                q_values,
                levels,
                mask,
                channel,
                ordered_amplitude,
            )
        elif dither == "adaptive-block":
            mask = adaptive_dither_mask(
                pixels,
                transformed,
                step,
                channel,
                adaptive_radius,
                std_step_threshold,
                dark_max,
            )
            q = quantize_block_average_dither(
                q_values,
                levels,
                mask,
                channel,
                block_size,
            )
        else:
            raise ValueError(f"unknown dither mode: {dither}")

        indices[:, :, channel] = q
        q_recon = q.astype(np.float64)
        if recon_bias != 0.0:
            if recon_bias_dark_max >= 0.0 and pixels.shape[2] >= 3:
                bias_mask = dark_luma_mask(pixels, recon_bias_dark_max)
                q_recon = q_recon.copy()
                q_recon[bias_mask] += float(recon_bias)
            else:
                q_recon = q_recon + float(recon_bias)
            q_recon = np.clip(q_recon, 0.0, float(levels))
        rec_t = lo + q_recon * (hi - lo) / levels
        if dark_bits > 0:
            if dark_bits < bits:
                raise ValueError("--dark-bits must be greater than or equal to --bits")
            if recon_table != "center":
                raise ValueError("--dark-bits is not supported with --recon-table yet")
            dark_mask = dark_router_mask(
                pixels,
                bits,
                dark_bits_max,
                adaptive_radius,
                std_step_threshold,
                dark_bits_mask,
            )
            if np.any(dark_mask):
                dark_scale = dark_levels / (hi - lo)
                q_dark_values = (transformed - lo) * dark_scale
                q_dark = quantize_no_dither(q_dark_values, dark_levels)
                rec_t_dark = lo + q_dark.astype(np.float64) * (hi - lo) / dark_levels
                rec_t = rec_t.copy()
                rec_t[dark_mask] = rec_t_dark[dark_mask]
        decoded[:, :, channel] = inverse_signed_log(rec_t).astype(np.float32)
        masks.append(mask)

    if recon_table != "center":
        decoded = table_reconstruct(pixels, indices, bits, recon_table)

    mask_stack = np.stack(masks, axis=2)
    info = {
        "ranges": ranges,
        "dithered_sample_rate": float(np.mean(mask_stack)),
        "dithered_pixel_rate": float(np.mean(np.any(mask_stack, axis=2))),
        "dithered_by_channel": [
            float(np.mean(mask_stack[:, :, channel]))
            for channel in range(mask_stack.shape[2])
        ],
        "dark_bits": dark_bits,
        "dark_bits_max": dark_bits_max,
        "dark_bits_mask": dark_bits_mask,
        "dark_bits_pixel_rate": (
            float(np.mean(dark_router_mask(
                pixels,
                bits,
                dark_bits_max,
                adaptive_radius,
                std_step_threshold,
                dark_bits_mask,
            ))) if dark_bits > 0 else 0.0
        ),
    }
    return np.ascontiguousarray(decoded), info


def parse_ints(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split(",") if part.strip())


def parse_strings(text: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in text.split(",") if part.strip())


def export_case(
    path: Path,
    pixels: np.ndarray,
    bits: int,
    dither: str,
    crop_size: int,
    white: float,
    gamma: float,
    output_dir: Path,
    adaptive_radius: int,
    std_step_threshold: float,
    dark_max: float,
    serpentine: bool,
    diffusion_strength: float,
    error_clip: float,
    ordered_amplitude: float,
    block_size: int,
    recon_bias: float,
    recon_bias_dark_max: float,
    recon_table: str,
    dark_bits: int,
    dark_bits_max: float,
    dark_bits_mask: str,
    write_original: bool,
) -> dict:
    started = time.perf_counter()
    decoded, dither_info = reconstruct_signed_log(
        pixels,
        bits,
        dither,
        adaptive_radius,
        std_step_threshold,
        dark_max,
        serpentine,
        diffusion_strength,
        error_clip,
        ordered_amplitude,
        block_size,
        recon_bias,
        recon_bias_dark_max,
        recon_table,
        dark_bits,
        dark_bits_max,
        dark_bits_mask,
    )
    elapsed = time.perf_counter() - started
    stem = path.stem.replace(" ", "_")
    dither_label = dither
    if dither == "adaptive-fs":
        dither_label = (
            f"{dither}_std{std_step_threshold:g}"
            f"_dark{dark_max:g}_r{adaptive_radius}"
            f"_s{diffusion_strength:g}_clip{error_clip:g}"
        )
    elif dither == "fs":
        dither_label = f"{dither}_s{diffusion_strength:g}_clip{error_clip:g}"
    elif dither == "adaptive-ordered":
        dither_label = (
            f"{dither}_std{std_step_threshold:g}"
            f"_dark{dark_max:g}_r{adaptive_radius}"
            f"_amp{ordered_amplitude:g}"
        )
    elif dither == "ordered":
        dither_label = f"{dither}_amp{ordered_amplitude:g}"
    elif dither == "adaptive-block":
        dither_label = (
            f"{dither}_std{std_step_threshold:g}"
            f"_dark{dark_max:g}_r{adaptive_radius}"
            f"_b{block_size}"
        )
    elif dither == "block":
        dither_label = f"{dither}_b{block_size}"
    if recon_bias != 0.0:
        dither_label = (
            f"{dither_label}_rb{recon_bias:g}"
            f"_rbdark{recon_bias_dark_max:g}"
        )
    if recon_table != "center":
        dither_label = f"{dither_label}_rt{recon_table}"
    if dark_bits > 0:
        dither_label = (
            f"{dither_label}_darkbits{dark_bits}"
            f"_darkmax{dark_bits_max:g}_darkmask{dark_bits_mask}"
        )
    suffix = (
        f"{stem}_crop{crop_size}_signed-log_bits{bits}_{dither_label}"
        f"_w{white:g}_g{gamma:g}"
    )
    original_png = output_dir / f"{stem}_crop{crop_size}_original_w{white:g}_g{gamma:g}.png"
    decoded_png = output_dir / f"{suffix}_decoded.png"
    if write_original and not original_png.exists():
        write_png(original_png, to_preview_u16(pixels, white, gamma))
    write_png(decoded_png, to_preview_u16(decoded, white, gamma))

    stats = {
        "image": path.name,
        "shape": list(pixels.shape),
        "crop_size": crop_size,
        "bits": bits,
        "transform": "signed-log",
        "dither": dither,
        "white": white,
        "gamma": gamma,
        "adaptive_radius": adaptive_radius,
        "std_step_threshold": std_step_threshold,
        "dark_max": dark_max,
        "serpentine": serpentine,
        "diffusion_strength": diffusion_strength,
        "error_clip": error_clip,
        "ordered_amplitude": ordered_amplitude,
        "block_size": block_size,
        "recon_bias": recon_bias,
        "recon_bias_dark_max": recon_bias_dark_max,
        "recon_table": recon_table,
        "dark_bits": dark_bits,
        "dark_bits_max": dark_bits_max,
        "dark_bits_mask": dark_bits_mask,
        "elapsed_sec": elapsed,
        "raw_bytes": pixels.nbytes,
        "decoded_png": str(decoded_png),
        "original_png": str(original_png) if write_original else None,
        "quality": error_stats(pixels, decoded),
        **dither_info,
    }
    manifest = output_dir / f"{suffix}_manifest.json"
    manifest.write_text(json.dumps(stats, indent=2))
    stats["manifest"] = str(manifest)
    return stats


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", default="sample_DSCF0009.EXR")
    parser.add_argument("--crop-size", type=int, default=1024)
    parser.add_argument("--bits", default="8,9,10")
    parser.add_argument("--dither", default="none,adaptive-fs")
    parser.add_argument("--white", type=float, default=4.0)
    parser.add_argument("--gamma", type=float, default=2.2)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_PREVIEW_DIR)
    parser.add_argument("--adaptive-radius", type=int, default=2)
    parser.add_argument("--std-step-threshold", type=float, default=0.5)
    parser.add_argument("--dark-max", type=float, default=0.25)
    parser.add_argument("--diffusion-strength", type=float, default=1.0)
    parser.add_argument("--error-clip", type=float, default=0.0)
    parser.add_argument("--ordered-amplitude", type=float, default=1.0)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--recon-bias", type=float, default=0.0)
    parser.add_argument("--recon-bias-dark-max", type=float, default=-1.0)
    parser.add_argument(
        "--recon-table",
        choices=("center", "value-mean", "signed-log-mean"),
        default="center",
    )
    parser.add_argument("--dark-bits", type=int, default=0)
    parser.add_argument("--dark-bits-max", type=float, default=0.18)
    parser.add_argument("--dark-bits-mask", choices=("luma", "banding"), default="luma")
    parser.add_argument("--no-serpentine", action="store_true")
    parser.add_argument("--no-original", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-save-summary", action="store_true")
    args = parser.parse_args()

    bits_values = parse_ints(args.bits)
    dither_values = parse_strings(args.dither)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for path in sorted(DATA_DIR.glob(args.glob)):
        pixels = read_exr(path, args.crop_size)
        print(path.name)
        for bits in bits_values:
            for dither in dither_values:
                row = export_case(
                    path,
                    pixels,
                    bits,
                    dither,
                    args.crop_size,
                    args.white,
                    args.gamma,
                    output_dir,
                    args.adaptive_radius,
                    args.std_step_threshold,
                    args.dark_max,
                    not args.no_serpentine,
                    args.diffusion_strength,
                    args.error_clip,
                    args.ordered_amplitude,
                    args.block_size,
                    args.recon_bias,
                    args.recon_bias_dark_max,
                    args.recon_table,
                    args.dark_bits,
                    args.dark_bits_max,
                    args.dark_bits_mask,
                    not args.no_original,
                )
                summaries.append(row)
                q = row["quality"]
                print(
                    f"  bits{bits} {dither:11s} "
                    f"log={q['signed_log2_rmse']:.3e} "
                    f"p99={q['signed_log2_p99']:.3e} "
                    f"grad={q['gradient_signed_log2_nrmse']:.3e} "
                    f"psnr={q['psnr_peak_abs']:.2f}dB "
                    f"dither_samples={row['dithered_sample_rate']:.2%} "
                    f"png={row['decoded_png']}"
                )
        if args.limit and len(summaries) >= args.limit:
            break

    if not args.no_save_summary:
        safe_glob = args.glob.replace("*", "star").replace(".", "_")
        output = RESULTS_DIR / (
            f"dithered_linear_index_previews_{safe_glob}"
            f"_crop{args.crop_size}_bits{args.bits.replace(',', '-')}.json"
        )
        output.write_text(json.dumps(summaries, indent=2))
        print(f"\nSaved: {output}")
    if not summaries:
        print(f"no files matched: {args.glob}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
