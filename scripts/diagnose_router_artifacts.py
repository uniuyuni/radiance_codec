"""Create small visual and numeric diagnostics for router artifacts.

This deliberately avoids writing full-resolution previews. Large PNGs can be
awkward to inspect in Codex, so this script writes only downsized previews and
small crops, plus a JSON file with display-space and texture-loss metrics.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import OpenImageIO as oiio

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs" / "diagnostics" / "router_artifacts"
RESULTS_DIR = ROOT / "results"
sys.path.insert(0, str(ROOT / "codec" / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

import radiance_codec  # noqa: E402


def read_exr(path: Path) -> np.ndarray:
    image_input = oiio.ImageInput.open(str(path))
    if image_input is None:
        raise RuntimeError(f"can't open {path}: {oiio.geterror()}")
    spec = image_input.spec()
    pixels = image_input.read_image(format=oiio.FLOAT)
    image_input.close()
    if pixels is None:
        raise RuntimeError(f"can't read {path}: {oiio.geterror()}")
    arr = np.asarray(pixels, dtype=np.float32).reshape(
        spec.height,
        spec.width,
        spec.nchannels,
    )
    return np.ascontiguousarray(arr[:, :, :3])


def write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if image.dtype != np.uint16:
        raise TypeError(f"expected uint16 image, got {image.dtype}")
    spec = oiio.ImageSpec(image.shape[1], image.shape[0], image.shape[2], oiio.UINT16)
    out = oiio.ImageOutput.create(str(path))
    if out is None:
        raise RuntimeError(f"can't create {path}: {oiio.geterror()}")
    if not out.open(str(path), spec):
        raise RuntimeError(f"can't open output {path}: {out.geterror()}")
    if not out.write_image(image):
        raise RuntimeError(f"can't write {path}: {out.geterror()}")
    out.close()


def display_map(pixels: np.ndarray, white: float, gamma: float) -> np.ndarray:
    rgb = pixels[:, :, :3].astype(np.float64)
    normalized = np.clip(rgb / float(white), 0.0, 1.0)
    return np.power(normalized, 1.0 / float(gamma))


def to_u16(display_rgb: np.ndarray) -> np.ndarray:
    return np.rint(np.clip(display_rgb, 0.0, 1.0) * 65535.0).astype(np.uint16)


def luma(display_rgb: np.ndarray) -> np.ndarray:
    return (
        display_rgb[:, :, 0] * 0.2126
        + display_rgb[:, :, 1] * 0.7152
        + display_rgb[:, :, 2] * 0.0722
    )


def sampled_preview(display_rgb: np.ndarray, max_width: int) -> np.ndarray:
    h, w, _ = display_rgb.shape
    if w <= max_width:
        return display_rgb
    out_w = max_width
    out_h = max(1, int(round(h * (out_w / w))))
    ys = np.linspace(0, h - 1, out_h).round().astype(np.int64)
    xs = np.linspace(0, w - 1, out_w).round().astype(np.int64)
    return display_rgb[ys][:, xs]


def diff_preview(original_display: np.ndarray, decoded_display: np.ndarray, scale: float) -> np.ndarray:
    diff = (decoded_display - original_display) * float(scale)
    return np.clip(0.5 + diff, 0.0, 1.0)


def abs_error_preview(original_display: np.ndarray, decoded_display: np.ndarray, scale: float) -> np.ndarray:
    err = np.max(np.abs(decoded_display - original_display), axis=2) * float(scale)
    gray = np.clip(err, 0.0, 1.0)
    return np.repeat(gray[:, :, None], 3, axis=2)


def crop_bounds(cx: int, cy: int, size: int, width: int, height: int) -> tuple[int, int, int, int]:
    half = size // 2
    x0 = min(max(0, cx - half), max(0, width - size))
    y0 = min(max(0, cy - half), max(0, height - size))
    x1 = min(width, x0 + size)
    y1 = min(height, y0 + size)
    return x0, y0, x1, y1


def gradient_rms(values: np.ndarray) -> float:
    if values.shape[0] < 2 or values.shape[1] < 2:
        return 0.0
    dx = values[:, 1:] - values[:, :-1]
    dy = values[1:, :] - values[:-1, :]
    return float(math.sqrt((np.mean(dx * dx) + np.mean(dy * dy)) * 0.5))


def window_metrics(
    original_luma: np.ndarray,
    decoded_luma: np.ndarray,
    crop_size: int,
    stride: int,
) -> list[dict]:
    h, w = original_luma.shape
    rows: list[dict] = []
    for y0 in range(0, max(1, h - crop_size + 1), stride):
        for x0 in range(0, max(1, w - crop_size + 1), stride):
            y1 = min(h, y0 + crop_size)
            x1 = min(w, x0 + crop_size)
            orig = original_luma[y0:y1, x0:x1]
            dec = decoded_luma[y0:y1, x0:x1]
            delta = dec - orig
            rmse = float(np.sqrt(np.mean(delta * delta)))
            mae = float(np.mean(np.abs(delta)))
            orig_grad = gradient_rms(orig)
            dec_grad = gradient_rms(dec)
            grad_ratio = dec_grad / orig_grad if orig_grad > 1e-12 else 1.0
            rows.append(
                {
                    "x": int(x0),
                    "y": int(y0),
                    "w": int(x1 - x0),
                    "h": int(y1 - y0),
                    "center_x": int((x0 + x1) // 2),
                    "center_y": int((y0 + y1) // 2),
                    "display_luma_rmse": rmse,
                    "display_luma_mae": mae,
                    "original_gradient_rms": orig_grad,
                    "decoded_gradient_rms": dec_grad,
                    "gradient_ratio_decoded_over_original": grad_ratio,
                    "texture_loss_score": rmse * max(0.0, 1.0 - grad_ratio),
                    "texture_gain_score": rmse * max(0.0, grad_ratio - 1.0),
                }
            )
    return rows


def top_unique(rows: list[dict], key: str, count: int, min_distance: int) -> list[dict]:
    selected: list[dict] = []
    for row in sorted(rows, key=lambda item: float(item[key]), reverse=True):
        cx = int(row["center_x"])
        cy = int(row["center_y"])
        if all(
            (cx - int(prev["center_x"])) ** 2 + (cy - int(prev["center_y"])) ** 2
            >= min_distance * min_distance
            for prev in selected
        ):
            selected.append(row)
            if len(selected) >= count:
                break
    return selected


def region_stats(original: np.ndarray, decoded: np.ndarray, original_display: np.ndarray, decoded_display: np.ndarray) -> dict:
    max_rgb = np.max(original[:, :, :3].astype(np.float64), axis=2)
    masks = {
        "dark_le_0.25": max_rgb <= 0.25,
        "mid_0.25_1": (max_rgb > 0.25) & (max_rgb <= 1.0),
        "highlight_1_4": (max_rgb > 1.0) & (max_rgb <= 4.0),
        "extreme_gt4": max_rgb > 4.0,
        "all": np.ones(max_rgb.shape, dtype=bool),
    }
    diff = decoded_display - original_display
    result = {}
    for name, mask in masks.items():
        if not bool(np.any(mask)):
            result[name] = {"pixels": 0}
            continue
        values = diff[mask]
        abs_values = np.abs(values)
        result[name] = {
            "pixels": int(mask.sum()),
            "display_rgb_rmse": float(np.sqrt(np.mean(values * values))),
            "display_rgb_mae": float(np.mean(abs_values)),
            "display_rgb_p99_abs": float(np.percentile(abs_values, 99.0)),
            "display_rgb_max_abs": float(np.max(abs_values)),
        }
    return result


def tile_boundary_scores(values: np.ndarray, periods: tuple[int, ...]) -> dict:
    scores = {}
    horizontal = np.abs(values[:, 1:] - values[:, :-1])
    vertical = np.abs(values[1:, :] - values[:-1, :])
    for period in periods:
        x_boundary = (np.arange(1, values.shape[1]) % period) == 0
        y_boundary = (np.arange(1, values.shape[0]) % period) == 0
        x_non = ~x_boundary
        y_non = ~y_boundary
        boundary = np.concatenate(
            [
                horizontal[:, x_boundary].ravel(),
                vertical[y_boundary, :].ravel(),
            ]
        )
        non_boundary = np.concatenate(
            [
                horizontal[:, x_non].ravel(),
                vertical[y_non, :].ravel(),
            ]
        )
        b_mean = float(np.mean(boundary)) if boundary.size else 0.0
        n_mean = float(np.mean(non_boundary)) if non_boundary.size else 0.0
        scores[str(period)] = {
            "boundary_mean_abs_delta": b_mean,
            "non_boundary_mean_abs_delta": n_mean,
            "ratio": b_mean / n_mean if n_mean > 1e-12 else 1.0,
        }
    return scores


def save_crop_set(
    output_dir: Path,
    prefix: str,
    original_display: np.ndarray,
    decoded_display: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    diff_scale: float,
    error_scale: float,
) -> dict:
    original_crop = original_display[y0:y1, x0:x1]
    decoded_crop = decoded_display[y0:y1, x0:x1]
    paths = {
        "original": output_dir / f"{prefix}_original.png",
        "candidate": output_dir / f"{prefix}_candidate.png",
        "diff_neutral_scale": output_dir / f"{prefix}_diff_neutral_x{diff_scale:g}.png",
        "abs_error": output_dir / f"{prefix}_abs_error_x{error_scale:g}.png",
    }
    write_png(paths["original"], to_u16(original_crop))
    write_png(paths["candidate"], to_u16(decoded_crop))
    write_png(paths["diff_neutral_scale"], to_u16(diff_preview(original_crop, decoded_crop, diff_scale)))
    write_png(paths["abs_error"], to_u16(abs_error_preview(original_crop, decoded_crop, error_scale)))
    return {key: str(value) for key, value in paths.items()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="sample_cat_noisy.EXR")
    parser.add_argument("--white", type=float, default=4.0)
    parser.add_argument("--gamma", type=float, default=2.2)
    parser.add_argument("--preview-width", type=int, default=1024)
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--top-crops", type=int, default=4)
    parser.add_argument("--anchor-bits", type=int, default=10)
    parser.add_argument("--diff-scale", type=float, default=4.0)
    parser.add_argument("--error-scale", type=float, default=8.0)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--result", type=Path, default=Path())
    parser.add_argument(
        "--codec-path",
        choices=("encode-decode", "reconstruct"),
        default="encode-decode",
    )
    args = parser.parse_args()

    image_path = DATA_DIR / args.image
    stem = image_path.stem.replace(" ", "_")
    out_dir = args.output_dir / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    pixels = read_exr(image_path)
    read_s = time.perf_counter() - started

    encoded_bytes = None
    encode_s = None
    decode_s = None
    report = {}
    if args.codec_path == "encode-decode":
        started = time.perf_counter()
        encoded = radiance_codec.encode_near_lossless_router_v1(pixels)
        encode_s = time.perf_counter() - started
        encoded_bytes = len(encoded)
        started = time.perf_counter()
        decoded = radiance_codec.decode(encoded, pixels.shape)
        decode_s = time.perf_counter() - started
    else:
        started = time.perf_counter()
        decoded, report = radiance_codec.reconstruct_near_lossless_router_v1(
            pixels,
            anchor_bits=args.anchor_bits,
            visual_guard_white=args.white,
            visual_guard_gamma=args.gamma,
        )
        decode_s = time.perf_counter() - started

    original_display = display_map(pixels, args.white, args.gamma)
    decoded_display = display_map(decoded, args.white, args.gamma)
    original_luma = luma(original_display)
    decoded_luma = luma(decoded_display)
    diff_luma = decoded_luma - original_luma

    preview_paths = {
        "original": out_dir / f"{stem}_preview_w{args.preview_width}_original.png",
        "candidate": out_dir / f"{stem}_preview_w{args.preview_width}_candidate.png",
        "diff_neutral_scale": out_dir / f"{stem}_preview_w{args.preview_width}_diff_neutral_x{args.diff_scale:g}.png",
        "abs_error": out_dir / f"{stem}_preview_w{args.preview_width}_abs_error_x{args.error_scale:g}.png",
    }
    write_png(preview_paths["original"], to_u16(sampled_preview(original_display, args.preview_width)))
    write_png(preview_paths["candidate"], to_u16(sampled_preview(decoded_display, args.preview_width)))
    write_png(
        preview_paths["diff_neutral_scale"],
        to_u16(sampled_preview(diff_preview(original_display, decoded_display, args.diff_scale), args.preview_width)),
    )
    write_png(
        preview_paths["abs_error"],
        to_u16(sampled_preview(abs_error_preview(original_display, decoded_display, args.error_scale), args.preview_width)),
    )

    rows = window_metrics(original_luma, decoded_luma, args.crop_size, args.stride)
    by_rmse = top_unique(rows, "display_luma_rmse", args.top_crops, args.crop_size)
    by_loss = top_unique(rows, "texture_loss_score", args.top_crops, args.crop_size)
    by_gain = top_unique(rows, "texture_gain_score", args.top_crops, args.crop_size)
    fixed = [
        {
            "label": "center",
            "center_x": pixels.shape[1] // 2,
            "center_y": pixels.shape[0] // 2,
        },
        {
            "label": "upper_left_third",
            "center_x": pixels.shape[1] // 3,
            "center_y": pixels.shape[0] // 3,
        },
        {
            "label": "lower_right_third",
            "center_x": pixels.shape[1] * 2 // 3,
            "center_y": pixels.shape[0] * 2 // 3,
        },
    ]

    crop_entries = []
    seen: set[tuple[int, int]] = set()
    for group, candidates in (
        ("display_rmse", by_rmse),
        ("texture_loss", by_loss),
        ("texture_gain", by_gain),
        ("fixed", fixed),
    ):
        for index, row in enumerate(candidates, start=1):
            cx = int(row["center_x"])
            cy = int(row["center_y"])
            x0, y0, x1, y1 = crop_bounds(cx, cy, args.crop_size, pixels.shape[1], pixels.shape[0])
            key = (x0, y0)
            if key in seen:
                continue
            seen.add(key)
            label = row.get("label", f"{group}_{index:02d}")
            prefix = f"{stem}_crop_{label}_x{x0}_y{y0}_s{x1 - x0}"
            paths = save_crop_set(
                out_dir,
                prefix,
                original_display,
                decoded_display,
                x0,
                y0,
                x1,
                y1,
                args.diff_scale,
                args.error_scale,
            )
            crop_entries.append(
                {
                    "group": group,
                    "label": label,
                    "x": x0,
                    "y": y0,
                    "w": x1 - x0,
                    "h": y1 - y0,
                    "metrics": row,
                    "paths": paths,
                }
            )

    result = {
        "image": args.image,
        "shape": list(pixels.shape),
        "white": args.white,
        "gamma": args.gamma,
        "anchor_bits": args.anchor_bits,
        "codec_path": args.codec_path,
        "encoded_bytes": encoded_bytes,
        "encoded_mib": None if encoded_bytes is None else encoded_bytes / 1048576.0,
        "environment_flags": {
            key: os.environ.get(key, "")
            for key in (
                "RADIANCE_CODEC_USE_METAL_GUIDED",
                "RADIANCE_CODEC_USE_METAL_DOWNSAMPLE",
                "RADIANCE_CODEC_USE_METAL_HIGHPASS",
                "RADIANCE_CODEC_USE_METAL_VISUAL_GUARD",
                "RADIANCE_CODEC_NO_METAL_GUIDED",
                "RADIANCE_CODEC_NO_METAL_DOWNSAMPLE",
                "RADIANCE_CODEC_NO_METAL_HIGHPASS",
                "RADIANCE_CODEC_NO_METAL_VISUAL_GUARD",
                "RADIANCE_CODEC_ROUTER_NO_DARK_SMOOTH_BYPASS",
                "RADIANCE_CODEC_ROUTER_DARK_NOISE_THRESHOLD",
                "OMP_WAIT_POLICY",
                "KMP_BLOCKTIME",
            )
        },
        "timing": {
            "read_s": read_s,
            "encode_s": encode_s,
            "decode_or_reconstruct_s": decode_s,
        },
        "router_report": report,
        "previews": {key: str(value) for key, value in preview_paths.items()},
        "regions": region_stats(pixels, decoded, original_display, decoded_display),
        "tile_boundary_scores_on_display_luma_diff": tile_boundary_scores(diff_luma, (8, 16, 32, 64)),
        "top_windows": {
            "display_rmse": by_rmse,
            "texture_loss": by_loss,
            "texture_gain": by_gain,
        },
        "crops": crop_entries,
    }

    result_path = args.result
    if not result_path:
        result_path = RESULTS_DIR / f"router_artifact_diagnosis_{stem}.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps({"result": str(result_path), "output_dir": str(out_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
