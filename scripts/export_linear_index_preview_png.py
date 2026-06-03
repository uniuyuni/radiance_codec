"""Export 16-bit gamma-corrected PNG previews after codec round-trip."""
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
sys.path.insert(0, str(ROOT / "codec" / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

import radiance_codec  # noqa: E402
from benchmark_linear_index_codec import error_stats  # noqa: E402
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


def to_preview_u16(pixels: np.ndarray, white: float, gamma: float) -> np.ndarray:
    rgb = pixels[:, :, :3].astype(np.float64)
    normalized = np.clip(rgb / float(white), 0.0, 1.0)
    corrected = np.power(normalized, 1.0 / float(gamma))
    return np.rint(corrected * 65535.0).clip(0, 65535).astype(np.uint16)


def write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    spec = oiio.ImageSpec(image.shape[1], image.shape[0], image.shape[2], oiio.UINT16)
    out = oiio.ImageOutput.create(str(path))
    if out is None:
        raise RuntimeError(f"can't create {path}: {oiio.geterror()}")
    if not out.open(str(path), spec):
        raise RuntimeError(f"can't open output {path}: {out.geterror()}")
    if not out.write_image(image):
        raise RuntimeError(f"can't write {path}: {out.geterror()}")
    out.close()


def export_roundtrip(
    path: Path,
    crop_size: int,
    bits: int,
    transform: str,
    mode: str,
    power_gamma: float,
    recon_table: str,
    white: float,
    gamma: float,
    output_dir: Path,
    decoded_only: bool,
) -> dict:
    pixels = read_exr(path, crop_size)
    if mode == "codec":
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
        encoded_bytes = len(encoded)
        encode_seconds = t1 - t0
        decode_seconds = t2 - t1
        mode_label = f"{transform}_bits{bits}"
    elif mode == "power-recon-table":
        t0 = time.perf_counter()
        indices, decoded = quantize_power(pixels, bits, power_gamma)
        if recon_table != "center":
            decoded = table_reconstruct(pixels, indices, bits, recon_table)
        t1 = time.perf_counter()
        encoded_bytes = None
        encode_seconds = None
        decode_seconds = t1 - t0
        mode_label = f"power{power_gamma:g}_bits{bits}_{recon_table}"
    else:
        raise ValueError(f"unknown mode: {mode}")

    stem = path.stem.replace(" ", "_")
    suffix = f"{stem}_crop{crop_size}_{mode_label}_w{white:g}_g{gamma:g}"
    original_png = output_dir / f"{suffix}_original.png"
    decoded_png = output_dir / f"{suffix}_decoded.png"
    if not decoded_only:
        write_png(original_png, to_preview_u16(pixels, white, gamma))
    write_png(decoded_png, to_preview_u16(decoded, white, gamma))

    stats = {
        "image": path.name,
        "shape": list(pixels.shape),
        "crop_size": crop_size,
        "mode": mode,
        "bits": bits,
        "transform": transform,
        "power_gamma": power_gamma,
        "recon_table": recon_table,
        "white": white,
        "gamma": gamma,
        "encoded_bytes": encoded_bytes,
        "raw_bytes": pixels.nbytes,
        "ratio_vs_original": None if encoded_bytes is None else pixels.nbytes / encoded_bytes,
        "encode_seconds": encode_seconds,
        "decode_seconds": decode_seconds,
        "original_png": None if decoded_only else str(original_png),
        "decoded_png": str(decoded_png),
        "quality": error_stats(pixels, decoded),
    }
    manifest = output_dir / f"{suffix}_manifest.json"
    manifest.write_text(json.dumps(stats, indent=2))
    stats["manifest"] = str(manifest)
    return stats


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", default="sample_DSCF0009.EXR")
    parser.add_argument("--crop-size", type=int, default=0)
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--transform", default="gamma075")
    parser.add_argument("--mode", choices=("codec", "power-recon-table"), default="codec")
    parser.add_argument("--power-gamma", type=float, default=0.8)
    parser.add_argument("--recon-table", default="signed-log-mean")
    parser.add_argument("--white", type=float, default=4.0)
    parser.add_argument("--gamma", type=float, default=2.2)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR / "previews")
    parser.add_argument("--decoded-only", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    rows = []
    for path in sorted(DATA_DIR.glob(args.glob)):
        row = export_roundtrip(
            path,
            args.crop_size,
            args.bits,
            args.transform,
            args.mode,
            args.power_gamma,
            args.recon_table,
            args.white,
            args.gamma,
            args.output_dir,
            args.decoded_only,
        )
        rows.append(row)
        q = row["quality"]
        ratio_text = (
            "n/a"
            if row["ratio_vs_original"] is None
            else f"{row['ratio_vs_original']:.2f}x"
        )
        enc_text = (
            "n/a"
            if row["encode_seconds"] is None
            else f"{row['encode_seconds']:.2f}s"
        )
        print(path.name)
        print(
            f"  mode={row['mode']} "
            f"encoded={row['encoded_bytes']} bytes "
            f"ratio={ratio_text} "
            f"enc={enc_text} dec={row['decode_seconds']:.2f}s"
        )
        print(
            f"  quality log_rmse={q['signed_log2_rmse']:.3e} "
            f"p99={q['signed_log2_p99']:.3e} "
            f"grad={q['gradient_signed_log2_nrmse']:.3e} "
            f"psnr={q['psnr_peak_abs']:.2f}dB"
        )
        if row["original_png"] is not None:
            print(f"  original_png={row['original_png']}")
        print(f"  decoded_png={row['decoded_png']}")
        print(f"  manifest={row['manifest']}")
        if args.limit and len(rows) >= args.limit:
            break
    if not rows:
        print(f"no files matched: {args.glob}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
