"""Small file-oriented CLI for radiance_codec frames.

Examples:
  pixi run python scripts/radiance_codec_file.py encode input.exr output.rcodec
  pixi run python scripts/radiance_codec_file.py decode output.rcodec decoded.exr
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import OpenImageIO as oiio

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "codec" / "python"))

import radiance_codec  # noqa: E402


def read_float_image(path: Path) -> np.ndarray:
    image_input = oiio.ImageInput.open(str(path))
    if image_input is None:
        raise RuntimeError(f"can't open {path}: {oiio.geterror()}")
    spec = image_input.spec()
    pixels = image_input.read_image(format=oiio.FLOAT)
    image_input.close()
    if pixels is None:
        raise RuntimeError(f"can't read {path}: {oiio.geterror()}")
    return np.ascontiguousarray(
        np.asarray(pixels, dtype=np.float32).reshape(
            spec.height,
            spec.width,
            spec.nchannels,
        )
    )


def write_float_image(path: Path, pixels: np.ndarray) -> None:
    pixels = np.ascontiguousarray(pixels, dtype=np.float32)
    if pixels.ndim == 2:
        pixels = pixels[:, :, np.newaxis]
    if pixels.ndim != 3:
        raise RuntimeError(f"expected HxWxC pixels, got shape {pixels.shape}")
    path.parent.mkdir(parents=True, exist_ok=True)
    spec = oiio.ImageSpec(pixels.shape[1], pixels.shape[0], pixels.shape[2], oiio.FLOAT)
    image_output = oiio.ImageOutput.create(str(path))
    if image_output is None:
        raise RuntimeError(f"can't create {path}: {oiio.geterror()}")
    if not image_output.open(str(path), spec):
        raise RuntimeError(f"can't open output {path}: {image_output.geterror()}")
    if not image_output.write_image(pixels):
        raise RuntimeError(f"can't write {path}: {image_output.geterror()}")
    image_output.close()


def encode_pixels(args: argparse.Namespace, pixels: np.ndarray) -> bytes:
    if args.mode == "lossless":
        return radiance_codec.encode_lossless(
            pixels,
            preset=args.preset,
            effort=args.effort,
        )
    if args.mode == "near":
        return radiance_codec.encode_near_lossless(
            pixels,
            low_bits=args.low_bits,
            effort=args.effort,
            policy=radiance_codec.NearLosslessPolicy(args.policy),
        )
    if args.mode == "router":
        return radiance_codec.encode_near_lossless_router_v1(
            pixels,
            effort=args.effort,
        )
    raise RuntimeError(f"unknown mode: {args.mode}")


def encode_file(args: argparse.Namespace) -> int:
    src = Path(args.input)
    dst = Path(args.output) if args.output else src.with_suffix(src.suffix + ".rcodec")
    pixels = read_float_image(src)

    t0 = time.perf_counter()
    encoded = encode_pixels(args, pixels)
    t1 = time.perf_counter()

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(encoded)

    print(
        f"encoded {src} -> {dst} "
        f"raw={pixels.nbytes:,} bytes encoded={len(encoded):,} bytes "
        f"ratio={pixels.nbytes / len(encoded):.2f}x time={t1 - t0:.3f}s"
    )
    return 0


def decode_file(args: argparse.Namespace) -> int:
    src = Path(args.input)
    dst = Path(args.output) if args.output else src.with_suffix(".exr")
    encoded = src.read_bytes()

    t0 = time.perf_counter()
    pixels = radiance_codec.decode(encoded)
    t1 = time.perf_counter()

    write_float_image(dst, pixels)
    print(
        f"decoded {src} -> {dst} "
        f"shape={pixels.shape} raw={pixels.nbytes:,} bytes time={t1 - t0:.3f}s"
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    encode = subparsers.add_parser("encode", help="encode an image file into a .rcodec frame")
    encode.add_argument("input", help="input image readable by OpenImageIO")
    encode.add_argument("output", nargs="?", help="output .rcodec path")
    encode.add_argument(
        "--mode",
        choices=("lossless", "near", "router"),
        default="lossless",
        help="codec route to use",
    )
    encode.add_argument(
        "--preset",
        choices=("fast", "compact", "balanced", "quality", "max"),
        default="quality",
        help="lossless preset used with --mode lossless",
    )
    encode.add_argument("--effort", type=int, default=None, help="override codec effort")
    encode.add_argument("--low-bits", type=int, default=12, help="low mantissa bits for --mode near")
    encode.add_argument(
        "--policy",
        choices=[policy.name.lower() for policy in radiance_codec.NearLosslessPolicy],
        default="fixed",
        help="near-lossless mantissa policy for --mode near",
    )
    encode.set_defaults(func=encode_file)

    decode = subparsers.add_parser("decode", help="decode a .rcodec frame into an image file")
    decode.add_argument("input", help="input .rcodec frame")
    decode.add_argument("output", nargs="?", help="output image path, default: input with .exr suffix")
    decode.set_defaults(func=decode_file)

    args = parser.parse_args()
    if args.command == "encode":
        if args.effort is None:
            args.effort = None if args.mode == "lossless" else 11
        if args.mode == "near":
            args.policy = radiance_codec.NearLosslessPolicy[args.policy.upper()]
    return args


def main() -> int:
    try:
        args = parse_args()
        return args.func(args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
