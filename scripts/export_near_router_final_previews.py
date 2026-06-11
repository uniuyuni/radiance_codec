"""Export full-resolution near-lossless router preview PNGs.

Outputs use:
  outputs/previews/<label>/original/<source-name>.png
  outputs/previews/<label>/candidate/<source-name>.png
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import OpenImageIO as oiio

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_ROOT = ROOT / "outputs" / "previews"
sys.path.insert(0, str(ROOT / "codec" / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

import radiance_codec  # noqa: E402
from export_linear_index_preview_png import to_preview_u16, write_png  # noqa: E402


def read_exr(path: Path) -> np.ndarray:
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


def export_one(path: Path, output_dir: Path, white: float, gamma: float) -> dict[str, object]:
    pixels = read_exr(path)
    t0 = time.perf_counter()
    encoded = radiance_codec.encode_near_lossless_router_v1(pixels)
    t1 = time.perf_counter()
    decoded = radiance_codec.decode(encoded, pixels.shape)
    t2 = time.perf_counter()

    png_name = f"{path.name}.png"
    original_png = output_dir / "original" / png_name
    candidate_png = output_dir / "candidate" / png_name
    write_png(original_png, to_preview_u16(pixels, white, gamma))
    write_png(candidate_png, to_preview_u16(decoded, white, gamma))

    return {
        "source": str(path),
        "original_png": str(original_png),
        "candidate_png": str(candidate_png),
        "shape": list(pixels.shape),
        "raw_bytes": int(pixels.nbytes),
        "encoded_bytes": len(encoded),
        "ratio": float(pixels.nbytes / len(encoded)),
        "encode_s": t1 - t0,
        "decode_s": t2 - t1,
        "white": white,
        "gamma": gamma,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--glob",
        action="append",
        default=[],
        help="Input glob under data/. Can be passed multiple times.",
    )
    parser.add_argument(
        "--label",
        default="near_lossless_router_final_20260611",
        help="Directory name under outputs/previews/.",
    )
    parser.add_argument("--white", type=float, default=4.0)
    parser.add_argument("--gamma", type=float, default=2.2)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    patterns = args.glob or ["sample_*.EXR", "sample_*.exr"]
    paths = sorted({path for pattern in patterns for path in DATA_DIR.glob(pattern)})
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        raise RuntimeError(f"no inputs matched: {patterns}")

    output_dir = OUTPUT_ROOT / args.label
    rows = []
    for path in paths:
        row = export_one(path, output_dir, args.white, args.gamma)
        rows.append(row)
        print(
            f"{path.name}: {row['encoded_bytes']:,} bytes "
            f"ratio={row['ratio']:.2f}x "
            f"enc={row['encode_s']:.3f}s dec={row['decode_s']:.3f}s"
        )

    manifest = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "label": args.label,
        "rows": rows,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
