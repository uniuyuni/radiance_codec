"""Run the C++ self-contained research codec stages on EXR inputs."""
from __future__ import annotations

import argparse
import fnmatch
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import OpenImageIO as oiio

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "codec" / "python"))

import radiance_codec  # noqa: E402

DATA_DIR = ROOT / "data"
RESULTS_JSON = ROOT / "results" / "results.json"

CORPUS_PATTERNS = {
    "custom": (),
    "all": ("*.exr",),
    "target-no-noise": ("*.exr",),
    "realistic-core": ("ph_*.exr", "oexr_ScanLines_*.exr"),
    "realistic-no-puresky": ("ph_*.exr", "oexr_ScanLines_*.exr"),
    "highres-sample": ("sample_*.exr",),
    "puresky-hard": ("ph_*puresky*.exr",),
    "easy": ("synth_gradient_1k.exr", "oexr_TestImages_GrayRampsHorizontal.exr"),
    "synthetic": ("synth_*.exr",),
    "noise-stress": ("synth_noise_1k.exr",),
}

CORPUS_EXCLUDES = {
    "target-no-noise": ("synth_noise_1k.exr",),
    "realistic-no-puresky": ("ph_*puresky*.exr",),
}


def read_exr(path: Path) -> np.ndarray:
    image_input = oiio.ImageInput.open(str(path))
    if image_input is None:
        raise RuntimeError(f"can't open {path}: {oiio.geterror()}")
    spec = image_input.spec()
    pixels = image_input.read_image(format=oiio.FLOAT)
    image_input.close()
    if pixels is None:
        raise RuntimeError(f"can't read {path}: {oiio.geterror()}")
    return np.asarray(pixels, dtype=np.float32).reshape(
        spec.height, spec.width, spec.nchannels
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", default="ph_studio_small_03_1k.exr")
    parser.add_argument(
        "--corpus",
        choices=sorted(CORPUS_PATTERNS),
        default="custom",
        help="Named corpus preset. Use custom to select inputs with --glob.",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Exclude matching filename/stem glob. Can be passed multiple times.",
    )
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--effort", type=int, default=5)
    parser.add_argument(
        "--stage",
        choices=("structural_context", "grouped_delta"),
        default="structural_context",
    )
    return parser.parse_args()


def selected_paths(args: argparse.Namespace) -> list[Path]:
    if args.corpus == "custom":
        paths = sorted(DATA_DIR.glob(args.glob))
    else:
        paths = sorted(
            {
                path
                for pattern in CORPUS_PATTERNS[args.corpus]
                for path in DATA_DIR.glob(pattern)
            }
        )

    excludes = list(CORPUS_EXCLUDES.get(args.corpus, ())) + args.exclude
    if excludes:
        paths = [
            path
            for path in paths
            if not any(
                fnmatch.fnmatch(path.name, pattern)
                or fnmatch.fnmatch(path.stem, pattern)
                for pattern in excludes
            )
        ]
    return paths


def geomean(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def load_blosc_ratios() -> dict[str, float]:
    if not RESULTS_JSON.exists():
        return {}
    rows = json.loads(RESULTS_JSON.read_text())
    return {
        Path(row["image"]).stem: float(row["ratio"])
        for row in rows
        if row["method"] == "blosc_zstd9_bitshuf"
    }


def main() -> int:
    args = parse_args()
    paths = selected_paths(args)
    if not paths:
        raise RuntimeError(
            f"no images matched corpus={args.corpus!r} glob={args.glob!r}"
        )

    blosc = load_blosc_ratios()
    stage = {
        "structural_context": radiance_codec.Stage.STRUCTURAL_CONTEXT,
        "grouped_delta": radiance_codec.Stage.GROUPED_DELTA,
    }[args.stage]
    results = []
    print(radiance_codec.version())
    selection = f"glob={args.glob}" if args.corpus == "custom" else "preset"
    print(f"corpus={args.corpus} {selection} crop={args.crop_size} effort={args.effort}")
    print(
        f"{'image':43s} {'shape':>16s} {'ratio':>8s} "
        f"{'Blosc':>8s} {'enc':>8s} {'dec':>8s}"
    )
    print("-" * 101)
    for path in paths:
        pixels = read_exr(path)
        if args.crop_size:
            pixels = pixels[:args.crop_size, :args.crop_size]
        pixels = np.ascontiguousarray(pixels, dtype=np.float32)
        t0 = time.perf_counter()
        encoded = radiance_codec.encode(
            pixels,
            stages=stage,
            effort=args.effort,
        )
        t1 = time.perf_counter()
        decoded = radiance_codec.decode(encoded, pixels.shape)
        t2 = time.perf_counter()
        ok = decoded.tobytes() == pixels.tobytes()
        if not ok:
            raise RuntimeError(f"roundtrip mismatch: {path.name}")
        ratio = pixels.nbytes / len(encoded)
        blosc_ratio = blosc.get(path.stem)
        results.append({"ratio": ratio, "blosc_ratio": blosc_ratio})
        blosc_text = f"{blosc_ratio:7.2f}x" if blosc_ratio is not None else "      -"
        print(
            f"{path.stem:43s} {str(tuple(pixels.shape)):>16s} "
            f"{ratio:7.2f}x {blosc_text} {t1 - t0:7.2f}s {t2 - t1:7.2f}s"
        )
    if results:
        print("-" * 101)
        print(f"geomean {args.stage}: {geomean([row['ratio'] for row in results]):.3f}x")
        blosc_values = [
            row["blosc_ratio"]
            for row in results
            if row["blosc_ratio"] is not None
        ]
        if blosc_values:
            print(f"geomean Blosc:      {geomean(blosc_values):.3f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
