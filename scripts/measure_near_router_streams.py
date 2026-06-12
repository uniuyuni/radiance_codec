"""Measure near-lossless router stream budgets for C1 sizing work."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import OpenImageIO as oiio

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "codec" / "python"))

import radiance_codec as rc  # noqa: E402

DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

OUTER_HEADER_V3 = 24
ROUTER_HEADER_BYTES = 92

STREAM_LABELS = (
    "route_mask",
    "high_mask",
    "Y",
    "Co_low",
    "Cg_low",
    "Co_high",
    "Cg_high",
    "SLog_R",
    "SLog_G",
    "SLog_B",
    "dark_refine_mask",
    "dark_refine_G",
)

STREAM_METHODS = {
    0: "raw",
    1: "rans0",
    2: "rans1",
    3: "zstd",
    4: "symbol_rans",
    5: "mask_binary",
    6: "mask_tiled",
    7: "symbol_context_rans",
    8: "symbol_parity_context_rans",
}


def read_exr(path: Path) -> np.ndarray:
    inp = oiio.ImageInput.open(str(path))
    if inp is None:
        raise RuntimeError(f"can't open {path}: {oiio.geterror()}")
    spec = inp.spec()
    pixels = inp.read_image(format=oiio.FLOAT)
    inp.close()
    if pixels is None:
        raise RuntimeError(f"can't read {path}: {oiio.geterror()}")
    return np.ascontiguousarray(
        np.asarray(pixels, dtype=np.float32).reshape(
            spec.height, spec.width, spec.nchannels
        )
    )


def parse_streams(blob: bytes) -> list[dict[str, int | str | float]]:
    if len(blob) < OUTER_HEADER_V3 + ROUTER_HEADER_BYTES:
        raise RuntimeError("compressed frame too small")
    if blob[:4] != b"HDR0":
        raise RuntimeError("missing outer HDR0 magic")
    p = OUTER_HEADER_V3
    if blob[p:p + 4] != b"NLR1":
        raise RuntimeError("missing router NLR1 magic")
    p += ROUTER_HEADER_BYTES
    rows: list[dict[str, int | str | float]] = []
    for label in STREAM_LABELS:
        if p + 5 > len(blob):
            raise RuntimeError(f"truncated stream header for {label}")
        method = blob[p]
        size = int.from_bytes(blob[p + 1:p + 5], "little")
        p += 5
        if p + size > len(blob):
            raise RuntimeError(f"truncated stream payload for {label}")
        total = size + 5
        rows.append(
            {
                "stream": label,
                "method": STREAM_METHODS.get(method, f"method_{method}"),
                "payload_size": size,
                "framed_bytes": total,
            }
        )
        p += size
    total_stream_bytes = sum(int(row["framed_bytes"]) for row in rows)
    for row in rows:
        row["share_of_streams"] = (
            float(row["framed_bytes"]) / total_stream_bytes
            if total_stream_bytes
            else 0.0
        )
    return rows


def measure(path: Path) -> dict[str, object]:
    pixels = read_exr(path)
    raw_bytes = int(pixels.nbytes)

    t0 = time.perf_counter()
    blob = rc.encode_near_lossless_router_v1(pixels)
    t1 = time.perf_counter()
    decoded = rc.decode(blob)
    t2 = time.perf_counter()
    if not np.isfinite(decoded).all():
        raise RuntimeError(f"non-finite decode: {path.name}")

    _, report = rc.reconstruct_near_lossless_router_v1(pixels)

    streams = parse_streams(blob)
    signed_escape_bytes = sum(
        int(row["framed_bytes"])
        for row in streams
        if str(row["stream"]).startswith("SLog_")
    )
    y_bytes = next(int(row["framed_bytes"]) for row in streams if row["stream"] == "Y")
    high_mask_bytes = next(
        int(row["framed_bytes"]) for row in streams if row["stream"] == "high_mask"
    )
    route_mask_bytes = next(
        int(row["framed_bytes"]) for row in streams if row["stream"] == "route_mask"
    )

    return {
        "image": path.name,
        "shape": list(pixels.shape),
        "raw_bytes": raw_bytes,
        "encoded_bytes": len(blob),
        "ratio": raw_bytes / len(blob),
        "encode_s": t1 - t0,
        "decode_s": t2 - t1,
        "route_mask_rate": report["route_mask_rate"],
        "dark_mask_rate": report["dark_mask_rate"],
        "outlier_mask_rate": report["outlier_mask_rate"],
        "outlier_active": report["outlier_active"],
        "y_step": report["y_step"],
        "route_mask_bytes": route_mask_bytes,
        "high_mask_bytes": high_mask_bytes,
        "y_bytes": y_bytes,
        "signed_escape_bytes": signed_escape_bytes,
        "streams": streams,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--glob",
        action="append",
        default=[],
        help="Input glob under data/. Can be passed multiple times.",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_DIR / "near_router_c1_stream_budget.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    patterns = args.glob or ["sample_1920×1280.exr"]
    paths = sorted({path for pattern in patterns for path in DATA_DIR.glob(pattern)})
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        raise RuntimeError(f"no inputs matched: {patterns}")

    rows = []
    for path in paths:
        row = measure(path)
        rows.append(row)
        print(
            f"{path.name}: {row['encoded_bytes']:,} bytes "
            f"ratio={row['ratio']:.2f}x "
            f"Y={row['y_bytes']:,} "
            f"SLog={row['signed_escape_bytes']:,} "
            f"route={row['route_mask_rate']:.2%}"
        )

    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
