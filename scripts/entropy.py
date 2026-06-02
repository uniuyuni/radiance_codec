"""
Estimate information-theoretic lower bounds for each test image.

Computes for each EXR:
  H0(raw)          : order-0 byte entropy of raw float32 bytes
                     — lower bound for byte-level context-free coders
  H0(planes)       : sum of per-byte-plane entropies after SHUFFLE
                     — what `blosc + shuffle` aspires to
  H0(residual)     : entropy of (x[i] - x[i-1]) per-pixel float diff,
                     re-interpreted as bytes
                     — what a simple left-predictor + entropy coder can hit
  H0(planes_resid) : SHUFFLE on the per-pixel residual
                     — combined predictor + shuffle ideal

All values reported as bits per pixel (bpp) for float32 RGBA-equivalent.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import OpenImageIO as oiio
from tabulate import tabulate

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def h0_bytes(b: bytes | np.ndarray) -> float:
    """Order-0 Shannon entropy of a byte stream, in bits per byte."""
    if isinstance(b, bytes):
        arr = np.frombuffer(b, dtype=np.uint8)
    else:
        arr = b.view(np.uint8)
    counts = np.bincount(arr, minlength=256).astype(np.float64)
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log2(p)).sum())


def analyze(path: Path) -> dict:
    inp = oiio.ImageInput.open(str(path))
    spec = inp.spec()
    A = np.asarray(inp.read_image(format=oiio.FLOAT),
                   dtype=np.float32).reshape(
        spec.height, spec.width, spec.nchannels)
    inp.close()

    pixels = spec.height * spec.width
    raw = A.tobytes()
    raw_bytes_per_pixel = A.nbytes / pixels   # = nchannels * 4

    # 1) order-0 entropy of raw bytes
    h_raw = h0_bytes(raw)

    # 2) per-byte-plane: split little-endian float32 into 4 planes
    flat_u8 = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 4)
    plane_h = [h0_bytes(flat_u8[:, i].copy()) for i in range(4)]
    # avg over the 4 planes
    h_planes = sum(plane_h) / 4

    # 3) per-pixel residual: A[i] - A[i-1] along scanline (raster),
    #    then re-interpret as bytes for entropy
    flat = A.reshape(-1)
    resid = np.empty_like(flat)
    resid[0] = flat[0]
    resid[1:] = flat[1:] - flat[:-1]
    h_resid = h0_bytes(resid.tobytes())

    # 4) shuffle of the residual
    resid_u8 = np.frombuffer(resid.tobytes(), dtype=np.uint8).reshape(-1, 4)
    plane_h_resid = [h0_bytes(resid_u8[:, i].copy()) for i in range(4)]
    h_planes_resid = sum(plane_h_resid) / 4

    return {
        "image": path.name,
        "width": spec.width,
        "height": spec.height,
        "channels": spec.nchannels,
        "raw_bpp": raw_bytes_per_pixel * 8,
        "h_raw_bits_per_byte": h_raw,
        "h_raw_bpp": h_raw * raw_bytes_per_pixel,
        "h_planes_bits_per_byte": h_planes,
        "h_planes_bpp": h_planes * raw_bytes_per_pixel,
        "h_resid_bits_per_byte": h_resid,
        "h_resid_bpp": h_resid * raw_bytes_per_pixel,
        "h_planes_resid_bits_per_byte": h_planes_resid,
        "h_planes_resid_bpp": h_planes_resid * raw_bytes_per_pixel,
        "per_plane_h": plane_h,
        "per_plane_h_resid": plane_h_resid,
    }


def main() -> int:
    images = sorted(DATA_DIR.glob("*.exr"))
    if not images:
        print(f"No EXR files in {DATA_DIR}", file=sys.stderr)
        return 1

    all_results = []
    rows = []
    for path in images:
        r = analyze(path)
        all_results.append(r)
        rows.append([
            r["image"][:40],
            f"{r['raw_bpp']:.0f}",
            f"{r['h_raw_bpp']:.2f}",
            f"{r['h_planes_bpp']:.2f}",
            f"{r['h_resid_bpp']:.2f}",
            f"{r['h_planes_resid_bpp']:.2f}",
            f"{r['raw_bpp'] / r['h_planes_resid_bpp']:.2f}x",
        ])

    headers = ["image", "raw bpp", "H₀(raw)",
               "H₀(planes)", "H₀(resid)", "H₀(planes+resid)",
               "max ratio"]
    print("\n=== Order-0 entropy lower bounds (bits per pixel) ===\n")
    print(tabulate(rows, headers=headers, tablefmt="github"))
    print(f"\nAll values are upper bounds on what an order-0 coder")
    print(f"(no inter-symbol context) can achieve on the given representation.")
    print(f"A real coder with context can go LOWER than these numbers.")

    # Aggregate row
    keys_bpp = ["raw_bpp", "h_raw_bpp", "h_planes_bpp",
                "h_resid_bpp", "h_planes_resid_bpp"]
    means = {k: sum(r[k] for r in all_results) / len(all_results)
             for k in keys_bpp}
    print(f"\n=== Means across {len(all_results)} images (bpp) ===")
    print(f"  raw                  : {means['raw_bpp']:6.2f}")
    print(f"  H₀(raw)              : {means['h_raw_bpp']:6.2f}  "
          f"({means['raw_bpp']/means['h_raw_bpp']:.2f}x)")
    print(f"  H₀(planes)           : {means['h_planes_bpp']:6.2f}  "
          f"({means['raw_bpp']/means['h_planes_bpp']:.2f}x)")
    print(f"  H₀(residual)         : {means['h_resid_bpp']:6.2f}  "
          f"({means['raw_bpp']/means['h_resid_bpp']:.2f}x)")
    print(f"  H₀(planes+residual)  : {means['h_planes_resid_bpp']:6.2f}  "
          f"({means['raw_bpp']/means['h_planes_resid_bpp']:.2f}x)")

    (RESULTS_DIR / "entropy.json").write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved: results/entropy.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
