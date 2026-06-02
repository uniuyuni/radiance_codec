"""
Generate synthetic float32 EXR images that bound the realistic range:

  synth_gradient_1k.exr : smooth 2D gradient (extremely compressible)
  synth_noise_1k.exr    : pure Gaussian noise (incompressible reference)
  synth_mixed_1k.exr    : gradient + small noise (realistic-ish low entropy)
  synth_rgba_hdr_1k.exr : 4-channel RGBA HDR scene
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import OpenImageIO as oiio

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

W, H = 1024, 512


def write_exr(path: Path, arr: np.ndarray, channels: list[str]) -> None:
    spec = oiio.ImageSpec(arr.shape[1], arr.shape[0], arr.shape[2], oiio.FLOAT)
    spec.channelnames = channels
    spec.attribute("compression", "none")
    out = oiio.ImageOutput.create(str(path))
    if out is None:
        raise RuntimeError(f"OIIO can't create: {path}")
    out.open(str(path), spec)
    out.write_image(arr.astype(np.float32))
    out.close()


def make_gradient() -> np.ndarray:
    yy, xx = np.meshgrid(np.linspace(0, 1, H), np.linspace(0, 1, W),
                          indexing="ij")
    r = xx
    g = yy
    b = (xx + yy) / 2
    return np.stack([r, g, b], axis=-1).astype(np.float32) * 10.0  # HDR range


def make_noise() -> np.ndarray:
    rng = np.random.default_rng(seed=42)
    arr = rng.standard_normal((H, W, 3)).astype(np.float32) * 5.0 + 10.0
    return arr


def make_mixed() -> np.ndarray:
    grad = make_gradient()
    rng = np.random.default_rng(seed=43)
    noise = rng.standard_normal(grad.shape).astype(np.float32) * 0.05
    return grad + noise


def make_rgba_hdr() -> np.ndarray:
    """Synthetic 4-channel HDR scene with bright sun region + sky gradient."""
    rng = np.random.default_rng(seed=44)
    yy, xx = np.meshgrid(np.linspace(0, 1, H), np.linspace(0, 1, W),
                          indexing="ij")
    # Sky gradient (top dark, bottom bright) + sun spot
    sky_r = 0.5 + 0.5 * yy
    sky_g = 0.6 + 0.4 * yy
    sky_b = 1.5 - 0.5 * yy
    # Sun (very bright spot)
    cx, cy = 0.7, 0.3
    d2 = (xx - cx) ** 2 + (yy - cy) ** 2
    sun = 1000.0 * np.exp(-d2 / 0.001)
    r = sky_r + sun
    g = sky_g + sun * 0.95
    b = sky_b + sun * 0.85
    a = np.ones_like(r)  # alpha
    # Add small natural noise
    out = np.stack([r, g, b, a], axis=-1).astype(np.float32)
    out[..., :3] += rng.standard_normal(out[..., :3].shape).astype(
        np.float32) * 0.02
    return out


def main() -> int:
    print(f"Generating synthetic test images in {DATA_DIR}")
    write_exr(DATA_DIR / "synth_gradient_1k.exr",
              make_gradient(), ["R", "G", "B"])
    print("  synth_gradient_1k.exr   (smooth gradient, low entropy)")
    write_exr(DATA_DIR / "synth_noise_1k.exr",
              make_noise(), ["R", "G", "B"])
    print("  synth_noise_1k.exr      (Gaussian noise, near-incompressible)")
    write_exr(DATA_DIR / "synth_mixed_1k.exr",
              make_mixed(), ["R", "G", "B"])
    print("  synth_mixed_1k.exr      (gradient + small noise)")
    write_exr(DATA_DIR / "synth_rgba_hdr_1k.exr",
              make_rgba_hdr(), ["R", "G", "B", "A"])
    print("  synth_rgba_hdr_1k.exr   (4-channel HDR sky + sun)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
