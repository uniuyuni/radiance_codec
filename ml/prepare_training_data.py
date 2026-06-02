"""
Stage all EXR images for the ML compression experiment.

Source                          → Destination                       Use
─────────────────────────────────────────────────────────────────────────
data/*.exr                      → ml/residuals/test/<name>.bin      held-out
ml/training_data/*.exr          → ml/residuals/train/<name>.bin     training

For each EXR:
  1. Read pixels via OIIO (float32 array, shape H x W x C)
  2. Run through our v4d pipeline stages = SpatialPredict | Bitshuffle
     to get residual bytes
  3. Strip the 27-byte framing header (HDR0 file header 21 + bitshuf 6)
     so the saved bin file contains ONLY the residual bytes that the
     ML model needs to predict

Train/test separation is by SOURCE DIRECTORY — no leakage.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import OpenImageIO as oiio

ROOT = Path(__file__).resolve().parent.parent
TEST_SRC  = ROOT / "data"
TRAIN_SRC = ROOT / "ml" / "training_data"
TEST_DST  = ROOT / "ml" / "residuals" / "test"
TRAIN_DST = ROOT / "ml" / "residuals" / "train"
TEST_DST.mkdir(parents=True, exist_ok=True)
TRAIN_DST.mkdir(parents=True, exist_ok=True)

# File framing header layout (kept in sync with codec.cpp / bitshuffle.cpp):
#   HDR0 file header     : 4 magic + 1 ver + 4 w + 4 h + 1 ch + 1 fmt
#                        + 4 stages + 1 rans_mode + 1 effort = 21 bytes
#   Bitshuffle stage hdr : 4 n_items + 1 typesize + 1 tail_items = 6 bytes
# Total prefix to strip when stages = SpatialPredict | Bitshuffle
HEADER_BYTES = 21 + 6

sys.path.insert(0, str(ROOT / "codec" / "python"))
import radiance_codec  # type: ignore


def read_exr(path: Path) -> tuple[np.ndarray, oiio.ImageSpec]:
    inp = oiio.ImageInput.open(str(path))
    if inp is None:
        raise RuntimeError(f"Can't open {path}: {oiio.geterror()}")
    spec = inp.spec()
    pix = inp.read_image(format=oiio.FLOAT)
    inp.close()
    arr = np.asarray(pix, dtype=np.float32).reshape(
        spec.height, spec.width, spec.nchannels)
    return arr, spec


def process_one(src: Path, dst_dir: Path) -> tuple[int, int]:
    """Returns (raw_bytes, residual_bytes_written)."""
    arr, spec = read_exr(src)
    raw_size = arr.nbytes
    compressed = radiance_codec.encode(
        arr,
        stages=radiance_codec.Stage.SPATIAL_PREDICT | radiance_codec.Stage.BITSHUFFLE,
        rans_mode=radiance_codec.RansMode.ORDER0,
    )
    # Strip the framing prefix so the bin file is pure residual bytes.
    if len(compressed) < HEADER_BYTES:
        raise RuntimeError(f"compressed too small for {src}")
    residual = compressed[HEADER_BYTES:]
    if len(residual) != raw_size:
        # Sanity: predict+bitshuf are size-preserving (modulo header).
        # For typesize=4 with n_items % 8 == 0 this should always hold.
        print(f"  warning: residual size {len(residual)} != raw {raw_size} "
              f"for {src.name}")
    dst = dst_dir / f"{src.stem}.bin"
    dst.write_bytes(residual)
    return raw_size, len(residual)


def process_dir(src_dir: Path, dst_dir: Path, label: str) -> None:
    files = sorted(src_dir.glob("*.exr"))
    if not files:
        print(f"[{label}] no EXR files in {src_dir}")
        return
    print(f"\n[{label}] {len(files)} files → {dst_dir}")
    total_raw = 0
    total_res = 0
    n_ok = 0
    n_fail = 0
    for p in files:
        try:
            raw, res = process_one(p, dst_dir)
        except Exception as e:
            n_fail += 1
            print(f"  SKIP {p.name}: {e}")
            continue
        total_raw += raw
        total_res += res
        n_ok += 1
        # Progress: terse for training corpus, verbose for test
        if label == "test":
            print(f"  {p.name}: {raw} -> {res} bytes")
    print(f"  ok={n_ok}, fail={n_fail}, "
          f"total: {total_raw/1e6:.1f} MB raw, "
          f"{total_res/1e6:.1f} MB residual (saved)")


def main() -> int:
    process_dir(TEST_SRC,  TEST_DST,  "test")
    process_dir(TRAIN_SRC, TRAIN_DST, "train")

    test_total  = sum(p.stat().st_size for p in TEST_DST.glob("*.bin"))
    train_total = sum(p.stat().st_size for p in TRAIN_DST.glob("*.bin"))
    print(f"\nSummary:")
    print(f"  test/  : {len(list(TEST_DST.glob('*.bin'))):3d} files, "
          f"{test_total/1e6:6.1f} MB")
    print(f"  train/ : {len(list(TRAIN_DST.glob('*.bin'))):3d} files, "
          f"{train_total/1e6:6.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
