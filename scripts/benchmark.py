"""
Lossless compression benchmark for float32 HDR EXR images.

For each input EXR, this script:
  1. Loads pixels into a numpy float32 array (the "raw" reference)
  2. Stages canonical inputs:
       - <work>/raw.bin            : raw float bytes  (for zstd/xz)
       - <work>/uncompressed.exr   : EXR with compression=none
  3. Runs each compression method, measuring:
       - encode time (wall clock, median of N runs)
       - decode time (wall clock, median of N runs)
       - output size in bytes
       - byte-exact lossless verification
  4. Writes results as CSV + JSON.

Methods compared:
  exr_rle / zips / zip / piz : OpenEXR built-in lossless compressors
  jxl_e1 / e3 / e7 / e9      : JPEG XL lossless (modular), effort levels
  zstd_3 / 19 / 22           : zstd applied to raw float bytes
  xz_9                       : xz/LZMA applied to raw float bytes
"""
from __future__ import annotations

import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable

import numpy as np
import OpenImageIO as oiio
import zfpy
import blosc
from tqdm import tqdm

# Add codec/python to path so we can import our work-in-progress codec.
# Skip silently if the library hasn't been built yet — the rest of the
# bench still runs.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "codec" / "python"))
try:
    import radiance_codec as our_codec  # type: ignore
    HAVE_OUR_CODEC = True
except (ImportError, FileNotFoundError) as e:
    HAVE_OUR_CODEC = False
    our_codec = None
    print(f"[note] our_codec not loaded: {e}", file=sys.stderr)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

N_RUNS = 3  # median of this many timed runs (after one warm-up)


# ─────────────────────────────────────────────────────────────────────────
# EXR I/O via OpenImageIO
# ─────────────────────────────────────────────────────────────────────────

def read_exr(path: Path) -> tuple[np.ndarray, oiio.ImageSpec]:
    inp = oiio.ImageInput.open(str(path))
    if inp is None:
        raise RuntimeError(f"OIIO failed to open: {path}: {oiio.geterror()}")
    spec = inp.spec()
    pixels = inp.read_image(format=oiio.FLOAT)
    inp.close()
    if pixels is None:
        raise RuntimeError(f"OIIO failed to read pixels: {path}")
    arr = np.asarray(pixels, dtype=np.float32)
    # OIIO returns shape (H, W, C) for multi-channel
    if arr.ndim == 1:
        arr = arr.reshape(spec.height, spec.width, spec.nchannels)
    elif arr.ndim == 2 and spec.nchannels == 1:
        arr = arr.reshape(spec.height, spec.width, 1)
    return arr, spec


def write_exr(path: Path, arr: np.ndarray, spec: oiio.ImageSpec,
              compression: str) -> None:
    new_spec = oiio.ImageSpec(spec)
    new_spec.attribute("compression", compression)
    out = oiio.ImageOutput.create(str(path))
    if out is None:
        raise RuntimeError(f"OIIO can't create output: {path}: {oiio.geterror()}")
    if not out.open(str(path), new_spec):
        raise RuntimeError(f"OIIO open failed: {path}: {out.geterror()}")
    if not out.write_image(arr):
        raise RuntimeError(f"OIIO write failed: {path}: {out.geterror()}")
    out.close()


def bit_equal(a: np.ndarray, b: np.ndarray) -> bool:
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    return a.tobytes() == b.tobytes()


# ─────────────────────────────────────────────────────────────────────────
# Method interface
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class MethodResult:
    image: str
    method: str
    raw_bytes: int
    out_bytes: int
    encode_ms: float          # median over N_RUNS
    decode_ms: float
    encode_throughput_mbps: float  # raw MB/s
    decode_throughput_mbps: float
    ratio: float              # raw / compressed
    bpp: float                # bits per pixel
    lossless: bool

    def to_dict(self) -> dict:
        return asdict(self)


def time_fn(fn: Callable[[], None], n: int = N_RUNS) -> float:
    # one warm-up
    fn()
    times = []
    for _ in range(n):
        t0 = time.perf_counter_ns()
        fn()
        times.append((time.perf_counter_ns() - t0) / 1e6)  # ms
    return statistics.median(times)


# ─────────────────────────────────────────────────────────────────────────
# Methods
# ─────────────────────────────────────────────────────────────────────────

def make_exr_method(name: str, compression: str):
    """OpenEXR with given internal compression. Encode/decode via OIIO."""
    def run(arr: np.ndarray, spec: oiio.ImageSpec, work: Path):
        out_path = work / f"{name}.exr"

        def encode():
            if out_path.exists():
                out_path.unlink()
            write_exr(out_path, arr, spec, compression)

        decoded_holder = {}

        def decode():
            decoded, _ = read_exr(out_path)
            decoded_holder["arr"] = decoded

        encode_ms = time_fn(encode)
        decode_ms = time_fn(decode)
        out_bytes = out_path.stat().st_size
        ok = bit_equal(arr, decoded_holder["arr"])
        return out_bytes, encode_ms, decode_ms, ok

    return name, run


def make_jxl_method(name: str, effort: int, true_lossless: bool = True):
    """JPEG XL via cjxl/djxl CLI. Input: uncompressed.exr.

    Note: cjxl with `-d 0` on float32 EXR is *near-lossless* by default
    (tiny mantissa quantization, ~1e-5 max diff). True bit-exact lossless
    for float32 requires `--override_bitdepth=32`.
    """
    def run(arr: np.ndarray, spec: oiio.ImageSpec, work: Path):
        uncompressed = work / "uncompressed.exr"
        out_path = work / f"{name}.jxl"
        decoded_path = work / f"{name}.decoded.exr"

        cmd = ["cjxl", str(uncompressed), str(out_path),
               "-d", "0", "-e", str(effort), "-m", "1", "--quiet"]
        if true_lossless:
            cmd += ["--override_bitdepth=32"]

        def encode():
            if out_path.exists():
                out_path.unlink()
            subprocess.run(cmd, check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        def decode():
            if decoded_path.exists():
                decoded_path.unlink()
            subprocess.run(
                ["djxl", str(out_path), str(decoded_path), "--quiet"],
                check=True,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )

        encode_ms = time_fn(encode)
        decode_ms = time_fn(decode)
        out_bytes = out_path.stat().st_size
        decoded, _ = read_exr(decoded_path)
        ok = bit_equal(arr, decoded)
        return out_bytes, encode_ms, decode_ms, ok

    return name, run


def make_raw_compressor_method(name: str, encode_cmd: list[str],
                                decode_cmd: list[str], ext: str):
    """Generic compressor (zstd/xz) on raw float bytes."""
    def run(arr: np.ndarray, spec: oiio.ImageSpec, work: Path):
        raw_path = work / "raw.bin"
        out_path = work / f"{name}.{ext}"
        decoded_path = work / f"{name}.decoded.bin"

        def encode():
            if out_path.exists():
                out_path.unlink()
            with open(raw_path, "rb") as fin, open(out_path, "wb") as fout:
                subprocess.run(encode_cmd, stdin=fin, stdout=fout, check=True,
                               stderr=subprocess.DEVNULL)

        def decode():
            if decoded_path.exists():
                decoded_path.unlink()
            with open(out_path, "rb") as fin, open(decoded_path, "wb") as fout:
                subprocess.run(decode_cmd, stdin=fin, stdout=fout, check=True,
                               stderr=subprocess.DEVNULL)

        encode_ms = time_fn(encode)
        decode_ms = time_fn(decode)
        out_bytes = out_path.stat().st_size
        decoded_bytes = decoded_path.read_bytes()
        ok = decoded_bytes == arr.tobytes()
        return out_bytes, encode_ms, decode_ms, ok

    return name, run


def make_zfp_method(name: str):
    """ZFP (lossless reversible mode) via Python zfpy."""
    def run(arr: np.ndarray, spec: oiio.ImageSpec, work: Path):
        out_path = work / f"{name}.zfp"
        compressed_holder = {}
        decoded_holder = {}

        def encode():
            # default mode is reversible (lossless)
            buf = zfpy.compress_numpy(arr, write_header=True)
            out_path.write_bytes(buf)
            compressed_holder["buf"] = buf

        def decode():
            buf = out_path.read_bytes()
            decoded_holder["arr"] = zfpy.decompress_numpy(buf)

        encode_ms = time_fn(encode)
        decode_ms = time_fn(decode)
        out_bytes = out_path.stat().st_size
        ok = bit_equal(arr, decoded_holder["arr"])
        return out_bytes, encode_ms, decode_ms, ok

    return name, run


def make_blosc_method(name: str, cname: str, clevel: int, shuffle: int):
    """Blosc with chosen internal codec, level, and shuffle filter."""
    def run(arr: np.ndarray, spec: oiio.ImageSpec, work: Path):
        out_path = work / f"{name}.blosc"
        raw = arr.tobytes()
        decoded_holder = {}

        def encode():
            buf = blosc.compress(
                raw, typesize=4, cname=cname, clevel=clevel, shuffle=shuffle)
            out_path.write_bytes(buf)

        def decode():
            buf = out_path.read_bytes()
            decoded_holder["bytes"] = blosc.decompress(buf)

        encode_ms = time_fn(encode)
        decode_ms = time_fn(decode)
        out_bytes = out_path.stat().st_size
        ok = decoded_holder["bytes"] == raw
        return out_bytes, encode_ms, decode_ms, ok

    return name, run


def make_byteplane_method(name: str, encode_cmd: list[str],
                           decode_cmd: list[str], ext: str):
    """Split float32 into 4 byte planes, compress each with given tool,
    concatenate. Mimics SHUFFLE filter."""
    def run(arr: np.ndarray, spec: oiio.ImageSpec, work: Path):
        out_path = work / f"{name}.{ext}"
        raw = arr.tobytes()
        # Byte-plane split: little-endian float32
        # plane 0 = LSB mantissa, plane 3 = sign + high exp
        flat = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 4)
        planes = [flat[:, i].tobytes() for i in range(4)]
        decoded_holder = {}

        def encode():
            chunks = []
            for p in planes:
                proc = subprocess.run(encode_cmd, input=p,
                                       capture_output=True, check=True)
                chunks.append(proc.stdout)
            # Header: 4 x u32 lengths, then concatenated payloads
            header = b"".join(len(c).to_bytes(4, "little") for c in chunks)
            out_path.write_bytes(header + b"".join(chunks))

        def decode():
            data = out_path.read_bytes()
            lengths = [int.from_bytes(data[i*4:(i+1)*4], "little")
                       for i in range(4)]
            offset = 16
            recovered_planes = []
            for L in lengths:
                proc = subprocess.run(decode_cmd, input=data[offset:offset+L],
                                       capture_output=True, check=True)
                recovered_planes.append(np.frombuffer(proc.stdout,
                                                      dtype=np.uint8))
                offset += L
            interleaved = np.stack(recovered_planes, axis=1).tobytes()
            decoded_holder["bytes"] = interleaved

        encode_ms = time_fn(encode)
        decode_ms = time_fn(decode)
        out_bytes = out_path.stat().st_size
        ok = decoded_holder["bytes"] == raw
        return out_bytes, encode_ms, decode_ms, ok

    return name, run


METHODS = [
    # OpenEXR built-in lossless codecs
    make_exr_method("exr_none", "none"),
    make_exr_method("exr_rle",  "rle"),
    make_exr_method("exr_zips", "zips"),
    make_exr_method("exr_zip",  "zip"),
    make_exr_method("exr_piz",  "piz"),

    # JPEG XL — true bit-exact lossless (override_bitdepth=32)
    make_jxl_method("jxl_e3_ll", 3, true_lossless=True),
    make_jxl_method("jxl_e7_ll", 7, true_lossless=True),
    # JPEG XL — near-lossless (default for HDR float; ~1e-5 mantissa error)
    make_jxl_method("jxl_e3_nl", 3, true_lossless=False),
    make_jxl_method("jxl_e7_nl", 7, true_lossless=False),

    # General-purpose compressors on raw float bytes (CLI)
    make_raw_compressor_method(
        "zstd_3", ["zstd", "-3", "--no-progress", "-c"],
        ["zstd", "-d", "--no-progress", "-c"], "zst"),
    make_raw_compressor_method(
        "zstd_19", ["zstd", "-19", "--no-progress", "-c"],
        ["zstd", "-d", "--no-progress", "-c"], "zst"),
    make_raw_compressor_method(
        "zstd_22", ["zstd", "-22", "--ultra", "--no-progress", "-c"],
        ["zstd", "-d", "--no-progress", "-c"], "zst"),
    make_raw_compressor_method(
        "xz_9", ["xz", "-9", "-c"], ["xz", "-d", "-c"], "xz"),
    make_raw_compressor_method(
        "bzip2_9", ["bzip2", "-9", "-c"], ["bzip2", "-d", "-c"], "bz2"),
    make_raw_compressor_method(
        "lz4_9", ["lz4", "-9", "-c"], ["lz4", "-d", "-c"], "lz4"),

    # ZFP — float-specific compressor (HPC)
    make_zfp_method("zfp"),

    # Blosc — meta-compressor with internal SHUFFLE filter for floats
    make_blosc_method("blosc_zstd9_shuf",    "zstd",  9, blosc.SHUFFLE),
    make_blosc_method("blosc_zstd9_bitshuf", "zstd",  9, blosc.BITSHUFFLE),
    make_blosc_method("blosc_zstd9_noshuf",  "zstd",  9, blosc.NOSHUFFLE),
    make_blosc_method("blosc_lz4hc9_shuf",   "lz4hc", 9, blosc.SHUFFLE),
    make_blosc_method("blosc_lz4_shuf",      "lz4",   9, blosc.SHUFFLE),

    # Experimental: byte-plane split + zstd (mimics SHUFFLE via CLI)
    make_byteplane_method(
        "planes_zstd19",
        ["zstd", "-19", "--no-progress", "-c"],
        ["zstd", "-d", "--no-progress", "-c"], "zst"),
]


def make_our_codec_method(name: str, stages_value: int,
                           rans_mode: int = 1):
    """Our work-in-progress C++ codec. Stages bitmask selects active stages,
    rans_mode picks the rANS variant when StageRans is enabled."""
    def run(arr: np.ndarray, spec: oiio.ImageSpec, work: Path):
        out_path = work / f"{name}.hdrc"
        decoded_holder = {}

        def encode():
            buf = our_codec.encode(
                arr,
                stages=our_codec.Stage(stages_value),
                rans_mode=our_codec.RansMode(rans_mode),
            )
            out_path.write_bytes(buf)

        def decode():
            buf = out_path.read_bytes()
            decoded_holder["arr"] = our_codec.decode(buf)

        encode_ms = time_fn(encode)
        decode_ms = time_fn(decode)
        out_bytes = out_path.stat().st_size
        ok = bit_equal(arr, decoded_holder["arr"])
        return out_bytes, encode_ms, decode_ms, ok

    return name, run


if HAVE_OUR_CODEC:
    METHODS.append(make_our_codec_method("our_v0_passthrough", 0))
    # Phase 1 rANS variants. StageRans = 0x10.
    METHODS.append(make_our_codec_method(
        "our_v1a_rans_static", 0x10, rans_mode=0))
    METHODS.append(make_our_codec_method(
        "our_v1b_rans_order0", 0x10, rans_mode=1))
    METHODS.append(make_our_codec_method(
        "our_v1c_rans_order1", 0x10, rans_mode=2))
    # Phase 2 — bitshuffle + rANS combinations.
    # StageBitshuffle = 0x08, StageRans = 0x10, combined = 0x18.
    METHODS.append(make_our_codec_method(
        "our_v2a_bitshuf_order0", 0x18, rans_mode=1))
    METHODS.append(make_our_codec_method(
        "our_v2b_bitshuf_order1", 0x18, rans_mode=2))
    # Phase 4 — MED spatial predictor combined with later stages.
    # StageSpatialPredict = 0x04
    METHODS.append(make_our_codec_method(
        "our_v4a_pred_order0",        0x14, rans_mode=1))
    METHODS.append(make_our_codec_method(
        "our_v4b_pred_order1",        0x14, rans_mode=2))
    METHODS.append(make_our_codec_method(
        "our_v4c_pred_bitshuf_order0", 0x1C, rans_mode=1))
    METHODS.append(make_our_codec_method(
        "our_v4d_pred_bitshuf_order1", 0x1C, rans_mode=2))
    # Phase 3 — color transform (XOR-based RCT) added.
    # StageColorTransform = 0x01
    # Full pipeline: 0x01 | 0x04 | 0x08 | 0x10 = 0x1D
    METHODS.append(make_our_codec_method(
        "our_v3a_rct_order1",                0x11, rans_mode=2))
    METHODS.append(make_our_codec_method(
        "our_v3b_rct_bitshuf_order1",        0x19, rans_mode=2))
    METHODS.append(make_our_codec_method(
        "our_v3c_rct_pred_bitshuf_order1",   0x1D, rans_mode=2))
    # Phase 5 — self-contained structural float32 context codec.
    # StageStructuralContext = 0x20.
    METHODS.append(make_our_codec_method(
        "our_v5_structural_context", 0x20, rans_mode=1))


# ─────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────

def process_image(exr_path: Path) -> list[MethodResult]:
    print(f"\n=== {exr_path.name} ===")
    arr, spec = read_exr(exr_path)
    raw_size = arr.nbytes  # W * H * C * 4
    pixels = spec.width * spec.height
    print(f"  {spec.width}x{spec.height}x{spec.nchannels} float32  "
          f"raw={raw_size/1e6:.2f} MB  ({raw_size*8/pixels:.0f} bpp)")

    results = []
    with tempfile.TemporaryDirectory(prefix="hdrbench_") as tmp:
        work = Path(tmp)
        # Stage canonical inputs
        raw_path = work / "raw.bin"
        raw_path.write_bytes(arr.tobytes())
        write_exr(work / "uncompressed.exr", arr, spec, "none")

        for name, fn in METHODS:
            try:
                out_bytes, enc_ms, dec_ms, ok = fn(arr, spec, work)
            except subprocess.CalledProcessError as e:
                print(f"  {name:12s}  ERROR: subprocess failed")
                continue
            except Exception as e:
                print(f"  {name:12s}  ERROR: {e}")
                continue

            r = MethodResult(
                image=exr_path.name,
                method=name,
                raw_bytes=raw_size,
                out_bytes=out_bytes,
                encode_ms=enc_ms,
                decode_ms=dec_ms,
                encode_throughput_mbps=(raw_size / 1e6) / (enc_ms / 1000),
                decode_throughput_mbps=(raw_size / 1e6) / (dec_ms / 1000),
                ratio=raw_size / out_bytes,
                bpp=out_bytes * 8 / pixels,
                lossless=ok,
            )
            results.append(r)
            status = "✓" if ok else "✗LOSSY"
            print(f"  {name:12s}  {r.bpp:5.2f} bpp  ratio={r.ratio:5.2f}x  "
                  f"enc={enc_ms:7.1f} ms ({r.encode_throughput_mbps:6.1f} MB/s)  "
                  f"dec={dec_ms:7.1f} ms ({r.decode_throughput_mbps:6.1f} MB/s)  "
                  f"{status}")
    return results


def main() -> int:
    images = sorted(DATA_DIR.glob("*.exr"))
    if not images:
        print(f"No EXR files in {DATA_DIR}. Run `pixi run fetch` first.",
              file=sys.stderr)
        return 1

    all_results: list[dict] = []
    for img in tqdm(images, desc="images", position=0):
        for r in process_image(img):
            all_results.append(r.to_dict())

    out_json = RESULTS_DIR / "results.json"
    out_json.write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved JSON: {out_json}")

    # CSV
    import csv
    out_csv = RESULTS_DIR / "results.csv"
    if all_results:
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
            w.writeheader()
            for r in all_results:
                w.writerow(r)
        print(f"Saved CSV:  {out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
