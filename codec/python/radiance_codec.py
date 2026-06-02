"""
Python ctypes bindings for libradiance_codec.

Loads the library built by `pixi run build` and exposes encode/decode
operating on numpy arrays. Designed for the benchmark harness, so it
favors simplicity over performance (one allocation per call is fine).

Stage bits mirror radiance_codec/codec.hpp. Pass combinations to enable
individual transforms for ablation studies. The grouped-delta path is the
current strongest exact-lossless backend; adding MANTISSA_QUANTIZE enables the
separate near-lossless low-mantissa mode.
"""
from __future__ import annotations

import ctypes
import enum
from pathlib import Path

import numpy as np


# ─── locate the shared library ─────────────────────────────────────

def _find_lib() -> Path:
    candidates = [
        Path(__file__).resolve().parent.parent / "build" / "libradiance_codec.dylib",
        Path(__file__).resolve().parent.parent / "build" / "libradiance_codec.so",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "libradiance_codec not found. Run `pixi run build` first. "
        f"Looked at: {[str(p) for p in candidates]}"
    )


_lib = ctypes.CDLL(str(_find_lib()))


# ─── C structs ────────────────────────────────────────────────────

class _Meta(ctypes.Structure):
    _fields_ = [
        ("width",    ctypes.c_uint32),
        ("height",   ctypes.c_uint32),
        ("channels", ctypes.c_uint8),
        ("format",   ctypes.c_uint8),
        ("_pad",     ctypes.c_uint8 * 2),
    ]


class _Config(ctypes.Structure):
    _fields_ = [
        ("stages",    ctypes.c_uint32),
        ("effort",    ctypes.c_uint8),
        ("rans_mode", ctypes.c_uint8),
        ("near_lossless_bits", ctypes.c_uint8),
        ("_pad",      ctypes.c_uint8 * 1),
    ]


class RansMode(enum.IntEnum):
    STATIC  = 0
    ORDER0  = 1
    ORDER1  = 2


class _Buffer(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.POINTER(ctypes.c_uint8)),
        ("size", ctypes.c_size_t),
    ]


# ─── function signatures ──────────────────────────────────────────

_lib.radiance_codec_version.restype = ctypes.c_char_p
_lib.radiance_codec_version.argtypes = []

_lib.radiance_codec_buffer_free.restype = None
_lib.radiance_codec_buffer_free.argtypes = [ctypes.POINTER(_Buffer)]

_lib.radiance_codec_encode.restype = ctypes.c_int
_lib.radiance_codec_encode.argtypes = [
    ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t,
    ctypes.POINTER(_Meta),
    ctypes.POINTER(_Config),
    ctypes.POINTER(_Buffer),
]

_lib.radiance_codec_decode.restype = ctypes.c_int
_lib.radiance_codec_decode.argtypes = [
    ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t,
    ctypes.POINTER(_Meta),
    ctypes.POINTER(_Config),
    ctypes.POINTER(_Buffer),
]


# ─── public API ───────────────────────────────────────────────────

class Stage(enum.IntFlag):
    NONE            = 0x0000
    COLOR_TRANSFORM = 0x0001
    LOG_MAGNITUDE   = 0x0002
    SPATIAL_PREDICT = 0x0004
    BITSHUFFLE      = 0x0008
    RANS            = 0x0010
    STRUCTURAL_CONTEXT = 0x0020
    GROUPED_DELTA   = 0x0040
    MANTISSA_QUANTIZE = 0x0080
    ALL = (COLOR_TRANSFORM | LOG_MAGNITUDE | SPATIAL_PREDICT
           | BITSHUFFLE | RANS)


class CodecError(RuntimeError):
    pass


_STATUS_NAMES = {
    -1: "INVALID_ARG",
    -2: "UNSUPPORTED_FORMAT",
    -3: "UNIMPLEMENTED_STAGE",
    -4: "DECOMPRESS_FAILED",
    -5: "SIZE_MISMATCH",
}


def version() -> str:
    return _lib.radiance_codec_version().decode("utf-8")


def encode(pixels: np.ndarray,
           stages: Stage = Stage.NONE,
           effort: int = 5,
           rans_mode: RansMode = RansMode.ORDER0,
           near_lossless_bits: int = 0) -> bytes:
    """Encode a float32 image array of shape (H, W, C) or (H, W).

    Returns the compressed byte string (including the framing header).
    """
    if pixels.dtype != np.float32:
        raise CodecError(f"expected float32, got {pixels.dtype}")
    if pixels.ndim == 2:
        h, w = pixels.shape
        c = 1
    elif pixels.ndim == 3:
        h, w, c = pixels.shape
    else:
        raise CodecError(f"expected 2D or 3D array, got shape {pixels.shape}")
    if c < 1 or c > 4:
        raise CodecError(f"channels must be 1..4, got {c}")

    arr = np.ascontiguousarray(pixels)
    raw = arr.tobytes()

    meta = _Meta(width=w, height=h, channels=c, format=1)
    if near_lossless_bits < 0 or near_lossless_bits > 23:
        raise CodecError(
            f"near_lossless_bits must be in 0..23, got {near_lossless_bits}")
    cfg = _Config(stages=int(stages), effort=effort,
                  rans_mode=int(rans_mode),
                  near_lossless_bits=near_lossless_bits)
    out = _Buffer()

    in_ptr = (ctypes.c_uint8 * len(raw)).from_buffer_copy(raw)
    rc = _lib.radiance_codec_encode(in_ptr, len(raw),
                                    ctypes.byref(meta), ctypes.byref(cfg),
                                    ctypes.byref(out))
    if rc != 0:
        raise CodecError(f"encode failed: {_STATUS_NAMES.get(rc, rc)}")
    try:
        return bytes(ctypes.string_at(out.data, out.size))
    finally:
        _lib.radiance_codec_buffer_free(ctypes.byref(out))


def decode(compressed: bytes,
           shape: tuple[int, ...],
           stages: Stage = Stage.NONE,
           effort: int = 5,
           rans_mode: RansMode = RansMode.ORDER0) -> np.ndarray:
    """Decode bytes produced by encode. shape is (H, W, C) or (H, W).

    Note: the codec stores the stage/config choices in the file header, so the
    explicit kwargs here are ignored on decode (kept for symmetry).
    """
    if len(shape) == 2:
        h, w = shape
        c = 1
    elif len(shape) == 3:
        h, w, c = shape
    else:
        raise CodecError(f"shape must be 2D or 3D, got {shape}")

    meta = _Meta(width=w, height=h, channels=c, format=1)
    cfg = _Config(stages=int(stages), effort=effort,
                  rans_mode=int(rans_mode),
                  near_lossless_bits=0)
    out = _Buffer()

    in_ptr = (ctypes.c_uint8 * len(compressed)).from_buffer_copy(compressed)
    rc = _lib.radiance_codec_decode(in_ptr, len(compressed),
                                    ctypes.byref(meta), ctypes.byref(cfg),
                                    ctypes.byref(out))
    if rc != 0:
        raise CodecError(f"decode failed: {_STATUS_NAMES.get(rc, rc)}")
    try:
        raw = bytes(ctypes.string_at(out.data, out.size))
    finally:
        _lib.radiance_codec_buffer_free(ctypes.byref(out))

    arr = np.frombuffer(raw, dtype=np.float32)
    return arr.reshape(shape)


def encode_near_lossless(pixels: np.ndarray,
                         low_bits: int,
                         effort: int = 11,
                         stages: Stage = Stage.GROUPED_DELTA) -> bytes:
    """Encode with low mantissa bits zeroed before compression.

    Decoding returns the quantized image, not the original bit-exact image.
    """
    return encode(
        pixels,
        stages=Stage.MANTISSA_QUANTIZE | stages,
        effort=effort,
        near_lossless_bits=low_bits,
    )


def quantize_mantissa(pixels: np.ndarray, low_bits: int) -> np.ndarray:
    """Return the image produced by the near-lossless mantissa quantizer."""
    if pixels.dtype != np.float32:
        raise CodecError(f"expected float32, got {pixels.dtype}")
    if low_bits < 0 or low_bits > 23:
        raise CodecError(f"low_bits must be in 0..23, got {low_bits}")
    bits = np.ascontiguousarray(pixels).view(np.uint32).copy()
    if low_bits:
        exponent = (bits >> np.uint32(23)) & np.uint32(0xff)
        finite = exponent != np.uint32(0xff)
        keep_mask = np.uint32(0xffffffff ^ ((1 << low_bits) - 1))
        bits[finite] &= keep_mask
    return bits.view(np.float32).reshape(pixels.shape)


# ─── self-test when run as a script ───────────────────────────────

if __name__ == "__main__":
    print(version())
    rng = np.random.default_rng(0)
    A = rng.standard_normal((32, 64, 3)).astype(np.float32) * 10.0
    enc = encode(A)
    dec = decode(enc, A.shape)
    assert A.tobytes() == dec.tobytes(), "round-trip mismatch"
    print(f"encode: {A.nbytes} -> {len(enc)} bytes "
          f"(ratio {A.nbytes / len(enc):.3f}x)")
    print("ROUND-TRIP OK")
