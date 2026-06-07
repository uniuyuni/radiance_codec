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
import os
from pathlib import Path

os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
os.environ.setdefault("KMP_BLOCKTIME", "0")

import numpy as np


# ─── locate the shared library ─────────────────────────────────────

def _find_lib() -> Path:
    module_dir = Path(__file__).resolve().parent
    env_path = os.environ.get("RADIANCE_CODEC_LIBRARY")
    candidates = [
        Path(env_path) if env_path else None,
        module_dir / "libradiance_codec.dylib",
        module_dir / "libradiance_codec.so",
        module_dir / "radiance_codec.dll",
        module_dir.parent / "build" / "libradiance_codec.dylib",
        module_dir.parent / "build" / "libradiance_codec.so",
    ]
    for p in candidates:
        if p is not None and p.exists():
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
        ("near_lossless_policy", ctypes.c_uint8),
    ]


class RansMode(enum.IntEnum):
    STATIC  = 0
    ORDER0  = 1
    ORDER1  = 2


class NearLosslessPolicy(enum.IntEnum):
    FIXED = 0
    TILE = 1
    EXPONENT = 2
    TILE_EXPONENT = 3
    LINEAR_RANGE = 4
    LOG_RANGE = 5
    SQRT_RANGE = 6
    GAMMA075_RANGE = 7
    GAMMA025_RANGE = 8
    ASINH_RANGE = 9


class LinearIndexPreset(enum.Enum):
    RATIO = "ratio"
    BALANCED = "balanced"
    QUALITY = "quality"
    QUALITY_PLUS = "quality_plus"


class LosslessPreset(enum.Enum):
    FAST = "fast"
    BALANCED = "balanced"
    QUALITY = "quality"
    MAX = "max"


class _Buffer(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.POINTER(ctypes.c_uint8)),
        ("size", ctypes.c_size_t),
    ]


class _NearLosslessRouterParams(ctypes.Structure):
    _fields_ = [
        ("y_bits", ctypes.c_uint8),
        ("chroma_low_bits", ctypes.c_uint8),
        ("high_bits", ctypes.c_uint8),
        ("anchor_bits", ctypes.c_uint8),
        ("low_scale", ctypes.c_uint8),
        ("guide_radius", ctypes.c_uint8),
        ("guide_eps", ctypes.c_float),
        ("threshold_mult", ctypes.c_float),
        ("dark_max", ctypes.c_float),
        ("mask_radius", ctypes.c_uint8),
        ("smooth_threshold", ctypes.c_float),
        ("target_y_step", ctypes.c_float),
        ("outlier_activation_ratio", ctypes.c_float),
        ("visual_guard_enabled", ctypes.c_uint8),
        ("visual_guard_dilate_radius", ctypes.c_uint8),
        ("visual_guard_luma_threshold", ctypes.c_float),
        ("visual_guard_rgb_threshold", ctypes.c_float),
        ("visual_guard_white", ctypes.c_float),
        ("visual_guard_gamma", ctypes.c_float),
    ]


class _NearLosslessRouterReport(ctypes.Structure):
    _fields_ = [
        ("route_mask_rate", ctypes.c_float),
        ("dark_mask_rate", ctypes.c_float),
        ("outlier_mask_rate", ctypes.c_float),
        ("outlier_active", ctypes.c_uint8),
        ("chosen_percentile", ctypes.c_float),
        ("threshold_maxrgb", ctypes.c_float),
        ("y_step", ctypes.c_float),
        ("max_over_p99", ctypes.c_float),
        ("p99_over_p97", ctypes.c_float),
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

_lib.radiance_codec_near_lossless_router_v1_reconstruct.restype = ctypes.c_int
_lib.radiance_codec_near_lossless_router_v1_reconstruct.argtypes = [
    ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t,
    ctypes.POINTER(_Meta),
    ctypes.POINTER(_NearLosslessRouterParams),
    ctypes.POINTER(_Buffer),
    ctypes.POINTER(_NearLosslessRouterReport),
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
    LINEAR_INDEX    = 0x0100
    NEAR_LOSSLESS_ROUTER = 0x0200
    BYTEPLANE_RANS = 0x0400
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
           near_lossless_bits: int = 0,
           near_lossless_policy: NearLosslessPolicy | int = NearLosslessPolicy.FIXED) -> bytes:
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
    policy = NearLosslessPolicy(near_lossless_policy)
    cfg = _Config(stages=int(stages), effort=effort,
                  rans_mode=int(rans_mode),
                  near_lossless_bits=near_lossless_bits,
                  near_lossless_policy=int(policy))
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
                  near_lossless_bits=0,
                  near_lossless_policy=int(NearLosslessPolicy.FIXED))
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


def lossless_effort_for_preset(preset: LosslessPreset | str) -> int:
    """Return the effort value used by a named exact-lossless preset."""
    preset = LosslessPreset(preset)
    return {
        LosslessPreset.FAST: 5,
        LosslessPreset.BALANCED: 10,
        LosslessPreset.QUALITY: 11,
        LosslessPreset.MAX: 12,
    }[preset]


def lossless_stage_for_preset(preset: LosslessPreset | str) -> tuple[Stage, RansMode]:
    """Return the stage and rANS mode used by a named exact-lossless preset."""
    preset = LosslessPreset(preset)
    if preset is LosslessPreset.FAST:
        return Stage.BYTEPLANE_RANS, RansMode.ORDER0
    return Stage.GROUPED_DELTA, RansMode.ORDER0


def encode_lossless(pixels: np.ndarray,
                    preset: LosslessPreset | str = LosslessPreset.QUALITY,
                    effort: int | None = None) -> bytes:
    """Encode with a named exact-lossless route.

    ``preset="fast"`` uses the chunked byteplane-rANS route for fast full-image
    coding. Other presets use the grouped-delta route with increasing search.
    """
    stages, rans_mode = lossless_stage_for_preset(preset)
    return encode(
        pixels,
        stages=stages,
        effort=lossless_effort_for_preset(preset) if effort is None else effort,
        rans_mode=rans_mode,
    )


def encode_near_lossless(pixels: np.ndarray,
                         low_bits: int,
                         effort: int = 11,
                         stages: Stage = Stage.GROUPED_DELTA,
                         policy: NearLosslessPolicy | int = NearLosslessPolicy.FIXED) -> bytes:
    """Encode with low mantissa bits zeroed before compression.

    Decoding returns the quantized image, not the original bit-exact image.
    """
    return encode(
        pixels,
        stages=Stage.MANTISSA_QUANTIZE | stages,
        effort=effort,
        near_lossless_bits=low_bits,
        near_lossless_policy=policy,
    )


def encode_near_lossless_router_v1(pixels: np.ndarray,
                                   effort: int = 11) -> bytes:
    """Encode with the visual near-lossless router v1.

    Decoding returns the router-reconstructed candidate, not the original
    bit-exact image. The current production path wraps that candidate in the
    existing grouped-delta entropy stage.
    """
    return encode(
        pixels,
        stages=Stage.NEAR_LOSSLESS_ROUTER,
        effort=effort,
        near_lossless_bits=0,
        near_lossless_policy=NearLosslessPolicy.FIXED,
    )


_LINEAR_INDEX_TRANSFORM_POLICIES = {
    "linear": NearLosslessPolicy.LINEAR_RANGE,
    "signed-log": NearLosslessPolicy.LOG_RANGE,
    "sqrt": NearLosslessPolicy.SQRT_RANGE,
    "gamma075": NearLosslessPolicy.GAMMA075_RANGE,
    "gamma025": NearLosslessPolicy.GAMMA025_RANGE,
    "asinh": NearLosslessPolicy.ASINH_RANGE,
}

_LINEAR_INDEX_TRANSFORM_ALIASES = {
    "lin": "linear",
    "signedlog": "signed-log",
    "log": "signed-log",
    "log1p": "signed-log",
    "squareroot": "sqrt",
    "gamma-075": "gamma075",
    "gamma0.75": "gamma075",
    "gamma-025": "gamma025",
    "gamma0.25": "gamma025",
}


def _linear_index_transform_key(transform: str) -> str:
    key = transform.strip().lower().replace("_", "-")
    key = _LINEAR_INDEX_TRANSFORM_ALIASES.get(key, key)
    if key not in _LINEAR_INDEX_TRANSFORM_POLICIES:
        raise CodecError(f"unknown linear index transform: {transform}")
    return key


def _linear_index_policy_for_transform(transform: str) -> NearLosslessPolicy:
    return _LINEAR_INDEX_TRANSFORM_POLICIES[
        _linear_index_transform_key(transform)]


def _linear_index_transform_values(values: np.ndarray, key: str) -> np.ndarray:
    if key == "linear":
        return values
    if key == "signed-log":
        return np.sign(values) * np.log2(1.0 + np.abs(values))
    if key == "sqrt":
        return np.sign(values) * np.sqrt(np.abs(values))
    if key == "gamma075":
        return np.sign(values) * np.power(np.abs(values), 0.75)
    if key == "gamma025":
        return np.sign(values) * np.power(np.abs(values), 0.25)
    if key == "asinh":
        return np.arcsinh(values)
    raise CodecError(f"unknown linear index transform: {key}")


def _linear_index_inverse_transform_values(values: np.ndarray, key: str) -> np.ndarray:
    if key == "linear":
        return values
    if key == "signed-log":
        return np.sign(values) * (np.exp2(np.abs(values)) - 1.0)
    if key == "sqrt":
        return np.sign(values) * np.power(np.abs(values), 2.0)
    if key == "gamma075":
        return np.sign(values) * np.power(np.abs(values), 1.0 / 0.75)
    if key == "gamma025":
        return np.sign(values) * np.power(np.abs(values), 4.0)
    if key == "asinh":
        return np.sinh(values)
    raise CodecError(f"unknown linear index transform: {key}")


def encode_linear_index_near_lossless(pixels: np.ndarray,
                                      bits: int = 7,
                                      effort: int = 9,
                                      transform: str = "linear") -> bytes:
    """Encode with the dedicated transform-index near-lossless route.

    Decoding returns the per-channel quantized image. ``transform="linear"``
    stores uniform value indices. ``transform="signed-log"`` stores uniform
    signed-log2(1+abs(x)) indices, which is useful for quality-first HDR photos.
    """
    if bits <= 0 or bits > 15:
        raise CodecError(f"linear index bits must be in 1..15, got {bits}")
    return encode(
        pixels,
        stages=Stage.LINEAR_INDEX,
        effort=effort,
        near_lossless_bits=bits,
        near_lossless_policy=_linear_index_policy_for_transform(transform),
    )


def reconstruct_near_lossless_router_v1(
    pixels: np.ndarray,
    *,
    y_bits: int = 8,
    chroma_low_bits: int = 8,
    high_bits: int = 5,
    anchor_bits: int = 10,
    low_scale: int = 2,
    guide_radius: int = 2,
    guide_eps: float = 0.1,
    threshold_mult: float = 2.5,
    dark_max: float = 0.5,
    mask_radius: int = 2,
    smooth_threshold: float = 0.0025,
    target_y_step: float = 0.0032,
    outlier_activation_ratio: float = 4.0,
    visual_guard_enabled: bool = True,
    visual_guard_dilate_radius: int = 0,
    visual_guard_luma_threshold: float = 0.010,
    visual_guard_rgb_threshold: float = 0.0,
    visual_guard_white: float = 4.0,
    visual_guard_gamma: float = 2.2,
) -> tuple[np.ndarray, dict[str, float | bool]]:
    """Reconstruct the current research near-lossless router v1 candidate.

    This is a C++ acceleration path for the VST/YCoCg + signed-log10 escape
    research route. It is not yet the final compressed bitstream API.
    """
    if pixels.dtype != np.float32:
        raise CodecError(f"expected float32, got {pixels.dtype}")
    if pixels.ndim != 3:
        raise CodecError(f"expected HxWxC array, got shape {pixels.shape}")
    h, w, c = pixels.shape
    if c < 3 or c > 4:
        raise CodecError(f"channels must be 3 or 4, got {c}")
    arr = np.ascontiguousarray(pixels)
    raw = arr.tobytes()
    meta = _Meta(width=w, height=h, channels=c, format=1)
    params = _NearLosslessRouterParams(
        y_bits=y_bits,
        chroma_low_bits=chroma_low_bits,
        high_bits=high_bits,
        anchor_bits=anchor_bits,
        low_scale=low_scale,
        guide_radius=guide_radius,
        guide_eps=guide_eps,
        threshold_mult=threshold_mult,
        dark_max=dark_max,
        mask_radius=mask_radius,
        smooth_threshold=smooth_threshold,
        target_y_step=target_y_step,
        outlier_activation_ratio=outlier_activation_ratio,
        visual_guard_enabled=1 if visual_guard_enabled else 0,
        visual_guard_dilate_radius=visual_guard_dilate_radius,
        visual_guard_luma_threshold=visual_guard_luma_threshold,
        visual_guard_rgb_threshold=visual_guard_rgb_threshold,
        visual_guard_white=visual_guard_white,
        visual_guard_gamma=visual_guard_gamma,
    )
    out = _Buffer()
    report = _NearLosslessRouterReport()
    in_ptr = (ctypes.c_uint8 * len(raw)).from_buffer_copy(raw)
    rc = _lib.radiance_codec_near_lossless_router_v1_reconstruct(
        in_ptr,
        len(raw),
        ctypes.byref(meta),
        ctypes.byref(params),
        ctypes.byref(out),
        ctypes.byref(report),
    )
    if rc != 0:
        raise CodecError(f"near-lossless router reconstruct failed: {_STATUS_NAMES.get(rc, rc)}")
    try:
        raw_out = bytes(ctypes.string_at(out.data, out.size))
    finally:
        _lib.radiance_codec_buffer_free(ctypes.byref(out))
    decoded = np.frombuffer(raw_out, dtype=np.float32).reshape(arr.shape).copy()
    return decoded, {
        "route_mask_rate": float(report.route_mask_rate),
        "dark_mask_rate": float(report.dark_mask_rate),
        "outlier_mask_rate": float(report.outlier_mask_rate),
        "outlier_active": bool(report.outlier_active),
        "chosen_percentile": float(report.chosen_percentile),
        "threshold_maxrgb": float(report.threshold_maxrgb),
        "y_step": float(report.y_step),
        "max_over_p99": float(report.max_over_p99),
        "p99_over_p97": float(report.p99_over_p97),
    }


def linear_index_bits_for_preset(preset: LinearIndexPreset | str) -> int:
    """Return the current research bit-depth for a linear-index preset."""
    preset = LinearIndexPreset(preset)
    return {
        LinearIndexPreset.RATIO: 7,
        LinearIndexPreset.BALANCED: 8,
        LinearIndexPreset.QUALITY: 9,
        LinearIndexPreset.QUALITY_PLUS: 10,
    }[preset]


def encode_linear_index_preset(pixels: np.ndarray,
                               preset: LinearIndexPreset | str = LinearIndexPreset.QUALITY,
                               effort: int = 9,
                               transform: str = "linear") -> bytes:
    """Encode with a named linear-index quality preset.

    Current research mapping:
    ratio -> bits7, balanced -> bits8, quality -> bits9, quality_plus -> bits10.
    """
    return encode_linear_index_near_lossless(
        pixels,
        bits=linear_index_bits_for_preset(preset),
        effort=effort,
        transform=transform,
    )


def _tile_random_bits(bits: np.ndarray, cap_bits: int) -> int:
    cap_bits = min(23, max(0, int(cap_bits)))
    if cap_bits == 0:
        return 0
    exponent = (bits >> np.uint32(23)) & np.uint32(0xff)
    finite = exponent != np.uint32(0xff)
    selected = bits[finite]
    if selected.size < 64:
        return 0
    width = 0
    for bit in range(cap_bits):
        ones = int(((selected >> np.uint32(bit)) & np.uint32(1)).sum())
        diff = abs(2 * ones - int(selected.size))
        if diff <= int(selected.size) // 10:
            width = bit + 1
        else:
            break
    return width


def _exponent_policy_bits(bits: np.ndarray, base_bits: int) -> np.ndarray:
    exponent = ((bits >> np.uint32(23)) & np.uint32(0xff)).astype(np.int16)
    finite = exponent != 0xff
    max_exp = int(exponent[finite].max()) if bool(np.any(finite)) else 0
    delta = np.maximum(0, max_exp - exponent)
    out = np.minimum(23, int(base_bits) + (delta // 2)).astype(np.uint8)
    out[~finite] = 0
    return out


def _clear_low_bits(bits: np.ndarray, low_bits: np.ndarray | int) -> np.ndarray:
    out = bits.copy()
    exponent = (out >> np.uint32(23)) & np.uint32(0xff)
    finite = exponent != np.uint32(0xff)
    if isinstance(low_bits, np.ndarray):
        for value in range(1, 24):
            selected = finite & (low_bits == value)
            if bool(np.any(selected)):
                mask = np.uint32(0xff800000 if value >= 23
                                 else (0xffffffff ^ ((1 << value) - 1)))
                out[selected] &= mask
        return out
    if low_bits:
        mask = np.uint32(0xff800000 if low_bits >= 23
                         else (0xffffffff ^ ((1 << int(low_bits)) - 1)))
        out[finite] &= mask
    return out


def _quantize_linear_range(bits: np.ndarray, value_bits: int) -> np.ndarray:
    out = bits.copy()
    if value_bits <= 0:
        return out
    levels = (1 << min(23, int(value_bits))) - 1
    values = bits.view(np.float32)
    if values.ndim == 2:
        work = values.reshape(values.shape[0], values.shape[1], 1)
        out_work = out.reshape(values.shape[0], values.shape[1], 1)
    else:
        work = values
        out_work = out
    for c in range(work.shape[2]):
        channel = work[:, :, c]
        finite = np.isfinite(channel)
        if not bool(np.any(finite)):
            continue
        mn = float(channel[finite].min())
        mx = float(channel[finite].max())
        if not mx > mn:
            continue
        q = np.floor((channel.astype(np.float64) - mn) / (mx - mn) * levels + 0.5)
        q = np.clip(q, 0, levels)
        rec = mn + q * (mx - mn) / levels
        quantized = rec.astype(np.float32).view(np.uint32)
        out_work[:, :, c][finite] = quantized[finite]
    return out


def _quantize_log_range(bits: np.ndarray, value_bits: int) -> np.ndarray:
    out = bits.copy()
    if value_bits <= 0:
        return out
    levels = (1 << min(23, int(value_bits))) - 1
    values = bits.view(np.float32)
    if values.ndim == 2:
        work = values.reshape(values.shape[0], values.shape[1], 1)
        out_work = out.reshape(values.shape[0], values.shape[1], 1)
    else:
        work = values
        out_work = out
    eps = 1e-8
    for c in range(work.shape[2]):
        channel = work[:, :, c]
        finite = np.isfinite(channel) & (channel >= 0.0)
        if not bool(np.any(finite)):
            continue
        log_values = np.log2(channel[finite].astype(np.float64) + eps)
        mn = float(log_values.min())
        mx = float(log_values.max())
        if not mx > mn:
            continue
        log_channel = np.zeros(channel.shape, dtype=np.float64)
        log_channel[finite] = np.log2(channel[finite].astype(np.float64) + eps)
        q = np.floor((log_channel - mn) / (mx - mn) * levels + 0.5)
        q = np.clip(q, 0, levels)
        rec = np.maximum(0.0, np.exp2(mn + q * (mx - mn) / levels) - eps)
        quantized = rec.astype(np.float32).view(np.uint32)
        out_work[:, :, c][finite] = quantized[finite]
    return out


def quantize_linear_index(pixels: np.ndarray,
                          bits: int = 7,
                          tile_size: int = 0,
                          transform: str = "linear") -> np.ndarray:
    """Return the image reconstructed by the dedicated transform-index route."""
    if pixels.dtype != np.float32:
        raise CodecError(f"expected float32, got {pixels.dtype}")
    if bits <= 0 or bits > 15:
        raise CodecError(f"linear index bits must be in 1..15, got {bits}")
    if not bool(np.all(np.isfinite(pixels))):
        raise CodecError("linear index route currently requires finite pixels")
    transform_key = _linear_index_transform_key(transform)
    levels = (1 << int(bits)) - 1
    work = np.ascontiguousarray(pixels)
    if work.ndim == 2:
        view = work.reshape(work.shape[0], work.shape[1], 1)
    else:
        view = work
    out = np.array(view, copy=True)
    del tile_size
    for c in range(view.shape[2]):
        channel = view[:, :, c].astype(np.float64)
        transformed = _linear_index_transform_values(channel, transform_key)
        range_values = transformed.astype(np.float32).astype(np.float64)
        lo = float(np.min(range_values))
        hi = float(np.max(range_values))
        target = out[:, :, c]
        if not hi > lo:
            target[:, :] = np.float32(
                _linear_index_inverse_transform_values(
                    np.asarray(lo, dtype=np.float64),
                    transform_key))
            continue
        q = np.floor((transformed - lo) / (hi - lo) * levels + 0.5)
        q = np.clip(q, 0, levels)
        rec_t = lo + q * (hi - lo) / levels
        rec = _linear_index_inverse_transform_values(rec_t, transform_key)
        target[:, :] = rec.astype(np.float32)
    if work.ndim == 2:
        return np.ascontiguousarray(out.reshape(work.shape))
    return np.ascontiguousarray(out.reshape(work.shape))


def quantize_mantissa(
    pixels: np.ndarray,
    low_bits: int,
    policy: NearLosslessPolicy | int = NearLosslessPolicy.FIXED,
) -> np.ndarray:
    """Return the image produced by the near-lossless mantissa quantizer."""
    if pixels.dtype != np.float32:
        raise CodecError(f"expected float32, got {pixels.dtype}")
    if low_bits < 0 or low_bits > 23:
        raise CodecError(f"low_bits must be in 0..23, got {low_bits}")
    policy = NearLosslessPolicy(policy)
    bits = np.ascontiguousarray(pixels).view(np.uint32)
    if low_bits == 0:
        return bits.copy().view(np.float32).reshape(pixels.shape)
    if policy == NearLosslessPolicy.LINEAR_RANGE:
        return _quantize_linear_range(bits, low_bits).view(np.float32).reshape(pixels.shape)
    if policy == NearLosslessPolicy.LOG_RANGE:
        return _quantize_log_range(bits, low_bits).view(np.float32).reshape(pixels.shape)
    if policy == NearLosslessPolicy.FIXED:
        return _clear_low_bits(bits, low_bits).view(np.float32).reshape(pixels.shape)

    flat_shape = bits.shape
    if pixels.ndim == 2:
        work = bits.reshape(bits.shape[0], bits.shape[1], 1)
    else:
        work = bits
    out = work.copy()
    exp_bits = _exponent_policy_bits(work, low_bits)
    tile_size = 128
    for y0 in range(0, work.shape[0], tile_size):
        y1 = min(work.shape[0], y0 + tile_size)
        for x0 in range(0, work.shape[1], tile_size):
            x1 = min(work.shape[1], x0 + tile_size)
            tile = work[y0:y1, x0:x1, :]
            if policy == NearLosslessPolicy.EXPONENT:
                clear = exp_bits[y0:y1, x0:x1, :]
            else:
                cap = low_bits if policy == NearLosslessPolicy.TILE else 23
                tile_bits = _tile_random_bits(tile, cap)
                if policy == NearLosslessPolicy.TILE:
                    clear = tile_bits
                else:
                    clear = np.minimum(tile_bits, exp_bits[y0:y1, x0:x1, :])
            out[y0:y1, x0:x1, :] = _clear_low_bits(tile, clear)
    return out.reshape(flat_shape).view(np.float32).reshape(pixels.shape)


def inspect_header(compressed: bytes) -> dict[str, int | str]:
    """Parse the small outer codec header for tests and research probes."""
    if len(compressed) < 21 or compressed[:4] != b"HDR0":
        raise CodecError("not a radiance_codec frame")
    p = 4
    version = compressed[p]; p += 1
    width = int.from_bytes(compressed[p:p + 4], "little"); p += 4
    height = int.from_bytes(compressed[p:p + 4], "little"); p += 4
    channels = compressed[p]; p += 1
    fmt = compressed[p]; p += 1
    stages = int.from_bytes(compressed[p:p + 4], "little"); p += 4
    rans_mode = compressed[p]; p += 1
    effort = compressed[p]; p += 1
    near_bits = compressed[p] if version >= 2 else 0
    if version >= 2:
        p += 1
    policy = compressed[p] if version >= 3 else 0
    sign_class = compressed[p + 1] if version >= 3 else 0
    sign_names = {0: "mixed", 1: "all_positive", 2: "all_negative"}
    policy_names = {
        0: "fixed",
        1: "tile",
        2: "exponent",
        3: "tile_exponent",
        4: "linear_range",
        5: "log_range",
    }
    return {
        "version": version,
        "width": width,
        "height": height,
        "channels": channels,
        "format": fmt,
        "stages": stages,
        "rans_mode": rans_mode,
        "effort": effort,
        "near_lossless_bits": near_bits,
        "near_lossless_policy": policy,
        "near_lossless_policy_name": policy_names.get(policy, "unknown"),
        "sign_class": sign_class,
        "sign_class_name": sign_names.get(sign_class, "unknown"),
    }


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
