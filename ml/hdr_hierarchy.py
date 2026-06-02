"""Reversible float32 field transforms and coarse-to-fine spatial passes."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FIELD_GROUPS = {
    "sign": (31,),
    "exponent": tuple(range(30, 22, -1)),
    "mantissa_hi": tuple(range(22, 14, -1)),
    "mantissa_mid": tuple(range(14, 6, -1)),
    "mantissa_lo": tuple(range(6, -1, -1)),
}


@dataclass(frozen=True)
class HierarchyPass:
    name: str
    step: int
    mask: np.ndarray


def float_to_bits(pixels: np.ndarray) -> np.ndarray:
    if pixels.dtype != np.float32:
        raise ValueError(f"expected float32 pixels, got {pixels.dtype}")
    return np.ascontiguousarray(pixels).view(np.uint32)


def bits_to_float(bits: np.ndarray) -> np.ndarray:
    if bits.dtype != np.uint32:
        raise ValueError(f"expected uint32 bits, got {bits.dtype}")
    return np.ascontiguousarray(bits).view(np.float32)


def lift_channels(bits: np.ndarray, mode: str) -> np.ndarray:
    """Apply a reversible modular transform to inter-channel uint32 values."""
    if bits.dtype != np.uint32 or bits.ndim != 3:
        raise ValueError("expected HxWxC uint32 bits")
    out = bits.copy()
    if mode == "none" or bits.shape[2] < 3:
        return out
    green = bits[..., 1]
    if mode == "green_delta":
        out[..., 0] = bits[..., 0] - green
        out[..., 2] = bits[..., 2] - green
        return out
    if mode == "green_xor":
        out[..., 0] = bits[..., 0] ^ green
        out[..., 2] = bits[..., 2] ^ green
        return out
    raise ValueError(f"unknown channel lifting mode: {mode}")


def unlift_channels(bits: np.ndarray, mode: str) -> np.ndarray:
    if bits.dtype != np.uint32 or bits.ndim != 3:
        raise ValueError("expected HxWxC uint32 bits")
    out = bits.copy()
    if mode == "none" or bits.shape[2] < 3:
        return out
    green = bits[..., 1]
    if mode == "green_delta":
        out[..., 0] = bits[..., 0] + green
        out[..., 2] = bits[..., 2] + green
        return out
    if mode == "green_xor":
        out[..., 0] = bits[..., 0] ^ green
        out[..., 2] = bits[..., 2] ^ green
        return out
    raise ValueError(f"unknown channel lifting mode: {mode}")


def make_hierarchy_passes(
    height: int,
    width: int,
    max_step: int = 16,
) -> list[HierarchyPass]:
    """Return anchor, edge, and center passes that cover every pixel once."""
    if max_step < 2 or max_step & (max_step - 1):
        raise ValueError("max_step must be a power of two >= 2")
    yy, xx = np.indices((height, width))
    claimed = np.zeros((height, width), dtype=bool)
    passes: list[HierarchyPass] = []

    def add(name: str, step: int, mask: np.ndarray) -> None:
        nonlocal claimed
        new_mask = mask & ~claimed
        if np.any(new_mask):
            passes.append(HierarchyPass(name=name, step=step, mask=new_mask))
            claimed |= new_mask

    add("anchors", max_step, (yy % max_step == 0) & (xx % max_step == 0))
    step = max_step
    while step >= 2:
        half = step // 2
        add(
            f"step{step}_horizontal",
            step,
            (yy % step == 0) & (xx % step == half),
        )
        add(
            f"step{step}_vertical",
            step,
            (yy % step == half) & (xx % step == 0),
        )
        add(
            f"step{step}_center",
            step,
            (yy % step == half) & (xx % step == half),
        )
        step //= 2
    if not np.all(claimed):
        raise RuntimeError("hierarchy passes did not cover every pixel")
    return passes


def make_pass_map(
    height: int,
    width: int,
    max_step: int = 16,
) -> tuple[np.ndarray, list[str]]:
    passes = make_hierarchy_passes(height, width, max_step)
    pass_map = np.full((height, width), 255, dtype=np.uint8)
    for index, hierarchy_pass in enumerate(passes):
        pass_map[hierarchy_pass.mask] = index
    if np.any(pass_map == 255):
        raise RuntimeError("invalid pass map")
    return pass_map, [hierarchy_pass.name for hierarchy_pass in passes]


def nearest_context_residuals(
    bits: np.ndarray,
    max_step: int = 16,
) -> tuple[np.ndarray, list[HierarchyPass]]:
    """XOR each value with the first available nearby decoded value."""
    if bits.dtype != np.uint32 or bits.ndim != 3:
        raise ValueError("expected HxWxC uint32 bits")
    height, width, _ = bits.shape
    residuals = np.empty_like(bits)
    known = np.zeros((height, width), dtype=bool)
    passes = make_hierarchy_passes(height, width, max_step)

    for hierarchy_pass in passes:
        mask = hierarchy_pass.mask
        prediction = _nearest_context_prediction(bits, known, hierarchy_pass)
        residuals[mask] = bits[mask] ^ prediction[mask]
        known |= mask
    return residuals, passes


def reconstruct_nearest_context(
    residuals: np.ndarray,
    max_step: int = 16,
) -> np.ndarray:
    """Invert nearest_context_residuals exactly."""
    if residuals.dtype != np.uint32 or residuals.ndim != 3:
        raise ValueError("expected HxWxC uint32 residuals")
    height, width, _ = residuals.shape
    bits = np.empty_like(residuals)
    known = np.zeros((height, width), dtype=bool)
    for hierarchy_pass in make_hierarchy_passes(height, width, max_step):
        mask = hierarchy_pass.mask
        prediction = _nearest_context_prediction(bits, known, hierarchy_pass)
        bits[mask] = residuals[mask] ^ prediction[mask]
        known |= mask
    return bits


def _nearest_context_prediction(
    bits: np.ndarray,
    known: np.ndarray,
    hierarchy_pass: HierarchyPass,
) -> np.ndarray:
    height, width, _ = bits.shape
    mask = hierarchy_pass.mask
    prediction = np.zeros_like(bits)
    assigned = np.zeros((height, width), dtype=bool)
    radius = max(1, hierarchy_pass.step // 2)
    offsets = [
        (0, -radius),
        (-radius, 0),
        (0, radius),
        (radius, 0),
        (-radius, -radius),
        (-radius, radius),
        (radius, -radius),
        (radius, radius),
    ]
    for dy, dx in offsets:
        src_y0 = max(0, -dy)
        src_y1 = min(height, height - dy)
        src_x0 = max(0, -dx)
        src_x1 = min(width, width - dx)
        dst_y0 = src_y0 + dy
        dst_y1 = src_y1 + dy
        dst_x0 = src_x0 + dx
        dst_x1 = src_x1 + dx
        available = (
            mask[dst_y0:dst_y1, dst_x0:dst_x1]
            & ~assigned[dst_y0:dst_y1, dst_x0:dst_x1]
            & known[src_y0:src_y1, src_x0:src_x1]
        )
        if not np.any(available):
            continue
        dst = prediction[dst_y0:dst_y1, dst_x0:dst_x1]
        src = bits[src_y0:src_y1, src_x0:src_x1]
        dst[available] = src[available]
        assigned[dst_y0:dst_y1, dst_x0:dst_x1] |= available
    return prediction
