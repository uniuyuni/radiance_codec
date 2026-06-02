"""Self-test for reversible V2 HDR hierarchy transforms."""
from __future__ import annotations

import numpy as np

from hdr_hierarchy import (
    lift_channels,
    make_pass_map,
    nearest_context_residuals,
    reconstruct_nearest_context,
    unlift_channels,
)


def main() -> int:
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 1 << 32, size=(37, 53, 4), dtype=np.uint32)
    bits[0, :4, 0] = np.array(
        [0x00000000, 0x80000000, 0x7F800000, 0x7FC01234],
        dtype=np.uint32,
    )

    for mode in ["none", "green_delta", "green_xor"]:
        lifted = lift_channels(bits, mode)
        restored = unlift_channels(lifted, mode)
        assert np.array_equal(bits, restored), f"lifting mismatch: {mode}"

    residuals, passes = nearest_context_residuals(bits, max_step=16)
    restored = reconstruct_nearest_context(residuals, max_step=16)
    assert np.array_equal(bits, restored), "hierarchy residual mismatch"

    pass_map, names = make_pass_map(37, 53, max_step=16)
    assert len(names) == len(passes)
    assert np.all(pass_map < len(passes))
    print(
        f"HDR hierarchy round-trip: OK "
        f"({bits.shape}, {len(passes)} spatial passes)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
