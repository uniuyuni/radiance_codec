"""Report whether the current process can run PyTorch on Metal."""
from __future__ import annotations

import platform

import torch


def main() -> int:
    print(f"platform:      {platform.platform()}")
    print(f"macOS:         {platform.mac_ver()[0]}")
    print(f"torch:         {torch.__version__}")
    print(f"MPS built:     {torch.backends.mps.is_built()}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"MPS devices:   {torch.mps.device_count()}")
    if not torch.backends.mps.is_available():
        print(
            "\nMetal is not visible to this process. On an Apple Silicon Mac, "
            "run outside restricted sandboxes before changing PyTorch builds."
        )
        return 1
    tensor = torch.ones(4, device="mps") * 2
    print(f"MPS smoke:     OK ({tensor})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
