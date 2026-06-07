from __future__ import annotations

import os
import platform
import shutil
import subprocess
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py as _build_py
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel


ROOT = Path(__file__).resolve().parent
CODEC_DIR = ROOT / "codec"
DEFAULT_BUILD_DIR = CODEC_DIR / "build"


class build_py(_build_py):
    """Build the native shared library and place it next to radiance_codec.py."""

    def run(self) -> None:
        build_dir = Path(os.environ.get("RADIANCE_CODEC_CMAKE_BUILD_DIR", DEFAULT_BUILD_DIR))
        if os.environ.get("RADIANCE_CODEC_SKIP_CMAKE_BUILD") != "1":
            build_type = os.environ.get("CMAKE_BUILD_TYPE", "Release")
            subprocess.check_call(
                [
                    "cmake",
                    "-G",
                    "Ninja",
                    "-S",
                    str(CODEC_DIR),
                    "-B",
                    str(build_dir),
                    f"-DCMAKE_BUILD_TYPE={build_type}",
                ],
                cwd=ROOT,
            )
            subprocess.check_call(
                ["cmake", "--build", str(build_dir), "--target", "radiance_codec"],
                cwd=ROOT,
            )

        super().run()

        lib = self._find_built_library(build_dir)
        target = Path(self.build_lib) / lib.name
        shutil.copy2(lib, target)

    @staticmethod
    def _find_built_library(build_dir: Path) -> Path:
        names = (
            "libradiance_codec.dylib",
            "libradiance_codec.so",
            "radiance_codec.dll",
        )
        for name in names:
            candidate = build_dir / name
            if candidate.exists():
                return candidate
        for pattern in ("libradiance_codec.*.dylib", "libradiance_codec.so.*"):
            matches = sorted(build_dir.glob(pattern))
            if matches:
                return matches[-1]
        raise FileNotFoundError(f"built radiance_codec library not found in {build_dir}")


class bdist_wheel(_bdist_wheel):
    """The wheel contains a platform-specific shared library."""

    def finalize_options(self) -> None:
        super().finalize_options()
        self.root_is_pure = False

    def get_tag(self) -> tuple[str, str, str]:
        python, abi, plat = super().get_tag()
        machine = platform.machine().lower()
        if plat.endswith("_universal2") and machine in {"arm64", "x86_64"}:
            plat = plat[: -len("_universal2")] + f"_{machine}"
        return python, abi, plat


setup(cmdclass={"build_py": build_py, "bdist_wheel": bdist_wheel})
