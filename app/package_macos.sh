#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
APP_DIR="$ROOT/app"
BUILD_DIR="$APP_DIR/build"
PACKAGE_DIR="$APP_DIR/radiance-codec-macos"
PREFIX="$ROOT/dist/radiance_codec"

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "error: run through pixi, e.g. pixi run bash app/package_macos.sh" >&2
  exit 1
fi

cmake -G Ninja -S "$ROOT/codec" -B "$ROOT/codec/build" -DCMAKE_BUILD_TYPE=Release
cmake --build "$ROOT/codec/build" --target radiance_codec
cmake --install "$ROOT/codec/build" --prefix "$PREFIX"

cmake -G Ninja -S "$APP_DIR" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$PREFIX;$CONDA_PREFIX"
cmake --build "$BUILD_DIR" --target radiance-codec

rm -rf "$PACKAGE_DIR"
mkdir -p "$PACKAGE_DIR/bin" "$PACKAGE_DIR/lib"
cp "$BUILD_DIR/radiance-codec" "$PACKAGE_DIR/bin/"

copy_lib() {
  local src="$1"
  cp -f "$src" "$PACKAGE_DIR/lib/"
}

copy_lib "$ROOT/codec/build/libradiance_codec.0.0.1.dylib"
copy_lib "$CONDA_PREFIX/lib/libOpenEXR.33.3.4.11.dylib"
copy_lib "$CONDA_PREFIX/lib/libOpenEXRCore.33.3.4.11.dylib"
copy_lib "$CONDA_PREFIX/lib/libOpenEXRUtil.33.3.4.11.dylib"
copy_lib "$CONDA_PREFIX/lib/libIlmThread.33.3.4.11.dylib"
copy_lib "$CONDA_PREFIX/lib/libIex.33.3.4.11.dylib"
copy_lib "$CONDA_PREFIX/lib/libImath.30.3.2.2.dylib"
copy_lib "$CONDA_PREFIX/lib/libzstd.1.5.7.dylib"
copy_lib "$CONDA_PREFIX/lib/libomp.dylib"
copy_lib "$CONDA_PREFIX/lib/libc++.1.0.dylib"
copy_lib "$CONDA_PREFIX/lib/libdeflate.0.dylib"
copy_lib "$CONDA_PREFIX/lib/libopenjph.0.27.3.dylib"

(
  cd "$PACKAGE_DIR/lib"
  ln -sf libradiance_codec.0.0.1.dylib libradiance_codec.0.dylib
  ln -sf libOpenEXR.33.3.4.11.dylib libOpenEXR.33.dylib
  ln -sf libOpenEXRCore.33.3.4.11.dylib libOpenEXRCore.33.dylib
  ln -sf libOpenEXRUtil.33.3.4.11.dylib libOpenEXRUtil.33.dylib
  ln -sf libIlmThread.33.3.4.11.dylib libIlmThread.33.dylib
  ln -sf libIex.33.3.4.11.dylib libIex.33.dylib
  ln -sf libImath.30.3.2.2.dylib libImath.30.dylib
  ln -sf libzstd.1.5.7.dylib libzstd.1.dylib
  ln -sf libc++.1.0.dylib libc++.1.dylib
  ln -sf libopenjph.0.27.3.dylib libopenjph.0.27.dylib
)

install_name_tool -add_rpath "@executable_path/../lib" "$PACKAGE_DIR/bin/radiance-codec" 2>/dev/null || true
install_name_tool -delete_rpath "$CONDA_PREFIX/lib" "$PACKAGE_DIR/bin/radiance-codec" 2>/dev/null || true
install_name_tool -delete_rpath "$PREFIX/lib" "$PACKAGE_DIR/bin/radiance-codec" 2>/dev/null || true
cp "$APP_DIR/THIRD_PARTY_NOTICES.txt" "$PACKAGE_DIR/"
cp "$APP_DIR/README.md" "$PACKAGE_DIR/"
cp "$ROOT/LICENSE" "$PACKAGE_DIR/"

echo "packaged: $PACKAGE_DIR"
"$PACKAGE_DIR/bin/radiance-codec" || true
