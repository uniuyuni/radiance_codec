# radiance-codec macOS CLI package

This folder builds and packages a self-contained macOS command line app for
EXR input/output and `.rcodec` frames.

Build the package from the repository root:

```bash
pixi run bash app/package_macos.sh
```

The packaged CLI is written to:

```text
app/radiance-codec-macos/
  bin/radiance-codec
  lib/*.dylib
  THIRD_PARTY_NOTICES.txt
```

Usage:

```bash
app/radiance-codec-macos/bin/radiance-codec encode input.exr output.rcodec
app/radiance-codec-macos/bin/radiance-codec decode output.rcodec decoded.exr
app/radiance-codec-macos/bin/radiance-codec info output.rcodec
```

Encoding modes:

```bash
--mode lossless --preset fast|compact|balanced|quality|max
--mode near --low-bits 12
--mode router
```

The package bundles `libradiance_codec`, OpenEXR/Imath, zstd, libdeflate,
openjph, libomp, and the conda-forge libc++ used by the build. macOS system
frameworks are not bundled.
