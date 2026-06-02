// Stage: bit-level transposition (BITSHUFFLE).
//
// Treats input as N items of typesize bytes each. Output is a bit-level
// transpose: for each bit position p (0 .. typesize*8 - 1), the output
// contains a contiguous stream of N bits — bit p of item 0, bit p of
// item 1, etc.
//
// For float32 (typesize=4) the 32 bit-planes tend to look like:
//   plane 31 (sign):                   ≈ all zeros (positive HDR data)
//   planes 23..30 (exponent):          slowly varying, low entropy
//   plane 22 (mantissa MSB):           moderate
//   planes 0..21 (mantissa low):       high entropy
// → downstream entropy coder gets very compressible high-bit streams.
//
// For typesize = 4 and item count divisible by 8, output size == input size.
// For ragged tails, this implementation processes blocks of 8 items and
// leaves any tail (< 8 items) untransposed at the END of the output.
//
// Header (in the stage's output payload):
//   [u32 LE n_items][u8 typesize][u8 tail_items]
//   [bitshuffled body: floor(n_items/8) * typesize * 8 bytes]
//   [raw tail: tail_items * typesize bytes]
//
// Future optimization: NEON / AVX SIMD blocked transposition (8x speedup).
// For Phase 2 we ship the scalar version and let the entropy coder
// downstream do most of the work.

#pragma once

#include "pipeline.hpp"

namespace radiance_codec {

class BitshuffleStage final : public IStage {
public:
    explicit BitshuffleStage(uint8_t typesize) noexcept
        : typesize_(typesize) {}

    Status encode(std::span<const std::uint8_t> in,
                  const ImageMeta& meta,
                  std::vector<std::uint8_t>& out) noexcept override;
    Status decode(std::span<const std::uint8_t> in,
                  const ImageMeta& meta,
                  std::vector<std::uint8_t>& out) noexcept override;
    const char* name() const noexcept override { return "bitshuffle"; }

private:
    uint8_t typesize_;
};

} // namespace radiance_codec
