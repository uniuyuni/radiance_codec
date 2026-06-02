// Stage: reversible color transform (RCT) for RGB(A) data.
//
// XOR-based to remain bit-exact regardless of float arithmetic:
//   encode: (R, G, B, [A]) → (R XOR G, G, B XOR G, [A])
//   decode: (R', G, B', [A]) → (R' XOR G, G, B' XOR G, [A])
//
// Rationale: in natural HDR images the three channels are strongly
// correlated (especially in gray-ish areas). XORing R and B against G
// yields bit patterns that are mostly zero where R ≈ G ≈ B. Subsequent
// spatial prediction + bitshuffle + entropy coding then exploit those
// flat planes.
//
// Channel layout passthrough: alpha (channel 3, if present) is left
// untouched. For 1-channel and 2-channel images the stage is a no-op.

#pragma once

#include "pipeline.hpp"

namespace radiance_codec {

class ColorTransformStage final : public IStage {
public:
    Status encode(std::span<const std::uint8_t> in,
                  const ImageMeta& meta,
                  std::vector<std::uint8_t>& out) noexcept override;
    Status decode(std::span<const std::uint8_t> in,
                  const ImageMeta& meta,
                  std::vector<std::uint8_t>& out) noexcept override;
    const char* name() const noexcept override { return "color_transform"; }
};

} // namespace radiance_codec
