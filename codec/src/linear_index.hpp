// Dedicated near-lossless transform-index codec.
//
// This stage quantizes finite float32 pixels per channel into N-bit indices,
// codes predictor residuals with adaptive rANS, and decodes to the quantized
// float32 image, not the original bit-exact image.

#pragma once

#include "pipeline.hpp"

namespace radiance_codec {

class LinearIndexStage final : public IStage {
public:
    LinearIndexStage(
        std::uint8_t bits,
        std::uint8_t effort = 5,
        std::uint8_t policy = static_cast<std::uint8_t>(
            NearLosslessPolicy::LinearRange)) noexcept
        : bits_(bits), effort_(effort), policy_(policy) {}

    Status encode(std::span<const std::uint8_t> in,
                  const ImageMeta& meta,
                  std::vector<std::uint8_t>& out) noexcept override;
    Status decode(std::span<const std::uint8_t> in,
                  const ImageMeta& meta,
                  std::vector<std::uint8_t>& out) noexcept override;
    const char* name() const noexcept override { return "linear_index"; }

private:
    std::uint8_t bits_ = 7;
    std::uint8_t effort_ = 5;
    std::uint8_t policy_ = static_cast<std::uint8_t>(
        NearLosslessPolicy::LinearRange);
};

} // namespace radiance_codec
