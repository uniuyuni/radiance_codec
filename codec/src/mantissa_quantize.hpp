#pragma once

#include "pipeline.hpp"

namespace radiance_codec {

class MantissaQuantizeStage final : public IStage {
public:
    explicit MantissaQuantizeStage(
        std::uint8_t low_bits,
        std::uint8_t policy) noexcept
        : low_bits_(low_bits > 23 ? 23 : low_bits),
          policy_(policy) {}

    Status encode(std::span<const std::uint8_t> in,
                  const ImageMeta& meta,
                  std::vector<std::uint8_t>& out) noexcept override;

    Status decode(std::span<const std::uint8_t> in,
                  const ImageMeta& meta,
                  std::vector<std::uint8_t>& out) noexcept override;

    const char* name() const noexcept override {
        return "mantissa_quantize";
    }

private:
    std::uint8_t low_bits_ = 0;
    std::uint8_t policy_ = 0;
};

} // namespace radiance_codec
