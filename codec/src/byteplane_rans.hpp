// Self-contained exact-lossless float32 codec stage.
//
// The stage splits raw float32 bytes into four byteplanes per chunk and codes
// each stream independently. Independent streams give us coarse-grained
// parallelism while preserving a compact order-0 model per plane.

#pragma once

#include "pipeline.hpp"

#include <cstdint>

namespace radiance_codec {

class ByteplaneRansStage final : public IStage {
public:
    explicit ByteplaneRansStage(std::uint8_t effort = 5) noexcept
        : effort_(effort) {}

    Status encode(std::span<const std::uint8_t> in,
                  const ImageMeta& meta,
                  std::vector<std::uint8_t>& out) noexcept override;
    Status decode(std::span<const std::uint8_t> in,
                  const ImageMeta& meta,
                  std::vector<std::uint8_t>& out) noexcept override;
    const char* name() const noexcept override { return "byteplane_rans"; }

private:
    std::uint8_t effort_;
};

} // namespace radiance_codec
