// Structural float32 HDR stage using decoder-safe bit-plane contexts.
//
// This research stage is intentionally self-contained: it consumes raw
// interleaved float32 bytes and emits a compact container with a mode map plus
// one adaptive binary rANS payload. It is meant to validate the current
// structural-codec direction in C++ before optimizing for speed.

#pragma once

#include "pipeline.hpp"

namespace radiance_codec {

class StructuralContextStage final : public IStage {
public:
    explicit StructuralContextStage(std::uint8_t effort = 5) noexcept
        : effort_(effort) {}

    Status encode(std::span<const std::uint8_t> in,
                  const ImageMeta& meta,
                  std::vector<std::uint8_t>& out) noexcept override;
    Status decode(std::span<const std::uint8_t> in,
                  const ImageMeta& meta,
                  std::vector<std::uint8_t>& out) noexcept override;
    const char* name() const noexcept override { return "structural_context"; }

private:
    std::uint8_t effort_ = 5;
};

} // namespace radiance_codec
