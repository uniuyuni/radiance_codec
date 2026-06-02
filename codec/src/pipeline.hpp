// Pipeline coordinator: owns the ordered list of stages,
// applies them in encode/decode direction.

#pragma once

#include "radiance_codec/codec.hpp"

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace radiance_codec {

// Most stages are reversible. The near-lossless quantizer is intentionally
// encode-only and leaves decode as a passthrough, producing quantized pixels.
// Stages receive arbitrary byte streams and produce arbitrary byte streams;
// they may use ImageMeta to interpret structure.
class IStage {
public:
    virtual ~IStage() = default;
    virtual Status encode(std::span<const std::uint8_t> in,
                          const ImageMeta& meta,
                          std::vector<std::uint8_t>& out) noexcept = 0;
    virtual Status decode(std::span<const std::uint8_t> in,
                          const ImageMeta& meta,
                          std::vector<std::uint8_t>& out) noexcept = 0;
    virtual const char* name() const noexcept = 0;
};

// Build the active pipeline from a config bitmask. Order matters:
// encode runs front-to-back, decode runs back-to-front (inverse stages).
std::vector<std::unique_ptr<IStage>> build_pipeline(
    const PipelineConfig& config);

} // namespace radiance_codec
