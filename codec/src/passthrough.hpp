// Stage 0: passthrough — copies input to output unchanged.
// Used as a sanity check for the pipeline scaffold.

#pragma once

#include "pipeline.hpp"

namespace radiance_codec {

class PassthroughStage final : public IStage {
public:
    Status encode(std::span<const std::uint8_t> in,
                  const ImageMeta& meta,
                  std::vector<std::uint8_t>& out) noexcept override;
    Status decode(std::span<const std::uint8_t> in,
                  const ImageMeta& meta,
                  std::vector<std::uint8_t>& out) noexcept override;
    const char* name() const noexcept override { return "passthrough"; }
};

} // namespace radiance_codec
