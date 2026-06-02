// Grouped ordered-float delta stage.
//
// This stage is the C++ counterpart of the Python "grouped body" probe:
// map float32 bits into an order-preserving uint32 domain, code bits 0..30
// (exponent + mantissa body) as one modular-delta payload, and code the sign
// bit as a separate XOR-like payload.  The payload bitplanes are then entropy
// coded with decoder-safe adaptive binary rANS contexts.

#pragma once

#include "pipeline.hpp"

namespace radiance_codec {

class GroupedDeltaStage final : public IStage {
public:
    explicit GroupedDeltaStage(std::uint8_t effort = 5) noexcept
        : effort_(effort) {}

    Status encode(std::span<const std::uint8_t> in,
                  const ImageMeta& meta,
                  std::vector<std::uint8_t>& out) noexcept override;
    Status decode(std::span<const std::uint8_t> in,
                  const ImageMeta& meta,
                  std::vector<std::uint8_t>& out) noexcept override;
    const char* name() const noexcept override { return "grouped_delta"; }

private:
    std::uint8_t effort_ = 5;
};

} // namespace radiance_codec
