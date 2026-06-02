// Pipeline stage: rANS entropy coder.
//
// Three variants selected by `mode`:
//   Static  : uniform 8-bit byte model (no header). Sanity check.
//             Output size ≈ input size.
//   Order0  : adaptive 8-bit byte model. Two-pass: histogram → renormalize
//             to PROB_SCALE → encode. Header = serialized freq table.
//   Order1  : (Phase 1-4) context-conditioned on previous byte.

#pragma once

#include "pipeline.hpp"

namespace radiance_codec {

enum class RansMode : uint8_t {
    Static = 0,
    Order0 = 1,
    Order1 = 2,
};

class RansStage final : public IStage {
public:
    explicit RansStage(RansMode mode) noexcept : mode_(mode) {}

    Status encode(std::span<const std::uint8_t> in,
                  const ImageMeta& meta,
                  std::vector<std::uint8_t>& out) noexcept override;
    Status decode(std::span<const std::uint8_t> in,
                  const ImageMeta& meta,
                  std::vector<std::uint8_t>& out) noexcept override;
    const char* name() const noexcept override;

private:
    RansMode mode_;
};

} // namespace radiance_codec
