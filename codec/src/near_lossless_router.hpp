#pragma once

#include "pipeline.hpp"

#include <cstdint>
#include <span>
#include <vector>

namespace radiance_codec {

struct NearLosslessRouterParams {
    std::uint8_t y_bits = 8;
    std::uint8_t chroma_low_bits = 8;
    std::uint8_t high_bits = 5;
    std::uint8_t anchor_bits = 10;
    std::uint8_t low_scale = 2;
    std::uint8_t guide_radius = 2;
    float guide_eps = 0.1f;
    float threshold_mult = 2.5f;
    float dark_max = 0.5f;
    std::uint8_t mask_radius = 2;
    float smooth_threshold = 0.0025f;
    float target_y_step = 0.0032f;
    float outlier_activation_ratio = 4.0f;
    std::uint8_t visual_guard_enabled = 1;
    std::uint8_t visual_guard_dilate_radius = 0;
    float visual_guard_luma_threshold = 0.010f;
    float visual_guard_rgb_threshold = 0.0f;
    float visual_guard_white = 4.0f;
    float visual_guard_gamma = 2.2f;
};

struct NearLosslessRouterReport {
    float route_mask_rate = 0.0f;
    float dark_mask_rate = 0.0f;
    float outlier_mask_rate = 0.0f;
    std::uint8_t outlier_active = 0;
    float chosen_percentile = 0.0f;
    float threshold_maxrgb = 0.0f;
    float y_step = 0.0f;
    float max_over_p99 = 0.0f;
    float p99_over_p97 = 0.0f;
};

Status reconstruct_near_lossless_router_v1(
    std::span<const std::uint8_t> raw,
    const ImageMeta& meta,
    const NearLosslessRouterParams& params,
    std::vector<std::uint8_t>& out,
    NearLosslessRouterReport* report) noexcept;

class NearLosslessRouterStage final : public IStage {
public:
    explicit NearLosslessRouterStage(
        NearLosslessRouterParams params = NearLosslessRouterParams{}) noexcept
        : params_(params) {}

    Status encode(std::span<const std::uint8_t> in,
                  const ImageMeta& meta,
                  std::vector<std::uint8_t>& out) noexcept override;

    Status decode(std::span<const std::uint8_t> in,
                  const ImageMeta& meta,
                  std::vector<std::uint8_t>& out) noexcept override;

    const char* name() const noexcept override {
        return "near_lossless_router_v1";
    }

private:
    NearLosslessRouterParams params_;
};

} // namespace radiance_codec
