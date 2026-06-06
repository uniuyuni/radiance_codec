#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace radiance_codec {

struct MetalVisualGuardConfig {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t channels = 0;
    std::uint32_t low_w = 0;
    std::uint32_t low_h = 0;
    std::uint32_t count = 0;
    std::uint8_t low_scale = 1;
    std::uint8_t y_bits = 8;
    std::uint8_t chroma_low_bits = 8;
    std::uint8_t high_bits = 5;
    std::uint8_t anchor_bits = 10;
    std::uint8_t visual_guard_dilate_radius = 0;
    float visual_guard_luma_threshold = 0.0f;
    float visual_guard_rgb_threshold = 0.0f;
    float visual_guard_white = 1.0f;
    float visual_guard_gamma = 2.2f;
    float base_y_lo = 0.0f;
    float base_y_hi = 0.0f;
    float base_co_low_lo = 0.0f;
    float base_co_low_hi = 0.0f;
    float base_cg_low_lo = 0.0f;
    float base_cg_low_hi = 0.0f;
    float base_co_high_lo = 0.0f;
    float base_co_high_hi = 0.0f;
    float base_cg_high_lo = 0.0f;
    float base_cg_high_hi = 0.0f;
    float base_log_lo[3] = {};
    float base_log_hi[3] = {};
};

bool metal_guided_low_pair(
    const std::vector<float>& first_plane,
    const std::vector<float>& second_plane,
    const std::vector<float>& guide,
    std::uint32_t width,
    std::uint32_t height,
    std::uint8_t radius,
    float eps,
    bool copy_outputs,
    std::vector<float>& first_low,
    std::vector<float>& second_low) noexcept;

bool metal_guided_low_downsample_pair(
    const std::vector<float>& first_plane,
    const std::vector<float>& second_plane,
    const std::vector<float>& guide,
    std::uint32_t width,
    std::uint32_t height,
    std::uint8_t radius,
    float eps,
    std::uint8_t scale,
    bool copy_low_outputs,
    std::vector<float>& first_low,
    std::vector<float>& second_low,
    std::uint32_t& out_w,
    std::uint32_t& out_h,
    std::vector<float>& first_coarse,
    std::vector<float>& second_coarse) noexcept;

bool metal_copy_cached_low_pair(
    std::size_t count,
    std::vector<float>& first_low,
    std::vector<float>& second_low) noexcept;

bool metal_cached_residual_threshold_pair(
    std::size_t count,
    float threshold_mult,
    float& first_threshold,
    float& second_threshold) noexcept;

bool metal_high_pass_shrink_pair(
    const std::vector<float>& first_plane,
    const std::vector<float>& second_plane,
    const std::vector<float>& first_low,
    const std::vector<float>& second_low,
    bool use_cached_low_pair,
    float first_threshold,
    float second_threshold,
    std::vector<float>& first_high,
    std::vector<float>& second_high) noexcept;

bool metal_block_mean_downsample_pair(
    const std::vector<float>& first_plane,
    const std::vector<float>& second_plane,
    std::uint32_t width,
    std::uint32_t height,
    std::uint8_t scale,
    bool use_cached_input_pair,
    std::uint32_t& out_w,
    std::uint32_t& out_h,
    std::vector<float>& first_out,
    std::vector<float>& second_out) noexcept;

bool metal_visual_guard(
    const std::uint8_t* raw,
    std::size_t raw_size,
    const std::vector<std::uint8_t>& base_route_mask,
    const std::vector<std::uint8_t>& base_high_mask,
    const std::vector<float>& y_plane,
    const std::vector<float>& co_coarse,
    const std::vector<float>& cg_coarse,
    const std::vector<float>& co_high,
    const std::vector<float>& cg_high,
    const std::vector<float>& source_display_luma,
    bool use_cached_guide,
    bool use_cached_coarse_pair,
    bool use_cached_high_pass,
    const MetalVisualGuardConfig& config,
    std::vector<std::uint8_t>& guard) noexcept;

} // namespace radiance_codec
