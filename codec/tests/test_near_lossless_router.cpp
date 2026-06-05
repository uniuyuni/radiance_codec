#include "near_lossless_router.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

namespace {

void write_f32(std::vector<std::uint8_t>& raw, std::size_t sample, float value) {
    std::memcpy(raw.data() + sample * 4, &value, sizeof(value));
}

float read_f32(const std::vector<std::uint8_t>& raw, std::size_t sample) {
    float value = 0.0f;
    std::memcpy(&value, raw.data() + sample * 4, sizeof(value));
    return value;
}

} // namespace

int main() {
    radiance_codec::ImageMeta meta;
    meta.width = 64;
    meta.height = 32;
    meta.channels = 3;
    meta.format = radiance_codec::PixelFormat::Float32;

    std::vector<std::uint8_t> raw(meta.raw_size(), 0);
    for (std::uint32_t y = 0; y < meta.height; ++y) {
        for (std::uint32_t x = 0; x < meta.width; ++x) {
            const auto pixel = (std::size_t(y) * meta.width + x) * meta.channels;
            const float fx = static_cast<float>(x) / static_cast<float>(meta.width - 1);
            const float fy = static_cast<float>(y) / static_cast<float>(meta.height - 1);
            float r = 0.02f + 0.8f * fx;
            float g = 0.03f + 0.7f * fy;
            float b = 0.04f + 0.6f * (fx + fy) * 0.5f;
            if (x > 58 && y > 26) {
                r = 120.0f;
                g = 80.0f;
                b = 60.0f;
            }
            write_f32(raw, pixel + 0, r);
            write_f32(raw, pixel + 1, g);
            write_f32(raw, pixel + 2, b);
        }
    }

    radiance_codec::NearLosslessRouterParams params;
    params.target_y_step = 0.0032f;
    params.outlier_activation_ratio = 4.0f;

    std::vector<std::uint8_t> decoded;
    radiance_codec::NearLosslessRouterReport report;
    const auto status = radiance_codec::reconstruct_near_lossless_router_v1(
        raw,
        meta,
        params,
        decoded,
        &report);
    if (status != radiance_codec::Status::Ok) {
        std::cerr << "router reconstruct failed\n";
        return 1;
    }
    if (decoded.size() != raw.size()) {
        std::cerr << "decoded size mismatch\n";
        return 1;
    }
    if (!report.outlier_active || report.route_mask_rate <= 0.0f) {
        std::cerr << "expected outlier router to activate\n";
        return 1;
    }
    for (std::size_t i = 0; i < decoded.size() / 4; ++i) {
        if (!std::isfinite(read_f32(decoded, i))) {
            std::cerr << "decoded non-finite value\n";
            return 1;
        }
    }

    radiance_codec::PipelineConfig config;
    config.stages = radiance_codec::StageNearLosslessRouter;
    config.effort = 1;

    std::vector<std::uint8_t> compressed;
    auto codec_status = radiance_codec::encode(raw, meta, config, compressed);
    if (codec_status != radiance_codec::Status::Ok || compressed.empty()) {
        std::cerr << "pipeline encode failed\n";
        return 1;
    }

    std::vector<std::uint8_t> roundtrip;
    codec_status = radiance_codec::decode(compressed, meta, config, roundtrip);
    if (codec_status != radiance_codec::Status::Ok) {
        std::cerr << "pipeline decode failed\n";
        return 1;
    }
    if (roundtrip.size() != raw.size()) {
        std::cerr << "pipeline output size mismatch\n";
        return 1;
    }
    for (std::size_t i = 0; i < roundtrip.size() / 4; ++i) {
        if (!std::isfinite(read_f32(roundtrip, i))) {
            std::cerr << "pipeline output has non-finite value\n";
            return 1;
        }
    }

    std::cout << "near_lossless_router OK: mask="
              << report.route_mask_rate
              << " p=" << report.chosen_percentile
              << " step=" << report.y_step << "\n";
    return 0;
}
