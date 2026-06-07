#include "pipeline.hpp"
#include "passthrough.hpp"
#include "rans.hpp"
#include "bitshuffle.hpp"
#include "predictor.hpp"
#include "color_transform.hpp"
#include "structural_context.hpp"
#include "grouped_delta.hpp"
#include "mantissa_quantize.hpp"
#include "linear_index.hpp"
#include "near_lossless_router.hpp"
#include "byteplane_rans.hpp"

#include <cstdlib>
#include <limits>
#include <memory>

namespace radiance_codec {
namespace {

std::uint8_t router_anchor_bits_from_env(std::uint8_t fallback) noexcept {
    const char* raw = std::getenv("RADIANCE_CODEC_ROUTER_ANCHOR_BITS");
    if (!raw || !*raw) return fallback;
    char* end = nullptr;
    const long value = std::strtol(raw, &end, 10);
    if (end == raw || value <= 0 || value > 16) return fallback;
    return static_cast<std::uint8_t>(value);
}

std::uint8_t router_u8_from_env(const char* name, std::uint8_t fallback, std::uint8_t min_value, std::uint8_t max_value) noexcept {
    const char* raw = std::getenv(name);
    if (!raw || !*raw) return fallback;
    char* end = nullptr;
    const long value = std::strtol(raw, &end, 10);
    if (end == raw || value < min_value || value > max_value) return fallback;
    return static_cast<std::uint8_t>(value);
}

std::uint8_t router_bool_from_env(const char* name, std::uint8_t fallback) noexcept {
    const char* raw = std::getenv(name);
    if (!raw || !*raw) return fallback;
    if (raw[0] == '0' || raw[0] == 'n' || raw[0] == 'N' || raw[0] == 'f' || raw[0] == 'F') {
        return 0;
    }
    return 1;
}

NearLosslessRouterParams router_params_from_env() noexcept {
    NearLosslessRouterParams params;
    params.anchor_bits = router_anchor_bits_from_env(params.anchor_bits);
    params.guide_radius = router_u8_from_env(
        "RADIANCE_CODEC_ROUTER_GUIDE_RADIUS",
        params.guide_radius,
        1,
        8);
    params.visual_guard_enabled = router_bool_from_env(
        "RADIANCE_CODEC_ROUTER_VISUAL_GUARD",
        params.visual_guard_enabled);
    return params;
}

} // namespace

std::vector<std::unique_ptr<IStage>> build_pipeline(
    const PipelineConfig& config) {

    std::vector<std::unique_ptr<IStage>> stages;

    // Order matters: this is the encode direction. The decode side
    // walks the list in reverse. Encode pipeline:
    //   ColorTransform → LogMagnitude → SpatialPredict → Bitshuffle → Rans
    //
    if (config.stages & StageLinearIndex) {
        stages.emplace_back(
            std::make_unique<LinearIndexStage>(
                config.near_lossless_bits,
                config.effort,
                config.near_lossless_policy));
        return stages;
    }

    if (config.stages & StageNearLosslessRouter) {
        stages.emplace_back(
            std::make_unique<NearLosslessRouterStage>(router_params_from_env()));
        return stages;
    }

    if (config.stages & StageMantissaQuantize) {
        stages.emplace_back(
            std::make_unique<MantissaQuantizeStage>(
                config.near_lossless_bits,
                config.near_lossless_policy));
    }

    // These research codec stages already include their own transform and
    // entropy coding, so keep them otherwise exclusive.
    if (config.stages & StageGroupedDelta) {
        stages.emplace_back(std::make_unique<GroupedDeltaStage>(config.effort));
        return stages;
    }

    if (config.stages & StageByteplaneRans) {
        stages.emplace_back(std::make_unique<ByteplaneRansStage>(config.effort));
        return stages;
    }

    if (config.stages & StageStructuralContext) {
        stages.emplace_back(std::make_unique<StructuralContextStage>(config.effort));
        return stages;
    }

    if (config.stages & StageColorTransform) {
        stages.emplace_back(std::make_unique<ColorTransformStage>());
    }

    if (config.stages & StageSpatialPredict) {
        stages.emplace_back(std::make_unique<PredictStage>());
    }

    if (config.stages & StageBitshuffle) {
        // float32 typesize. Channels are interleaved at the byte level;
        // bitshuffle works on the raw byte stream treating every 4 bytes
        // as one item, which interleaves channels into bit-planes.
        stages.emplace_back(std::make_unique<BitshuffleStage>(4));
    }

    if (config.stages & StageRans) {
        RansMode mode = RansMode::Order0;
        switch (config.rans_mode) {
            case 0: mode = RansMode::Static; break;
            case 1: mode = RansMode::Order0; break;
            case 2: mode = RansMode::Order1; break;
            default: mode = RansMode::Order0; break;
        }
        stages.emplace_back(std::make_unique<RansStage>(mode));
    }

    // If no real stages selected, fall back to passthrough so we still
    // exercise the framing layer.
    if (stages.empty()) {
        stages.emplace_back(std::make_unique<PassthroughStage>());
    }

    return stages;
}

} // namespace radiance_codec
