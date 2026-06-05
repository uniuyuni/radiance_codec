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

#include <memory>

namespace radiance_codec {

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
        stages.emplace_back(std::make_unique<NearLosslessRouterStage>());
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
