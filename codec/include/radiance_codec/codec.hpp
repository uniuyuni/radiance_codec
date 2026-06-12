// C++ public API for the HDR codec.
//
// The codec is built from a pipeline of independent stages. Most stages are
// bit-exact reversible; StageMantissaQuantize and StageNearLosslessRouter are
// intentionally near-lossless and decode to the quantized pixels. The classic
// pipeline order (encoding
// direction):
//
//   raw float32 bytes
//      -> color_transform (RCT)
//      -> log_magnitude
//      -> spatial_predictor (MED-like)
//      -> bitshuffle
//      -> rans (entropy coder)
//      -> compressed bytes
//
// Decoding runs the inverse stages in reverse order.
//
// Stages can be individually enabled/disabled via PipelineConfig.stages,
// which is essential for ablation studies. StageGroupedDelta,
// StageByteplaneRans, and StageStructuralContext are self-contained research
// codecs and should normally be used as alternatives to the classic transform
// stack.

#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace radiance_codec {

enum class PixelFormat : uint8_t {
    Float32 = 1,   // IEEE 754 binary32, little-endian, interleaved
};

struct ImageMeta {
    uint32_t width    = 0;
    uint32_t height   = 0;
    uint8_t  channels = 0;          // 1..4
    PixelFormat format = PixelFormat::Float32;

    constexpr std::size_t raw_size() const noexcept {
        return std::size_t(width) * height * channels * 4;
    }
};

// Bitmask of pipeline stages to enable.
enum Stage : uint32_t {
    StageNone           = 0x0000,
    StageColorTransform = 0x0001,
    StageLogMagnitude   = 0x0002,
    StageSpatialPredict = 0x0004,
    StageBitshuffle     = 0x0008,
    StageRans           = 0x0010,
    StageStructuralContext = 0x0020,
    StageGroupedDelta   = 0x0040,
    StageMantissaQuantize = 0x0080,
    StageLinearIndex    = 0x0100,
    StageNearLosslessRouter = 0x0200,
    StageByteplaneRans = 0x0400,

    // Convenience presets
    StageAll = StageColorTransform | StageLogMagnitude
             | StageSpatialPredict | StageBitshuffle | StageRans,
};

enum class NearLosslessPolicy : uint8_t {
    Fixed        = 0, // clear the same number of low mantissa bits everywhere
    Tile         = 1, // clear only bitplanes that look random within a tile
    Exponent     = 2, // clear more bits for darker values
    TileExponent = 3, // tile random-tail cap plus exponent adaptation
    LinearRange  = 4, // quantize finite values per channel to N linear levels
    LogRange     = 5, // quantize finite non-negative values in log2 space
    SqrtRange    = 6, // transform-index experiment: sign(x)*sqrt(abs(x))
    Gamma075Range = 7, // transform-index experiment: sign(x)*abs(x)^0.75
    Gamma025Range = 8, // transform-index experiment: sign(x)*abs(x)^0.25
    AsinhRange   = 9, // transform-index experiment: asinh(x)
};

enum class SignClass : uint8_t {
    Mixed       = 0,
    AllPositive = 1, // every stored IEEE sign bit is 0
    AllNegative = 2, // every stored IEEE sign bit is 1
};

struct PipelineConfig {
    uint32_t stages = StageNone;    // bitmask of Stage values
    uint8_t  effort = 5;            // codec-specific search level
    uint8_t  rans_mode = 1;         // 0=Static, 1=Order0, 2=Order1, 3/4=interleaved
    uint8_t  near_lossless_bits = 0; // low mantissa bits to zero before coding
    uint8_t  near_lossless_policy =
        static_cast<uint8_t>(NearLosslessPolicy::Fixed);
};

enum class Status : int32_t {
    Ok                  = 0,
    InvalidArg          = -1,
    UnsupportedFormat   = -2,
    UnimplementedStage  = -3,
    DecompressFailed    = -4,
    SizeMismatch        = -5,
};

// Encode raw pixels into compressed buffer.
// On success returns Status::Ok and out is resized to the compressed size.
Status encode(std::span<const std::uint8_t> raw,
              const ImageMeta& meta,
              const PipelineConfig& config,
              std::vector<std::uint8_t>& out) noexcept;

// Decode compressed buffer back into raw pixels.
// On success returns Status::Ok and out is resized to meta.raw_size().
Status decode(std::span<const std::uint8_t> compressed,
              const ImageMeta& meta,
              const PipelineConfig& config,
              std::vector<std::uint8_t>& out) noexcept;

// Decode compressed buffer using the ImageMeta stored in the frame header.
// On success returns Status::Ok, out is resized to decoded raw bytes, and
// meta_out is populated when non-null.
Status decode(std::span<const std::uint8_t> compressed,
              std::vector<std::uint8_t>& out,
              ImageMeta* meta_out = nullptr) noexcept;

// Library identification
const char* version() noexcept;

} // namespace radiance_codec
