// Round-trip tests for the public codec pipeline.
//
// Generates a deterministic float32 buffer, checks passthrough bit-exactness,
// and checks the near-lossless mantissa-quantize + grouped-delta path.

#include "radiance_codec/codec.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

static int fail(const char* what) {
    std::fprintf(stderr, "TEST FAIL: %s\n", what);
    return 1;
}

static std::uint32_t read_le32(const std::uint8_t* p) {
    return static_cast<std::uint32_t>(p[0])
        | (static_cast<std::uint32_t>(p[1]) << 8)
        | (static_cast<std::uint32_t>(p[2]) << 16)
        | (static_cast<std::uint32_t>(p[3]) << 24);
}

static void write_le32(std::uint8_t* p, std::uint32_t value) {
    p[0] = static_cast<std::uint8_t>(value & 0xffu);
    p[1] = static_cast<std::uint8_t>((value >> 8) & 0xffu);
    p[2] = static_cast<std::uint8_t>((value >> 16) & 0xffu);
    p[3] = static_cast<std::uint8_t>((value >> 24) & 0xffu);
}

static std::uint8_t header_byte(
    const std::vector<std::uint8_t>& compressed,
    std::size_t offset) {
    return offset < compressed.size() ? compressed[offset] : 255;
}

static std::vector<std::uint8_t> quantized_mantissa_copy(
    const std::vector<std::uint8_t>& raw,
    std::uint8_t low_bits) {
    std::vector<std::uint8_t> out = raw;
    if (low_bits == 0) return out;

    const std::uint32_t clear_mask =
        low_bits >= 23
            ? 0xff800000u
            : (0xffffffffu << low_bits);
    const std::size_t count = out.size() / 4;
    for (std::size_t i = 0; i < count; ++i) {
        auto* p = out.data() + i * 4;
        auto bits = read_le32(p);
        const auto exponent = (bits >> 23) & 0xffu;
        if (exponent != 0xffu) {
            bits &= clear_mask;
            write_le32(p, bits);
        }
    }
    return out;
}

static std::vector<std::uint8_t> quantized_linear_index_copy(
    const std::vector<float>& floats,
    const radiance_codec::ImageMeta& meta,
    std::uint8_t bits) {
    constexpr std::uint32_t TILE = 128;
    const std::uint32_t levels = (std::uint32_t(1) << bits) - 1;
    std::vector<float> out = floats;
    for (std::uint32_t y0 = 0; y0 < meta.height; y0 += TILE) {
        const auto y1 = std::min<std::uint32_t>(meta.height, y0 + TILE);
        for (std::uint32_t x0 = 0; x0 < meta.width; x0 += TILE) {
            const auto x1 = std::min<std::uint32_t>(meta.width, x0 + TILE);
            for (std::uint8_t c = 0; c < meta.channels; ++c) {
                float lo = std::numeric_limits<float>::infinity();
                float hi = -std::numeric_limits<float>::infinity();
                for (std::uint32_t y = y0; y < y1; ++y) {
                    for (std::uint32_t x = x0; x < x1; ++x) {
                        const auto off =
                            (std::size_t(y) * meta.width + x) * meta.channels + c;
                        lo = std::min(lo, floats[off]);
                        hi = std::max(hi, floats[off]);
                    }
                }
                for (std::uint32_t y = y0; y < y1; ++y) {
                    for (std::uint32_t x = x0; x < x1; ++x) {
                        const auto off =
                            (std::size_t(y) * meta.width + x) * meta.channels + c;
                        if (!(hi > lo)) {
                            out[off] = lo;
                            continue;
                        }
                        double q = std::floor(
                            (static_cast<double>(floats[off]) - static_cast<double>(lo))
                            / (static_cast<double>(hi) - static_cast<double>(lo))
                            * levels
                            + 0.5);
                        q = std::clamp(q, 0.0, static_cast<double>(levels));
                        out[off] = static_cast<float>(
                            static_cast<double>(lo)
                            + q * (static_cast<double>(hi) - static_cast<double>(lo))
                                / static_cast<double>(levels));
                    }
                }
            }
        }
    }
    std::vector<std::uint8_t> raw(out.size() * 4);
    std::memcpy(raw.data(), out.data(), raw.size());
    return raw;
}

int main() {
    std::printf("%s\n", radiance_codec::version());

    // 64x32 RGB float32 = 24576 bytes
    constexpr std::uint32_t W = 64, H = 32;
    constexpr std::uint8_t  C = 3;
    radiance_codec::ImageMeta meta{
        .width = W, .height = H, .channels = C,
        .format = radiance_codec::PixelFormat::Float32,
    };

    // Deterministic non-trivial input (mix gradient + small noise)
    std::vector<float> floats(W * H * C);
    std::mt19937 rng(12345);
    std::normal_distribution<float> noise(0.0f, 0.01f);
    for (std::uint32_t y = 0; y < H; ++y) {
        for (std::uint32_t x = 0; x < W; ++x) {
            for (std::uint8_t c = 0; c < C; ++c) {
                float base = (x + y * 0.5f + c * 100.0f) * 0.01f;
                floats[(y * W + x) * C + c] = base + noise(rng);
            }
        }
    }
    floats[5] = std::numeric_limits<float>::infinity();
    floats[17] = std::numeric_limits<float>::quiet_NaN();
    floats[23] = -0.0f;

    std::vector<std::uint8_t> raw(meta.raw_size());
    std::memcpy(raw.data(), floats.data(), raw.size());

    radiance_codec::PipelineConfig cfg{
        .stages = radiance_codec::StageNone,
        .effort = 5,
    };

    std::vector<std::uint8_t> compressed;
    if (radiance_codec::encode(raw, meta, cfg, compressed) != radiance_codec::Status::Ok) {
        return fail("encode returned non-Ok");
    }
    std::printf("encode: %zu -> %zu bytes (ratio %.3fx)\n",
                raw.size(), compressed.size(),
                double(raw.size()) / double(compressed.size()));

    std::vector<std::uint8_t> roundtrip;
    if (radiance_codec::decode(compressed, meta, cfg, roundtrip) != radiance_codec::Status::Ok) {
        return fail("decode returned non-Ok");
    }

    if (roundtrip.size() != raw.size()) return fail("size mismatch");
    if (std::memcmp(roundtrip.data(), raw.data(), raw.size()) != 0) {
        return fail("byte mismatch (passthrough should be bit-exact)");
    }
    if (header_byte(compressed, 4) != 3) {
        return fail("new frames should use header version 3");
    }
    if (header_byte(compressed, 23) !=
        static_cast<std::uint8_t>(radiance_codec::SignClass::Mixed)) {
        return fail("mixed-sign test image should write mixed sign class");
    }

    std::printf("ROUND-TRIP OK: %zu bytes bit-exact\n", raw.size());

    constexpr std::uint8_t LOW_BITS = 12;
    radiance_codec::PipelineConfig near_cfg{
        .stages = radiance_codec::StageMantissaQuantize | radiance_codec::StageGroupedDelta,
        .effort = 11,
        .rans_mode = 1,
        .near_lossless_bits = LOW_BITS,
        .near_lossless_policy =
            static_cast<std::uint8_t>(radiance_codec::NearLosslessPolicy::Fixed),
    };

    std::vector<std::uint8_t> near_compressed;
    if (radiance_codec::encode(raw, meta, near_cfg, near_compressed)
        != radiance_codec::Status::Ok) {
        return fail("near-lossless encode returned non-Ok");
    }

    std::vector<std::uint8_t> near_roundtrip;
    if (radiance_codec::decode(near_compressed, meta, near_cfg, near_roundtrip)
        != radiance_codec::Status::Ok) {
        return fail("near-lossless decode returned non-Ok");
    }

    const auto expected = quantized_mantissa_copy(raw, LOW_BITS);
    if (near_roundtrip.size() != expected.size()) {
        return fail("near-lossless size mismatch");
    }
    if (std::memcmp(near_roundtrip.data(), expected.data(), expected.size()) != 0) {
        return fail("near-lossless output should match quantized input");
    }
    if (std::memcmp(near_roundtrip.data(), raw.data(), raw.size()) == 0) {
        return fail("near-lossless test input was not changed by quantization");
    }

    std::printf("NEAR-LOSSLESS OK: low%u %zu -> %zu bytes (ratio %.3fx)\n",
                unsigned(LOW_BITS), raw.size(), near_compressed.size(),
                double(raw.size()) / double(near_compressed.size()));

    std::vector<float> finite_floats(W * H * C);
    for (std::uint32_t y = 0; y < H; ++y) {
        for (std::uint32_t x = 0; x < W; ++x) {
            for (std::uint8_t c = 0; c < C; ++c) {
                finite_floats[(y * W + x) * C + c] =
                    0.02f * float(x) + 0.01f * float(y) + 0.25f * float(c);
            }
        }
    }
    std::vector<std::uint8_t> finite_raw(meta.raw_size());
    std::memcpy(finite_raw.data(), finite_floats.data(), finite_raw.size());
    radiance_codec::PipelineConfig index_cfg{
        .stages = radiance_codec::StageLinearIndex,
        .effort = 9,
        .rans_mode = 1,
        .near_lossless_bits = 7,
        .near_lossless_policy =
            static_cast<std::uint8_t>(radiance_codec::NearLosslessPolicy::LinearRange),
    };
    std::vector<std::uint8_t> index_compressed;
    if (radiance_codec::encode(finite_raw, meta, index_cfg, index_compressed)
        != radiance_codec::Status::Ok) {
        return fail("linear-index encode returned non-Ok");
    }
    std::vector<std::uint8_t> index_roundtrip;
    if (radiance_codec::decode(index_compressed, meta, index_cfg, index_roundtrip)
        != radiance_codec::Status::Ok) {
        return fail("linear-index decode returned non-Ok");
    }
    const auto index_expected =
        quantized_linear_index_copy(finite_floats, meta, 7);
    if (index_roundtrip.size() != index_expected.size()) {
        return fail("linear-index size mismatch");
    }
    if (std::memcmp(index_roundtrip.data(), index_expected.data(),
                    index_expected.size()) != 0) {
        return fail("linear-index output should match local linear quantization");
    }
    std::printf("LINEAR-INDEX OK: bits7 %zu -> %zu bytes (ratio %.3fx)\n",
                finite_raw.size(), index_compressed.size(),
                double(finite_raw.size()) / double(index_compressed.size()));

    radiance_codec::PipelineConfig adaptive_cfg = near_cfg;
    adaptive_cfg.near_lossless_policy =
        static_cast<std::uint8_t>(radiance_codec::NearLosslessPolicy::TileExponent);
    std::vector<std::uint8_t> adaptive_compressed;
    if (radiance_codec::encode(raw, meta, adaptive_cfg, adaptive_compressed)
        != radiance_codec::Status::Ok) {
        return fail("adaptive near-lossless encode returned non-Ok");
    }
    if (header_byte(adaptive_compressed, 22) !=
        static_cast<std::uint8_t>(radiance_codec::NearLosslessPolicy::TileExponent)) {
        return fail("adaptive near-lossless policy was not written to the header");
    }
    std::vector<std::uint8_t> adaptive_roundtrip;
    if (radiance_codec::decode(adaptive_compressed, meta, adaptive_cfg, adaptive_roundtrip)
        != radiance_codec::Status::Ok) {
        return fail("adaptive near-lossless decode returned non-Ok");
    }
    if (adaptive_roundtrip.size() != raw.size()) {
        return fail("adaptive near-lossless size mismatch");
    }

    std::vector<float> positive_floats(8, 1.0f);
    std::vector<std::uint8_t> positive_raw(positive_floats.size() * 4);
    std::memcpy(positive_raw.data(), positive_floats.data(), positive_raw.size());
    radiance_codec::ImageMeta positive_meta{
        .width = 4, .height = 2, .channels = 1,
        .format = radiance_codec::PixelFormat::Float32,
    };
    std::vector<std::uint8_t> positive_compressed;
    if (radiance_codec::encode(positive_raw, positive_meta, cfg, positive_compressed)
        != radiance_codec::Status::Ok) {
        return fail("positive sign-class encode returned non-Ok");
    }
    if (header_byte(positive_compressed, 23) !=
        static_cast<std::uint8_t>(radiance_codec::SignClass::AllPositive)) {
        return fail("all-positive sign class was not written to the header");
    }

    std::vector<float> negative_floats(8, -1.0f);
    std::vector<std::uint8_t> negative_raw(negative_floats.size() * 4);
    std::memcpy(negative_raw.data(), negative_floats.data(), negative_raw.size());
    std::vector<std::uint8_t> negative_compressed;
    if (radiance_codec::encode(negative_raw, positive_meta, cfg, negative_compressed)
        != radiance_codec::Status::Ok) {
        return fail("negative sign-class encode returned non-Ok");
    }
    if (header_byte(negative_compressed, 23) !=
        static_cast<std::uint8_t>(radiance_codec::SignClass::AllNegative)) {
        return fail("all-negative sign class was not written to the header");
    }
    return 0;
}
