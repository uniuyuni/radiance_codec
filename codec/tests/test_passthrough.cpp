// Round-trip tests for the public codec pipeline.
//
// Generates a deterministic float32 buffer, checks passthrough bit-exactness,
// and checks the near-lossless mantissa-quantize + grouped-delta path.

#include "radiance_codec/codec.hpp"

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

    std::printf("ROUND-TRIP OK: %zu bytes bit-exact\n", raw.size());

    constexpr std::uint8_t LOW_BITS = 12;
    radiance_codec::PipelineConfig near_cfg{
        .stages = radiance_codec::StageMantissaQuantize | radiance_codec::StageGroupedDelta,
        .effort = 11,
        .rans_mode = 1,
        .near_lossless_bits = LOW_BITS,
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
    return 0;
}
