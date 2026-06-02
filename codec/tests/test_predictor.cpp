// Round-trip tests for the MED predictor stage in isolation.

#include "radiance_codec/codec.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {

int run_case(const char* label,
             const std::vector<uint8_t>& raw,
             const radiance_codec::ImageMeta& meta) {
    radiance_codec::PipelineConfig cfg{};
    cfg.stages = radiance_codec::StageSpatialPredict;

    std::vector<uint8_t> compressed;
    if (radiance_codec::encode(raw, meta, cfg, compressed)
        != radiance_codec::Status::Ok) {
        std::fprintf(stderr, "  [%s] encode failed\n", label);
        return 1;
    }
    std::vector<uint8_t> roundtrip;
    if (radiance_codec::decode(compressed, meta, cfg, roundtrip)
        != radiance_codec::Status::Ok) {
        std::fprintf(stderr, "  [%s] decode failed\n", label);
        return 1;
    }
    if (roundtrip.size() != raw.size()
        || std::memcmp(roundtrip.data(), raw.data(), raw.size()) != 0) {
        std::fprintf(stderr, "  [%s] round-trip mismatch\n", label);
        // Find first differing position for diagnosis.
        for (std::size_t i = 0; i < raw.size(); ++i) {
            if (raw[i] != roundtrip[i]) {
                std::fprintf(stderr,
                    "    first diff at offset %zu: raw=%02x decoded=%02x\n",
                    i, raw[i], roundtrip[i]);
                break;
            }
        }
        return 1;
    }
    std::printf("  [%s] %zu bytes round-trip OK\n",
                label, raw.size());
    return 0;
}

} // namespace

int main() {
    std::printf("%s\n", radiance_codec::version());

    int errors = 0;

    // Case 1: smooth gradient — predictor should work very well
    {
        std::printf("\nCase: 16x16 smooth gradient RGB\n");
        radiance_codec::ImageMeta meta{
            .width = 16, .height = 16, .channels = 3,
            .format = radiance_codec::PixelFormat::Float32,
        };
        std::vector<float> f(16 * 16 * 3);
        for (uint32_t y = 0; y < 16; ++y) {
            for (uint32_t x = 0; x < 16; ++x) {
                f[(y * 16 + x) * 3 + 0] = x * 1.0f;
                f[(y * 16 + x) * 3 + 1] = y * 1.0f;
                f[(y * 16 + x) * 3 + 2] = (x + y) * 0.5f;
            }
        }
        std::vector<uint8_t> raw(f.size() * 4);
        std::memcpy(raw.data(), f.data(), raw.size());
        errors += run_case("gradient", raw, meta);
    }

    // Case 2: random HDR-ish data
    {
        std::printf("\nCase: 32x16 random HDR\n");
        radiance_codec::ImageMeta meta{
            .width = 32, .height = 16, .channels = 3,
            .format = radiance_codec::PixelFormat::Float32,
        };
        std::vector<float> f(32 * 16 * 3);
        std::mt19937 rng(7);
        std::normal_distribution<float> dist(5.0f, 50.0f);
        for (auto& v : f) v = dist(rng);
        std::vector<uint8_t> raw(f.size() * 4);
        std::memcpy(raw.data(), f.data(), raw.size());
        errors += run_case("random_hdr", raw, meta);
    }

    // Case 3: single channel
    {
        std::printf("\nCase: 32x16 single channel\n");
        radiance_codec::ImageMeta meta{
            .width = 32, .height = 16, .channels = 1,
            .format = radiance_codec::PixelFormat::Float32,
        };
        std::vector<float> f(32 * 16);
        for (std::size_t i = 0; i < f.size(); ++i)
            f[i] = static_cast<float>(i) * 0.1f;
        std::vector<uint8_t> raw(f.size() * 4);
        std::memcpy(raw.data(), f.data(), raw.size());
        errors += run_case("single_ch", raw, meta);
    }

    // Case 4: RGBA 4-channel
    {
        std::printf("\nCase: 16x16 RGBA HDR\n");
        radiance_codec::ImageMeta meta{
            .width = 16, .height = 16, .channels = 4,
            .format = radiance_codec::PixelFormat::Float32,
        };
        std::vector<float> f(16 * 16 * 4);
        std::mt19937 rng(11);
        std::normal_distribution<float> dist(10.0f, 100.0f);
        for (auto& v : f) v = dist(rng);
        std::vector<uint8_t> raw(f.size() * 4);
        std::memcpy(raw.data(), f.data(), raw.size());
        errors += run_case("rgba", raw, meta);
    }

    if (errors == 0) {
        std::printf("\nALL TESTS PASSED\n");
        return 0;
    } else {
        std::printf("\n%d failures\n", errors);
        return 1;
    }
}
