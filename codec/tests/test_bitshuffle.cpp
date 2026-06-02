// Round-trip tests for the bitshuffle stage in isolation
// (no rANS, just transform-only via full pipeline).

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
    // Stage: just bitshuffle (no rANS).
    // Note: pipeline framing always prepends a tiny file header.
    radiance_codec::PipelineConfig cfg{};
    cfg.stages = radiance_codec::StageBitshuffle;

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
        return 1;
    }
    std::printf("  [%s] %zu -> %zu bytes  (transform-only, no entropy coder)\n",
                label, raw.size(), compressed.size());
    return 0;
}

} // namespace

int main() {
    std::printf("%s\n", radiance_codec::version());

    int errors = 0;

    // Case 1: 8x4x3 floats — n_items = 96, divisible by 8
    {
        std::printf("\nCase: 8x4x3 floats (divisible by 8)\n");
        radiance_codec::ImageMeta meta{
            .width = 8, .height = 4, .channels = 3,
            .format = radiance_codec::PixelFormat::Float32,
        };
        std::vector<float> floats(8 * 4 * 3);
        for (std::size_t i = 0; i < floats.size(); ++i)
            floats[i] = static_cast<float>(i) * 0.5f;
        std::vector<uint8_t> raw(floats.size() * 4);
        std::memcpy(raw.data(), floats.data(), raw.size());
        errors += run_case("aligned", raw, meta);
    }

    // Case 2: 7x3x3 floats — n_items = 63, NOT divisible by 8 (tail = 7)
    {
        std::printf("\nCase: 7x3x3 floats (n_items=63, tail=7)\n");
        radiance_codec::ImageMeta meta{
            .width = 7, .height = 3, .channels = 3,
            .format = radiance_codec::PixelFormat::Float32,
        };
        std::vector<float> floats(7 * 3 * 3);
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-100.0f, 100.0f);
        for (auto& f : floats) f = dist(rng);
        std::vector<uint8_t> raw(floats.size() * 4);
        std::memcpy(raw.data(), floats.data(), raw.size());
        errors += run_case("ragged", raw, meta);
    }

    // Case 3: realistic 64x32 RGB HDR-like
    {
        std::printf("\nCase: 64x32 RGB HDR-like\n");
        radiance_codec::ImageMeta meta{
            .width = 64, .height = 32, .channels = 3,
            .format = radiance_codec::PixelFormat::Float32,
        };
        std::vector<float> floats(64 * 32 * 3);
        std::mt19937 rng(7);
        std::normal_distribution<float> dist(5.0f, 50.0f);
        for (auto& f : floats) f = dist(rng);
        std::vector<uint8_t> raw(floats.size() * 4);
        std::memcpy(raw.data(), floats.data(), raw.size());
        errors += run_case("hdr_64x32", raw, meta);
    }

    if (errors == 0) {
        std::printf("\nALL TESTS PASSED\n");
        return 0;
    } else {
        std::printf("\n%d failures\n", errors);
        return 1;
    }
}
