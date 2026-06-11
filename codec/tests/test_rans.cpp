// Round-trip tests for the rANS stage. Exercises both Static (uniform)
// and Order0 (adaptive) on three different input distributions:
//   uniform random bytes  : Static ≈ 1.0x, Order0 ≈ 1.0x
//   skewed (mostly zeros) : Static ≈ 1.0x, Order0 >> 1.0x
//   float32 image bytes   : Static ≈ 1.0x, Order0 ~ 1.1-1.5x

#include "radiance_codec/codec.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {

int fail(const char* what) {
    std::fprintf(stderr, "TEST FAIL: %s\n", what);
    return 1;
}

int run_case(const char* label,
             const std::vector<uint8_t>& raw,
             const radiance_codec::ImageMeta& meta,
             uint8_t rans_mode) {
    radiance_codec::PipelineConfig cfg{};
    cfg.stages    = radiance_codec::StageRans;
    cfg.rans_mode = rans_mode;

    std::vector<uint8_t> compressed;
    if (radiance_codec::encode(raw, meta, cfg, compressed)
        != radiance_codec::Status::Ok) {
        std::fprintf(stderr, "  [%s mode=%u] encode failed\n",
                     label, rans_mode);
        return 1;
    }
    std::vector<uint8_t> roundtrip;
    if (radiance_codec::decode(compressed, meta, cfg, roundtrip)
        != radiance_codec::Status::Ok) {
        std::fprintf(stderr, "  [%s mode=%u] decode failed\n",
                     label, rans_mode);
        return 1;
    }
    if (roundtrip.size() != raw.size()
        || std::memcmp(roundtrip.data(), raw.data(), raw.size()) != 0) {
        std::fprintf(stderr, "  [%s mode=%u] round-trip mismatch\n",
                     label, rans_mode);
        return 1;
    }
    std::printf("  [%s mode=%u] %zu -> %zu bytes (ratio %.3fx)\n",
                label, rans_mode, raw.size(), compressed.size(),
                double(raw.size()) / double(compressed.size()));
    return 0;
}

} // namespace

int main() {
    std::printf("%s\n", radiance_codec::version());

    // Common meta: 64x32 RGB float32, but raw bytes are arbitrary.
    constexpr uint32_t W = 64, H = 32;
    constexpr uint8_t  C = 3;
    radiance_codec::ImageMeta meta{
        .width = W, .height = H, .channels = C,
        .format = radiance_codec::PixelFormat::Float32,
    };
    const std::size_t raw_size = std::size_t(W) * H * C * 4;  // 24576

    int errors = 0;

    // Case 1: uniform random bytes — should compress poorly
    {
        std::printf("\nCase: uniform random bytes\n");
        std::vector<uint8_t> raw(raw_size);
        std::mt19937 rng(1);
        std::uniform_int_distribution<int> dist(0, 255);
        for (auto& b : raw) b = static_cast<uint8_t>(dist(rng));
        errors += run_case("uniform", raw, meta, 0);
        errors += run_case("uniform", raw, meta, 1);
        errors += run_case("uniform", raw, meta, 2);
        errors += run_case("uniform", raw, meta, 3);
        errors += run_case("uniform", raw, meta, 4);
    }

    // Case 2: skewed (mostly zeros, occasional symbols)
    {
        std::printf("\nCase: skewed (90%% zeros)\n");
        std::vector<uint8_t> raw(raw_size);
        std::mt19937 rng(2);
        std::uniform_int_distribution<int> dist(0, 99);
        std::uniform_int_distribution<int> sym(0, 255);
        for (auto& b : raw) b = (dist(rng) < 90)
            ? uint8_t(0)
            : static_cast<uint8_t>(sym(rng));
        errors += run_case("skewed", raw, meta, 0);
        errors += run_case("skewed", raw, meta, 1);
        errors += run_case("skewed", raw, meta, 2);
        errors += run_case("skewed", raw, meta, 3);
        errors += run_case("skewed", raw, meta, 4);
    }

    // Case 3: float32 gradient + noise (realistic-ish HDR pattern)
    {
        std::printf("\nCase: float32 gradient + noise\n");
        std::vector<float> floats(W * H * C);
        std::mt19937 rng(3);
        std::normal_distribution<float> noise(0.0f, 0.01f);
        for (uint32_t y = 0; y < H; ++y) {
            for (uint32_t x = 0; x < W; ++x) {
                for (uint8_t c = 0; c < C; ++c) {
                    float base = (x + y * 0.5f + c * 100.0f) * 0.01f;
                    floats[(y * W + x) * C + c] = base + noise(rng);
                }
            }
        }
        std::vector<uint8_t> raw(raw_size);
        std::memcpy(raw.data(), floats.data(), raw_size);
        errors += run_case("float32", raw, meta, 0);
        errors += run_case("float32", raw, meta, 1);
        errors += run_case("float32", raw, meta, 2);
        errors += run_case("float32", raw, meta, 3);
        errors += run_case("float32", raw, meta, 4);
    }

    if (errors == 0) {
        std::printf("\nALL TESTS PASSED\n");
        return 0;
    } else {
        std::printf("\n%d failures\n", errors);
        return 1;
    }
}
