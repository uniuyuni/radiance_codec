// Round-trip test for the structural float32 context stage.

#include "radiance_codec/codec.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

int fail(const char* what) {
    std::fprintf(stderr, "TEST FAIL: %s\n", what);
    return 1;
}

int run_case(const char* label, std::uint32_t width, std::uint32_t height,
             std::uint8_t channels) {
    radiance_codec::ImageMeta meta{
        .width = width,
        .height = height,
        .channels = channels,
        .format = radiance_codec::PixelFormat::Float32,
    };

    std::vector<float> pixels(std::size_t(width) * height * channels);
    for (std::uint32_t y = 0; y < height; ++y) {
        for (std::uint32_t x = 0; x < width; ++x) {
            for (std::uint8_t c = 0; c < channels; ++c) {
                const float base = 0.01f * static_cast<float>(x)
                    + 0.006f * static_cast<float>(y)
                    + 0.25f * static_cast<float>(c);
                const float ripple = 0.001f * std::sin(float(x + y + c) * 0.17f);
                pixels[(std::size_t(y) * width + x) * channels + c] = base + ripple;
            }
        }
    }

    std::vector<std::uint8_t> raw(meta.raw_size());
    std::memcpy(raw.data(), pixels.data(), raw.size());

    radiance_codec::PipelineConfig cfg{
        .stages = radiance_codec::StageStructuralContext,
        .effort = 5,
    };

    std::vector<std::uint8_t> compressed;
    if (radiance_codec::encode(raw, meta, cfg, compressed) != radiance_codec::Status::Ok) {
        return fail("encode failed");
    }
    std::vector<std::uint8_t> decoded;
    if (radiance_codec::decode(compressed, meta, cfg, decoded) != radiance_codec::Status::Ok) {
        return fail("decode failed");
    }
    if (decoded.size() != raw.size()) return fail("size mismatch");
    if (std::memcmp(decoded.data(), raw.data(), raw.size()) != 0) {
        return fail("roundtrip mismatch");
    }
    std::printf("  [%s] %zu -> %zu bytes (%.3fx)\n",
                label, raw.size(), compressed.size(),
                double(raw.size()) / double(compressed.size()));
    return 0;
}

} // namespace

int main() {
    int errors = 0;
    errors += run_case("rgb_gradient_64", 64, 64, 3);
    errors += run_case("rgba_gradient_64", 64, 64, 4);

    if (errors == 0) {
        std::printf("\nALL TESTS PASSED\n");
    }
    return errors == 0 ? 0 : 1;
}
