#include "rans_binary.hpp"

#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

namespace {

int fail(const char* what) {
    std::fprintf(stderr, "TEST FAIL: %s\n", what);
    return 1;
}

std::vector<std::uint16_t> make_contexts(std::size_t n) {
    std::vector<std::uint16_t> contexts(n);
    std::uint16_t rolling = 0;
    for (std::size_t i = 0; i < n; ++i) {
        contexts[i] = rolling;
        rolling = static_cast<std::uint16_t>((rolling * 5 + i + 1) & 31);
    }
    return contexts;
}

int run_case(const char* label, const std::vector<std::uint8_t>& bits) {
    constexpr std::uint32_t context_count = 32;
    const auto contexts = make_contexts(bits.size());
    const auto payload = radiance_codec::rans::encode_adaptive_binary(
        bits, contexts, context_count);
    if (payload.empty()) {
        return fail("encode returned empty payload");
    }
    std::vector<std::uint8_t> decoded;
    if (!radiance_codec::rans::decode_adaptive_binary(
            payload, contexts, context_count, decoded)) {
        return fail("decode failed");
    }
    if (decoded != bits) {
        return fail("roundtrip mismatch");
    }
    const double raw_bytes = double(bits.size()) / 8.0;
    std::printf("  [%s] %.0f raw bytes -> %zu bytes (%.3fx)\n",
                label, raw_bytes, payload.size(),
                raw_bytes / double(payload.size()));
    return 0;
}

} // namespace

int main() {
    int errors = 0;

    {
        std::vector<std::uint8_t> bits(1 << 16);
        std::mt19937 rng(1);
        std::bernoulli_distribution dist(0.5);
        for (auto& bit : bits) bit = dist(rng) ? 1 : 0;
        errors += run_case("uniform", bits);
    }

    {
        std::vector<std::uint8_t> bits(1 << 16);
        std::mt19937 rng(2);
        std::bernoulli_distribution dist(0.03);
        for (auto& bit : bits) bit = dist(rng) ? 1 : 0;
        errors += run_case("sparse", bits);
    }

    {
        std::vector<std::uint8_t> bits(1 << 16);
        for (std::size_t i = 0; i < bits.size(); ++i) {
            bits[i] = ((i * 13) ^ (i >> 3)) & 1;
        }
        errors += run_case("structured", bits);
    }

    if (errors == 0) {
        std::printf("\nALL TESTS PASSED\n");
    }
    return errors == 0 ? 0 : 1;
}
