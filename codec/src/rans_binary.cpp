#include "rans_binary.hpp"

#include "rans_internal.hpp"

#include <algorithm>
#include <cstdint>
#include <vector>

namespace radiance_codec::rans {
namespace {

struct Counts {
    std::uint32_t zeros = 0;
    std::uint32_t ones = 0;
};

std::uint32_t freq0_from_counts(const Counts& counts) noexcept {
    // KT-like symmetric half-count smoothing:
    //   p0 = (zeros + 1/2) / (zeros + ones + 1)
    // expressed without floating point:
    //   p0 = (2*zeros + 1) / (2*(zeros+ones) + 2)
    const std::uint64_t numerator = std::uint64_t(2) * counts.zeros + 1;
    const std::uint64_t denominator =
        std::uint64_t(2) * (counts.zeros + counts.ones) + 2;
    std::uint32_t freq0 = static_cast<std::uint32_t>(
        (numerator * PROB_SCALE + denominator / 2) / denominator);
    freq0 = std::clamp<std::uint32_t>(freq0, 1, PROB_SCALE - 1);
    return freq0;
}

void update(Counts& counts, std::uint8_t bit) noexcept {
    if (bit) {
        ++counts.ones;
    } else {
        ++counts.zeros;
    }
}

} // namespace

std::vector<std::uint8_t> encode_adaptive_binary(
    std::span<const std::uint8_t> bits,
    std::span<const std::uint16_t> contexts,
    std::uint32_t context_count) {

    if (bits.size() != contexts.size() || context_count == 0) {
        return {};
    }

    std::vector<Counts> counts(context_count);
    std::vector<std::uint16_t> freq0_by_symbol(bits.size());
    for (std::size_t i = 0; i < bits.size(); ++i) {
        const std::uint16_t context = contexts[i];
        if (context >= context_count || bits[i] > 1) {
            return {};
        }
        Counts& c = counts[context];
        freq0_by_symbol[i] = static_cast<std::uint16_t>(freq0_from_counts(c));
        update(c, bits[i]);
    }

    std::vector<std::uint8_t> buffer(bits.size() * 4 + 32);
    std::uint8_t* end = buffer.data() + buffer.size();
    std::uint8_t* write_ptr = end;
    std::uint32_t state = RANS_L;

    for (std::size_t i = bits.size(); i-- > 0;) {
        const std::uint32_t freq0 = freq0_by_symbol[i];
        const std::uint8_t bit = bits[i];
        const std::uint32_t cum = bit ? freq0 : 0;
        const std::uint32_t freq = bit ? (PROB_SCALE - freq0) : freq0;
        encode_renorm_and_put(state, write_ptr, cum, freq);
    }
    encode_flush(state, write_ptr);
    return std::vector<std::uint8_t>(write_ptr, end);
}

bool decode_adaptive_binary(
    std::span<const std::uint8_t> payload,
    std::span<const std::uint16_t> contexts,
    std::uint32_t context_count,
    std::vector<std::uint8_t>& bits) {

    if (payload.size() < 4 || context_count == 0) {
        return false;
    }

    bits.assign(contexts.size(), 0);
    std::vector<Counts> counts(context_count);
    const std::uint8_t* read_ptr = payload.data();
    const std::uint8_t* read_end = payload.data() + payload.size();
    std::uint32_t state = decode_init(read_ptr);

    for (std::size_t i = 0; i < contexts.size(); ++i) {
        const std::uint16_t context = contexts[i];
        if (context >= context_count) {
            return false;
        }
        Counts& c = counts[context];
        const std::uint32_t freq0 = freq0_from_counts(c);
        const std::uint32_t slot = decode_get_slot(state);
        const std::uint8_t bit = slot >= freq0 ? 1 : 0;
        const std::uint32_t cum = bit ? freq0 : 0;
        const std::uint32_t freq = bit ? (PROB_SCALE - freq0) : freq0;
        bits[i] = bit;
        decode_advance(state, read_ptr, cum, freq);
        if (read_ptr > read_end) {
            return false;
        }
        update(c, bit);
    }
    return true;
}

} // namespace radiance_codec::rans
