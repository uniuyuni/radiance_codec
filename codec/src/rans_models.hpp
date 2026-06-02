// Frequency models for the rANS entropy coder.
//
// All models normalize a histogram of 8-bit symbols to sum exactly
// PROB_SCALE (= 2^14 = 16384), then derive per-symbol (cum, freq) and
// a slot→symbol lookup table.

#pragma once

#include "rans_internal.hpp"

#include <array>
#include <cstdint>
#include <span>

namespace radiance_codec::rans {

// 256-symbol byte model. After build_*(), the freq/cum arrays describe
// a valid distribution summing to PROB_SCALE. The slot_to_sym table
// maps any slot in [0, PROB_SCALE) back to its symbol in O(1).
struct ByteModel {
    std::array<uint32_t, 256> freq{};
    std::array<uint32_t, 256> cum{};
    std::array<uint8_t,  PROB_SCALE> slot_to_sym{};

    // Initialize as uniform (freq[i] == PROB_SCALE/256 for all i).
    void build_uniform() noexcept;

    // Initialize from a histogram (256 entries). Empty buckets get a
    // minimum frequency of 1 to remain decodable. Renormalized so that
    // sum(freq) == PROB_SCALE.
    void build_from_histogram(const std::array<uint64_t, 256>& hist) noexcept;

    // After freq[] is populated, fill cum[] and slot_to_sym[].
    void finalize_lookup_tables() noexcept;
};

// Compute byte histogram over a span (single pass).
inline std::array<uint64_t, 256> compute_histogram(
    std::span<const uint8_t> data) noexcept {
    std::array<uint64_t, 256> hist{};
    for (uint8_t b : data) ++hist[b];
    return hist;
}

} // namespace radiance_codec::rans
