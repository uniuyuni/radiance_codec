#include "rans_models.hpp"

#include <algorithm>
#include <cstdint>

namespace radiance_codec::rans {

void ByteModel::build_uniform() noexcept {
    constexpr uint32_t per_symbol = PROB_SCALE / 256;  // == 64
    static_assert(per_symbol * 256 == PROB_SCALE);
    for (auto& f : freq) f = per_symbol;
    finalize_lookup_tables();
}

void ByteModel::build_from_histogram(
    const std::array<uint64_t, 256>& hist) noexcept {

    // Sum of histogram entries.
    uint64_t total = 0;
    for (auto h : hist) total += h;

    if (total == 0) {
        build_uniform();
        return;
    }

    // Initial proportional allocation (may round to 0 for tiny entries).
    // We track which symbols have nonzero histogram so we can guarantee
    // every "seen" symbol gets at least freq=1.
    uint64_t assigned = 0;
    for (uint32_t s = 0; s < 256; ++s) {
        if (hist[s] == 0) {
            freq[s] = 0;
            continue;
        }
        // Round to nearest, but enforce minimum of 1 for seen symbols.
        uint64_t f =
            (hist[s] * uint64_t(PROB_SCALE) + total / 2) / total;
        if (f == 0) f = 1;
        if (f > PROB_SCALE - 1) f = PROB_SCALE - 1;  // leave room for others
        freq[s] = static_cast<uint32_t>(f);
        assigned += f;
    }

    // Adjust to exactly PROB_SCALE by nudging the largest entries.
    while (assigned > PROB_SCALE) {
        // Find the largest freq and decrement it.
        uint32_t best = 0;
        for (uint32_t s = 1; s < 256; ++s) {
            if (freq[s] > freq[best]) best = s;
        }
        if (freq[best] <= 1) break;  // shouldn't happen, defensive
        --freq[best];
        --assigned;
    }
    while (assigned < PROB_SCALE) {
        // Add to the largest freq.
        uint32_t best = 0;
        for (uint32_t s = 1; s < 256; ++s) {
            if (freq[s] > freq[best]) best = s;
        }
        ++freq[best];
        ++assigned;
    }

    finalize_lookup_tables();
}

void ByteModel::finalize_lookup_tables() noexcept {
    uint32_t c = 0;
    for (uint32_t s = 0; s < 256; ++s) {
        cum[s] = c;
        const uint32_t f = freq[s];
        // Fill the slot table for symbols with non-zero frequency.
        for (uint32_t i = 0; i < f; ++i) {
            slot_to_sym[c + i] = static_cast<uint8_t>(s);
        }
        c += f;
    }
    // Post-condition: c == PROB_SCALE.
}

} // namespace radiance_codec::rans
