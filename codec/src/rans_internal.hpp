// Low-level rANS state management (header-only for inlining).
//
// Adapted from ryg_rans (Fabian Giesen, public domain).
// Reference: https://github.com/rygorous/ryg_rans
//
// 32-bit state, 8-bit renormalization. Encoder writes BACKWARD into a
// preallocated buffer; decoder reads FORWARD. This makes the symbol
// processing order:
//   encode: iterate symbols last-to-first
//   decode: iterate symbols first-to-last
//
// Probability scale: PROB_SCALE = 2^PROB_BITS. All frequencies (freq, cum)
// are in this scale.

#pragma once

#include <cstdint>
#include <cstring>

namespace radiance_codec::rans {

constexpr uint32_t PROB_BITS  = 14;
constexpr uint32_t PROB_SCALE = 1u << PROB_BITS;          // 16384
constexpr uint32_t RANS_L     = 1u << 23;                 // lower bound
constexpr uint32_t RANS_B     = 256;                      // I/O base

// ─── Encoder ──────────────────────────────────────────────────────
//
// Usage:
//   uint32_t state = RANS_L;
//   for (each symbol in REVERSE) put(state, &write_ptr, cum, freq);
//   flush(state, &write_ptr);
//   bytes are at [write_ptr .. buffer_end)
//
// write_ptr is decremented by put/flush. The output bytes are at the
// END of the preallocated buffer; the caller knows the final length
// from (buffer_end - write_ptr).

inline void encode_renorm_and_put(
    uint32_t& state,
    uint8_t*& write_ptr,
    uint32_t  cum,
    uint32_t  freq) {

    // Renormalize state so that (state / freq) fits in (RANS_L / PROB_SCALE) * B
    uint32_t x_max = ((RANS_L >> PROB_BITS) << 8) * freq;
    while (state >= x_max) {
        *(--write_ptr) = static_cast<uint8_t>(state & 0xFF);
        state >>= 8;
    }
    // Bias state by cumulative + offset within frequency interval
    state = ((state / freq) << PROB_BITS) + (state % freq) + cum;
}

inline void encode_flush(uint32_t state, uint8_t*& write_ptr) {
    // Write 4 bytes of final state (little-endian)
    write_ptr -= 4;
    write_ptr[0] = static_cast<uint8_t>(state >>  0);
    write_ptr[1] = static_cast<uint8_t>(state >>  8);
    write_ptr[2] = static_cast<uint8_t>(state >> 16);
    write_ptr[3] = static_cast<uint8_t>(state >> 24);
}


// ─── Decoder ──────────────────────────────────────────────────────
//
// Usage:
//   uint32_t state = decode_init(&read_ptr);
//   for (each symbol in FORWARD order) {
//       uint32_t slot = decode_get_slot(state);
//       // model: slot -> symbol -> (cum, freq)
//       decode_advance(state, &read_ptr, cum, freq);
//       emit symbol;
//   }

inline uint32_t decode_init(const uint8_t*& read_ptr) {
    uint32_t state =
          (static_cast<uint32_t>(read_ptr[0]) <<  0)
        | (static_cast<uint32_t>(read_ptr[1]) <<  8)
        | (static_cast<uint32_t>(read_ptr[2]) << 16)
        | (static_cast<uint32_t>(read_ptr[3]) << 24);
    read_ptr += 4;
    return state;
}

inline uint32_t decode_get_slot(uint32_t state) {
    return state & (PROB_SCALE - 1);
}

inline void decode_advance(
    uint32_t& state,
    const uint8_t*& read_ptr,
    uint32_t cum,
    uint32_t freq) {

    // Inverse of encode_renorm_and_put
    state = freq * (state >> PROB_BITS) + (state & (PROB_SCALE - 1)) - cum;
    // Renormalize by reading bytes
    while (state < RANS_L) {
        state = (state << 8) | *read_ptr;
        ++read_ptr;
    }
}

} // namespace radiance_codec::rans
