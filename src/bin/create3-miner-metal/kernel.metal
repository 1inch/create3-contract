// CREATE3 vanity-mining kernels for Apple GPUs.
//
// Each thread derives one candidate:
//
//   salt    = prefix[24] ++ counter (big-endian u64), counter = base + tid
//   proxy   = keccak256(0xff ++ factory ++ salt ++ code_hash)[12..]
//   address = keccak256(0xd6 0x94 ++ proxy ++ 0x01)[12..]
//
// keccak-f[1600] runs in bit-interleaved form (even bits of every 64-bit lane
// in .x, odd bits in .y), so a 64-bit rotation is two 32-bit rotations - the
// representation the bench/metal-keccak benchmark showed to be ~40% faster
// than plain ulong lanes on Apple GPUs. The state lives in named local
// variables so it can never be demoted to scratch memory.
//
// The host precomputes the stage-1 sponge template (17 rate words, already
// interleaved, with the counter bytes zeroed) plus plain words 5/6 for the
// counter splice, and the match pattern/mask over final-state words 1..3 in
// interleaved form (an equality-under-mask test survives any fixed bit
// permutation, so no deinterleave is needed to check a candidate). Nothing in
// the per-candidate path leaves the interleaved domain: the stage-2 input is
// spliced from the stage-1 digest words in place (stage2_word0..2).
//
// `mine` reports matching counters through an atomic hit buffer; the host
// re-derives and verifies every candidate on the CPU before accepting it.
// `debug_addresses` dumps every thread's final words 1..3 and exists for the
// GPU-vs-scalar cross-check tests only.

#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// keccak-f[1600], bit-interleaved (same tables as bench/metal-keccak)
// ---------------------------------------------------------------------------

// Round constants in bit-interleaved form: .x = even bits, .y = odd bits.
constant uint2 RCI[24] = {
    uint2(0x00000001u, 0x00000000u),
    uint2(0x00000000u, 0x00000089u),
    uint2(0x00000000u, 0x8000008bu),
    uint2(0x00000000u, 0x80008080u),
    uint2(0x00000001u, 0x0000008bu),
    uint2(0x00000001u, 0x00008000u),
    uint2(0x00000001u, 0x80008088u),
    uint2(0x00000001u, 0x80000082u),
    uint2(0x00000000u, 0x0000000bu),
    uint2(0x00000000u, 0x0000000au),
    uint2(0x00000001u, 0x00008082u),
    uint2(0x00000000u, 0x00008003u),
    uint2(0x00000001u, 0x0000808bu),
    uint2(0x00000001u, 0x8000000bu),
    uint2(0x00000001u, 0x8000008au),
    uint2(0x00000001u, 0x80000081u),
    uint2(0x00000000u, 0x80000081u),
    uint2(0x00000000u, 0x80000008u),
    uint2(0x00000000u, 0x00000083u),
    uint2(0x00000000u, 0x80008003u),
    uint2(0x00000001u, 0x80008088u),
    uint2(0x00000000u, 0x80000088u),
    uint2(0x00000001u, 0x00008000u),
    uint2(0x00000000u, 0x80008082u),
};

// rotl64 in bit-interleaved form. Even rotation 2m: both halves rotate by m.
// Odd rotation 2m+1: halves swap, the (new) even half rotates by m+1 and the
// odd half by m.
template <uint N>
static inline uint2 irotl(uint2 v)
{
    if ((N & 1u) == 0u) {
        return uint2(rotate(v.x, N / 2u), rotate(v.y, N / 2u));
    } else {
        return uint2(rotate(v.y, (N + 1u) / 2u), rotate(v.x, N / 2u));
    }
}

// One keccak-f round over lanes a0..a24 (theta, rho+pi fused with the theta
// add, chi, iota). Lane numbering, rho offsets and the pi mapping match the
// NEON miner (src/bin/create3-miner-neon.rs): index = x + 5*y.
#define KROUND_IL(rc) {                     \
    uint2 c0 = a0 ^ a5 ^ a10 ^ a15 ^ a20;   \
    uint2 c1 = a1 ^ a6 ^ a11 ^ a16 ^ a21;   \
    uint2 c2 = a2 ^ a7 ^ a12 ^ a17 ^ a22;   \
    uint2 c3 = a3 ^ a8 ^ a13 ^ a18 ^ a23;   \
    uint2 c4 = a4 ^ a9 ^ a14 ^ a19 ^ a24;   \
    uint2 d0 = c4 ^ irotl<1>(c1);           \
    uint2 d1 = c0 ^ irotl<1>(c2);           \
    uint2 d2 = c1 ^ irotl<1>(c3);           \
    uint2 d3 = c2 ^ irotl<1>(c4);           \
    uint2 d4 = c3 ^ irotl<1>(c0);           \
    uint2 b0  = a0 ^ d0;                    \
    uint2 b10 = irotl<1>(a1 ^ d1);          \
    uint2 b20 = irotl<62>(a2 ^ d2);         \
    uint2 b5  = irotl<28>(a3 ^ d3);         \
    uint2 b15 = irotl<27>(a4 ^ d4);         \
    uint2 b16 = irotl<36>(a5 ^ d0);         \
    uint2 b1  = irotl<44>(a6 ^ d1);         \
    uint2 b11 = irotl<6>(a7 ^ d2);          \
    uint2 b21 = irotl<55>(a8 ^ d3);         \
    uint2 b6  = irotl<20>(a9 ^ d4);         \
    uint2 b7  = irotl<3>(a10 ^ d0);         \
    uint2 b17 = irotl<10>(a11 ^ d1);        \
    uint2 b2  = irotl<43>(a12 ^ d2);        \
    uint2 b12 = irotl<25>(a13 ^ d3);        \
    uint2 b22 = irotl<39>(a14 ^ d4);        \
    uint2 b23 = irotl<41>(a15 ^ d0);        \
    uint2 b8  = irotl<45>(a16 ^ d1);        \
    uint2 b18 = irotl<15>(a17 ^ d2);        \
    uint2 b3  = irotl<21>(a18 ^ d3);        \
    uint2 b13 = irotl<8>(a19 ^ d4);         \
    uint2 b14 = irotl<18>(a20 ^ d0);        \
    uint2 b24 = irotl<2>(a21 ^ d1);         \
    uint2 b9  = irotl<61>(a22 ^ d2);        \
    uint2 b19 = irotl<56>(a23 ^ d3);        \
    uint2 b4  = irotl<14>(a24 ^ d4);        \
    a0  = b0  ^ (~b1  & b2) ^ (rc);         \
    a1  = b1  ^ (~b2  & b3);                \
    a2  = b2  ^ (~b3  & b4);                \
    a3  = b3  ^ (~b4  & b0);                \
    a4  = b4  ^ (~b0  & b1);                \
    a5  = b5  ^ (~b6  & b7);                \
    a6  = b6  ^ (~b7  & b8);                \
    a7  = b7  ^ (~b8  & b9);                \
    a8  = b8  ^ (~b9  & b5);                \
    a9  = b9  ^ (~b5  & b6);                \
    a10 = b10 ^ (~b11 & b12);               \
    a11 = b11 ^ (~b12 & b13);               \
    a12 = b12 ^ (~b13 & b14);               \
    a13 = b13 ^ (~b14 & b10);               \
    a14 = b14 ^ (~b10 & b11);               \
    a15 = b15 ^ (~b16 & b17);               \
    a16 = b16 ^ (~b17 & b18);               \
    a17 = b17 ^ (~b18 & b19);               \
    a18 = b18 ^ (~b19 & b15);               \
    a19 = b19 ^ (~b15 & b16);               \
    a20 = b20 ^ (~b21 & b22);               \
    a21 = b21 ^ (~b22 & b23);               \
    a22 = b22 ^ (~b23 & b24);               \
    a23 = b23 ^ (~b24 & b20);               \
    a24 = b24 ^ (~b20 & b21);               \
}

#define KECCAKF_IL() for (uint r = 0; r < 24; ++r) { KROUND_IL(RCI[r]); }

// ---------------------------------------------------------------------------
// Bit-interleaving helpers
// ---------------------------------------------------------------------------

static inline ulong bswap64(ulong x)
{
    x = ((x & 0x00ff00ff00ff00fful) << 8) | ((x >> 8) & 0x00ff00ff00ff00fful);
    x = ((x & 0x0000ffff0000fffful) << 16) | ((x >> 16) & 0x0000ffff0000fffful);
    return (x << 32) | (x >> 32);
}

// Gathers the even-position bits of x into the low 32 bits (Morton compact).
static inline uint even_bits(ulong x)
{
    x &= 0x5555555555555555ul;
    x = (x | (x >> 1)) & 0x3333333333333333ul;
    x = (x | (x >> 2)) & 0x0f0f0f0f0f0f0f0ful;
    x = (x | (x >> 4)) & 0x00ff00ff00ff00fful;
    x = (x | (x >> 8)) & 0x0000ffff0000fffful;
    x = (x | (x >> 16)) & 0x00000000fffffffful;
    return (uint)x;
}

static inline uint2 interleave64(ulong x)
{
    return uint2(even_bits(x), even_bits(x >> 1));
}

// Spreads 32 bits over the even positions of a 64-bit word (Morton spread).
static inline ulong spread_bits(uint v)
{
    ulong x = v;
    x = (x | (x << 16)) & 0x0000ffff0000fffful;
    x = (x | (x << 8)) & 0x00ff00ff00ff00fful;
    x = (x | (x << 4)) & 0x0f0f0f0f0f0f0f0ful;
    x = (x | (x << 2)) & 0x3333333333333333ul;
    x = (x | (x << 1)) & 0x5555555555555555ul;
    return x;
}

static inline ulong deinterleave64(uint2 v)
{
    return spread_bits(v.x) | (spread_bits(v.y) << 1);
}

// ---------------------------------------------------------------------------
// Stage-2 input assembly, in the interleaved domain
// ---------------------------------------------------------------------------

// Interleaved constants of the stage-2 frame: the leading `0xd6 0x94` RLP
// header, and the `0x01` nonce plus `0x01` sponge terminator at input bytes
// 22..23. These are il(0x94d6) and il(0x0101 << 48).
constant uint2 S2_HEAD = uint2(0x0000006eu, 0x00000089u);
constant uint2 S2_TAIL = uint2(0x11000000u, 0x00000000u);

// The three nonzero rate words of stage 2 (`0xd6 0x94 ++ proxy[20] ++ 0x01`,
// padded), built straight from the interleaved stage-1 digest words h1..h3 -
// the proxy address is hash bytes 12..31, i.e. the high half of word 1 plus
// words 2 and 3. In the plain 64-bit domain (stage2_words() in the NEON miner):
//
//   s0 = 0x94d6 | ((h1 >> 32) << 16) | ((h2 & 0xffff) << 48)
//   s1 = (h2 >> 16) | ((h3 & 0xffff) << 48)
//   s2 = (h3 >> 16) | (0x0101 << 48)
//
// Every shift here is by an even number of bits, and a shift by 2k is a shift
// by k of both interleaved halves, so the whole splice maps over without
// leaving the interleaved domain. The `& 0xffff` masks drop out: the following
// `<< 48` discards everything above bit 15 anyway.
static inline uint2 stage2_word0(uint2 h1, uint2 h2)
{
    return uint2(S2_HEAD.x | ((h1.x >> 16) << 8) | (h2.x << 24),
                 S2_HEAD.y | ((h1.y >> 16) << 8) | (h2.y << 24));
}

static inline uint2 stage2_word1(uint2 h2, uint2 h3)
{
    return uint2((h2.x >> 8) | (h3.x << 24), (h2.y >> 8) | (h3.y << 24));
}

static inline uint2 stage2_word2(uint2 h3)
{
    return uint2((h3.x >> 8) | S2_TAIL.x, (h3.y >> 8) | S2_TAIL.y);
}

// ---------------------------------------------------------------------------
// Mining kernels
// ---------------------------------------------------------------------------

constant uint MAX_HITS = 16;

// Layout mirrored by MineParams in src/bin/create3-miner-metal/main.rs.
struct MineParams {
    // Stage-1 rate words (interleaved), counter bytes zeroed; [5] and [6] are
    // placeholders overridden per thread.
    uint2 tmpl[17];
    // Match pattern/mask over final-state words 1..3, interleaved.
    uint2 pattern[3];
    uint2 mask[3];
    // Plain stage-1 words 5/6 with the counter bytes zeroed. The 8-byte
    // big-endian counter occupies input bytes 45..53, i.e. the top 3 bytes of
    // word 5 and the low 5 bytes of word 6 (see counter_words() in the NEON
    // miner).
    ulong w5_base;
    ulong w6_base;
    ulong base_counter;
};

struct HitBuffer {
    atomic_uint count;
    uint pad;
    ulong counters[MAX_HITS];
};

// Derives the candidate for counter `c` (in scope) from the template in `P`
// (in scope), leaving the final keccak state words 1..3 - which carry address
// bytes - in a1..a3 (interleaved).
#define MINE_CORE()                                                     \
    ulong cr = bswap64(c);                                              \
    ulong w5 = P.w5_base | ((cr & 0x0000000000fffffful) << 40);         \
    ulong w6 = P.w6_base | (cr >> 24);                                  \
    uint2 a0  = P.tmpl[0];                                              \
    uint2 a1  = P.tmpl[1];                                              \
    uint2 a2  = P.tmpl[2];                                              \
    uint2 a3  = P.tmpl[3];                                              \
    uint2 a4  = P.tmpl[4];                                              \
    uint2 a5  = interleave64(w5);                                       \
    uint2 a6  = interleave64(w6);                                       \
    uint2 a7  = P.tmpl[7];                                              \
    uint2 a8  = P.tmpl[8];                                              \
    uint2 a9  = P.tmpl[9];                                              \
    uint2 a10 = P.tmpl[10];                                             \
    uint2 a11 = P.tmpl[11];                                             \
    uint2 a12 = P.tmpl[12];                                             \
    uint2 a13 = P.tmpl[13];                                             \
    uint2 a14 = P.tmpl[14];                                             \
    uint2 a15 = P.tmpl[15];                                             \
    uint2 a16 = P.tmpl[16];                                             \
    uint2 a17 = uint2(0u);                                              \
    uint2 a18 = uint2(0u);                                              \
    uint2 a19 = uint2(0u);                                              \
    uint2 a20 = uint2(0u);                                              \
    uint2 a21 = uint2(0u);                                              \
    uint2 a22 = uint2(0u);                                              \
    uint2 a23 = uint2(0u);                                              \
    uint2 a24 = uint2(0u);                                              \
    KECCAKF_IL();                                                       \
    /* stage 2 (see stage2_word0..2): assembled from the stage-1 digest    */ \
    /* words without a round trip through the plain 64-bit domain.         */ \
    uint2 s0 = stage2_word0(a1, a2);                                    \
    uint2 s1 = stage2_word1(a2, a3);                                    \
    uint2 s2 = stage2_word2(a3);                                        \
    a0 = s0;                                                            \
    a1 = s1;                                                            \
    a2 = s2;                                                            \
    a3 = uint2(0u);                                                     \
    a4 = uint2(0u);                                                     \
    a5 = uint2(0u);                                                     \
    a6 = uint2(0u);                                                     \
    a7 = uint2(0u);                                                     \
    a8 = uint2(0u);                                                     \
    a9 = uint2(0u);                                                     \
    a10 = uint2(0u);                                                    \
    a11 = uint2(0u);                                                    \
    a12 = uint2(0u);                                                    \
    a13 = uint2(0u);                                                    \
    a14 = uint2(0u);                                                    \
    a15 = uint2(0u);                                                    \
    /* 0x80 block pad at byte 135: bit 63 of word 16 = odd bit 31 */    \
    a16 = uint2(0x00000000u, 0x80000000u);                              \
    a17 = uint2(0u);                                                    \
    a18 = uint2(0u);                                                    \
    a19 = uint2(0u);                                                    \
    a20 = uint2(0u);                                                    \
    a21 = uint2(0u);                                                    \
    a22 = uint2(0u);                                                    \
    a23 = uint2(0u);                                                    \
    a24 = uint2(0u);                                                    \
    KECCAKF_IL();

kernel void mine(constant MineParams& P [[buffer(0)]],
                 device HitBuffer& hits [[buffer(1)]],
                 uint tid [[thread_position_in_grid]])
{
    ulong c = P.base_counter + (ulong)tid;
    MINE_CORE();

    // Equality under mask is invariant under the bit interleave, so the
    // address bytes are checked without leaving the interleaved domain.
    uint2 bad = ((a1 ^ P.pattern[0]) & P.mask[0])
              | ((a2 ^ P.pattern[1]) & P.mask[1])
              | ((a3 ^ P.pattern[2]) & P.mask[2]);
    if ((bad.x | bad.y) == 0u) {
        uint slot = atomic_fetch_add_explicit(&hits.count, 1u, memory_order_relaxed);
        if (slot < MAX_HITS) {
            hits.counters[slot] = c;
        }
    }
}

// Test-only kernel: dumps final-state words 1..3 (plain u64) per thread so
// the host can cross-check GPU derivation against the scalar reference.
kernel void debug_addresses(constant MineParams& P [[buffer(0)]],
                            device ulong* out [[buffer(1)]],
                            uint tid [[thread_position_in_grid]])
{
    ulong c = P.base_counter + (ulong)tid;
    MINE_CORE();

    out[3 * tid + 0] = deinterleave64(a1);
    out[3 * tid + 1] = deinterleave64(a2);
    out[3 * tid + 2] = deinterleave64(a3);
}
