// Metal microbenchmark kernels for keccak-f[1600].
//
// Two register-resident implementations of the permutation:
//
//  - u64:         lanes as `ulong`; the compiler lowers 64-bit ops onto the
//                 32-bit ALUs. Simplest possible port.
//  - interleaved: bit-interleaved lanes as `uint2` (even bits of the 64-bit
//                 lane in .x, odd bits in .y), so every 64-bit rotation
//                 becomes two independent 32-bit rotations.
//
// The state lives in named local variables (a0..a24), never in an indexable
// array, so it cannot be demoted to thread (scratch) memory by a failed
// SROA pass — register allocation is left entirely to the compiler.
//
// `bench_*` kernels chain ITERS dependent permutations per thread (modelling
// the miner, where stage-2 keccak depends on stage-1) and write an
// xor-reduction of the final state so nothing is dead-code-eliminated.
// `verify_*` kernels permute an explicit state buffer so the host can
// cross-check the kernels against a scalar reference implementation.

#include <metal_stdlib>
using namespace metal;

constant uint ITERS [[function_constant(0)]];

// ---------------------------------------------------------------------------
// Shared tables
// ---------------------------------------------------------------------------

// Keccak-f[1600] round constants.
constant ulong RC[24] = {
    0x0000000000000001ul, 0x0000000000008082ul, 0x800000000000808aul,
    0x8000000080008000ul, 0x000000000000808bul, 0x0000000080000001ul,
    0x8000000080008081ul, 0x8000000000008009ul, 0x000000000000008aul,
    0x0000000000000088ul, 0x0000000080008009ul, 0x000000008000000aul,
    0x000000008000808bul, 0x800000000000008bul, 0x8000000000008089ul,
    0x8000000000008003ul, 0x8000000000008002ul, 0x8000000000000080ul,
    0x000000000000800aul, 0x800000008000000aul, 0x8000000080008081ul,
    0x8000000000008080ul, 0x0000000080000001ul, 0x8000000080008008ul,
};

// The same constants in bit-interleaved form: .x holds the even bits of the
// 64-bit constant, .y the odd bits (generated offline, cross-checked by the
// verify kernels against the scalar reference).
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

// ---------------------------------------------------------------------------
// Variant 1: ulong lanes
// ---------------------------------------------------------------------------

// Only ever used with 1 <= n <= 63.
#define ROTL64(x, n) (((x) << (n)) | ((x) >> (64 - (n))))

// One keccak-f round over lanes a0..a24 (theta, rho+pi fused with the theta
// add, chi, iota). Lane numbering, rho offsets and the pi mapping match the
// NEON miner (src/bin/create3-miner-neon.rs): index = x + 5*y.
#define KROUND_U64(rc) {                    \
    ulong c0 = a0 ^ a5 ^ a10 ^ a15 ^ a20;   \
    ulong c1 = a1 ^ a6 ^ a11 ^ a16 ^ a21;   \
    ulong c2 = a2 ^ a7 ^ a12 ^ a17 ^ a22;   \
    ulong c3 = a3 ^ a8 ^ a13 ^ a18 ^ a23;   \
    ulong c4 = a4 ^ a9 ^ a14 ^ a19 ^ a24;   \
    ulong d0 = c4 ^ ROTL64(c1, 1);          \
    ulong d1 = c0 ^ ROTL64(c2, 1);          \
    ulong d2 = c1 ^ ROTL64(c3, 1);          \
    ulong d3 = c2 ^ ROTL64(c4, 1);          \
    ulong d4 = c3 ^ ROTL64(c0, 1);          \
    ulong b0  = a0 ^ d0;                    \
    ulong b10 = ROTL64(a1 ^ d1, 1);         \
    ulong b20 = ROTL64(a2 ^ d2, 62);        \
    ulong b5  = ROTL64(a3 ^ d3, 28);        \
    ulong b15 = ROTL64(a4 ^ d4, 27);        \
    ulong b16 = ROTL64(a5 ^ d0, 36);        \
    ulong b1  = ROTL64(a6 ^ d1, 44);        \
    ulong b11 = ROTL64(a7 ^ d2, 6);         \
    ulong b21 = ROTL64(a8 ^ d3, 55);        \
    ulong b6  = ROTL64(a9 ^ d4, 20);        \
    ulong b7  = ROTL64(a10 ^ d0, 3);        \
    ulong b17 = ROTL64(a11 ^ d1, 10);       \
    ulong b2  = ROTL64(a12 ^ d2, 43);       \
    ulong b12 = ROTL64(a13 ^ d3, 25);       \
    ulong b22 = ROTL64(a14 ^ d4, 39);       \
    ulong b23 = ROTL64(a15 ^ d0, 41);       \
    ulong b8  = ROTL64(a16 ^ d1, 45);       \
    ulong b18 = ROTL64(a17 ^ d2, 15);       \
    ulong b3  = ROTL64(a18 ^ d3, 21);       \
    ulong b13 = ROTL64(a19 ^ d4, 8);        \
    ulong b14 = ROTL64(a20 ^ d0, 18);       \
    ulong b24 = ROTL64(a21 ^ d1, 2);        \
    ulong b9  = ROTL64(a22 ^ d2, 61);       \
    ulong b19 = ROTL64(a23 ^ d3, 56);       \
    ulong b4  = ROTL64(a24 ^ d4, 14);       \
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

#define KECCAKF_U64() for (uint r = 0; r < 24; ++r) { KROUND_U64(RC[r]); }

kernel void bench_u64(device ulong* out [[buffer(0)]],
                      uint tid [[thread_position_in_grid]])
{
    ulong s = ((ulong)tid + 1ul) * 0x9e3779b97f4a7c15ul;
    ulong a0  = s ^ RC[0];
    ulong a1  = s ^ RC[1];
    ulong a2  = s ^ RC[2];
    ulong a3  = s ^ RC[3];
    ulong a4  = s ^ RC[4];
    ulong a5  = s ^ RC[5];
    ulong a6  = s ^ RC[6];
    ulong a7  = s ^ RC[7];
    ulong a8  = s ^ RC[8];
    ulong a9  = s ^ RC[9];
    ulong a10 = s ^ RC[10];
    ulong a11 = s ^ RC[11];
    ulong a12 = s ^ RC[12];
    ulong a13 = s ^ RC[13];
    ulong a14 = s ^ RC[14];
    ulong a15 = s ^ RC[15];
    ulong a16 = s ^ RC[16];
    ulong a17 = s ^ RC[17];
    ulong a18 = s ^ RC[18];
    ulong a19 = s ^ RC[19];
    ulong a20 = s ^ RC[20];
    ulong a21 = s ^ RC[21];
    ulong a22 = s ^ RC[22];
    ulong a23 = s ^ RC[23];
    ulong a24 = s * 0x2545f4914f6cdd1dul;

    for (uint it = 0; it < ITERS; ++it) {
        KECCAKF_U64();
    }

    out[tid] = a0 ^ a1 ^ a2 ^ a3 ^ a4 ^ a5 ^ a6 ^ a7 ^ a8 ^ a9 ^ a10 ^ a11
             ^ a12 ^ a13 ^ a14 ^ a15 ^ a16 ^ a17 ^ a18 ^ a19 ^ a20 ^ a21
             ^ a22 ^ a23 ^ a24;
}

kernel void verify_u64(device ulong* st [[buffer(0)]],
                       uint tid [[thread_position_in_grid]])
{
    if (tid != 0) {
        return;
    }
    ulong a0  = st[0];
    ulong a1  = st[1];
    ulong a2  = st[2];
    ulong a3  = st[3];
    ulong a4  = st[4];
    ulong a5  = st[5];
    ulong a6  = st[6];
    ulong a7  = st[7];
    ulong a8  = st[8];
    ulong a9  = st[9];
    ulong a10 = st[10];
    ulong a11 = st[11];
    ulong a12 = st[12];
    ulong a13 = st[13];
    ulong a14 = st[14];
    ulong a15 = st[15];
    ulong a16 = st[16];
    ulong a17 = st[17];
    ulong a18 = st[18];
    ulong a19 = st[19];
    ulong a20 = st[20];
    ulong a21 = st[21];
    ulong a22 = st[22];
    ulong a23 = st[23];
    ulong a24 = st[24];

    for (uint it = 0; it < ITERS; ++it) {
        KECCAKF_U64();
    }

    st[0]  = a0;
    st[1]  = a1;
    st[2]  = a2;
    st[3]  = a3;
    st[4]  = a4;
    st[5]  = a5;
    st[6]  = a6;
    st[7]  = a7;
    st[8]  = a8;
    st[9]  = a9;
    st[10] = a10;
    st[11] = a11;
    st[12] = a12;
    st[13] = a13;
    st[14] = a14;
    st[15] = a15;
    st[16] = a16;
    st[17] = a17;
    st[18] = a18;
    st[19] = a19;
    st[20] = a20;
    st[21] = a21;
    st[22] = a22;
    st[23] = a23;
    st[24] = a24;
}

// ---------------------------------------------------------------------------
// Variant 2: bit-interleaved uint2 lanes
// ---------------------------------------------------------------------------

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

kernel void bench_interleaved(device uint* out [[buffer(0)]],
                              uint tid [[thread_position_in_grid]])
{
    uint2 s = uint2((tid + 1u) * 0x9e3779b9u, (tid + 1u) * 0x85ebca6bu);
    uint2 a0  = s ^ RCI[0];
    uint2 a1  = s ^ RCI[1];
    uint2 a2  = s ^ RCI[2];
    uint2 a3  = s ^ RCI[3];
    uint2 a4  = s ^ RCI[4];
    uint2 a5  = s ^ RCI[5];
    uint2 a6  = s ^ RCI[6];
    uint2 a7  = s ^ RCI[7];
    uint2 a8  = s ^ RCI[8];
    uint2 a9  = s ^ RCI[9];
    uint2 a10 = s ^ RCI[10];
    uint2 a11 = s ^ RCI[11];
    uint2 a12 = s ^ RCI[12];
    uint2 a13 = s ^ RCI[13];
    uint2 a14 = s ^ RCI[14];
    uint2 a15 = s ^ RCI[15];
    uint2 a16 = s ^ RCI[16];
    uint2 a17 = s ^ RCI[17];
    uint2 a18 = s ^ RCI[18];
    uint2 a19 = s ^ RCI[19];
    uint2 a20 = s ^ RCI[20];
    uint2 a21 = s ^ RCI[21];
    uint2 a22 = s ^ RCI[22];
    uint2 a23 = s ^ RCI[23];
    uint2 a24 = s ^ uint2(0x5bd1e995u, 0xcc9e2d51u);

    for (uint it = 0; it < ITERS; ++it) {
        KECCAKF_IL();
    }

    uint2 acc = a0 ^ a1 ^ a2 ^ a3 ^ a4 ^ a5 ^ a6 ^ a7 ^ a8 ^ a9 ^ a10 ^ a11
              ^ a12 ^ a13 ^ a14 ^ a15 ^ a16 ^ a17 ^ a18 ^ a19 ^ a20 ^ a21
              ^ a22 ^ a23 ^ a24;
    out[tid] = acc.x ^ acc.y;
}

kernel void verify_interleaved(device uint2* st [[buffer(0)]],
                               uint tid [[thread_position_in_grid]])
{
    if (tid != 0) {
        return;
    }
    uint2 a0  = st[0];
    uint2 a1  = st[1];
    uint2 a2  = st[2];
    uint2 a3  = st[3];
    uint2 a4  = st[4];
    uint2 a5  = st[5];
    uint2 a6  = st[6];
    uint2 a7  = st[7];
    uint2 a8  = st[8];
    uint2 a9  = st[9];
    uint2 a10 = st[10];
    uint2 a11 = st[11];
    uint2 a12 = st[12];
    uint2 a13 = st[13];
    uint2 a14 = st[14];
    uint2 a15 = st[15];
    uint2 a16 = st[16];
    uint2 a17 = st[17];
    uint2 a18 = st[18];
    uint2 a19 = st[19];
    uint2 a20 = st[20];
    uint2 a21 = st[21];
    uint2 a22 = st[22];
    uint2 a23 = st[23];
    uint2 a24 = st[24];

    for (uint it = 0; it < ITERS; ++it) {
        KECCAKF_IL();
    }

    st[0]  = a0;
    st[1]  = a1;
    st[2]  = a2;
    st[3]  = a3;
    st[4]  = a4;
    st[5]  = a5;
    st[6]  = a6;
    st[7]  = a7;
    st[8]  = a8;
    st[9]  = a9;
    st[10] = a10;
    st[11] = a11;
    st[12] = a12;
    st[13] = a13;
    st[14] = a14;
    st[15] = a15;
    st[16] = a16;
    st[17] = a17;
    st[18] = a18;
    st[19] = a19;
    st[20] = a20;
    st[21] = a21;
    st[22] = a22;
    st[23] = a23;
    st[24] = a24;
}
