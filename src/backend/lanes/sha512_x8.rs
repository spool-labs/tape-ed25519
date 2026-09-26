// Vendored from the solana-ed25519 lanes fork, BSD-3-Clause. See LICENSE.

//! Eight- and sixteen-message multibuffer SHA-512 in AVX-512 intrinsics.
//!
//! One message per 64-bit lane. Lanes of different lengths run to the longest
//! lane's block count with per-lane masked state updates, so a shorter lane
//! freezes after its final padded block. A SHA-512 round is a serial chain on
//! `a` and `e`, so two independent groups of eight run through the same round
//! bodies interleaved, which is what [`challenge_digests_x16`] is for.

#![allow(clippy::needless_range_loop)]
// Nightly and stable currently disagree on the requirement of unsafe blocks
// when intrinsics are called inside `#[target_feature]` functions.
#![allow(unused_unsafe)]

use core::arch::x86_64::{
    __m512i, _mm512_add_epi64, _mm512_load_si512, _mm512_loadu_si512, _mm512_mask_add_epi64,
    _mm512_ror_epi64, _mm512_set1_epi64, _mm512_shuffle_epi8, _mm512_shuffle_i64x2,
    _mm512_srli_epi64, _mm512_storeu_si512, _mm512_ternarylogic_epi64, _mm512_unpackhi_epi64,
    _mm512_unpacklo_epi64,
};
use core::mem::MaybeUninit;

use super::field_x8::LANES;

#[rustfmt::skip]
const K64: [u64; 80] = [
    0x428a2f98d728ae22, 0x7137449123ef65cd, 0xb5c0fbcfec4d3b2f, 0xe9b5dba58189dbbc,
    0x3956c25bf348b538, 0x59f111f1b605d019, 0x923f82a4af194f9b, 0xab1c5ed5da6d8118,
    0xd807aa98a3030242, 0x12835b0145706fbe, 0x243185be4ee4b28c, 0x550c7dc3d5ffb4e2,
    0x72be5d74f27b896f, 0x80deb1fe3b1696b1, 0x9bdc06a725c71235, 0xc19bf174cf692694,
    0xe49b69c19ef14ad2, 0xefbe4786384f25e3, 0x0fc19dc68b8cd5b5, 0x240ca1cc77ac9c65,
    0x2de92c6f592b0275, 0x4a7484aa6ea6e483, 0x5cb0a9dcbd41fbd4, 0x76f988da831153b5,
    0x983e5152ee66dfab, 0xa831c66d2db43210, 0xb00327c898fb213f, 0xbf597fc7beef0ee4,
    0xc6e00bf33da88fc2, 0xd5a79147930aa725, 0x06ca6351e003826f, 0x142929670a0e6e70,
    0x27b70a8546d22ffc, 0x2e1b21385c26c926, 0x4d2c6dfc5ac42aed, 0x53380d139d95b3df,
    0x650a73548baf63de, 0x766a0abb3c77b2a8, 0x81c2c92e47edaee6, 0x92722c851482353b,
    0xa2bfe8a14cf10364, 0xa81a664bbc423001, 0xc24b8b70d0f89791, 0xc76c51a30654be30,
    0xd192e819d6ef5218, 0xd69906245565a910, 0xf40e35855771202a, 0x106aa07032bbd1b8,
    0x19a4c116b8d2d0c8, 0x1e376c085141ab53, 0x2748774cdf8eeb99, 0x34b0bcb5e19b48a8,
    0x391c0cb3c5c95a63, 0x4ed8aa4ae3418acb, 0x5b9cca4f7763e373, 0x682e6ff3d6b2b8a3,
    0x748f82ee5defb2fc, 0x78a5636f43172f60, 0x84c87814a1f0ab72, 0x8cc702081a6439ec,
    0x90befffa23631e28, 0xa4506cebde82bde9, 0xbef9a3f7b2c67915, 0xc67178f2e372532b,
    0xca273eceea26619c, 0xd186b8c721c0c207, 0xeada7dd6cde0eb1e, 0xf57d4f7fee6ed178,
    0x06f067aa72176fba, 0x0a637dc5a2c898a6, 0x113f9804bef90dae, 0x1b710b35131c471b,
    0x28db77f523047d84, 0x32caab7b40c72493, 0x3c9ebe0a15c9bebc, 0x431d67c49c100d4c,
    0x4cc5d4becb3e42b6, 0x597f299cfc657e2a, 0x5fcb6fab3ad6faec, 0x6c44198c4a475817,
];

/// Cache-line aligned so every row is a legal aligned 512-bit load.
#[repr(align(64))]
struct KLanes([[u64; LANES]; 80]);

/// The round constants pre-broadcast across the lanes: a load instead of a
/// broadcast, both one uop, here so the compiler cannot decide otherwise.
static K64_X8: KLanes = {
    let mut rows = [[0u64; LANES]; 80];
    let mut t = 0;
    while t < 80 {
        let mut l = 0;
        while l < LANES {
            rows[t][l] = K64[t];
            l += 1;
        }
        t += 1;
    }
    KLanes(rows)
};

#[rustfmt::skip]
const H512: [u64; 8] = [
    0x6a09e667f3bcc908, 0xbb67ae8584caa73b, 0x3c6ef372fe94f82b, 0xa54ff53a5f1d36f1,
    0x510e527fade682d1, 0x9b05688c2b3e6c1f, 0x1f83d9abfb41bd6b, 0x5be0cd19137e2179,
];

/// The SHA-512 block size in bytes.
const BLOCK: usize = 128;

/// Blocks staged per lane: the `R || A` prefix block plus the one or two padding
/// blocks. Everything between them is read where it lies.
const STAGE_BLOCKS: usize = 3;

/// `VPSHUFB` control reversing each 64-bit group, applied per 128-bit lane.
#[rustfmt::skip]
const BSWAP64: [u8; 64] = [
    7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8,
    7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8,
    7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8,
    7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8,
];

/// Transpose eight 64-byte lane rows into eight per-word vectors, byte-swapped
/// into the big-endian values the schedule operates on.
#[target_feature(enable = "avx512f,avx512bw")]
#[inline]
unsafe fn transpose_words_x8(rows: &[__m512i; 8], bswap: __m512i) -> [__m512i; 8] {
    unsafe {
        let t0 = _mm512_unpacklo_epi64(rows[0], rows[1]);
        let t1 = _mm512_unpackhi_epi64(rows[0], rows[1]);
        let t2 = _mm512_unpacklo_epi64(rows[2], rows[3]);
        let t3 = _mm512_unpackhi_epi64(rows[2], rows[3]);
        let t4 = _mm512_unpacklo_epi64(rows[4], rows[5]);
        let t5 = _mm512_unpackhi_epi64(rows[4], rows[5]);
        let t6 = _mm512_unpacklo_epi64(rows[6], rows[7]);
        let t7 = _mm512_unpackhi_epi64(rows[6], rows[7]);

        let s0 = _mm512_shuffle_i64x2::<0x88>(t0, t2);
        let s1 = _mm512_shuffle_i64x2::<0x88>(t1, t3);
        let s2 = _mm512_shuffle_i64x2::<0xdd>(t0, t2);
        let s3 = _mm512_shuffle_i64x2::<0xdd>(t1, t3);
        let s4 = _mm512_shuffle_i64x2::<0x88>(t4, t6);
        let s5 = _mm512_shuffle_i64x2::<0x88>(t5, t7);
        let s6 = _mm512_shuffle_i64x2::<0xdd>(t4, t6);
        let s7 = _mm512_shuffle_i64x2::<0xdd>(t5, t7);

        [
            _mm512_shuffle_epi8(_mm512_shuffle_i64x2::<0x88>(s0, s4), bswap),
            _mm512_shuffle_epi8(_mm512_shuffle_i64x2::<0x88>(s1, s5), bswap),
            _mm512_shuffle_epi8(_mm512_shuffle_i64x2::<0x88>(s2, s6), bswap),
            _mm512_shuffle_epi8(_mm512_shuffle_i64x2::<0x88>(s3, s7), bswap),
            _mm512_shuffle_epi8(_mm512_shuffle_i64x2::<0xdd>(s0, s4), bswap),
            _mm512_shuffle_epi8(_mm512_shuffle_i64x2::<0xdd>(s1, s5), bswap),
            _mm512_shuffle_epi8(_mm512_shuffle_i64x2::<0xdd>(s2, s6), bswap),
            _mm512_shuffle_epi8(_mm512_shuffle_i64x2::<0xdd>(s3, s7), bswap),
        ]
    }
}

/// One group's staged blocks and per-lane block plan.
struct Staged {
    /// The `R || A` prefix block and the padded tail, per lane.
    stage: [[MaybeUninit<u8>; STAGE_BLOCKS * BLOCK]; LANES],
    /// Padded block count, per lane.
    blocks: [usize; LANES],
    /// Index of the block holding the `0x80` marker, per lane.
    pad_block: [usize; LANES],
    /// Stage slot that marker block sits in, per lane.
    pad_slot: [usize; LANES],
    /// The longest lane's block count.
    max_blocks: usize,
}

impl Staged {
    fn new() -> Staged {
        Staged {
            stage: [[MaybeUninit::uninit(); STAGE_BLOCKS * BLOCK]; LANES],
            blocks: [0; LANES],
            pad_block: [0; LANES],
            pad_slot: [0; LANES],
            max_blocks: 0,
        }
    }

    /// Stage `R || A || M` per lane: block 0 and the padded tail are copied,
    /// the middle of the message is hashed where it lies.
    fn fill(
        &mut self,
        r_bytes: &[[u8; 32]; LANES],
        a_bytes: &[[u8; 32]; LANES],
        msgs: &[&[u8]; LANES],
    ) {
        for l in 0..LANES {
            let msg = msgs[l];
            let len = 64 + msg.len();
            let padded = (len + 1 + 16).div_ceil(BLOCK) * BLOCK;
            self.blocks[l] = padded / BLOCK;
            self.max_blocks = self.max_blocks.max(self.blocks[l]);

            // The block holding the 0x80 marker, and where it is staged.
            let pb = len / BLOCK;
            self.pad_block[l] = pb;
            self.pad_slot[l] = usize::from(pb != 0);

            unsafe {
                let base = self.stage[l].as_mut_ptr().cast::<u8>();
                core::ptr::copy_nonoverlapping(r_bytes[l].as_ptr(), base, 32);
                core::ptr::copy_nonoverlapping(a_bytes[l].as_ptr(), base.add(32), 32);
                core::ptr::copy_nonoverlapping(
                    msg.as_ptr(),
                    base.add(64),
                    msg.len().min(BLOCK - 64),
                );

                let pad = base.add(self.pad_slot[l] * BLOCK);
                let used = len - pb * BLOCK;
                if pb != 0 {
                    let off = pb * BLOCK - 64;
                    core::ptr::copy_nonoverlapping(msg.as_ptr().add(off), pad, used);
                }
                *pad.add(used) = 0x80;
                core::ptr::write_bytes(pad.add(used + 1), 0, BLOCK - used - 1);

                // The 128-bit length lands in the final block, one past the
                // marker's block when the marker left no room.
                let last = base.add((self.pad_slot[l] + self.blocks[l] - 1 - pb) * BLOCK);
                if last != pad {
                    core::ptr::write_bytes(last, 0, BLOCK - 8);
                }
                core::ptr::copy_nonoverlapping(
                    ((len as u64) * 8).to_be_bytes().as_ptr(),
                    last.add(BLOCK - 8),
                    8,
                );
            }
        }
    }

    /// Block `b`'s source pointer per lane, and the mask of lanes still inside
    /// their stream. Spent lanes read `zero` and are masked out of the update.
    #[inline]
    fn block(&self, msgs: &[&[u8]; LANES], b: usize, zero: *const u8) -> ([*const u8; LANES], u8) {
        let mut active: u8 = 0;
        let mut block = [zero; LANES];
        for l in 0..LANES {
            if b >= self.blocks[l] {
                continue;
            }
            active |= 1 << l;
            let base = self.stage[l].as_ptr().cast::<u8>();
            block[l] = unsafe {
                if b == 0 {
                    base
                } else if b >= self.pad_block[l] {
                    base.add((self.pad_slot[l] + b - self.pad_block[l]) * BLOCK)
                } else {
                    msgs[l].as_ptr().add(b * BLOCK - 64)
                }
            };
        }
        (block, active)
    }
}

/// The initial SHA-512 state, one message per lane.
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn init_state() -> [__m512i; 8] {
    unsafe { core::array::from_fn(|i| _mm512_set1_epi64(H512[i] as i64)) }
}

/// The block's sixteen schedule words, transposed out of the lane rows.
#[target_feature(enable = "avx512f,avx512bw")]
#[inline]
unsafe fn load_schedule(block: &[*const u8; LANES], bswap: __m512i) -> [__m512i; 16] {
    unsafe {
        let mut w = [_mm512_set1_epi64(0); 16];
        for half in 0..2 {
            let mut rows = [_mm512_set1_epi64(0); LANES];
            for l in 0..LANES {
                rows[l] = _mm512_loadu_si512(block[l].add(half * 64) as *const _);
            }
            w[half * 8..half * 8 + 8].copy_from_slice(&transpose_words_x8(&rows, bswap));
        }
        w
    }
}

/// Extract per-lane big-endian digests.
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn digests(state: &[__m512i; 8]) -> [[u8; 64]; LANES] {
    unsafe {
        let mut words = [[0u64; LANES]; 8];
        for i in 0..8 {
            _mm512_storeu_si512(words[i].as_mut_ptr() as *mut _, state[i]);
        }
        core::array::from_fn(|l| {
            let mut digest = [0u8; 64];
            for i in 0..8 {
                digest[i * 8..i * 8 + 8].copy_from_slice(&words[i][l].to_be_bytes());
            }
            digest
        })
    }
}

/// Compute the eight digests `SHA-512(R_l || A_l || M_l)`.
///
/// # Safety
///
/// The caller must have verified AVX-512 support (`ifma_x8::available()`,
/// which covers the `avx512f` and `avx512bw` operations used here).
#[target_feature(enable = "avx512f,avx512bw")]
pub unsafe fn challenge_digests_x8(
    r_bytes: &[[u8; 32]; LANES],
    a_bytes: &[[u8; 32]; LANES],
    msgs: &[&[u8]; LANES],
) -> [[u8; 64]; LANES] {
    unsafe {
        let mut staged = Staged::new();
        staged.fill(r_bytes, a_bytes, msgs);

        let zero_block = [0u8; BLOCK];
        let bswap = _mm512_loadu_si512(BSWAP64.as_ptr() as *const _);
        let mut state = init_state();

        for b in 0..staged.max_blocks {
            let (block, active) = staged.block(msgs, b, zero_block.as_ptr());
            block_in!(state, block, bswap, w0, a0, b0, c0, d0, e0, f0, g0, h0);
            compress_rounds!([w0, a0, b0, c0, d0, e0, f0, g0, h0]);
            block_out!(state, active, a0, b0, c0, d0, e0, f0, g0, h0);
        }

        digests(&state)
    }
}

/// [`challenge_digests_x8`] on two groups at once, their round bodies
/// interleaved. Blocks past the shorter group's end run on the single-group
/// body, so an uneven pair costs no more than the two groups apart.
///
/// # Safety
///
/// As [`challenge_digests_x8`].
#[target_feature(enable = "avx512f,avx512bw")]
pub unsafe fn challenge_digests_x16(
    r_bytes: &[[[u8; 32]; LANES]; 2],
    a_bytes: &[[[u8; 32]; LANES]; 2],
    msgs: &[[&[u8]; LANES]; 2],
) -> [[[u8; 64]; LANES]; 2] {
    unsafe {
        let mut staged0 = Staged::new();
        let mut staged1 = Staged::new();
        staged0.fill(&r_bytes[0], &a_bytes[0], &msgs[0]);
        staged1.fill(&r_bytes[1], &a_bytes[1], &msgs[1]);

        let zero_block = [0u8; BLOCK];
        let zero = zero_block.as_ptr();
        let bswap = _mm512_loadu_si512(BSWAP64.as_ptr() as *const _);
        let mut state0 = init_state();
        let mut state1 = init_state();

        let paired = staged0.max_blocks.min(staged1.max_blocks);
        for b in 0..paired {
            let (block0, active0) = staged0.block(&msgs[0], b, zero);
            let (block1, active1) = staged1.block(&msgs[1], b, zero);
            block_in!(state0, block0, bswap, w0, a0, b0, c0, d0, e0, f0, g0, h0);
            block_in!(state1, block1, bswap, w1, a1, b1, c1, d1, e1, f1, g1, h1);
            compress_rounds!(
                [w0, a0, b0, c0, d0, e0, f0, g0, h0],
                [w1, a1, b1, c1, d1, e1, f1, g1, h1]
            );
            block_out!(state0, active0, a0, b0, c0, d0, e0, f0, g0, h0);
            block_out!(state1, active1, a1, b1, c1, d1, e1, f1, g1, h1);
        }
        for b in paired..staged0.max_blocks {
            let (block, active) = staged0.block(&msgs[0], b, zero);
            block_in!(state0, block, bswap, w0, a0, b0, c0, d0, e0, f0, g0, h0);
            compress_rounds!([w0, a0, b0, c0, d0, e0, f0, g0, h0]);
            block_out!(state0, active, a0, b0, c0, d0, e0, f0, g0, h0);
        }
        for b in paired..staged1.max_blocks {
            let (block, active) = staged1.block(&msgs[1], b, zero);
            block_in!(state1, block, bswap, w1, a1, b1, c1, d1, e1, f1, g1, h1);
            compress_rounds!([w1, a1, b1, c1, d1, e1, f1, g1, h1]);
            block_out!(state1, active, a1, b1, c1, d1, e1, f1, g1, h1);
        }

        [digests(&state0), digests(&state1)]
    }
}

#[cfg(all(test, target_arch = "x86_64"))]
mod test {
    use super::*;
    use alloc::vec::Vec;
    use sha2::{Digest, Sha512};

    /// The digest `sha2` gives for one lane's inputs.
    fn expected(r: &[u8; 32], a: &[u8; 32], msg: &[u8]) -> [u8; 64] {
        let d = Sha512::new()
            .chain_update(r)
            .chain_update(a)
            .chain_update(msg)
            .finalize();
        let mut out = [0u8; 64];
        out.copy_from_slice(&d[..]);
        out
    }

    /// xorshift64, so a failing round replays from its index alone.
    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }

        fn bytes(&mut self, n: usize) -> Vec<u8> {
            (0..n).map(|_| self.next() as u8).collect()
        }

        fn fill32(&mut self) -> [u8; 32] {
            core::array::from_fn(|_| self.next() as u8)
        }
    }

    /// The `R` and `A` encodings of eight lanes, and the message each one covers
    type Group = ([[u8; 32]; LANES], [[u8; 32]; LANES], Vec<Vec<u8>>);

    /// One group of eight random lanes
    fn random_group(rng: &mut Rng) -> Group {
        let r: [[u8; 32]; LANES] = core::array::from_fn(|_| rng.fill32());
        let a: [[u8; 32]; LANES] = core::array::from_fn(|_| rng.fill32());
        let msgs: Vec<Vec<u8>> = (0..LANES)
            .map(|_| {
                let len = (rng.next() % 601) as usize;
                rng.bytes(len)
            })
            .collect();
        (r, a, msgs)
    }

    // digests match the sha2 crate across padding boundaries and mixed lengths
    #[test]
    fn digests_match_sha2() {
        if !crate::backend::lanes::ifma_x8::available() {
            return;
        }

        // Hashed length is 64 + msg len, so these straddle both padding boundaries.
        let msg_lens: [usize; LANES] = [0, 1, 47, 48, 63, 175, 176, 1232];
        let msgs_data: Vec<Vec<u8>> = msg_lens
            .iter()
            .enumerate()
            .map(|(i, &n)| alloc::vec![i as u8 ^ 0xA7; n])
            .collect();

        let r_bytes: [[u8; 32]; LANES] = core::array::from_fn(|l| [l as u8 + 1; 32]);
        let a_bytes: [[u8; 32]; LANES] = core::array::from_fn(|l| [l as u8 + 101; 32]);
        let msgs: [&[u8]; LANES] = core::array::from_fn(|l| msgs_data[l].as_slice());

        let digests = unsafe { challenge_digests_x8(&r_bytes, &a_bytes, &msgs) };

        for l in 0..LANES {
            assert_eq!(
                digests[l][..],
                expected(&r_bytes[l], &a_bytes[l], msgs[l])[..],
                "lane {l}"
            );
        }
    }

    // digests match sha2 with every staging boundary in every lane position
    #[test]
    fn digests_match_sha2_ragged() {
        if !crate::backend::lanes::ifma_x8::available() {
            return;
        }

        const LENS: [usize; 18] = [
            0, 1, 47, 48, 63, 64, 65, 111, 127, 128, 129, 175, 176, 191, 192, 200, 1232, 4096,
        ];

        for shift in 0..LENS.len() {
            let msg_lens: [usize; LANES] =
                core::array::from_fn(|l| LENS[(l * 5 + shift) % LENS.len()]);
            let msgs_data: Vec<Vec<u8>> = msg_lens
                .iter()
                .enumerate()
                .map(|(i, &n)| (0..n).map(|j| (i * 31 + j) as u8).collect())
                .collect();

            let r_bytes: [[u8; 32]; LANES] = core::array::from_fn(|l| [(l * 7 + shift) as u8; 32]);
            let a_bytes: [[u8; 32]; LANES] =
                core::array::from_fn(|l| [(l * 13 + shift + 3) as u8; 32]);
            let msgs: [&[u8]; LANES] = core::array::from_fn(|l| msgs_data[l].as_slice());

            let digests = unsafe { challenge_digests_x8(&r_bytes, &a_bytes, &msgs) };

            for l in 0..LANES {
                assert_eq!(
                    digests[l][..],
                    expected(&r_bytes[l], &a_bytes[l], msgs[l])[..],
                    "shift {shift} lane {l} len {}",
                    msg_lens[l]
                );
            }
        }
    }

    // random lengths and bytes match sha2, one group at a time
    #[test]
    fn digests_match_sha2_random_x8() {
        if !crate::backend::lanes::ifma_x8::available() {
            return;
        }

        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        for round in 0..2_000 {
            let (r, a, msgs_data) = random_group(&mut rng);
            let msgs: [&[u8]; LANES] = core::array::from_fn(|l| msgs_data[l].as_slice());
            let digests = unsafe { challenge_digests_x8(&r, &a, &msgs) };
            for l in 0..LANES {
                assert_eq!(
                    digests[l][..],
                    expected(&r[l], &a[l], msgs[l])[..],
                    "round {round} lane {l} len {}",
                    msgs[l].len()
                );
            }
        }
    }

    // random lengths and bytes match sha2, two ragged groups per call
    #[test]
    fn digests_match_sha2_random_x16() {
        if !crate::backend::lanes::ifma_x8::available() {
            return;
        }

        let mut rng = Rng(0x2545_F491_4F6C_DD1D);
        for round in 0..2_000 {
            let (r0, a0, msgs0_data) = random_group(&mut rng);
            let (r1, a1, msgs1_data) = random_group(&mut rng);
            let msgs0: [&[u8]; LANES] = core::array::from_fn(|l| msgs0_data[l].as_slice());
            let msgs1: [&[u8]; LANES] = core::array::from_fn(|l| msgs1_data[l].as_slice());

            let r = [r0, r1];
            let a = [a0, a1];
            let msgs = [msgs0, msgs1];
            let digests = unsafe { challenge_digests_x16(&r, &a, &msgs) };

            for g in 0..2 {
                for l in 0..LANES {
                    assert_eq!(
                        digests[g][l][..],
                        expected(&r[g][l], &a[g][l], msgs[g][l])[..],
                        "round {round} group {g} lane {l} len {}",
                        msgs[g][l].len()
                    );
                }
            }
        }
    }

    // the paired kernel agrees with the single-group kernel lane for lane
    #[test]
    fn x16_matches_x8() {
        if !crate::backend::lanes::ifma_x8::available() {
            return;
        }

        let mut rng = Rng(0xD1B5_4A32_D192_ED03);
        for _ in 0..200 {
            let (r0, a0, msgs0_data) = random_group(&mut rng);
            let (r1, a1, msgs1_data) = random_group(&mut rng);
            let msgs0: [&[u8]; LANES] = core::array::from_fn(|l| msgs0_data[l].as_slice());
            let msgs1: [&[u8]; LANES] = core::array::from_fn(|l| msgs1_data[l].as_slice());

            let paired = unsafe { challenge_digests_x16(&[r0, r1], &[a0, a1], &[msgs0, msgs1]) };
            let single0 = unsafe { challenge_digests_x8(&r0, &a0, &msgs0) };
            let single1 = unsafe { challenge_digests_x8(&r1, &a1, &msgs1) };

            assert_eq!(paired[0], single0);
            assert_eq!(paired[1], single1);
        }
    }
}
