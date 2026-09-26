//! Multibuffer SHA-512 in AVX-512, one message per 64-bit lane for eight or sixteen messages

// Vendored from the solana-ed25519 lanes fork, BSD-3-Clause

#![allow(clippy::needless_range_loop)]
// Stable and nightly disagree on unsafe blocks for intrinsics inside target_feature functions
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

/// Round constants with each row aligned for a 512-bit aligned load
#[repr(align(64))]
struct KLanes([[u64; LANES]; 80]);

/// The round constants broadcast across the lanes ahead of time so each round does one load
static K64_X8: KLanes = {
    let mut rows = [[0u64; LANES]; 80];
    let mut round = 0;
    while round < 80 {
        let mut lane = 0;
        while lane < LANES {
            rows[round][lane] = K64[round];
            lane += 1;
        }
        round += 1;
    }
    KLanes(rows)
};

#[rustfmt::skip]
const H512: [u64; 8] = [
    0x6a09e667f3bcc908, 0xbb67ae8584caa73b, 0x3c6ef372fe94f82b, 0xa54ff53a5f1d36f1,
    0x510e527fade682d1, 0x9b05688c2b3e6c1f, 0x1f83d9abfb41bd6b, 0x5be0cd19137e2179,
];

/// The SHA-512 block size in bytes
const BLOCK: usize = 128;

/// Each lane row loads half a block at a time
const HALF_BLOCK: usize = BLOCK / 2;

/// Bytes in one SHA-512 word
const WORD_BYTES: usize = 8;

/// Half a block holds this many schedule words
const HALF_BLOCK_WORDS: usize = HALF_BLOCK / WORD_BYTES;

/// The SHA-512 state holds eight words
const STATE_WORDS: usize = 8;

/// Bytes in a SHA-512 digest
const DIGEST_LENGTH: usize = 64;

/// Bytes in an encoded `R` or `A`
const ENCODING_LENGTH: usize = 32;

/// The `R || A` prefix takes this many bytes of the first block
const PREFIX_LENGTH: usize = 2 * ENCODING_LENGTH;

/// Padding ends with the message length as a 128-bit big-endian integer
const LENGTH_FIELD: usize = 16;

/// Padding starts with this byte
const PAD_MARKER: u8 = 0x80;

/// The length field counts bits
const BITS_PER_BYTE: u64 = 8;

/// Each lane stages the `R || A` prefix block and up to two padding blocks
const STAGE_BLOCKS: usize = 3;

/// The `VPSHUFB` control that reverses the bytes of each 64-bit word
#[rustfmt::skip]
const BSWAP64: [u8; 64] = [
    7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8,
    7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8,
    7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8,
    7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8,
];

/// Transpose eight 64-byte lane rows into eight big-endian word vectors
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

/// Staged blocks and the block plan for one group of eight lanes
struct Staged {
    /// The `R || A` prefix block and the padded tail, per lane
    stage: [[MaybeUninit<u8>; STAGE_BLOCKS * BLOCK]; LANES],
    /// Padded block count, per lane
    blocks: [usize; LANES],
    /// Index of the block that holds the `0x80` marker, per lane
    pad_block: [usize; LANES],
    /// Stage slot of the marker block, per lane
    pad_slot: [usize; LANES],
    /// Block count of the longest lane
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

    /// Stage block 0 and the padded tail of `R || A || M` per lane, leaving the middle in place
    fn fill(
        &mut self,
        r_bytes: &[[u8; ENCODING_LENGTH]; LANES],
        a_bytes: &[[u8; ENCODING_LENGTH]; LANES],
        msgs: &[&[u8]; LANES],
    ) {
        for lane in 0..LANES {
            let msg = msgs[lane];
            let length = PREFIX_LENGTH + msg.len();
            let padded = (length + 1 + LENGTH_FIELD).div_ceil(BLOCK) * BLOCK;
            self.blocks[lane] = padded / BLOCK;
            self.max_blocks = self.max_blocks.max(self.blocks[lane]);

            let pad_block = length / BLOCK;
            self.pad_block[lane] = pad_block;
            self.pad_slot[lane] = usize::from(pad_block != 0);

            // SAFETY: every write stays inside this lane's stage of STAGE_BLOCKS blocks and every read stays inside `msg`
            unsafe {
                let base = self.stage[lane].as_mut_ptr().cast::<u8>();
                core::ptr::copy_nonoverlapping(r_bytes[lane].as_ptr(), base, ENCODING_LENGTH);
                core::ptr::copy_nonoverlapping(
                    a_bytes[lane].as_ptr(),
                    base.add(ENCODING_LENGTH),
                    ENCODING_LENGTH,
                );
                core::ptr::copy_nonoverlapping(
                    msg.as_ptr(),
                    base.add(PREFIX_LENGTH),
                    msg.len().min(BLOCK - PREFIX_LENGTH),
                );

                let pad = base.add(self.pad_slot[lane] * BLOCK);
                let used = length - pad_block * BLOCK;
                if pad_block != 0 {
                    let offset = pad_block * BLOCK - PREFIX_LENGTH;
                    core::ptr::copy_nonoverlapping(msg.as_ptr().add(offset), pad, used);
                }
                *pad.add(used) = PAD_MARKER;
                core::ptr::write_bytes(pad.add(used + 1), 0, BLOCK - used - 1);

                // The length goes one block past the marker when the marker block has no room
                let last =
                    base.add((self.pad_slot[lane] + self.blocks[lane] - 1 - pad_block) * BLOCK);
                if last != pad {
                    core::ptr::write_bytes(last, 0, BLOCK - WORD_BYTES);
                }
                core::ptr::copy_nonoverlapping(
                    ((length as u64) * BITS_PER_BYTE).to_be_bytes().as_ptr(),
                    last.add(BLOCK - WORD_BYTES),
                    WORD_BYTES,
                );
            }
        }
    }

    /// Return one block's pointer per lane and a mask of live lanes, with spent lanes reading `zero`
    #[inline]
    fn block(
        &self,
        msgs: &[&[u8]; LANES],
        block_index: usize,
        zero: *const u8,
    ) -> ([*const u8; LANES], u8) {
        let mut active: u8 = 0;
        let mut block = [zero; LANES];
        for lane in 0..LANES {
            if block_index >= self.blocks[lane] {
                continue;
            }
            active |= 1 << lane;
            let base = self.stage[lane].as_ptr().cast::<u8>();
            let is_padded = block_index >= self.pad_block[lane];
            // SAFETY: fill sized the stage for every padded block and the middle blocks lie inside the message
            block[lane] = unsafe {
                match (block_index == 0, is_padded) {
                    (true, _) => base,
                    (false, true) => {
                        base.add((self.pad_slot[lane] + block_index - self.pad_block[lane]) * BLOCK)
                    }
                    (false, false) => msgs[lane].as_ptr().add(block_index * BLOCK - PREFIX_LENGTH),
                }
            };
        }
        (block, active)
    }
}

/// The initial SHA-512 state, one message per lane
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn init_state() -> [__m512i; STATE_WORDS] {
    unsafe { core::array::from_fn(|word| _mm512_set1_epi64(H512[word] as i64)) }
}

/// Load the block's sixteen schedule words, transposed out of the lane rows
#[target_feature(enable = "avx512f,avx512bw")]
#[inline]
unsafe fn load_schedule(block: &[*const u8; LANES], bswap: __m512i) -> [__m512i; 16] {
    unsafe {
        let mut w = [_mm512_set1_epi64(0); 16];
        for half in 0..2 {
            let mut rows = [_mm512_set1_epi64(0); LANES];
            for lane in 0..LANES {
                // SAFETY: block pointed every lane at a full readable block
                rows[lane] = _mm512_loadu_si512(block[lane].add(half * HALF_BLOCK) as *const _);
            }
            let words = half * HALF_BLOCK_WORDS..(half + 1) * HALF_BLOCK_WORDS;
            w[words].copy_from_slice(&transpose_words_x8(&rows, bswap));
        }
        w
    }
}

/// Extract the big-endian digest of each lane
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn digests(state: &[__m512i; STATE_WORDS]) -> [[u8; DIGEST_LENGTH]; LANES] {
    unsafe {
        let mut words = [[0u64; LANES]; STATE_WORDS];
        for word in 0..STATE_WORDS {
            // SAFETY: each row of `words` holds LANES u64 values, one full vector
            _mm512_storeu_si512(words[word].as_mut_ptr() as *mut _, state[word]);
        }
        core::array::from_fn(|lane| {
            let mut digest = [0u8; DIGEST_LENGTH];
            for word in 0..STATE_WORDS {
                digest[word * WORD_BYTES..(word + 1) * WORD_BYTES]
                    .copy_from_slice(&words[word][lane].to_be_bytes());
            }
            digest
        })
    }
}

/// Compute the eight digests `SHA-512(R || A || M)`, once `ifma_x8::available()` holds
#[target_feature(enable = "avx512f,avx512bw")]
pub unsafe fn challenge_digests_x8(
    r_bytes: &[[u8; ENCODING_LENGTH]; LANES],
    a_bytes: &[[u8; ENCODING_LENGTH]; LANES],
    msgs: &[&[u8]; LANES],
) -> [[u8; DIGEST_LENGTH]; LANES] {
    // SAFETY: the target features of this function cover every kernel it calls
    unsafe {
        let mut staged = Staged::new();
        staged.fill(r_bytes, a_bytes, msgs);

        let zero_block = [0u8; BLOCK];
        let bswap = _mm512_loadu_si512(BSWAP64.as_ptr() as *const _);
        let mut state = init_state();

        for block_index in 0..staged.max_blocks {
            let (block, active) = staged.block(msgs, block_index, zero_block.as_ptr());
            block_in!(state, block, bswap, w0, a0, b0, c0, d0, e0, f0, g0, h0);
            compress_rounds!([w0, a0, b0, c0, d0, e0, f0, g0, h0]);
            block_out!(state, active, a0, b0, c0, d0, e0, f0, g0, h0);
        }

        digests(&state)
    }
}

/// Compute the digests of two groups with their rounds interleaved, once `ifma_x8::available()` holds
#[target_feature(enable = "avx512f,avx512bw")]
pub unsafe fn challenge_digests_x16(
    r_bytes: &[[[u8; ENCODING_LENGTH]; LANES]; 2],
    a_bytes: &[[[u8; ENCODING_LENGTH]; LANES]; 2],
    msgs: &[[&[u8]; LANES]; 2],
) -> [[[u8; DIGEST_LENGTH]; LANES]; 2] {
    // SAFETY: the target features of this function cover every kernel it calls
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
        for block_index in 0..paired {
            let (block0, active0) = staged0.block(&msgs[0], block_index, zero);
            let (block1, active1) = staged1.block(&msgs[1], block_index, zero);
            block_in!(state0, block0, bswap, w0, a0, b0, c0, d0, e0, f0, g0, h0);
            block_in!(state1, block1, bswap, w1, a1, b1, c1, d1, e1, f1, g1, h1);
            compress_rounds!(
                [w0, a0, b0, c0, d0, e0, f0, g0, h0],
                [w1, a1, b1, c1, d1, e1, f1, g1, h1]
            );
            block_out!(state0, active0, a0, b0, c0, d0, e0, f0, g0, h0);
            block_out!(state1, active1, a1, b1, c1, d1, e1, f1, g1, h1);
        }
        for block_index in paired..staged0.max_blocks {
            let (block, active) = staged0.block(&msgs[0], block_index, zero);
            block_in!(state0, block, bswap, w0, a0, b0, c0, d0, e0, f0, g0, h0);
            compress_rounds!([w0, a0, b0, c0, d0, e0, f0, g0, h0]);
            block_out!(state0, active, a0, b0, c0, d0, e0, f0, g0, h0);
        }
        for block_index in paired..staged1.max_blocks {
            let (block, active) = staged1.block(&msgs[1], block_index, zero);
            block_in!(state1, block, bswap, w1, a1, b1, c1, d1, e1, f1, g1, h1);
            compress_rounds!([w1, a1, b1, c1, d1, e1, f1, g1, h1]);
            block_out!(state1, active, a1, b1, c1, d1, e1, f1, g1, h1);
        }

        [digests(&state0), digests(&state1)]
    }
}

#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use sha2::{Digest, Sha512};

    use super::*;
    use crate::backend::lanes::ifma_x8::available;

    /// Hash one lane's inputs with the `sha2` crate
    fn expected(r: &[u8; 32], a: &[u8; 32], msg: &[u8]) -> [u8; 64] {
        let digest = Sha512::new()
            .chain_update(r)
            .chain_update(a)
            .chain_update(msg)
            .finalize();
        let mut out = [0u8; 64];
        out.copy_from_slice(&digest[..]);
        out
    }

    /// An xorshift64 generator, so a failing round replays from its seed
    struct Xorshift(u64);

    impl Xorshift {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }

        fn bytes(&mut self, length: usize) -> Vec<u8> {
            (0..length).map(|_| self.next() as u8).collect()
        }

        fn encoding(&mut self) -> [u8; 32] {
            core::array::from_fn(|_| self.next() as u8)
        }
    }

    /// The `R` and `A` encodings of eight lanes and their messages
    type Group = ([[u8; 32]; LANES], [[u8; 32]; LANES], Vec<Vec<u8>>);

    /// One group of eight random lanes
    fn random_group(random: &mut Xorshift) -> Group {
        let r: [[u8; 32]; LANES] = core::array::from_fn(|_| random.encoding());
        let a: [[u8; 32]; LANES] = core::array::from_fn(|_| random.encoding());
        let msgs: Vec<Vec<u8>> = (0..LANES)
            .map(|_| {
                let length = (random.next() % 601) as usize;
                random.bytes(length)
            })
            .collect();
        (r, a, msgs)
    }

    // digests match the sha2 crate across padding boundaries and mixed lengths
    #[test]
    fn fixed_lengths() {
        if !available() {
            return;
        }

        // The hashed length is 64 plus the message length, so these straddle both padding boundaries
        let message_lengths: [usize; LANES] = [0, 1, 47, 48, 63, 175, 176, 1232];
        let message_data: Vec<Vec<u8>> = message_lengths
            .iter()
            .enumerate()
            .map(|(index, &length)| vec![index as u8 ^ 0xA7; length])
            .collect();

        let r_bytes: [[u8; 32]; LANES] = core::array::from_fn(|lane| [lane as u8 + 1; 32]);
        let a_bytes: [[u8; 32]; LANES] = core::array::from_fn(|lane| [lane as u8 + 101; 32]);
        let msgs: [&[u8]; LANES] = core::array::from_fn(|lane| message_data[lane].as_slice());

        // SAFETY: available() confirmed the AVX-512 features
        let digests = unsafe { challenge_digests_x8(&r_bytes, &a_bytes, &msgs) };

        for lane in 0..LANES {
            assert_eq!(
                digests[lane][..],
                expected(&r_bytes[lane], &a_bytes[lane], msgs[lane])[..],
                "lane {lane}"
            );
        }
    }

    // digests match sha2 with every staging boundary in every lane position
    #[test]
    fn ragged_lengths() {
        if !available() {
            return;
        }

        const LENGTHS: [usize; 18] = [
            0, 1, 47, 48, 63, 64, 65, 111, 127, 128, 129, 175, 176, 191, 192, 200, 1232, 4096,
        ];

        for shift in 0..LENGTHS.len() {
            let message_lengths: [usize; LANES] =
                core::array::from_fn(|lane| LENGTHS[(lane * 5 + shift) % LENGTHS.len()]);
            let message_data: Vec<Vec<u8>> = message_lengths
                .iter()
                .enumerate()
                .map(|(index, &length)| {
                    (0..length)
                        .map(|offset| (index * 31 + offset) as u8)
                        .collect()
                })
                .collect();

            let r_bytes: [[u8; 32]; LANES] =
                core::array::from_fn(|lane| [(lane * 7 + shift) as u8; 32]);
            let a_bytes: [[u8; 32]; LANES] =
                core::array::from_fn(|lane| [(lane * 13 + shift + 3) as u8; 32]);
            let msgs: [&[u8]; LANES] = core::array::from_fn(|lane| message_data[lane].as_slice());

            // SAFETY: available() confirmed the AVX-512 features
            let digests = unsafe { challenge_digests_x8(&r_bytes, &a_bytes, &msgs) };

            for lane in 0..LANES {
                assert_eq!(
                    digests[lane][..],
                    expected(&r_bytes[lane], &a_bytes[lane], msgs[lane])[..],
                    "shift {shift} lane {lane} length {}",
                    message_lengths[lane]
                );
            }
        }
    }

    // random lengths and bytes match sha2, one group at a time
    #[test]
    fn random_x8() {
        if !available() {
            return;
        }

        let mut random = Xorshift(0x9E37_79B9_7F4A_7C15);
        for round in 0..2_000 {
            let (r, a, message_data) = random_group(&mut random);
            let msgs: [&[u8]; LANES] = core::array::from_fn(|lane| message_data[lane].as_slice());
            // SAFETY: available() confirmed the AVX-512 features
            let digests = unsafe { challenge_digests_x8(&r, &a, &msgs) };
            for lane in 0..LANES {
                assert_eq!(
                    digests[lane][..],
                    expected(&r[lane], &a[lane], msgs[lane])[..],
                    "round {round} lane {lane} length {}",
                    msgs[lane].len()
                );
            }
        }
    }

    // random lengths and bytes match sha2, two ragged groups per call
    #[test]
    fn random_x16() {
        if !available() {
            return;
        }

        let mut random = Xorshift(0x2545_F491_4F6C_DD1D);
        for round in 0..2_000 {
            let (r0, a0, message_data0) = random_group(&mut random);
            let (r1, a1, message_data1) = random_group(&mut random);
            let msgs0: [&[u8]; LANES] = core::array::from_fn(|lane| message_data0[lane].as_slice());
            let msgs1: [&[u8]; LANES] = core::array::from_fn(|lane| message_data1[lane].as_slice());

            let r = [r0, r1];
            let a = [a0, a1];
            let msgs = [msgs0, msgs1];
            // SAFETY: available() confirmed the AVX-512 features
            let digests = unsafe { challenge_digests_x16(&r, &a, &msgs) };

            for group in 0..2 {
                for lane in 0..LANES {
                    assert_eq!(
                        digests[group][lane][..],
                        expected(&r[group][lane], &a[group][lane], msgs[group][lane])[..],
                        "round {round} group {group} lane {lane} length {}",
                        msgs[group][lane].len()
                    );
                }
            }
        }
    }

    // the paired kernel agrees with the single-group kernel lane for lane
    #[test]
    fn paired_groups() {
        if !available() {
            return;
        }

        let mut random = Xorshift(0xD1B5_4A32_D192_ED03);
        for _ in 0..200 {
            let (r0, a0, message_data0) = random_group(&mut random);
            let (r1, a1, message_data1) = random_group(&mut random);
            let msgs0: [&[u8]; LANES] = core::array::from_fn(|lane| message_data0[lane].as_slice());
            let msgs1: [&[u8]; LANES] = core::array::from_fn(|lane| message_data1[lane].as_slice());

            // SAFETY: available() confirmed the AVX-512 features
            let paired = unsafe { challenge_digests_x16(&[r0, r1], &[a0, a1], &[msgs0, msgs1]) };
            let single0 = unsafe { challenge_digests_x8(&r0, &a0, &msgs0) };
            let single1 = unsafe { challenge_digests_x8(&r1, &a1, &msgs1) };

            assert_eq!(paired[0], single0);
            assert_eq!(paired[1], single1);
        }
    }
}
