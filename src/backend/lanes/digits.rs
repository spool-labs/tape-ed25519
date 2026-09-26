//! Digit schedules for the radix-16 rounds of the lane verifier

// Vendored from the solana-ed25519 lanes fork, BSD-3-Clause

use crate::backend::lanes::field_x8::LANES;
use crate::scalar::Scalar;

/// A 129-bit scalar fills this many radix-16 positions
pub const HEEA_DIGITS: usize = 33;

/// Each half of `tau s` fills this many radix-16 positions
pub const JOINT_DIGITS: usize = 32;

/// The high half of `tau s` starts at this bit
const JOINT_SPLIT: usize = 128;

/// Bits in one radix-16 digit
const WINDOW: usize = 4;

/// Per-lane radix-16 digit schedules of the HEEA verification equation
pub struct GroupDigits {
    /// Joint fixed-base table index per lane, one row per position
    pub b: [[u64; LANES]; HEEA_DIGITS],
    /// Digits for A with the HEEA sign folded in
    pub a: [[i8; HEEA_DIGITS]; LANES],
    /// Digits for R
    pub r: [[i8; HEEA_DIGITS]; LANES],
    /// Highest position that holds a nonzero digit
    pub start: usize,
}

/// Signed radix-16 digits of a scalar below 2^129, truncated to the HEEA window
pub fn signed_digits(scalar: &Scalar) -> [i8; HEEA_DIGITS] {
    let full = scalar.as_radix_16();
    debug_assert!(full[HEEA_DIGITS..].iter().all(|&d| d == 0));
    let mut out = [0i8; HEEA_DIGITS];
    out.copy_from_slice(&full[..HEEA_DIGITS]);
    out
}

/// Negate a schedule digit by digit, which keeps every digit in -8..=8
pub fn negate(digits: &mut [i8; HEEA_DIGITS]) {
    for d in digits.iter_mut() {
        *d = -*d;
    }
}

/// Radix-16 digits of `tau s` as joint table indices, with the low half in the low four bits
pub fn joint_digits(bytes: &[u8; 32]) -> [u8; JOINT_DIGITS] {
    let nibble = |bit: usize| (bytes[bit / 8] >> (bit % 8)) & 15;
    let mut out = [0u8; JOINT_DIGITS];
    for (i, digit) in out.iter_mut().enumerate() {
        *digit = nibble(WINDOW * i) | nibble(JOINT_SPLIT + WINDOW * i) << WINDOW;
    }
    out
}

#[cfg(test)]
mod test {
    use super::*;

    fn xorshift(state: &mut u64) -> u64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    }

    // both halves come back from the joint digits
    #[test]
    fn joint_halves() {
        let mut state = 0x1234_5678_9abc_def1u64;
        for _ in 0..20_000u32 {
            let mut bytes = [0u8; 32];
            for chunk in bytes.chunks_mut(8) {
                chunk.copy_from_slice(&xorshift(&mut state).to_le_bytes());
            }
            bytes[31] &= 0x1f;
            let digits = joint_digits(&bytes);
            let mut lo: u128 = 0;
            let mut hi: u128 = 0;
            for &d in digits.iter().rev() {
                lo = lo << 4 | (d & 15) as u128;
                hi = hi << 4 | (d >> 4) as u128;
            }
            let mut half = [0u8; 16];
            half.copy_from_slice(&bytes[..16]);
            assert_eq!(lo, u128::from_le_bytes(half));
            half.copy_from_slice(&bytes[16..]);
            assert_eq!(hi, u128::from_le_bytes(half));
        }
    }
}
