// Vendored from the solana-ed25519 lanes fork, BSD-3-Clause. See LICENSE.

//! Lane-per-signature verification backend.
//!
//! One independent signature per SIMD lane: eight verification equations
//! advance in lockstep, each with its own verdict, and no field operation
//! ever crosses lanes.

pub mod digits;
pub mod edwards_x8;
pub mod field_x8;
#[cfg(curve25519_backend = "simd")]
pub mod ifma_x8;
pub mod joint;
mod joint_data;
#[cfg(curve25519_backend = "simd")]
pub mod sha512_x8;

use digits::HEEA_DIGITS;
use field_x8::LANES;

/// The per-lane radix-16 digit schedules of the HEEA-transformed verification equation.
pub struct GroupDigits {
    /// Joint fixed-base table index per lane, one row per position.
    pub b: [[u64; LANES]; HEEA_DIGITS],
    /// Digits applied to A, with the HEEA sign folded in.
    pub a: [[i8; HEEA_DIGITS]; LANES],
    /// Digits applied to R.
    pub r: [[i8; HEEA_DIGITS]; LANES],
    /// Highest position with any nonzero digit.
    pub start: usize,
}
