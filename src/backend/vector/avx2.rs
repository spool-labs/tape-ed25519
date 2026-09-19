// Vendored from curve25519-dalek, BSD-3-Clause. See LICENSE.

//! The AVX2 half of the vectorized strategy: one 64-bit lane per field element, 32-bit limbs

pub mod field;

pub mod edwards;

pub mod constants;

pub use self::edwards::{CachedPoint, ExtendedPoint};
