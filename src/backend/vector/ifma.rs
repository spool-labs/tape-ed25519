// Vendored from curve25519-dalek, BSD-3-Clause. See LICENSE.

//! The AVX-512 IFMA half of the vectorized strategy, on the two VPMADD52 instructions

#[allow(missing_docs)]
pub mod field;

#[allow(missing_docs)]
pub mod edwards;

pub mod constants;

pub use self::edwards::{CachedPoint, ExtendedPoint};
