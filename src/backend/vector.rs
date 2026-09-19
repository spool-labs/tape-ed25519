// Vendored from curve25519-dalek, BSD-3-Clause. See LICENSE.

//! Vectorized field and point operations, on a modification of the 4-way parallel formulas of
//! Hisil, Wong, Carter and Dawson

#[allow(missing_docs)]
pub mod packed_simd;

pub mod avx2;

pub mod ifma;

pub mod scalar_mul;
