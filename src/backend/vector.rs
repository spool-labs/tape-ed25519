//! Vectorized field and point operations on the 4-way parallel Edwards formulas

// Vendored from curve25519-dalek, BSD-3-Clause

#[macro_use]
mod macros;

#[allow(missing_docs)]
pub mod packed_simd;

pub mod avx2;

pub mod ifma;

pub mod scalar_mul;
