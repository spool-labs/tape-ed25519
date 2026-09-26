// Vendored from curve25519-dalek, BSD-3-Clause. See LICENSE.

//! Pluggable arithmetic for the architecture underneath.
//!
//! The [`serial`] backend holds 64-bit field and scalar arithmetic and point operations in the
//! mixed-model strategy. The [`vector`] backend holds vectorized field arithmetic and the point
//! operations that follow from the parallel formulas of Hisil, Wong, Carter and Dawson. The two
//! strategies give different curve models, so each holds its own scalar multiplication.
//!
//! The vector backend is compiled in on x86_64, and [`get_selected_backend`] picks between the two
//! per host on the AVX2 and AVX-512 IFMA cpuid bits. The [`lanes`] backend is separate again: one
//! signature per SIMD lane, eight equations in lockstep.

pub mod lanes;
pub mod serial;

pub mod dispatch;
#[cfg(curve25519_backend = "simd")]
pub mod vector;
