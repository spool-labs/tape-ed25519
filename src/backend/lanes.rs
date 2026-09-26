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
pub mod joint_data;
#[cfg(curve25519_backend = "simd")]
pub mod sha512_x8;
