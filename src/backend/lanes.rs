//! Lane-per-signature verification with eight equations in lockstep, one signature per SIMD lane

// Vendored from the solana-ed25519 lanes fork, BSD-3-Clause

#[cfg(curve25519_backend = "simd")]
#[macro_use]
mod macros;

pub mod digits;
pub mod edwards_x8;
pub mod field_x8;
#[cfg(curve25519_backend = "simd")]
pub mod ifma_x8;
pub mod joint;
pub mod joint_data;
#[cfg(curve25519_backend = "simd")]
pub mod sha512_x8;
