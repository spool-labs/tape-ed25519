//! Field arithmetic backends and the per-host choice between them

// Vendored from curve25519-dalek, BSD-3-Clause

pub mod dispatch;
pub mod lanes;
pub mod serial;
#[cfg(curve25519_backend = "simd")]
pub mod vector;
