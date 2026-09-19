// Vendored from curve25519-dalek, BSD-3-Clause. See LICENSE.

//! Serial implementations of field, scalar, point arithmetic.
//!
//! When the vector backend is disabled, the crate uses the mixed-model strategy
//! for implementing point operations and scalar multiplication; see the
//! [`curve_models`] and [`scalar_mul`] documentation for more information.
//!
//! When the vector backend is enabled, the field and scalar
//! implementations are still used for non-vectorized operations.

#[doc(hidden)]
pub mod u64;

pub mod curve_models;

pub mod scalar_mul;
