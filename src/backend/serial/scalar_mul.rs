// Vendored from curve25519-dalek, BSD-3-Clause. See LICENSE.

//! Implementations of various scalar multiplication algorithms.
//!
//! Note that all of these implementations use serial code for field
//! arithmetic with the multi-model strategy described in the
//! `curve_models` module.  The vectorized AVX2 backend has its own
//! scalar multiplication implementations, since it only uses one
//! curve model.

#[allow(missing_docs)]
pub mod variable_base;

#[allow(missing_docs)]
pub mod vartime_double_base;

#[allow(missing_docs)]
pub mod vartime_triple_base;



