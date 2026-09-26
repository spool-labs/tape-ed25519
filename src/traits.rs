// Vendored from the solana-ed25519 lanes fork, BSD-3-Clause. See LICENSE.

//! Module for common traits.

#![allow(non_snake_case)]

use crate::scalar::Scalar;
use subtle::ConstantTimeEq;

// ------------------------------------------------------------------------
// Public Traits
// ------------------------------------------------------------------------

/// Trait for getting the identity element of a point type.
pub trait Identity {
    /// Returns the identity element of the curve.
    /// Can be used as a constructor.
    fn identity() -> Self;
}

/// Trait for testing if a curve point is equivalent to the identity point.
pub trait IsIdentity {
    /// Return true if this element is the identity element of the curve.
    fn is_identity(&self) -> bool;
}

/// Implement generic identity equality testing for a point representations
/// which have constant-time equality testing and a defined identity
/// constructor.
impl<T> IsIdentity for T
where
    T: ConstantTimeEq + Identity,
{
    fn is_identity(&self) -> bool {
        self.ct_eq(&T::identity()).into()
    }
}

/// A precomputed table of basepoints, for optimising scalar multiplications.
pub trait BasepointTable {
    /// The type of point contained within this table.
    type Point;

    /// Generate a new precomputed basepoint table from the given basepoint.
    fn create(basepoint: &Self::Point) -> Self;

    /// Retrieve the original basepoint from this table.
    fn basepoint(&self) -> Self::Point;

    /// Multiply a `scalar` by this precomputed basepoint table, in constant time.
    fn mul_base(&self, scalar: &Scalar) -> Self::Point;
}

/// A trait for decomposing scalars for the HEEA method.
///
/// The HEEA method is used by EdDSA verification to replace multiplication by
/// a full-width challenge scalar with multiplications by two shorter scalars.
/// For a challenge scalar `h`, the decomposition returns non-negative scalars
/// `rho` and `tau` that satisfy either `rho == tau * h (mod ell)` or
/// `rho == -tau * h (mod ell)`.
///
/// This trait provides a method for performing this decomposition.
pub trait HEEADecomposition {
    /// Decomposes a scalar `h` into two approximately half-size scalars.
    ///
    /// Returns `(rho, tau, flip_h)`. If `flip_h` is false, then
    /// `rho == tau * h (mod ell)`. If `flip_h` is true, then
    /// `rho == -tau * h (mod ell)`.
    fn heea_decompose(&self) -> (Scalar, Scalar, bool);
}

// ------------------------------------------------------------------------
// Private Traits
// ------------------------------------------------------------------------

/// Trait for checking whether a point is on the curve.
///
/// This trait is only for debugging/testing, since it should be
/// impossible for a `curve25519-dalek` user to construct an invalid
/// point.
#[cfg(test)]
pub trait ValidityCheck {
    /// Checks whether the point is on the curve. Not CT.
    fn is_valid(&self) -> bool;
}
