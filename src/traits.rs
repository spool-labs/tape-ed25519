//! Traits shared by the point and scalar types

// Vendored from the solana-ed25519 lanes fork, BSD-3-Clause

use subtle::ConstantTimeEq;

use crate::scalar::Scalar;

/// A point type with an identity element
pub trait Identity {
    /// The identity element of the curve
    fn identity() -> Self;
}

/// A check for the identity element
pub trait IsIdentity {
    /// Return true when this element is the identity
    fn is_identity(&self) -> bool;
}

/// Any point with constant-time equality and an identity compares against the identity
impl<Point> IsIdentity for Point
where
    Point: ConstantTimeEq + Identity,
{
    fn is_identity(&self) -> bool {
        self.ct_eq(&Point::identity()).into()
    }
}

/// A precomputed table of basepoint multiples for fixed-base multiplication
pub trait BasepointTable {
    /// The type of the points in the table
    type Point;

    /// Build a table from `basepoint`
    fn create(basepoint: &Self::Point) -> Self;

    /// Recover the basepoint used to build the table
    fn basepoint(&self) -> Self::Point;

    /// Multiply the basepoint by `scalar` in constant time
    fn mul_base(&self, scalar: &Scalar) -> Self::Point;
}

/// Split a challenge scalar h into half-size rho and tau with rho equal to plus or minus tau h mod l
pub trait HEEADecomposition {
    /// Return `(rho, tau, flip_h)`, where `flip_h` is true when rho equals -tau h mod l
    fn heea_decompose(&self) -> (Scalar, Scalar, bool);
}

/// A check that a point lies on the curve, used by tests
#[cfg(test)]
pub trait ValidityCheck {
    /// Return true when the point lies on the curve, in variable time
    fn is_valid(&self) -> bool;
}
