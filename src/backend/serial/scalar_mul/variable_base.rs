//! Constant-time variable-base scalar multiplication on the serial backend

// Vendored from curve25519-dalek, BSD-3-Clause

#![allow(non_snake_case)]

use crate::backend::serial::curve_models::ProjectiveNielsPoint;
use crate::edwards::EdwardsPoint;
use crate::scalar::Scalar;
use crate::traits::Identity;
use crate::window::LookupTable;

/// Index of the top radix-16 digit of a scalar
const TOP_DIGIT: usize = 63;

/// Multiply `point` by `scalar` in constant time
#[rustfmt::skip]
pub fn mul(point: &EdwardsPoint, scalar: &Scalar) -> EdwardsPoint {
    let lookup_table = LookupTable::<ProjectiveNielsPoint>::from(point);
    // Radix-16 digits need s < 2^255, which every Scalar holds
    let scalar_digits = scalar.as_radix_16();

    // Peeling the top digit skips sixteen doublings of the identity
    let mut projective;
    let mut extended = EdwardsPoint::identity();
    let mut completed = &extended + &lookup_table.select(scalar_digits[TOP_DIGIT]);
    for i in (0..TOP_DIGIT).rev() {
        projective = completed.as_projective();
        completed = projective.double();
        projective = completed.as_projective();
        completed = projective.double();
        projective = completed.as_projective();
        completed = projective.double();
        projective = completed.as_projective();
        completed = projective.double();
        extended = completed.as_extended();
        completed = &extended + &lookup_table.select(scalar_digits[i]);
    }
    completed.as_extended()
}
