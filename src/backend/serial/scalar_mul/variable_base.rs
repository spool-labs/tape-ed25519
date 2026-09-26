//! Constant-time variable-base scalar multiplication on the serial backend

// Vendored from curve25519-dalek, BSD-3-Clause

#![allow(non_snake_case)]

use crate::backend::serial::curve_models::ProjectiveNielsPoint;
use crate::edwards::EdwardsPoint;
use crate::scalar::Scalar;
use crate::traits::Identity;
use crate::window::LookupTable;

/// Multiply `point` by `scalar` in constant time
#[rustfmt::skip]
pub fn mul(point: &EdwardsPoint, scalar: &Scalar) -> EdwardsPoint {
    let lookup_table = LookupTable::<ProjectiveNielsPoint>::from(point);
    // Radix-16 digits need s < 2^255, which every Scalar holds
    let scalar_digits = scalar.as_radix_16();

    // Peeling the top digit skips sixteen doublings of the identity
    let mut tmp2;
    let mut tmp3 = EdwardsPoint::identity();
    let mut tmp1 = &tmp3 + &lookup_table.select(scalar_digits[63]);
    for i in (0..63).rev() {
        tmp2 = tmp1.as_projective();
        tmp1 = tmp2.double();
        tmp2 = tmp1.as_projective();
        tmp1 = tmp2.double();
        tmp2 = tmp1.as_projective();
        tmp1 = tmp2.double();
        tmp2 = tmp1.as_projective();
        tmp1 = tmp2.double();
        tmp3 = tmp1.as_extended();
        tmp1 = &tmp3 + &lookup_table.select(scalar_digits[i]);
    }
    tmp1.as_extended()
}
