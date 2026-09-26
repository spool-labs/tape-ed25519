//! Constant-time variable-base scalar multiplication on the SIMD backends

// Vendored from curve25519-dalek, BSD-3-Clause

#![allow(non_snake_case)]

#[curve25519_dalek_derive::unsafe_target_feature_specialize("avx2", "avx512ifma,avx512vl")]
pub mod spec {

    #[for_target_feature("avx2")]
    use crate::backend::vector::avx2::edwards::{CachedPoint, ExtendedPoint};

    #[for_target_feature("avx512ifma")]
    use crate::backend::vector::ifma::edwards::{CachedPoint, ExtendedPoint};

    use crate::edwards::EdwardsPoint;
    use crate::scalar::Scalar;
    use crate::traits::Identity;
    use crate::window::LookupTable;

    /// Multiply `point` by `scalar` in constant time
    pub fn mul(point: &EdwardsPoint, scalar: &Scalar) -> EdwardsPoint {
        let lookup_table = LookupTable::<CachedPoint>::from(point);
        let scalar_digits = scalar.as_radix_16();
        // Horner's rule over the signed radix-16 digits, top digit first
        let mut Q = ExtendedPoint::identity();
        for i in (0..64).rev() {
            Q = Q.mul_by_pow_2(4);
            Q = &Q + &lookup_table.select(scalar_digits[i]);
        }
        Q.into()
    }
}
