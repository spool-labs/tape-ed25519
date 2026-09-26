//! Variable-time double-base scalar multiplication on the SIMD backends

// Vendored from curve25519-dalek, BSD-3-Clause

#![allow(non_snake_case)]

#[curve25519_dalek_derive::unsafe_target_feature_specialize("avx2", "avx512ifma,avx512vl")]
pub mod spec {

    use core::cmp::Ordering;

    #[for_target_feature("avx2")]
    use crate::backend::vector::avx2::edwards::{CachedPoint, ExtendedPoint};

    #[for_target_feature("avx512ifma")]
    use crate::backend::vector::ifma::edwards::{CachedPoint, ExtendedPoint};

    #[for_target_feature("avx2")]
    use crate::backend::vector::avx2::constants::BASEPOINT_ODD_LOOKUP_TABLE;

    #[for_target_feature("avx512ifma")]
    use crate::backend::vector::ifma::constants::BASEPOINT_ODD_LOOKUP_TABLE;

    use crate::edwards::EdwardsPoint;
    use crate::scalar::Scalar;
    use crate::traits::Identity;
    use crate::window::NafLookupTable5;

    /// The point table holds odd multiples up to 15
    const A_NAF_WINDOW: usize = 5;

    /// The static basepoint table holds odd multiples up to 127
    const B_NAF_WINDOW: usize = 8;

    /// A NAF of a scalar below 2^255 has this many digits
    const NAF_LENGTH: usize = 256;

    /// Compute `aA + bB` in variable time, where `B` is the Ed25519 basepoint
    pub fn mul(a: &Scalar, A: &EdwardsPoint, b: &Scalar) -> EdwardsPoint {
        let a_naf = a.non_adjacent_form(A_NAF_WINDOW);

        let b_naf = b.non_adjacent_form(B_NAF_WINDOW);

        // Skip the leading positions where both digit schedules are zero
        let mut i: usize = NAF_LENGTH - 1;
        for j in (0..NAF_LENGTH).rev() {
            i = j;
            if a_naf[i] != 0 || b_naf[i] != 0 {
                break;
            }
        }

        let table_A = NafLookupTable5::<CachedPoint>::from(A);

        let table_B = &BASEPOINT_ODD_LOOKUP_TABLE;

        let mut Q = ExtendedPoint::identity();

        loop {
            Q = Q.double();

            match a_naf[i].cmp(&0) {
                Ordering::Greater => {
                    Q = &Q + &table_A.select(a_naf[i] as usize);
                }
                Ordering::Less => {
                    Q = &Q - &table_A.select(-a_naf[i] as usize);
                }
                Ordering::Equal => {}
            }

            match b_naf[i].cmp(&0) {
                Ordering::Greater => {
                    Q = &Q + &table_B.select(b_naf[i] as usize);
                }
                Ordering::Less => {
                    Q = &Q - &table_B.select(-b_naf[i] as usize);
                }
                Ordering::Equal => {}
            }

            if i == 0 {
                break;
            }
            i -= 1;
        }

        Q.into()
    }
}
