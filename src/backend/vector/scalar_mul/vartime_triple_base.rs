//! Variable-time triple-base scalar multiplication with half-size scalars on the SIMD backends

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
    use crate::backend::vector::avx2::constants::BASEPOINT_128_ODD_LOOKUP_TABLE;

    #[for_target_feature("avx2")]
    use crate::backend::vector::avx2::constants::BASEPOINT_ODD_LOOKUP_TABLE;
    #[for_target_feature("avx512ifma")]
    use crate::backend::vector::ifma::constants::BASEPOINT_128_ODD_LOOKUP_TABLE;

    #[for_target_feature("avx512ifma")]
    use crate::backend::vector::ifma::constants::BASEPOINT_ODD_LOOKUP_TABLE;

    use crate::edwards::EdwardsPoint;
    use crate::heea::HEEA_MAX_INDEX;
    use crate::scalar::Scalar;
    use crate::traits::Identity;
    use crate::window::{NafLookupTable5, NafLookupTable8};

    /// The tables built per call for A1 and A2 are width 5
    const DYNAMIC_NAF_WINDOW: usize = 5;

    /// The static B table is width 8
    const B_LO_NAF_WINDOW: usize = 8;

    /// The static B' table is width 8
    const B_HI_NAF_WINDOW: usize = 8;

    /// Bytes in each 128-bit half of b
    const HALF_LENGTH: usize = 16;

    /// Compute `a1 A1 + a2 A2 + b B` in variable time for `a1` and `a2` below 2^128
    pub fn mul_128_128_256_prechecked(
        a1: &Scalar,
        A1: &EdwardsPoint,
        a2: &Scalar,
        A2: &EdwardsPoint,
        b: &Scalar,
    ) -> EdwardsPoint {
        // Split b as b_lo + b_hi 2^128 so every scalar in the loop is 128 bits
        let b_bytes = b.as_bytes();

        let mut b_lo_bytes = [0u8; 32];
        let mut b_hi_bytes = [0u8; 32];

        b_lo_bytes[..HALF_LENGTH].copy_from_slice(&b_bytes[..HALF_LENGTH]);
        b_hi_bytes[..HALF_LENGTH].copy_from_slice(&b_bytes[HALF_LENGTH..]);

        let b_lo = Scalar::from_canonical_bytes_unchecked(b_lo_bytes);
        let b_hi = Scalar::from_canonical_bytes_unchecked(b_hi_bytes);

        let a1_naf = a1.non_adjacent_form_128(DYNAMIC_NAF_WINDOW);
        let a2_naf = a2.non_adjacent_form_128(DYNAMIC_NAF_WINDOW);
        let b_lo_naf = b_lo.non_adjacent_form_128(B_LO_NAF_WINDOW);
        let b_hi_naf = b_hi.non_adjacent_form_128(B_HI_NAF_WINDOW);

        // Skip the leading positions where every digit schedule is zero
        let mut i: usize = HEEA_MAX_INDEX;
        for j in (0..=HEEA_MAX_INDEX).rev() {
            i = j;
            if a1_naf[i] != 0 || a2_naf[i] != 0 || b_lo_naf[i] != 0 || b_hi_naf[i] != 0 {
                break;
            }
        }

        let table_A1 = NafLookupTable5::<CachedPoint>::from(A1);
        let table_A2 = NafLookupTable5::<CachedPoint>::from(A2);

        let table_B = &BASEPOINT_ODD_LOOKUP_TABLE;

        // This table holds odd multiples of B' = B 2^128
        let table_B_128 = &BASEPOINT_128_ODD_LOOKUP_TABLE;

        let mut Q = ExtendedPoint::identity();

        loop {
            Q = Q.double();

            match a1_naf[i].cmp(&0) {
                Ordering::Greater => {
                    Q = &Q + &table_A1.select(a1_naf[i] as usize);
                }
                Ordering::Less => {
                    Q = &Q - &table_A1.select(-a1_naf[i] as usize);
                }
                Ordering::Equal => {}
            }

            match a2_naf[i].cmp(&0) {
                Ordering::Greater => {
                    Q = &Q + &table_A2.select(a2_naf[i] as usize);
                }
                Ordering::Less => {
                    Q = &Q - &table_A2.select(-a2_naf[i] as usize);
                }
                Ordering::Equal => {}
            }

            match b_lo_naf[i].cmp(&0) {
                Ordering::Greater => {
                    Q = &Q + &table_B.select(b_lo_naf[i] as usize);
                }
                Ordering::Less => {
                    Q = &Q - &table_B.select(-b_lo_naf[i] as usize);
                }
                Ordering::Equal => {}
            }

            match b_hi_naf[i].cmp(&0) {
                Ordering::Greater => {
                    Q = &Q + &table_B_128.select(b_hi_naf[i] as usize);
                }
                Ordering::Less => {
                    Q = &Q - &table_B_128.select(-b_hi_naf[i] as usize);
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

    /// Build the width-8 odd-multiples table of `A` for the prepared multiplication
    pub fn build_prepared_a2_table(A: &EdwardsPoint) -> NafLookupTable8<CachedPoint> {
        NafLookupTable8::<CachedPoint>::from(A)
    }

    /// Compute `a1 A1 + a2 A2 + b B` with `A2` given as a prebuilt table, negated when `negate_A2` is set
    pub fn mul_128_128_256_prechecked_prepared(
        a1: &Scalar,
        A1: &EdwardsPoint,
        a2: &Scalar,
        table_A2: &NafLookupTable8<CachedPoint>,
        negate_A2: bool,
        b: &Scalar,
    ) -> EdwardsPoint {
        // Split b as b_lo + b_hi 2^128 so every scalar in the loop is 128 bits
        let b_bytes = b.as_bytes();

        let mut b_lo_bytes = [0u8; 32];
        let mut b_hi_bytes = [0u8; 32];

        b_lo_bytes[..HALF_LENGTH].copy_from_slice(&b_bytes[..HALF_LENGTH]);
        b_hi_bytes[..HALF_LENGTH].copy_from_slice(&b_bytes[HALF_LENGTH..]);

        let b_lo = Scalar::from_canonical_bytes_unchecked(b_lo_bytes);
        let b_hi = Scalar::from_canonical_bytes_unchecked(b_hi_bytes);

        let a1_naf = a1.non_adjacent_form_128(DYNAMIC_NAF_WINDOW);
        // The prebuilt A2 table is width 8
        const PREPARED_NAF_WINDOW: usize = 8;
        let mut a2_naf = a2.non_adjacent_form_128(PREPARED_NAF_WINDOW);
        if negate_A2 {
            // Digits are odd with magnitude at most 127, so negation cannot overflow
            for digit in a2_naf.iter_mut() {
                *digit = -*digit;
            }
        }
        let b_lo_naf = b_lo.non_adjacent_form_128(B_LO_NAF_WINDOW);
        let b_hi_naf = b_hi.non_adjacent_form_128(B_HI_NAF_WINDOW);

        // Skip the leading positions where every digit schedule is zero
        let mut i: usize = HEEA_MAX_INDEX;
        for j in (0..=HEEA_MAX_INDEX).rev() {
            i = j;
            if a1_naf[i] != 0 || a2_naf[i] != 0 || b_lo_naf[i] != 0 || b_hi_naf[i] != 0 {
                break;
            }
        }

        let table_A1 = NafLookupTable5::<CachedPoint>::from(A1);

        let table_B = &BASEPOINT_ODD_LOOKUP_TABLE;

        // This table holds odd multiples of B' = B 2^128
        let table_B_128 = &BASEPOINT_128_ODD_LOOKUP_TABLE;

        let mut Q = ExtendedPoint::identity();

        loop {
            Q = Q.double();

            match a1_naf[i].cmp(&0) {
                Ordering::Greater => {
                    Q = &Q + &table_A1.select(a1_naf[i] as usize);
                }
                Ordering::Less => {
                    Q = &Q - &table_A1.select(-a1_naf[i] as usize);
                }
                Ordering::Equal => {}
            }

            match a2_naf[i].cmp(&0) {
                Ordering::Greater => {
                    Q = &Q + &table_A2.select(a2_naf[i] as usize);
                }
                Ordering::Less => {
                    Q = &Q - &table_A2.select(-a2_naf[i] as usize);
                }
                Ordering::Equal => {}
            }

            match b_lo_naf[i].cmp(&0) {
                Ordering::Greater => {
                    Q = &Q + &table_B.select(b_lo_naf[i] as usize);
                }
                Ordering::Less => {
                    Q = &Q - &table_B.select(-b_lo_naf[i] as usize);
                }
                Ordering::Equal => {}
            }

            match b_hi_naf[i].cmp(&0) {
                Ordering::Greater => {
                    Q = &Q + &table_B_128.select(b_hi_naf[i] as usize);
                }
                Ordering::Less => {
                    Q = &Q - &table_B_128.select(-b_hi_naf[i] as usize);
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

#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
    use super::{spec_avx2, spec_avx512ifma_avx512vl};
    use crate::backend::serial::scalar_mul::vartime_triple_base;
    use crate::constants::ED25519_BASEPOINT_POINT;
    use crate::edwards::EdwardsPoint;
    use crate::scalar::Scalar;

    const HALF_LENGTH: usize = 16;

    /// Scalars and points for one triple-base multiplication
    struct Case {
        first_scalar: Scalar,
        first_point: EdwardsPoint,
        second_scalar: Scalar,
        second_point: EdwardsPoint,
        base_scalar: Scalar,
    }

    fn scalar_from_low_half(low: [u8; HALF_LENGTH]) -> Scalar {
        let mut bytes = [0u8; 32];
        bytes[..HALF_LENGTH].copy_from_slice(&low);
        Scalar::from_canonical_bytes(bytes).expect("a 128-bit scalar is canonical")
    }

    fn fixed_case() -> Case {
        Case {
            first_scalar: scalar_from_low_half([
                0xff, 0xee, 0xdd, 0xcc, 0xbb, 0xaa, 0x99, 0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22,
                0x11, 0x00,
            ]),
            first_point: ED25519_BASEPOINT_POINT * Scalar::from(31u64),
            second_scalar: scalar_from_low_half([
                0x10, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc, 0xfe, 0x0f, 0x1e, 0x2d, 0x3c, 0x4b, 0x5a,
                0x69, 0x78,
            ]),
            second_point: ED25519_BASEPOINT_POINT * Scalar::from(37u64),
            base_scalar: Scalar::from_bytes_mod_order([
                0x42, 0x91, 0x0a, 0xbe, 0xef, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99,
                0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff, 0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef,
                0x10, 0x20, 0x30, 0x40,
            ]),
        }
    }

    fn serial_result(case: &Case) -> EdwardsPoint {
        vartime_triple_base::mul_128_128_256_prechecked(
            &case.first_scalar,
            &case.first_point,
            &case.second_scalar,
            &case.second_point,
            &case.base_scalar,
        )
    }

    fn naive_result(case: &Case) -> EdwardsPoint {
        (case.first_scalar * case.first_point + case.second_scalar * case.second_point)
            + case.base_scalar * ED25519_BASEPOINT_POINT
    }

    fn has_ifma() -> bool {
        std::is_x86_feature_detected!("avx512ifma") && std::is_x86_feature_detected!("avx512vl")
    }

    // the AVX2 triple-base multiplication matches the serial and naive results
    #[test]
    fn avx2() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        let case = fixed_case();

        let result = spec_avx2::mul_128_128_256_prechecked(
            &case.first_scalar,
            &case.first_point,
            &case.second_scalar,
            &case.second_point,
            &case.base_scalar,
        );

        assert_eq!(result, serial_result(&case));
        assert_eq!(result, naive_result(&case));
    }

    // the AVX-512 triple-base multiplication matches the serial and naive results
    #[test]
    fn avx512() {
        if !has_ifma() {
            return;
        }
        let case = fixed_case();

        let result = spec_avx512ifma_avx512vl::mul_128_128_256_prechecked(
            &case.first_scalar,
            &case.first_point,
            &case.second_scalar,
            &case.second_point,
            &case.base_scalar,
        );

        assert_eq!(result, serial_result(&case));
        assert_eq!(result, naive_result(&case));
    }

    // a prepared second-point table gives the same result as passing the point, for both signs
    #[test]
    fn prepared() {
        if !has_ifma() {
            return;
        }
        let first_scalar = scalar_from_low_half([
            0x21, 0x43, 0x65, 0x87, 0xa9, 0xcb, 0xed, 0x0f, 0xf0, 0xe1, 0xd2, 0xc3, 0xb4, 0xa5,
            0x96, 0x07,
        ]);
        let second_scalar = scalar_from_low_half([
            0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e,
            0x0f, 0x10,
        ]);
        let base_scalar = Scalar::from_bytes_mod_order([0x5Au8; 32]);
        let first_point = ED25519_BASEPOINT_POINT * Scalar::from(41u64);
        let second_point = ED25519_BASEPOINT_POINT * Scalar::from(43u64);
        let second_table = spec_avx512ifma_avx512vl::build_prepared_a2_table(&second_point);

        for should_negate in [false, true] {
            let prepared = spec_avx512ifma_avx512vl::mul_128_128_256_prechecked_prepared(
                &first_scalar,
                &first_point,
                &second_scalar,
                &second_table,
                should_negate,
                &base_scalar,
            );
            let signed_point = match should_negate {
                true => -second_point,
                false => second_point,
            };
            let unprepared = spec_avx512ifma_avx512vl::mul_128_128_256_prechecked(
                &first_scalar,
                &first_point,
                &second_scalar,
                &signed_point,
                &base_scalar,
            );

            assert_eq!(prepared, unprepared, "should_negate = {should_negate}");
        }
    }
}
