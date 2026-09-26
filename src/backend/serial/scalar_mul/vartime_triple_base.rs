//! Variable-time four-term multiplication with two 128-bit scalars and a split 256-bit basepoint scalar

// Vendored from the solana-ed25519 lanes fork, BSD-3-Clause

#![allow(non_snake_case)]

use core::cmp::Ordering;

use crate::backend::serial::curve_models::{ProjectiveNielsPoint, ProjectivePoint};
use crate::constants;
use crate::edwards::EdwardsPoint;
use crate::heea::HEEA_MAX_INDEX;
use crate::scalar::Scalar;
use crate::traits::Identity;
use crate::window::NafLookupTable5;

/// NAF window for the two variable points, sized to their eight-entry tables
const DYNAMIC_NAF_WINDOW: usize = 5;

/// NAF window for the low half of the basepoint scalar, sized to its static table
const B_LO_NAF_WINDOW: usize = 8;

/// NAF window for the high half of the basepoint scalar, sized to its static table
const B_HI_NAF_WINDOW: usize = 8;

/// Bytes in each half of the split basepoint scalar
const HALF_SCALAR_LENGTH: usize = 16;

/// Compute `a1 A1 + a2 A2 + b B` in variable time for `a1` and `a2` below 2^128
pub fn mul_128_128_256_prechecked(
    a1: &Scalar,
    A1: &EdwardsPoint,
    a2: &Scalar,
    A2: &EdwardsPoint,
    b: &Scalar,
) -> EdwardsPoint {
    // Split b as b_lo + b_hi * 2^128 so every scalar spans 128 bits
    let b_bytes = b.as_bytes();

    let mut b_lo_bytes = [0u8; 32];
    let mut b_hi_bytes = [0u8; 32];

    b_lo_bytes[..HALF_SCALAR_LENGTH].copy_from_slice(&b_bytes[..HALF_SCALAR_LENGTH]);
    b_hi_bytes[..HALF_SCALAR_LENGTH].copy_from_slice(&b_bytes[HALF_SCALAR_LENGTH..]);

    let b_lo = Scalar::from_canonical_bytes_unchecked(b_lo_bytes);
    let b_hi = Scalar::from_canonical_bytes_unchecked(b_hi_bytes);

    let a1_naf = a1.non_adjacent_form_128(DYNAMIC_NAF_WINDOW);
    let a2_naf = a2.non_adjacent_form_128(DYNAMIC_NAF_WINDOW);
    let b_lo_naf = b_lo.non_adjacent_form_128(B_LO_NAF_WINDOW);
    let b_hi_naf = b_hi.non_adjacent_form_128(B_HI_NAF_WINDOW);

    // Start at the highest position where any digit is nonzero, up to bit 129
    let mut i: usize = HEEA_MAX_INDEX;
    for j in (0..=HEEA_MAX_INDEX).rev() {
        i = j;
        if a1_naf[i] != 0 || a2_naf[i] != 0 || b_lo_naf[i] != 0 || b_hi_naf[i] != 0 {
            break;
        }
    }

    let table_A1 = NafLookupTable5::<ProjectiveNielsPoint>::from(A1);
    let table_A2 = NafLookupTable5::<ProjectiveNielsPoint>::from(A2);

    let table_B = &constants::AFFINE_ODD_MULTIPLES_OF_BASEPOINT;

    // This table holds odd multiples of B * 2^128
    let table_B_128 = &constants::AFFINE_ODD_MULTIPLES_OF_BASEPOINT_128;

    let mut r = ProjectivePoint::identity();

    loop {
        let mut t = r.double();

        match a1_naf[i].cmp(&0) {
            Ordering::Greater => t = &t.as_extended() + &table_A1.select(a1_naf[i] as usize),
            Ordering::Less => t = &t.as_extended() - &table_A1.select(-a1_naf[i] as usize),
            Ordering::Equal => {}
        }

        match a2_naf[i].cmp(&0) {
            Ordering::Greater => t = &t.as_extended() + &table_A2.select(a2_naf[i] as usize),
            Ordering::Less => t = &t.as_extended() - &table_A2.select(-a2_naf[i] as usize),
            Ordering::Equal => {}
        }

        match b_lo_naf[i].cmp(&0) {
            Ordering::Greater => t = &t.as_extended() + &table_B.select(b_lo_naf[i] as usize),
            Ordering::Less => t = &t.as_extended() - &table_B.select(-b_lo_naf[i] as usize),
            Ordering::Equal => {}
        }

        match b_hi_naf[i].cmp(&0) {
            Ordering::Greater => t = &t.as_extended() + &table_B_128.select(b_hi_naf[i] as usize),
            Ordering::Less => t = &t.as_extended() - &table_B_128.select(-b_hi_naf[i] as usize),
            Ordering::Equal => {}
        }

        r = t.as_projective();

        if i == 0 {
            break;
        }
        i -= 1;
    }

    r.as_extended()
}

#[cfg(test)]
mod tests {
    use rand::Rng;

    use super::*;
    use crate::scalar::Scalar;

    /// A uniformly random scalar
    fn random_scalar() -> Scalar {
        let mut wide = [0u8; 64];
        let mut rng = rand::rng();
        rng.fill_bytes(&mut wide);
        Scalar::from_bytes_mod_order_wide(&wide)
    }

    /// A multiple of the basepoint
    fn basepoint_times(multiple: u64) -> EdwardsPoint {
        constants::ED25519_BASEPOINT_POINT * Scalar::from(multiple)
    }

    /// A scalar with the low sixteen bytes from `low` and the rest zero
    fn half_width_scalar(low: [u8; HALF_SCALAR_LENGTH]) -> Scalar {
        let mut bytes = [0u8; 32];
        bytes[..HALF_SCALAR_LENGTH].copy_from_slice(&low);
        Scalar::from_bytes_mod_order(bytes)
    }

    /// The naive sum of the three products
    fn naive(
        first_scalar: Scalar,
        first_point: EdwardsPoint,
        second_scalar: Scalar,
        second_point: EdwardsPoint,
        base_scalar: Scalar,
    ) -> EdwardsPoint {
        (first_scalar * first_point + second_scalar * second_point)
            + base_scalar * constants::ED25519_BASEPOINT_POINT
    }

    // small scalars on small multiples of the basepoint match the naive sum
    #[test]
    fn small_scalars() {
        let first_scalar = Scalar::from(12345u64);
        let second_scalar = Scalar::from(67890u64);
        let base_scalar = random_scalar();
        let first_point = basepoint_times(2);
        let second_point = basepoint_times(3);

        let result = mul_128_128_256_prechecked(
            &first_scalar,
            &first_point,
            &second_scalar,
            &second_point,
            &base_scalar,
        );

        let expected = naive(
            first_scalar,
            first_point,
            second_scalar,
            second_point,
            base_scalar,
        );
        assert_eq!(result, expected);
    }

    // scalars that fill all 128 bits match the naive sum
    #[test]
    fn half_width() {
        let first_scalar = half_width_scalar([0xFF; 16]);
        let second_scalar = half_width_scalar([
            0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66,
            0x77, 0x88,
        ]);
        let base_scalar = random_scalar();
        let first_point = basepoint_times(5);
        let second_point = basepoint_times(7);

        let result = mul_128_128_256_prechecked(
            &first_scalar,
            &first_point,
            &second_scalar,
            &second_point,
            &base_scalar,
        );

        let expected = naive(
            first_scalar,
            first_point,
            second_scalar,
            second_point,
            base_scalar,
        );
        assert_eq!(result, expected);
    }

    // a zero scalar drops its term from the sum
    #[test]
    fn zero_scalar() {
        let first_scalar = Scalar::ZERO;
        let second_scalar = Scalar::from(123u64);
        let base_scalar = random_scalar();
        let first_point = basepoint_times(2);
        let second_point = basepoint_times(3);

        let result = mul_128_128_256_prechecked(
            &first_scalar,
            &first_point,
            &second_scalar,
            &second_point,
            &base_scalar,
        );

        let expected =
            second_scalar * second_point + base_scalar * constants::ED25519_BASEPOINT_POINT;
        assert_eq!(result, expected);
    }

    // an identity point drops its term from the sum
    #[test]
    fn identity_point() {
        let first_scalar = Scalar::from(111u64);
        let second_scalar = Scalar::from(222u64);
        let base_scalar = random_scalar();
        let first_point = EdwardsPoint::identity();
        let second_point = basepoint_times(3);

        let result = mul_128_128_256_prechecked(
            &first_scalar,
            &first_point,
            &second_scalar,
            &second_point,
            &base_scalar,
        );

        let expected =
            second_scalar * second_point + base_scalar * constants::ED25519_BASEPOINT_POINT;
        assert_eq!(result, expected);
    }

    // sixty-bit scalars match the naive sum
    #[test]
    fn medium_scalars() {
        let first_scalar = Scalar::from(0x123456789ABCDEFu64);
        let second_scalar = Scalar::from(0xFEDCBA987654321u64);
        let base_scalar = random_scalar();
        let first_point = basepoint_times(11);
        let second_point = basepoint_times(13);

        let result = mul_128_128_256_prechecked(
            &first_scalar,
            &first_point,
            &second_scalar,
            &second_point,
            &base_scalar,
        );

        let expected = naive(
            first_scalar,
            first_point,
            second_scalar,
            second_point,
            base_scalar,
        );
        assert_eq!(result, expected);
    }

    // repeated-byte 128-bit scalars match the naive sum
    #[test]
    fn large_scalars() {
        let first_scalar = half_width_scalar([0xFF; 16]);
        let second_scalar = half_width_scalar([0xAA; 16]);
        let base_scalar = random_scalar();
        let first_point = basepoint_times(17);
        let second_point = basepoint_times(19);

        let result = mul_128_128_256_prechecked(
            &first_scalar,
            &first_point,
            &second_scalar,
            &second_point,
            &base_scalar,
        );

        let expected = naive(
            first_scalar,
            first_point,
            second_scalar,
            second_point,
            base_scalar,
        );
        assert_eq!(result, expected);
    }

    proptest::proptest! {
        // random 128-bit scalars on random points match the naive sum
        #[test]
        fn random_inputs(
            first_low in proptest::array::uniform16(proptest::num::u8::ANY),
            second_low in proptest::array::uniform16(proptest::num::u8::ANY),
            base_bytes in proptest::array::uniform32(proptest::num::u8::ANY),
            first_point_bytes in proptest::array::uniform32(proptest::num::u8::ANY),
            second_point_bytes in proptest::array::uniform32(proptest::num::u8::ANY),
        ) {
            let first_scalar = half_width_scalar(first_low);
            let second_scalar = half_width_scalar(second_low);
            let base_scalar = Scalar::from_bytes_mod_order(base_bytes);
            let first_point =
                constants::ED25519_BASEPOINT_POINT * Scalar::from_bytes_mod_order(first_point_bytes);
            let second_point =
                constants::ED25519_BASEPOINT_POINT * Scalar::from_bytes_mod_order(second_point_bytes);

            let result = mul_128_128_256_prechecked(
                &first_scalar,
                &first_point,
                &second_scalar,
                &second_point,
                &base_scalar,
            );

            let expected = naive(first_scalar, first_point, second_scalar, second_point, base_scalar);
            proptest::prop_assert_eq!(result, expected);
        }
    }
}
