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

const DYNAMIC_NAF_WINDOW: usize = 5;

// Both basepoint tables hold width-8 odd multiples
const B_LO_NAF_WINDOW: usize = 8;

const B_HI_NAF_WINDOW: usize = 8;

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

    b_lo_bytes[..16].copy_from_slice(&b_bytes[..16]);
    b_hi_bytes[..16].copy_from_slice(&b_bytes[16..]);

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
mod test {
    use rand::Rng;

    use super::*;
    use crate::scalar::Scalar;

    fn random_scalar() -> Scalar {
        let mut wide = [0u8; 64];
        let mut rng = rand::rng();
        rng.fill_bytes(&mut wide);
        Scalar::from_bytes_mod_order_wide(&wide)
    }

    #[test]
    fn test_triple_base_multiplication() {
        let a1 = Scalar::from(12345u64);
        let a2 = Scalar::from(67890u64);
        let b = random_scalar();

        let A1 = constants::ED25519_BASEPOINT_POINT * Scalar::from(2u64);
        let A2 = constants::ED25519_BASEPOINT_POINT * Scalar::from(3u64);

        let result = mul_128_128_256_prechecked(&a1, &A1, &a2, &A2, &b);

        let expected = (a1 * A1 + a2 * A2) + b * constants::ED25519_BASEPOINT_POINT;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_triple_base_multiplication_128() {
        let a1_bytes = [
            0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
            0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00,
        ];
        let a1 = Scalar::from_bytes_mod_order(a1_bytes);

        let a2_bytes = [
            0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66,
            0x77, 0x88, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00,
        ];
        let a2 = Scalar::from_bytes_mod_order(a2_bytes);

        let b = random_scalar();

        let A1 = constants::ED25519_BASEPOINT_POINT * Scalar::from(5u64);
        let A2 = constants::ED25519_BASEPOINT_POINT * Scalar::from(7u64);

        let result_128 = mul_128_128_256_prechecked(&a1, &A1, &a2, &A2, &b);

        let expected = (a1 * A1 + a2 * A2) + b * constants::ED25519_BASEPOINT_POINT;

        assert_eq!(result_128, expected, "Optimized 128-bit version failed");
    }

    #[test]
    fn test_triple_base_with_zero_scalars() {
        let a1 = Scalar::ZERO;
        let a2 = Scalar::from(123u64);
        let b = random_scalar();

        let A1 = constants::ED25519_BASEPOINT_POINT * Scalar::from(2u64);
        let A2 = constants::ED25519_BASEPOINT_POINT * Scalar::from(3u64);

        let result = mul_128_128_256_prechecked(&a1, &A1, &a2, &A2, &b);
        let expected = a2 * A2 + b * constants::ED25519_BASEPOINT_POINT;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_triple_base_with_identity_points() {
        let a1 = Scalar::from(111u64);
        let a2 = Scalar::from(222u64);
        let b = random_scalar();

        let A1 = EdwardsPoint::identity();
        let A2 = constants::ED25519_BASEPOINT_POINT * Scalar::from(3u64);

        let result = mul_128_128_256_prechecked(&a1, &A1, &a2, &A2, &b);
        let expected = a2 * A2 + b * constants::ED25519_BASEPOINT_POINT;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_triple_base_consistency() {
        let a1 = Scalar::from(0x123456789ABCDEFu64);
        let a2 = Scalar::from(0xFEDCBA987654321u64);
        let b = random_scalar();

        let A1 = constants::ED25519_BASEPOINT_POINT * Scalar::from(11u64);
        let A2 = constants::ED25519_BASEPOINT_POINT * Scalar::from(13u64);

        let result_optimized = mul_128_128_256_prechecked(&a1, &A1, &a2, &A2, &b);
        let result_general = (a1 * A1 + a2 * A2) + b * constants::ED25519_BASEPOINT_POINT;

        assert_eq!(result_optimized, result_general);
    }

    #[test]
    fn test_triple_base_large_scalars() {
        let mut a1_bytes = [0xFFu8; 32];
        a1_bytes[16..].fill(0x00);
        let a1 = Scalar::from_bytes_mod_order(a1_bytes);

        let mut a2_bytes = [0xAAu8; 32];
        a2_bytes[16..].fill(0x00);
        let a2 = Scalar::from_bytes_mod_order(a2_bytes);

        let b = random_scalar();

        let A1 = constants::ED25519_BASEPOINT_POINT * Scalar::from(17u64);
        let A2 = constants::ED25519_BASEPOINT_POINT * Scalar::from(19u64);

        let result = mul_128_128_256_prechecked(&a1, &A1, &a2, &A2, &b);
        let expected = (a1 * A1 + a2 * A2) + b * constants::ED25519_BASEPOINT_POINT;

        assert_eq!(result, expected);
    }

    proptest::proptest! {
        #[test]
        fn proptest_triple_scalar_mul_equivalence(
            a1_bytes_16 in proptest::array::uniform16(proptest::num::u8::ANY),
            a2_bytes_16 in proptest::array::uniform16(proptest::num::u8::ANY),
            b_bytes in proptest::array::uniform32(proptest::num::u8::ANY),
            A1_scalar_bytes in proptest::array::uniform32(proptest::num::u8::ANY),
            A2_scalar_bytes in proptest::array::uniform32(proptest::num::u8::ANY),
        ) {
            let mut a1_bytes = [0u8; 32];
            let mut a2_bytes = [0u8; 32];
            a1_bytes[..16].copy_from_slice(&a1_bytes_16);
            a2_bytes[..16].copy_from_slice(&a2_bytes_16);

            let a1 = Scalar::from_bytes_mod_order(a1_bytes);
            let a2 = Scalar::from_bytes_mod_order(a2_bytes);

            let b = Scalar::from_bytes_mod_order(b_bytes);

            let A1_scalar = Scalar::from_bytes_mod_order(A1_scalar_bytes);
            let A2_scalar = Scalar::from_bytes_mod_order(A2_scalar_bytes);
            let A1 = constants::ED25519_BASEPOINT_POINT * A1_scalar;
            let A2 = constants::ED25519_BASEPOINT_POINT * A2_scalar;

            let result_optimized = mul_128_128_256_prechecked(&a1, &A1, &a2, &A2, &b);

            let expected = (a1 * A1 + a2 * A2) + b * constants::ED25519_BASEPOINT_POINT;

            proptest::prop_assert_eq!(result_optimized, expected, "Optimized triple scalar mul should equal raw operations");
        }
    }
}
