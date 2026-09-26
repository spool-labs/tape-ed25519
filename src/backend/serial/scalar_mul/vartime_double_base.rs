//! Variable-time double-base multiplication `aA + bB` on the serial backend

// Vendored from curve25519-dalek, BSD-3-Clause

#![allow(non_snake_case)]

use core::cmp::Ordering;

use crate::backend::serial::curve_models::{ProjectiveNielsPoint, ProjectivePoint};
use crate::constants;
use crate::edwards::EdwardsPoint;
use crate::scalar::Scalar;
use crate::traits::Identity;
use crate::window::NafLookupTable5;

/// NAF window for the variable point, sized to its eight-entry table
const POINT_NAF_WINDOW: usize = 5;

/// NAF window for the basepoint, sized to its sixty-four-entry static table
const BASEPOINT_NAF_WINDOW: usize = 8;

/// Digit positions in a full-width NAF
const NAF_DIGITS: usize = 256;

/// Compute `aA + bB` in variable time, where `B` is the Ed25519 basepoint
pub fn mul(a: &Scalar, A: &EdwardsPoint, b: &Scalar) -> EdwardsPoint {
    let a_naf = a.non_adjacent_form(POINT_NAF_WINDOW);

    let b_naf = b.non_adjacent_form(BASEPOINT_NAF_WINDOW);

    // Start at the highest position where either digit is nonzero
    let mut i: usize = NAF_DIGITS - 1;
    for j in (0..NAF_DIGITS).rev() {
        i = j;
        if a_naf[i] != 0 || b_naf[i] != 0 {
            break;
        }
    }

    let table_A = NafLookupTable5::<ProjectiveNielsPoint>::from(A);
    let table_B = &constants::AFFINE_ODD_MULTIPLES_OF_BASEPOINT;

    let mut r = ProjectivePoint::identity();
    loop {
        let mut t = r.double();

        match a_naf[i].cmp(&0) {
            Ordering::Greater => t = &t.as_extended() + &table_A.select(a_naf[i] as usize),
            Ordering::Less => t = &t.as_extended() - &table_A.select(-a_naf[i] as usize),
            Ordering::Equal => {}
        }

        match b_naf[i].cmp(&0) {
            Ordering::Greater => t = &t.as_extended() + &table_B.select(b_naf[i] as usize),
            Ordering::Less => t = &t.as_extended() - &table_B.select(-b_naf[i] as usize),
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
