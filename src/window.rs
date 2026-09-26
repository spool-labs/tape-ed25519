//! Code for fixed- and sliding-window functionality

#![allow(non_snake_case)]

use core::fmt::Debug;

use subtle::Choice;
use subtle::ConditionallyNegatable;
use subtle::ConditionallySelectable;
use subtle::ConstantTimeEq;

use crate::traits::Identity;

use crate::backend::serial::curve_models::AffineNielsPoint;
use crate::backend::serial::curve_models::ProjectiveNielsPoint;
use crate::edwards::EdwardsPoint;

// The first one has to be named "LookupTable" because it's used as a constructor for consts.
// This is radix-16
impl_lookup_table! {
    Name = LookupTable,
    Size = 8,
    SizeNeg = -8,
    SizeRange = 1..9,
    ConversionRange = 0..7
}

// The rest only get used to make basepoint tables
// radix-32
impl_lookup_table! {
    Name = LookupTableRadix32,
    Size = 16,
    SizeNeg = -16,
    SizeRange = 1..17,
    ConversionRange = 0..15
}
// radix-64
impl_lookup_table! {
    Name = LookupTableRadix64,
    Size = 32,
    SizeNeg = -32,
    SizeRange = 1..33,
    ConversionRange = 0..31
}
// radix-128
impl_lookup_table! {
    Name = LookupTableRadix128,
    Size = 64,
    SizeNeg = -64,
    SizeRange = 1..65,
    ConversionRange = 0..63
}
// radix-256
impl_lookup_table! {
    Name = LookupTableRadix256,
    Size = 128,
    SizeNeg = -128,
    SizeRange = 1..129,
    ConversionRange = 0..127
}

// For homogeneity we then alias it to "LookupTableRadix16".
pub type LookupTableRadix16<Point> = LookupTable<Point>;

/// Holds odd multiples 1A, 3A, ..., 15A of a point A.
#[derive(Copy, Clone)]
pub struct NafLookupTable5<Point>(pub [Point; 8]);

impl<Point: Copy> NafLookupTable5<Point> {
    /// Given public, odd \\( x \\) with \\( 0 < x < 2^4 \\), return \\(xA\\).
    pub fn select(&self, x: usize) -> Point {
        debug_assert_eq!(x & 1, 1);
        debug_assert!(x < 16);

        self.0[x / 2]
    }
}

impl<Point: Debug> Debug for NafLookupTable5<Point> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "NafLookupTable5({:?})", self.0)
    }
}

impl From<&EdwardsPoint> for NafLookupTable5<ProjectiveNielsPoint> {
    fn from(A: &EdwardsPoint) -> Self {
        let mut Ai = [A.as_projective_niels(); 8];
        let A2 = A.double();
        for i in 0..7 {
            Ai[i + 1] = (&A2 + &Ai[i]).as_extended().as_projective_niels();
        }
        // Now Ai = [A, 3A, 5A, 7A, 9A, 11A, 13A, 15A]
        NafLookupTable5(Ai)
    }
}

impl From<&EdwardsPoint> for NafLookupTable5<AffineNielsPoint> {
    fn from(A: &EdwardsPoint) -> Self {
        let mut Ai = [A.as_affine_niels(); 8];
        let A2 = A.double();
        for i in 0..7 {
            Ai[i + 1] = (&A2 + &Ai[i]).as_extended().as_affine_niels();
        }
        // Now Ai = [A, 3A, 5A, 7A, 9A, 11A, 13A, 15A]
        NafLookupTable5(Ai)
    }
}

/// Holds stuff up to 8. The only time we use tables this big is for precomputed basepoint tables
/// and multiscalar multiplication (which requires alloc).
#[derive(Copy, Clone)]
pub struct NafLookupTable8<Point>(pub [Point; 64]);

impl<Point: Copy> NafLookupTable8<Point> {
    pub fn select(&self, x: usize) -> Point {
        debug_assert_eq!(x & 1, 1);
        debug_assert!(x < 128);

        self.0[x / 2]
    }
}

impl<Point: Debug> Debug for NafLookupTable8<Point> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        writeln!(f, "NafLookupTable8([")?;
        for i in 0..64 {
            writeln!(f, "\t{:?},", self.0[i])?;
        }
        write!(f, "])")
    }
}

impl NafLookupTable8<AffineNielsPoint> {}

impl From<&EdwardsPoint> for NafLookupTable8<ProjectiveNielsPoint> {
    fn from(A: &EdwardsPoint) -> Self {
        let mut Ai = [A.as_projective_niels(); 64];
        let A2 = A.double();
        for i in 0..63 {
            Ai[i + 1] = (&A2 + &Ai[i]).as_extended().as_projective_niels();
        }
        // Now Ai = [A, 3A, 5A, 7A, 9A, 11A, 13A, 15A, ..., 127A]
        NafLookupTable8(Ai)
    }
}

impl From<&EdwardsPoint> for NafLookupTable8<AffineNielsPoint> {
    fn from(A: &EdwardsPoint) -> Self {
        let mut Ai = [A.as_affine_niels(); 64];
        let A2 = A.double();
        for i in 0..63 {
            Ai[i + 1] = (&A2 + &Ai[i]).as_extended().as_affine_niels();
        }
        // Now Ai = [A, 3A, 5A, 7A, 9A, 11A, 13A, 15A, ..., 127A]
        NafLookupTable8(Ai)
    }
}
