//! Lookup tables of point multiples for fixed and sliding window multiplication

// Vendored from curve25519-dalek, BSD-3-Clause

#![allow(non_snake_case)]

use core::fmt::Debug;

use subtle::{Choice, ConditionallyNegatable, ConditionallySelectable, ConstantTimeEq};

use crate::backend::serial::curve_models::{AffineNielsPoint, ProjectiveNielsPoint};
use crate::edwards::EdwardsPoint;
use crate::traits::Identity;

// The radix-16 table is plain `LookupTable` because the precomputed constants build it
impl_lookup_table! {
    Name = LookupTable,
    Size = 8,
    SizeNeg = -8,
    SizeRange = 1..9,
    ConversionRange = 0..7
}

// The larger radices only build basepoint tables
impl_lookup_table! {
    Name = LookupTableRadix32,
    Size = 16,
    SizeNeg = -16,
    SizeRange = 1..17,
    ConversionRange = 0..15
}
impl_lookup_table! {
    Name = LookupTableRadix64,
    Size = 32,
    SizeNeg = -32,
    SizeRange = 1..33,
    ConversionRange = 0..31
}
impl_lookup_table! {
    Name = LookupTableRadix128,
    Size = 64,
    SizeNeg = -64,
    SizeRange = 1..65,
    ConversionRange = 0..63
}
impl_lookup_table! {
    Name = LookupTableRadix256,
    Size = 128,
    SizeNeg = -128,
    SizeRange = 1..129,
    ConversionRange = 0..127
}

/// The radix-16 lookup table aliased to match the other radix tables
pub type LookupTableRadix16<Point> = LookupTable<Point>;

/// The odd multiples A, 3A, ..., 15A of a point A
#[derive(Clone, Copy)]
pub struct NafLookupTable5<Point>(pub [Point; 8]);

impl<Point: Copy> NafLookupTable5<Point> {
    /// Return xA for a public odd x below 16
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
        NafLookupTable5(Ai)
    }
}

/// The odd multiples A, 3A, ..., 127A of a point A
#[derive(Clone, Copy)]
pub struct NafLookupTable8<Point>(pub [Point; 64]);

impl<Point: Copy> NafLookupTable8<Point> {
    /// Return xA for a public odd x below 128
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

impl From<&EdwardsPoint> for NafLookupTable8<ProjectiveNielsPoint> {
    fn from(A: &EdwardsPoint) -> Self {
        let mut Ai = [A.as_projective_niels(); 64];
        let A2 = A.double();
        for i in 0..63 {
            Ai[i + 1] = (&A2 + &Ai[i]).as_extended().as_projective_niels();
        }
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
        NafLookupTable8(Ai)
    }
}
