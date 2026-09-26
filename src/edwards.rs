//! Points on the Edwards form of Curve25519 and their group operations

// Vendored from curve25519-dalek, BSD-3-Clause

#![allow(non_snake_case)]

use core::array::TryFromSliceError;
use core::borrow::Borrow;
use core::fmt::Debug;
use core::iter::Sum;
use core::ops::{Add, Neg, Sub};
use core::ops::{AddAssign, SubAssign};
use core::ops::{Mul, MulAssign};

use subtle::Choice;
use subtle::ConditionallySelectable;
use subtle::ConstantTimeEq;

use crate::backend::dispatch;
use crate::constants;

use crate::field::FieldElement;
use crate::scalar::{clamp_integer, Scalar};

use crate::backend::serial::curve_models::AffineNielsPoint;
use crate::backend::serial::curve_models::CompletedPoint;
use crate::backend::serial::curve_models::ProjectiveNielsPoint;
use crate::backend::serial::curve_models::ProjectivePoint;

use crate::window::{
    LookupTableRadix128, LookupTableRadix16, LookupTableRadix256, LookupTableRadix32,
    LookupTableRadix64,
};

use crate::traits::BasepointTable;

#[cfg(test)]
use crate::traits::ValidityCheck;
use crate::traits::{Identity, IsIdentity};

use crate::affine::AffinePoint;

use alloc::vec::Vec;

/// A point encoded as its y coordinate with the sign of x in the top bit
#[allow(clippy::derived_hash_with_manual_eq)]
#[derive(Clone, Copy, Hash)]
pub struct CompressedEdwardsY(pub [u8; 32]);

impl ConstantTimeEq for CompressedEdwardsY {
    fn ct_eq(&self, other: &CompressedEdwardsY) -> Choice {
        self.as_bytes().ct_eq(other.as_bytes())
    }
}

impl Eq for CompressedEdwardsY {}
impl PartialEq for CompressedEdwardsY {
    fn eq(&self, other: &Self) -> bool {
        self.ct_eq(other).into()
    }
}

impl Debug for CompressedEdwardsY {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "CompressedEdwardsY: {:?}", self.as_bytes())
    }
}

impl CompressedEdwardsY {
    /// The encoded bytes
    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// A copy of the encoded bytes
    pub const fn to_bytes(self) -> [u8; 32] {
        self.0
    }

    /// Decompress to a point, or `None` when y is not the y coordinate of a curve point
    pub fn decompress(&self) -> Option<EdwardsPoint> {
        let (is_valid_y_coord, X, Y, Z) = decompress::step_1(self);

        if is_valid_y_coord.into() {
            Some(decompress::step_2(self, X, Y, Z))
        } else {
            None
        }
    }
}

mod decompress {
    use subtle::{Choice, ConditionallyNegatable};

    use crate::constants;
    use crate::edwards::{CompressedEdwardsY, EdwardsPoint};
    use crate::field::FieldElement;

    #[rustfmt::skip]
    pub fn step_1(
        repr: &CompressedEdwardsY,
    ) -> (Choice, FieldElement, FieldElement, FieldElement) {
        let Y = FieldElement::from_bytes(repr.as_bytes());
        let Z = FieldElement::ONE;
        let YY = Y.square();
        let u = &YY - &Z;
        let v = &(&YY * &constants::EDWARDS_D) + &Z;
        let (is_valid_y_coord, X) = FieldElement::sqrt_ratio_i(&u, &v);

        (is_valid_y_coord, X, Y, Z)
    }

    #[rustfmt::skip]
    pub fn step_2(
        repr: &CompressedEdwardsY,
        mut X: FieldElement,
        Y: FieldElement,
        Z: FieldElement,
    ) -> EdwardsPoint {
        // sqrt_ratio_i returns the nonnegative root, so the encoded sign bit picks the other one
        let compressed_sign_bit = Choice::from(repr.as_bytes()[31] >> 7);
        X.conditional_negate(compressed_sign_bit);

        EdwardsPoint {
            X,
            Y,
            Z,
            T: &X * &Y,
        }
    }
}

impl TryFrom<&[u8]> for CompressedEdwardsY {
    type Error = TryFromSliceError;

    fn try_from(slice: &[u8]) -> Result<CompressedEdwardsY, TryFromSliceError> {
        Self::from_slice(slice)
    }
}

/// A point on the Edwards form of Curve25519 in extended coordinates
#[derive(Clone, Copy)]
#[allow(missing_docs)]
pub struct EdwardsPoint {
    pub X: FieldElement,
    pub Y: FieldElement,
    pub Z: FieldElement,
    pub T: FieldElement,
}

impl Identity for CompressedEdwardsY {
    fn identity() -> CompressedEdwardsY {
        CompressedEdwardsY([
            1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0,
        ])
    }
}

impl Default for CompressedEdwardsY {
    fn default() -> CompressedEdwardsY {
        CompressedEdwardsY::identity()
    }
}

impl CompressedEdwardsY {
    /// Wrap a slice of bytes, failing when it is not 32 bytes long
    pub fn from_slice(bytes: &[u8]) -> Result<CompressedEdwardsY, TryFromSliceError> {
        bytes.try_into().map(CompressedEdwardsY)
    }
}

impl Identity for EdwardsPoint {
    fn identity() -> EdwardsPoint {
        EdwardsPoint {
            X: FieldElement::ZERO,
            Y: FieldElement::ONE,
            Z: FieldElement::ONE,
            T: FieldElement::ZERO,
        }
    }
}

impl Default for EdwardsPoint {
    fn default() -> EdwardsPoint {
        EdwardsPoint::identity()
    }
}

#[cfg(test)]
impl ValidityCheck for EdwardsPoint {
    fn is_valid(&self) -> bool {
        let point_on_curve = self.as_projective().is_valid();
        let on_segre_image = (&self.X * &self.Y) == (&self.Z * &self.T);

        point_on_curve && on_segre_image
    }
}

impl ConditionallySelectable for EdwardsPoint {
    fn conditional_select(a: &EdwardsPoint, b: &EdwardsPoint, choice: Choice) -> EdwardsPoint {
        EdwardsPoint {
            X: FieldElement::conditional_select(&a.X, &b.X, choice),
            Y: FieldElement::conditional_select(&a.Y, &b.Y, choice),
            Z: FieldElement::conditional_select(&a.Z, &b.Z, choice),
            T: FieldElement::conditional_select(&a.T, &b.T, choice),
        }
    }
}

impl ConstantTimeEq for EdwardsPoint {
    fn ct_eq(&self, other: &EdwardsPoint) -> Choice {
        // Cross-multiplying by the other Z compares affine coordinates without an inversion
        (&self.X * &other.Z).ct_eq(&(&other.X * &self.Z))
            & (&self.Y * &other.Z).ct_eq(&(&other.Y * &self.Z))
    }
}

impl PartialEq for EdwardsPoint {
    fn eq(&self, other: &EdwardsPoint) -> bool {
        self.ct_eq(other).into()
    }
}

impl Eq for EdwardsPoint {}

impl EdwardsPoint {
    /// Convert to the projective Niels form used as an addition operand
    pub fn as_projective_niels(&self) -> ProjectiveNielsPoint {
        ProjectiveNielsPoint {
            Y_plus_X: &self.Y + &self.X,
            Y_minus_X: &self.Y - &self.X,
            Z: self.Z,
            T2d: &self.T * &constants::EDWARDS_D2,
        }
    }

    /// Drop T to get projective coordinates
    pub const fn as_projective(&self) -> ProjectivePoint {
        ProjectivePoint {
            X: self.X,
            Y: self.Y,
            Z: self.Z,
        }
    }

    /// Convert to the affine Niels form, which costs one inversion
    pub fn as_affine_niels(&self) -> AffineNielsPoint {
        let recip = self.Z.invert();
        let x = &self.X * &recip;
        let y = &self.Y * &recip;
        let xy2d = &(&x * &y) * &constants::EDWARDS_D2;
        AffineNielsPoint {
            y_plus_x: &y + &x,
            y_minus_x: &y - &x,
            xy2d,
        }
    }

    /// Convert to affine coordinates
    pub fn to_affine(self) -> AffinePoint {
        let recip = self.Z.invert();
        let x = &self.X * &recip;
        let y = &self.Y * &recip;
        AffinePoint { x, y }
    }

    /// Encode this point as its compressed y coordinate
    pub fn compress(&self) -> CompressedEdwardsY {
        self.to_affine().compress()
    }

    /// Encode several points with one shared batch inversion
    pub fn compress_batch(inputs: &[EdwardsPoint]) -> Vec<CompressedEdwardsY> {
        let mut zs = inputs.iter().map(|input| input.Z).collect::<Vec<_>>();
        FieldElement::invert_batch_alloc(&mut zs);

        inputs
            .iter()
            .zip(&zs)
            .map(|(input, recip)| {
                let x = &input.X * recip;
                let y = &input.Y * recip;
                AffinePoint { x, y }.compress()
            })
            .collect()
    }
}

impl EdwardsPoint {
    /// Add this point to itself
    pub fn double(&self) -> EdwardsPoint {
        self.as_projective().double().as_extended()
    }
}

impl Add<&EdwardsPoint> for &EdwardsPoint {
    type Output = EdwardsPoint;
    fn add(self, other: &EdwardsPoint) -> EdwardsPoint {
        (self + &other.as_projective_niels()).as_extended()
    }
}

define_add_variants!(
    LHS = EdwardsPoint,
    RHS = EdwardsPoint,
    Output = EdwardsPoint
);

impl AddAssign<&EdwardsPoint> for EdwardsPoint {
    fn add_assign(&mut self, _rhs: &EdwardsPoint) {
        *self = (self as &EdwardsPoint) + _rhs;
    }
}

define_add_assign_variants!(LHS = EdwardsPoint, RHS = EdwardsPoint);

impl Sub<&EdwardsPoint> for &EdwardsPoint {
    type Output = EdwardsPoint;
    fn sub(self, other: &EdwardsPoint) -> EdwardsPoint {
        (self - &other.as_projective_niels()).as_extended()
    }
}

define_sub_variants!(
    LHS = EdwardsPoint,
    RHS = EdwardsPoint,
    Output = EdwardsPoint
);

impl SubAssign<&EdwardsPoint> for EdwardsPoint {
    fn sub_assign(&mut self, _rhs: &EdwardsPoint) {
        *self = (self as &EdwardsPoint) - _rhs;
    }
}

define_sub_assign_variants!(LHS = EdwardsPoint, RHS = EdwardsPoint);

impl<Term> Sum<Term> for EdwardsPoint
where
    Term: Borrow<EdwardsPoint>,
{
    fn sum<Items>(iter: Items) -> Self
    where
        Items: Iterator<Item = Term>,
    {
        iter.fold(EdwardsPoint::identity(), |acc, item| acc + item.borrow())
    }
}

impl Neg for &EdwardsPoint {
    type Output = EdwardsPoint;

    fn neg(self) -> EdwardsPoint {
        EdwardsPoint {
            X: -(&self.X),
            Y: self.Y,
            Z: self.Z,
            T: -(&self.T),
        }
    }
}

impl Neg for EdwardsPoint {
    type Output = EdwardsPoint;

    fn neg(self) -> EdwardsPoint {
        -&self
    }
}

impl MulAssign<&Scalar> for EdwardsPoint {
    fn mul_assign(&mut self, scalar: &Scalar) {
        let result = (self as &EdwardsPoint) * scalar;
        *self = result;
    }
}

define_mul_assign_variants!(LHS = EdwardsPoint, RHS = Scalar);

define_mul_variants!(LHS = EdwardsPoint, RHS = Scalar, Output = EdwardsPoint);
define_mul_variants!(LHS = Scalar, RHS = EdwardsPoint, Output = EdwardsPoint);

impl Mul<&Scalar> for &EdwardsPoint {
    type Output = EdwardsPoint;
    /// Multiply this point by `scalar` in constant time
    fn mul(self, scalar: &Scalar) -> EdwardsPoint {
        dispatch::variable_base_mul(self, scalar)
    }
}

impl Mul<&EdwardsPoint> for &Scalar {
    type Output = EdwardsPoint;

    /// Multiply `point` by this scalar in constant time
    fn mul(self, point: &EdwardsPoint) -> EdwardsPoint {
        point * self
    }
}

impl EdwardsPoint {
    /// Multiply the Ed25519 basepoint by `scalar` with the precomputed table
    pub fn mul_base(scalar: &Scalar) -> Self {
        {
            scalar * constants::ED25519_BASEPOINT_TABLE
        }
    }

    /// Multiply this point by `clamp_integer(bytes)`
    pub fn mul_clamped(self, bytes: [u8; 32]) -> Self {
        // Point multiplication only needs the scalar below 2^255, which clamping guarantees
        let s = Scalar {
            bytes: clamp_integer(bytes),
        };
        s * self
    }

    /// Multiply the basepoint by `clamp_integer(bytes)`
    pub fn mul_base_clamped(bytes: [u8; 32]) -> Self {
        // Fixed-base multiplication also only needs the scalar below 2^255
        let s = Scalar {
            bytes: clamp_integer(bytes),
        };
        Self::mul_base(&s)
    }
}

impl EdwardsPoint {
    /// Compute aA + bB in variable time, where B is the Ed25519 basepoint
    pub fn vartime_double_scalar_mul_basepoint(
        a: &Scalar,
        A: &EdwardsPoint,
        b: &Scalar,
    ) -> EdwardsPoint {
        dispatch::vartime_double_base_mul(a, A, b)
    }
}

// A radix 2^w table needs ceil(256 / w) additions, plus one for the top digit at w = 8
impl_basepoint_table! {
    Name = EdwardsBasepointTable,
    LookupTable = LookupTableRadix16,
    Point = EdwardsPoint,
    Radix = 4,
    Additions = 64
}
impl_basepoint_table! {
    Name = EdwardsBasepointTableRadix32,
    LookupTable = LookupTableRadix32,
    Point = EdwardsPoint,
    Radix = 5,
    Additions = 52
}
impl_basepoint_table! {
    Name = EdwardsBasepointTableRadix64,
    LookupTable = LookupTableRadix64,
    Point = EdwardsPoint,
    Radix = 6,
    Additions = 43
}
impl_basepoint_table! {
    Name = EdwardsBasepointTableRadix128,
    LookupTable = LookupTableRadix128,
    Point = EdwardsPoint,
    Radix = 7,
    Additions = 37
}
impl_basepoint_table! {
    Name = EdwardsBasepointTableRadix256,
    LookupTable = LookupTableRadix256,
    Point = EdwardsPoint,
    Radix = 8,
    Additions = 33
}

/// The radix-16 basepoint table aliased to match the other radix tables
pub type EdwardsBasepointTableRadix16 = EdwardsBasepointTable;

impl_basepoint_table_conversions! {
    LHS = EdwardsBasepointTableRadix16,
    RHS = EdwardsBasepointTableRadix32
}
impl_basepoint_table_conversions! {
    LHS = EdwardsBasepointTableRadix16,
    RHS = EdwardsBasepointTableRadix64
}
impl_basepoint_table_conversions! {
    LHS = EdwardsBasepointTableRadix16,
    RHS = EdwardsBasepointTableRadix128
}
impl_basepoint_table_conversions! {
    LHS = EdwardsBasepointTableRadix16,
    RHS = EdwardsBasepointTableRadix256
}

impl_basepoint_table_conversions! {
    LHS = EdwardsBasepointTableRadix32,
    RHS = EdwardsBasepointTableRadix64
}
impl_basepoint_table_conversions! {
    LHS = EdwardsBasepointTableRadix32,
    RHS = EdwardsBasepointTableRadix128
}
impl_basepoint_table_conversions! {
    LHS = EdwardsBasepointTableRadix32,
    RHS = EdwardsBasepointTableRadix256
}

impl_basepoint_table_conversions! {
    LHS = EdwardsBasepointTableRadix64,
    RHS = EdwardsBasepointTableRadix128
}
impl_basepoint_table_conversions! {
    LHS = EdwardsBasepointTableRadix64,
    RHS = EdwardsBasepointTableRadix256
}

impl_basepoint_table_conversions! {
    LHS = EdwardsBasepointTableRadix128,
    RHS = EdwardsBasepointTableRadix256
}

impl EdwardsPoint {
    /// Multiply by the cofactor 8
    pub fn mul_by_cofactor(&self) -> EdwardsPoint {
        self.mul_by_pow_2(3)
    }

    /// Check for the identity in variable time, which needs no field multiplications
    pub fn is_identity_vartime(&self) -> bool {
        bool::from(self.X.is_zero()) && bool::from((&self.Y - &self.Z).is_zero())
    }

    /// Compute [2^k]P by k successive doublings, where k is at least one
    pub fn mul_by_pow_2(&self, k: u32) -> EdwardsPoint {
        debug_assert!(k > 0);
        let mut r: CompletedPoint;
        let mut s = self.as_projective();
        for _ in 0..(k - 1) {
            r = s.double();
            s = r.as_projective();
        }
        // The last doubling converts straight to extended coordinates
        s.double().as_extended()
    }

    /// Check whether this point lies in the torsion subgroup of order 8
    pub fn is_small_order(&self) -> bool {
        self.mul_by_cofactor().is_identity()
    }

    /// Check whether this point lies in the prime-order subgroup
    pub fn is_torsion_free(&self) -> bool {
        (self * constants::BASEPOINT_ORDER).is_identity()
    }
}

impl Debug for EdwardsPoint {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "EdwardsPoint{{\n\tX: {:?},\n\tY: {:?},\n\tZ: {:?},\n\tT: {:?}\n}}",
            self.X, self.Y, self.Z, self.T
        )
    }
}

#[cfg(test)]
mod test {
    use super::*;

    use crate::constants::ED25519_BASEPOINT_TABLE;
    use crate::scalar::test::LARGEST_UNREDUCED_SCALAR;

    /// The x coordinate of the basepoint
    static BASE_X_COORD_BYTES: [u8; 32] = [
        0x1a, 0xd5, 0x25, 0x8f, 0x60, 0x2d, 0x56, 0xc9, 0xb2, 0xa7, 0x25, 0x95, 0x60, 0xc7, 0x2c,
        0x69, 0x5c, 0xdc, 0xd6, 0xfd, 0x31, 0xe2, 0xa4, 0xc0, 0xfe, 0x53, 0x6e, 0xcd, 0xd3, 0x36,
        0x69, 0x21,
    ];

    /// The encoding of 2B
    static BASE2_CMPRSSD: CompressedEdwardsY = CompressedEdwardsY([
        0xc9, 0xa3, 0xf8, 0x6a, 0xae, 0x46, 0x5f, 0xe, 0x56, 0x51, 0x38, 0x64, 0x51, 0x0f, 0x39,
        0x97, 0x56, 0x1f, 0xa2, 0xc9, 0xe8, 0x5e, 0xa2, 0x1d, 0xc2, 0x29, 0x23, 0x09, 0xf3, 0xcd,
        0x60, 0x22,
    ]);

    /// The encoding of 16B
    static BASE16_CMPRSSD: CompressedEdwardsY = CompressedEdwardsY([
        0xeb, 0x27, 0x67, 0xc1, 0x37, 0xab, 0x7a, 0xd8, 0x27, 0x9c, 0x07, 0x8e, 0xff, 0x11, 0x6a,
        0xb0, 0x78, 0x6e, 0xad, 0x3a, 0x2e, 0x0f, 0x98, 0x9f, 0x72, 0xc3, 0x7f, 0x82, 0xf2, 0x96,
        0x96, 0x70,
    ]);

    /// A fixed test scalar
    pub static A_SCALAR: Scalar = Scalar {
        bytes: [
            0x1a, 0x0e, 0x97, 0x8a, 0x90, 0xf6, 0x62, 0x2d, 0x37, 0x47, 0x02, 0x3f, 0x8a, 0xd8,
            0x26, 0x4d, 0xa7, 0x58, 0xaa, 0x1b, 0x88, 0xe0, 0x40, 0xd1, 0x58, 0x9e, 0x7b, 0x7f,
            0x23, 0x76, 0xef, 0x09,
        ],
    };

    /// A second fixed test scalar
    pub static B_SCALAR: Scalar = Scalar {
        bytes: [
            0x91, 0x26, 0x7a, 0xcf, 0x25, 0xc2, 0x09, 0x1b, 0xa2, 0x17, 0x74, 0x7b, 0x66, 0xf0,
            0xb3, 0x2e, 0x9d, 0xf2, 0xa5, 0x67, 0x41, 0xcf, 0xda, 0xc4, 0x56, 0xa7, 0xd4, 0xaa,
            0xb8, 0x60, 0x8a, 0x05,
        ],
    };

    /// The encoding of `A_SCALAR` times the basepoint
    pub static A_TIMES_BASEPOINT: CompressedEdwardsY = CompressedEdwardsY([
        0xea, 0x27, 0xe2, 0x60, 0x53, 0xdf, 0x1b, 0x59, 0x56, 0xf1, 0x4d, 0x5d, 0xec, 0x3c, 0x34,
        0xc3, 0x84, 0xa2, 0x69, 0xb7, 0x4c, 0xc3, 0x80, 0x3e, 0xa8, 0xe2, 0xe7, 0xc9, 0x42, 0x5e,
        0x40, 0xa5,
    ]);

    /// The encoding of `A_SCALAR` times `A_TIMES_BASEPOINT` plus `B_SCALAR` times the basepoint
    static DOUBLE_SCALAR_MULT_RESULT: CompressedEdwardsY = CompressedEdwardsY([
        0x7d, 0xfd, 0x6c, 0x45, 0xaf, 0x6d, 0x6e, 0x0e, 0xba, 0x20, 0x37, 0x1a, 0x23, 0x64, 0x59,
        0xc4, 0xc0, 0x46, 0x83, 0x43, 0xde, 0x70, 0x4b, 0x85, 0x09, 0x6f, 0xfe, 0x35, 0x4f, 0x13,
        0x2b, 0x42,
    ]);

    /// The basepoint decompresses to the known x and compresses back
    #[test]
    fn basepoint_decompression_compression() {
        let base_X = FieldElement::from_bytes(&BASE_X_COORD_BYTES);
        let bp = constants::ED25519_BASEPOINT_COMPRESSED
            .decompress()
            .expect("basepoint should decompress");
        assert!(bp.is_valid());
        assert_eq!(base_X, bp.X);
        assert_eq!(bp.compress(), constants::ED25519_BASEPOINT_COMPRESSED);
    }

    /// Setting the sign bit decompresses to the negated basepoint
    #[test]
    fn decompression_sign_handling() {
        let mut minus_basepoint_bytes = *constants::ED25519_BASEPOINT_COMPRESSED.as_bytes();
        minus_basepoint_bytes[31] |= 1 << 7;
        let minus_basepoint = CompressedEdwardsY(minus_basepoint_bytes)
            .decompress()
            .expect("minus basepoint should decompress");
        assert_eq!(minus_basepoint.X, -(&constants::ED25519_BASEPOINT_POINT.X));
        assert_eq!(minus_basepoint.Y, constants::ED25519_BASEPOINT_POINT.Y);
        assert_eq!(minus_basepoint.Z, constants::ED25519_BASEPOINT_POINT.Z);
        assert_eq!(minus_basepoint.T, -(&constants::ED25519_BASEPOINT_POINT.T));
    }

    /// One times the basepoint table gives the basepoint
    #[test]
    fn basepoint_mult_one_vs_basepoint() {
        let bp = ED25519_BASEPOINT_TABLE * &Scalar::ONE;
        let compressed = bp.compress();
        assert_eq!(compressed, constants::ED25519_BASEPOINT_COMPRESSED);
    }

    /// The table hands back the basepoint it was built from
    #[test]
    fn basepoint_table_basepoint_function_correct() {
        let bp = ED25519_BASEPOINT_TABLE.basepoint();
        assert_eq!(bp.compress(), constants::ED25519_BASEPOINT_COMPRESSED);
    }

    /// Extended addition of B to itself gives 2B
    #[test]
    fn basepoint_plus_basepoint_vs_basepoint2() {
        let bp = constants::ED25519_BASEPOINT_POINT;
        let bp_added = bp + bp;
        assert_eq!(bp_added.compress(), BASE2_CMPRSSD);
    }

    /// Adding B in projective Niels form gives 2B
    #[test]
    fn basepoint_plus_basepoint_projective_niels_vs_basepoint2() {
        let bp = constants::ED25519_BASEPOINT_POINT;
        let bp_added = (&bp + &bp.as_projective_niels()).as_extended();
        assert_eq!(bp_added.compress(), BASE2_CMPRSSD);
    }

    /// Adding B in affine Niels form gives 2B
    #[test]
    fn basepoint_plus_basepoint_affine_niels_vs_basepoint2() {
        let bp = constants::ED25519_BASEPOINT_POINT;
        let bp_affine_niels = bp.as_affine_niels();
        let bp_added = (&bp + &bp_affine_niels).as_extended();
        assert_eq!(bp_added.compress(), BASE2_CMPRSSD);
    }

    /// Equality holds across different projective scalings
    #[test]
    fn extended_point_equality_handles_scaling() {
        let mut two_bytes = [0u8; 32];
        two_bytes[0] = 2;
        let id1 = EdwardsPoint::identity();
        let id2 = EdwardsPoint {
            X: FieldElement::ZERO,
            Y: FieldElement::from_bytes(&two_bytes),
            Z: FieldElement::from_bytes(&two_bytes),
            T: FieldElement::ZERO,
        };
        assert!(bool::from(id1.ct_eq(&id2)));
    }

    /// The affine Niels form of aB adds back to aB
    #[test]
    fn to_affine_niels_clears_denominators() {
        // aB has Z other than one, so the conversion has a denominator to clear
        let aB = ED25519_BASEPOINT_TABLE * &A_SCALAR;
        let aB_affine_niels = aB.as_affine_niels();
        let also_aB = (&EdwardsPoint::identity() + &aB_affine_niels).as_extended();
        assert_eq!(aB.compress(), also_aB.compress());
    }

    /// Fixed-base multiplication matches the known multiple
    #[test]
    fn basepoint_mult_vs_ed25519py() {
        let aB = EdwardsPoint::mul_base(&A_SCALAR);
        assert_eq!(aB.compress(), A_TIMES_BASEPOINT);
    }

    /// The basepoint order times the basepoint is the identity
    #[test]
    fn basepoint_mult_by_basepoint_order() {
        let should_be_id = EdwardsPoint::mul_base(&constants::BASEPOINT_ORDER);
        assert!(should_be_id.is_identity());
    }

    /// The table and variable-base multiplication agree
    #[test]
    fn test_precomputed_basepoint_mult() {
        let aB_1 = ED25519_BASEPOINT_TABLE * &A_SCALAR;
        let aB_2 = constants::ED25519_BASEPOINT_POINT * A_SCALAR;
        assert_eq!(aB_1.compress(), aB_2.compress());
    }

    /// Variable-base multiplication matches the known multiple
    #[test]
    fn scalar_mul_vs_ed25519py() {
        let aB = constants::ED25519_BASEPOINT_POINT * A_SCALAR;
        assert_eq!(aB.compress(), A_TIMES_BASEPOINT);
    }

    /// Doubling the basepoint gives 2B
    #[test]
    fn basepoint_double_vs_basepoint2() {
        assert_eq!(
            constants::ED25519_BASEPOINT_POINT.double().compress(),
            BASE2_CMPRSSD
        );
    }

    /// Two times the basepoint gives 2B
    #[test]
    fn basepoint_mult_two_vs_basepoint2() {
        let two = Scalar::from(2u64);
        let bp2 = EdwardsPoint::mul_base(&two);
        assert_eq!(bp2.compress(), BASE2_CMPRSSD);
    }

    /// Every table radix gives the same multiple
    #[test]
    fn basepoint_tables() {
        let P = &constants::ED25519_BASEPOINT_POINT;
        let a = A_SCALAR;

        let table_radix16 = EdwardsBasepointTableRadix16::create(P);
        let table_radix32 = EdwardsBasepointTableRadix32::create(P);
        let table_radix64 = EdwardsBasepointTableRadix64::create(P);
        let table_radix128 = EdwardsBasepointTableRadix128::create(P);
        let table_radix256 = EdwardsBasepointTableRadix256::create(P);

        let aP = (ED25519_BASEPOINT_TABLE * &a).compress();
        let aP16 = (&table_radix16 * &a).compress();
        let aP32 = (&table_radix32 * &a).compress();
        let aP64 = (&table_radix64 * &a).compress();
        let aP128 = (&table_radix128 * &a).compress();
        let aP256 = (&table_radix256 * &a).compress();

        assert_eq!(aP, aP16);
        assert_eq!(aP16, aP32);
        assert_eq!(aP32, aP64);
        assert_eq!(aP64, aP128);
        assert_eq!(aP128, aP256);
    }

    /// Every table radix gives the same multiple of the largest unreduced scalar
    #[test]
    fn basepoint_tables_unreduced_scalar() {
        let P = &constants::ED25519_BASEPOINT_POINT;
        let a = LARGEST_UNREDUCED_SCALAR;

        let table_radix16 = EdwardsBasepointTableRadix16::create(P);
        let table_radix32 = EdwardsBasepointTableRadix32::create(P);
        let table_radix64 = EdwardsBasepointTableRadix64::create(P);
        let table_radix128 = EdwardsBasepointTableRadix128::create(P);
        let table_radix256 = EdwardsBasepointTableRadix256::create(P);

        let aP = (ED25519_BASEPOINT_TABLE * &a).compress();
        let aP16 = (&table_radix16 * &a).compress();
        let aP32 = (&table_radix32 * &a).compress();
        let aP64 = (&table_radix64 * &a).compress();
        let aP128 = (&table_radix128 * &a).compress();
        let aP256 = (&table_radix256 * &a).compress();

        assert_eq!(aP, aP16);
        assert_eq!(aP16, aP32);
        assert_eq!(aP32, aP64);
        assert_eq!(aP64, aP128);
        assert_eq!(aP128, aP256);
    }

    /// Projective and back to extended keeps the basepoint
    #[test]
    fn basepoint_projective_extended_round_trip() {
        assert_eq!(
            constants::ED25519_BASEPOINT_POINT
                .as_projective()
                .as_extended()
                .compress(),
            constants::ED25519_BASEPOINT_COMPRESSED
        );
    }

    /// Four doublings of the basepoint give 16B
    #[test]
    fn basepoint16_vs_mul_by_pow_2_4() {
        let bp16 = constants::ED25519_BASEPOINT_POINT.mul_by_pow_2(4);
        assert_eq!(bp16.compress(), BASE16_CMPRSSD);
    }

    #[test]
    fn impl_sum() {
        let BASE = constants::ED25519_BASEPOINT_POINT;

        let s1 = Scalar::from(999u64);
        let P1 = BASE * s1;

        let s2 = Scalar::from(333u64);
        let P2 = BASE * s2;

        let arr = [P1, P2];
        let sum: EdwardsPoint = arr.iter().sum();

        assert_eq!(sum, P1 + P2);

        // An empty sum is the identity
        let empty_array: [EdwardsPoint; 0] = [];
        let sum: EdwardsPoint = empty_array.iter().sum();

        assert_eq!(sum, EdwardsPoint::identity());

        // Sum also takes owned points
        let s = Scalar::from(2u64);
        let mapped = arr.iter().map(|x| x * s);
        let sum: EdwardsPoint = mapped.sum();

        assert_eq!(sum, P1 * s + P2 * s);
    }

    /// Conditional assignment copies only when the choice is set
    #[test]
    fn conditional_assign_for_affine_niels_point() {
        let id = AffineNielsPoint::identity();
        let mut p1 = AffineNielsPoint::identity();
        let bp = constants::ED25519_BASEPOINT_POINT.as_affine_niels();

        p1.conditional_assign(&bp, Choice::from(0));
        assert_eq!(p1, id);
        p1.conditional_assign(&bp, Choice::from(1));
        assert_eq!(p1, bp);
    }

    #[test]
    fn is_small_order() {
        assert!(!constants::ED25519_BASEPOINT_POINT.is_small_order());
        for torsion_point in &constants::EIGHT_TORSION {
            assert!(torsion_point.is_small_order());
        }
    }

    #[test]
    fn compressed_identity() {
        assert_eq!(
            EdwardsPoint::identity().compress(),
            CompressedEdwardsY::identity()
        );

        {
            let compressed = EdwardsPoint::compress_batch(&[EdwardsPoint::identity()]);
            assert_eq!(&compressed, &[CompressedEdwardsY::identity()]);
        }
    }

    #[test]
    fn is_identity() {
        assert!(EdwardsPoint::identity().is_identity());
        assert!(!constants::ED25519_BASEPOINT_POINT.is_identity());
    }

    /// Many multiplications in a row trip no overflow check or debug assertion
    #[test]
    fn monte_carlo_overflow_underflow_debug_assert_test() {
        let mut P = constants::ED25519_BASEPOINT_POINT;
        for _ in 0..1_000 {
            P *= &A_SCALAR;
        }
    }

    #[test]
    fn scalarmult_extended_point_works_both_ways() {
        let G: EdwardsPoint = constants::ED25519_BASEPOINT_POINT;
        let s: Scalar = A_SCALAR;

        let P1 = G * s;
        let P2 = s * G;

        assert!(P1.compress().to_bytes() == P2.compress().to_bytes());
    }

    /// The double-base multiplication matches the known result
    #[test]
    fn double_scalar_mul_basepoint_vs_ed25519py() {
        let A = A_TIMES_BASEPOINT
            .decompress()
            .expect("test point should decompress");
        let result = EdwardsPoint::vartime_double_scalar_mul_basepoint(&A_SCALAR, &A, &B_SCALAR);
        assert_eq!(result.compress(), DOUBLE_SCALAR_MULT_RESULT);
    }
}
