//! AVX2 extended and cached points on the parallel Edwards formulas

// Vendored from curve25519-dalek, BSD-3-Clause

#![allow(non_snake_case)]

use core::ops::{Add, Neg, Sub};

use curve25519_dalek_derive::unsafe_target_feature;
use subtle::{Choice, ConditionallySelectable};

use crate::backend::vector::avx2::constants;
use crate::backend::vector::avx2::field::{FieldElement2625x4, Lanes, Shuffle};
use crate::edwards;
use crate::traits::Identity;
use crate::window::{LookupTable, NafLookupTable5, NafLookupTable8};

/// A point in extended coordinates with every limb bounded by b < 0.007
#[derive(Clone, Copy, Debug)]
pub struct ExtendedPoint(pub FieldElement2625x4);

#[unsafe_target_feature("avx2")]
impl From<edwards::EdwardsPoint> for ExtendedPoint {
    fn from(P: edwards::EdwardsPoint) -> ExtendedPoint {
        ExtendedPoint(FieldElement2625x4::new(&P.X, &P.Y, &P.Z, &P.T))
    }
}

#[unsafe_target_feature("avx2")]
impl From<ExtendedPoint> for edwards::EdwardsPoint {
    fn from(P: ExtendedPoint) -> edwards::EdwardsPoint {
        let tmp = P.0.split();
        edwards::EdwardsPoint {
            X: tmp[0],
            Y: tmp[1],
            Z: tmp[2],
            T: tmp[3],
        }
    }
}

#[unsafe_target_feature("avx2")]
impl ConditionallySelectable for ExtendedPoint {
    fn conditional_select(a: &Self, b: &Self, choice: Choice) -> Self {
        ExtendedPoint(FieldElement2625x4::conditional_select(&a.0, &b.0, choice))
    }

    fn conditional_assign(&mut self, other: &Self, choice: Choice) {
        self.0.conditional_assign(&other.0, choice);
    }
}

#[unsafe_target_feature("avx2")]
impl Default for ExtendedPoint {
    fn default() -> ExtendedPoint {
        ExtendedPoint::identity()
    }
}

#[unsafe_target_feature("avx2")]
impl Identity for ExtendedPoint {
    fn identity() -> ExtendedPoint {
        constants::EXTENDEDPOINT_IDENTITY
    }
}

#[unsafe_target_feature("avx2")]
impl ExtendedPoint {
    /// Double this point
    pub fn double(&self) -> ExtendedPoint {
        // Build (X1 Y1 Z1 X1+Y1) from two shuffles and an add
        let mut tmp0 = self.0.shuffle(Shuffle::ABAB);

        let mut tmp1 = tmp0.shuffle(Shuffle::BADC);

        tmp0 = self.0.blend(tmp0 + tmp1, Lanes::D);

        // tmp1 = (S1 S2 S3 -S4) with b < 0.007
        tmp1 = tmp0.square_and_negate_D();

        // Sum into (S5 S6 S8 S9) = (S1 + S2, S1 - S2, S1 - S2 + 2 S3, S1 + S2 - S4)
        let zero = FieldElement2625x4::ZERO;
        let S_1 = tmp1.shuffle(Shuffle::AAAA);
        let S_2 = tmp1.shuffle(Shuffle::BBBB);

        tmp0 = zero.blend(tmp1 + tmp1, Lanes::C);
        tmp0 = tmp0.blend(tmp1, Lanes::D);
        tmp0 = tmp0 + S_1;
        tmp0 = tmp0 + zero.blend(S_2, Lanes::AD);
        tmp0 = tmp0 + zero.blend(S_2.negate_lazy(), Lanes::BC);

        tmp1 = tmp0.shuffle(Shuffle::DBBD);
        tmp0 = tmp0.shuffle(Shuffle::CACA);

        // (S8 S5 S8 S5) times (S9 S6 S6 S9) stays within the multiply bounds (2.5, 1.75)
        ExtendedPoint(&tmp0 * &tmp1)
    }

    pub fn mul_by_pow_2(&self, k: u32) -> ExtendedPoint {
        let mut tmp: ExtendedPoint = *self;
        for _ in 0..k {
            tmp = tmp.double();
        }
        tmp
    }
}

/// A point prepared for addition, bounded by b < 1.0 until negated twice
#[derive(Clone, Copy, Debug)]
pub struct CachedPoint(pub FieldElement2625x4);

#[unsafe_target_feature("avx2")]
impl From<ExtendedPoint> for CachedPoint {
    fn from(P: ExtendedPoint) -> CachedPoint {
        let mut x = P.0;

        x = x.blend(x.diff_sum(), Lanes::AB);

        x = x * (121666, 121666, 2 * 121666, 2 * 121665);

        x = x.blend(-x, Lanes::D);

        CachedPoint(x)
    }
}

#[unsafe_target_feature("avx2")]
impl Default for CachedPoint {
    fn default() -> CachedPoint {
        CachedPoint::identity()
    }
}

#[unsafe_target_feature("avx2")]
impl Identity for CachedPoint {
    fn identity() -> CachedPoint {
        constants::CACHEDPOINT_IDENTITY
    }
}

#[unsafe_target_feature("avx2")]
impl ConditionallySelectable for CachedPoint {
    fn conditional_select(a: &Self, b: &Self, choice: Choice) -> Self {
        CachedPoint(FieldElement2625x4::conditional_select(&a.0, &b.0, choice))
    }

    fn conditional_assign(&mut self, other: &Self, choice: Choice) {
        self.0.conditional_assign(&other.0, choice);
    }
}

#[unsafe_target_feature("avx2")]
impl Neg for &CachedPoint {
    type Output = CachedPoint;
    /// Negate without reducing, which is safe once per point
    fn neg(self) -> CachedPoint {
        let swapped = self.0.shuffle(Shuffle::BACD);
        CachedPoint(swapped.blend(swapped.negate_lazy(), Lanes::D))
    }
}

#[unsafe_target_feature("avx2")]
impl Add<&CachedPoint> for &ExtendedPoint {
    type Output = ExtendedPoint;

    /// Add a cached point to this point
    fn add(self, other: &CachedPoint) -> ExtendedPoint {
        let mut tmp = self.0;

        // tmp = (S0 S1 Z1 T1) with b < 1.6, and `other` stays below b < 1.0
        tmp = tmp.blend(tmp.diff_sum(), Lanes::AB);

        tmp = &tmp * &other.0;

        tmp = tmp.shuffle(Shuffle::ABDC);

        // tmp = (S12 S13 S14 S15)
        tmp = tmp.diff_sum();

        let t0 = tmp.shuffle(Shuffle::ADDA);
        let t1 = tmp.shuffle(Shuffle::CBCB);

        // (S12 S15 S15 S12) times (S14 S13 S14 S13) gives (X3 Y3 Z3 T3)
        ExtendedPoint(&t0 * &t1)
    }
}

#[unsafe_target_feature("avx2")]
impl Sub<&CachedPoint> for &ExtendedPoint {
    type Output = ExtendedPoint;

    /// Subtract by adding the lazy negation
    fn sub(self, other: &CachedPoint) -> ExtendedPoint {
        self + &(-other)
    }
}

#[unsafe_target_feature("avx2")]
impl From<&edwards::EdwardsPoint> for LookupTable<CachedPoint> {
    fn from(point: &edwards::EdwardsPoint) -> Self {
        let P = ExtendedPoint::from(*point);
        let mut points = [CachedPoint::from(P); 8];
        for i in 0..7 {
            points[i + 1] = (&P + &points[i]).into();
        }
        LookupTable(points)
    }
}

#[unsafe_target_feature("avx2")]
impl From<&edwards::EdwardsPoint> for NafLookupTable5<CachedPoint> {
    fn from(point: &edwards::EdwardsPoint) -> Self {
        let A = ExtendedPoint::from(*point);
        let mut Ai = [CachedPoint::from(A); 8];
        let A2 = A.double();
        for i in 0..7 {
            Ai[i + 1] = (&A2 + &Ai[i]).into();
        }
        NafLookupTable5(Ai)
    }
}

#[unsafe_target_feature("avx2")]
impl From<&edwards::EdwardsPoint> for NafLookupTable8<CachedPoint> {
    fn from(point: &edwards::EdwardsPoint) -> Self {
        let A = ExtendedPoint::from(*point);
        let mut Ai = [CachedPoint::from(A); 64];
        let A2 = A.double();
        for i in 0..63 {
            Ai[i + 1] = (&A2 + &Ai[i]).into();
        }
        NafLookupTable8(Ai)
    }
}

#[cfg(target_feature = "avx2")]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::serial::u64::field::FieldElement51;
    use crate::backend::vector::avx2::constants::BASEPOINT_ODD_LOOKUP_TABLE;
    use crate::constants as curve_constants;
    use crate::scalar::Scalar;

    /// A scalar for a basepoint multiple with a nontrivial Z
    const SAMPLE_SCALAR: u64 = 8475983829;

    /// The parallel addition formulas evaluated with serial field arithmetic
    #[rustfmt::skip]
    fn serial_add(first: edwards::EdwardsPoint, second: edwards::EdwardsPoint) -> edwards::EdwardsPoint {
        let (X1, Y1, Z1, T1) = (first.X, first.Y, first.Z, first.T);
        let (X2, Y2, Z2, T2) = (second.X, second.Y, second.Z, second.T);

        let S0 = &Y1 - &X1;
        let S1 = &Y1 + &X1;
        let S2 = &Y2 - &X2;
        let S3 = &Y2 + &X2;

        let S4 = &S0 * &S2;
        let S5 = &S1 * &S3;
        let S6 = &Z1 * &Z2;
        let S7 = &T1 * &T2;

        let S8  =  &S4 *    &FieldElement51([  121666,0,0,0,0]);
        let S9  =  &S5 *    &FieldElement51([  121666,0,0,0,0]);
        let S10 =  &S6 *    &FieldElement51([2*121666,0,0,0,0]);
        let S11 =  &S7 * &(-&FieldElement51([2*121665,0,0,0,0]));

        let S12 =  &S9 - &S8;
        let S13 =  &S9 + &S8;
        let S14 = &S10 - &S11;
        let S15 = &S10 + &S11;

        edwards::EdwardsPoint {
            X: &S12 * &S14,
            Y: &S15 * &S13,
            Z: &S15 * &S14,
            T: &S12 * &S13,
        }
    }

    /// The parallel doubling formulas evaluated with serial field arithmetic
    fn serial_double(point: edwards::EdwardsPoint) -> edwards::EdwardsPoint {
        let (X1, Y1, Z1) = (point.X, point.Y, point.Z);

        let S0 = &X1 + &Y1;

        let S1 = X1.square();
        let S2 = Y1.square();
        let S3 = Z1.square();
        let S4 = S0.square();

        let S5 = &S1 + &S2;
        let S6 = &S1 - &S2;
        let S7 = &S3 + &S3;
        let S8 = &S7 + &S6;
        let S9 = &S5 - &S4;

        edwards::EdwardsPoint {
            X: &S8 * &S9,
            Y: &S5 * &S6,
            Z: &S8 * &S6,
            T: &S5 * &S9,
        }
    }

    /// Check the serial formulas and the vector sum and difference against `EdwardsPoint`
    fn check_addition(first: edwards::EdwardsPoint, second: edwards::EdwardsPoint) {
        let cached_second = CachedPoint::from(ExtendedPoint::from(second));

        let serial_sum = serial_add(first, second);
        let vector_sum: edwards::EdwardsPoint =
            (&ExtendedPoint::from(first) + &cached_second).into();
        let vector_difference: edwards::EdwardsPoint =
            (&ExtendedPoint::from(first) - &cached_second).into();

        assert_eq!(serial_sum.compress(), (first + second).compress());
        assert_eq!(vector_sum.compress(), (first + second).compress());
        assert_eq!(vector_difference.compress(), (first - second).compress());
    }

    /// Check the serial formulas and the vector double against `EdwardsPoint`
    fn check_doubling(point: edwards::EdwardsPoint) {
        let serial_double = serial_double(point);
        let vector_double: edwards::EdwardsPoint = ExtendedPoint::from(point).double().into();

        assert_eq!(serial_double.compress(), (point + point).compress());
        assert_eq!(vector_double.compress(), (point + point).compress());
    }

    // vector addition and subtraction agree with serial addition on identity and basepoint multiples
    #[test]
    fn addition() {
        let identity = edwards::EdwardsPoint::identity();
        let basepoint = curve_constants::ED25519_BASEPOINT_POINT;
        let multiple = curve_constants::ED25519_BASEPOINT_TABLE * &Scalar::from(SAMPLE_SCALAR);

        check_addition(identity, identity);
        check_addition(identity, basepoint);
        check_addition(basepoint, basepoint);
        check_addition(basepoint, multiple);
    }

    // vector doubling agrees with serial doubling on identity and basepoint multiples
    #[test]
    fn doubling() {
        check_doubling(edwards::EdwardsPoint::identity());
        check_doubling(curve_constants::ED25519_BASEPOINT_POINT);
        check_doubling(curve_constants::ED25519_BASEPOINT_TABLE * &Scalar::from(SAMPLE_SCALAR));
    }

    // the static odd-multiples table matches one built at runtime
    #[test]
    fn basepoint_table() {
        let runtime_table =
            NafLookupTable8::<CachedPoint>::from(&curve_constants::ED25519_BASEPOINT_POINT);

        for (static_entry, runtime_entry) in BASEPOINT_ODD_LOOKUP_TABLE
            .0
            .iter()
            .zip(runtime_table.0.iter())
        {
            assert_eq!(runtime_entry.0.split(), static_entry.0.split());
        }
    }
}
