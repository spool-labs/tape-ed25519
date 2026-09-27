//! Extended and cached Edwards points on the AVX-512 IFMA backend, four coordinates per vector

// Vendored from curve25519-dalek, BSD-3-Clause

#![allow(non_snake_case)]

use core::ops::{Add, Neg, Sub};

use curve25519_dalek_derive::unsafe_target_feature;
use subtle::{Choice, ConditionallySelectable};

use super::constants;
use super::field::{F51x4Reduced, F51x4Unreduced, Lanes, Shuffle};
use crate::edwards;
use crate::traits::Identity;
use crate::window::{LookupTable, NafLookupTable5, NafLookupTable8};

#[derive(Clone, Copy, Debug)]
pub struct ExtendedPoint(pub F51x4Unreduced);

#[derive(Clone, Copy, Debug)]
pub struct CachedPoint(pub F51x4Reduced);

#[unsafe_target_feature("avx512ifma,avx512vl")]
impl From<edwards::EdwardsPoint> for ExtendedPoint {
    fn from(P: edwards::EdwardsPoint) -> ExtendedPoint {
        ExtendedPoint(F51x4Unreduced::new(&P.X, &P.Y, &P.Z, &P.T))
    }
}

#[unsafe_target_feature("avx512ifma,avx512vl")]
impl From<ExtendedPoint> for edwards::EdwardsPoint {
    fn from(P: ExtendedPoint) -> edwards::EdwardsPoint {
        let reduced = F51x4Reduced::from(P.0);
        let tmp = F51x4Unreduced::from(reduced).split();
        edwards::EdwardsPoint {
            X: tmp[0],
            Y: tmp[1],
            Z: tmp[2],
            T: tmp[3],
        }
    }
}

#[unsafe_target_feature("avx512ifma,avx512vl")]
impl From<ExtendedPoint> for CachedPoint {
    fn from(P: ExtendedPoint) -> CachedPoint {
        let mut x = P.0;

        x = x.blend(&x.diff_sum(), Lanes::AB);
        x = &F51x4Reduced::from(x) * (121666, 121666, 2 * 121666, 2 * 121665);
        x = x.blend(&x.negate_lazy(), Lanes::D);

        CachedPoint(F51x4Reduced::from(x))
    }
}

#[unsafe_target_feature("avx512ifma,avx512vl")]
impl Default for ExtendedPoint {
    fn default() -> ExtendedPoint {
        ExtendedPoint::identity()
    }
}

#[unsafe_target_feature("avx512ifma,avx512vl")]
impl Identity for ExtendedPoint {
    fn identity() -> ExtendedPoint {
        constants::EXTENDEDPOINT_IDENTITY
    }
}

#[unsafe_target_feature("avx512ifma,avx512vl")]
impl ExtendedPoint {
    pub fn double(&self) -> ExtendedPoint {
        // (Y1 X1 T1 Z1)
        let mut tmp0 = self.0.shuffle(Shuffle::BADC);

        // (X1+Y1 X1+Y1 X1+Y1 X1+Y1)
        let mut tmp1 = (self.0 + tmp0).shuffle(Shuffle::ABAB);

        // (X1 Y1 Z1 X1+Y1)
        tmp0 = self.0.blend(&tmp1, Lanes::D);

        // tmp1 = (S1 S2 S3 S4)
        tmp1 = F51x4Reduced::from(tmp0).square();

        // Combine the squares lane by lane into (S5 S6 S8 S9)

        let zero = F51x4Unreduced::ZERO;

        let S1_S1_S1_S1 = tmp1.shuffle(Shuffle::AAAA);
        let S2_S2_S2_S2 = tmp1.shuffle(Shuffle::BBBB);

        let S2_S2_S2_S4 = S2_S2_S2_S2.blend(&tmp1, Lanes::D).negate_lazy();

        tmp0 = S1_S1_S1_S1 + zero.blend(&(tmp1 + tmp1), Lanes::C);
        tmp0 = tmp0 + zero.blend(&S2_S2_S2_S2, Lanes::AD);
        tmp0 = tmp0 + zero.blend(&S2_S2_S2_S4, Lanes::BCD);

        let tmp2 = F51x4Reduced::from(tmp0);

        ExtendedPoint(&tmp2.shuffle(Shuffle::DBBD) * &tmp2.shuffle(Shuffle::CACA))
    }

    pub fn mul_by_pow_2(&self, k: u32) -> ExtendedPoint {
        let mut tmp: ExtendedPoint = *self;
        for _ in 0..k {
            tmp = tmp.double();
        }
        tmp
    }
}

#[unsafe_target_feature("avx512ifma,avx512vl")]
impl Add<&CachedPoint> for &ExtendedPoint {
    type Output = ExtendedPoint;

    /// Add a cached point to an extended point
    fn add(self, other: &CachedPoint) -> ExtendedPoint {
        let mut tmp = self.0;

        // tmp = (Y1-X1 Y1+X1 Z1 T1) = (S0 S1 Z1 T1)
        tmp = tmp.blend(&tmp.diff_sum(), Lanes::AB);

        // tmp = (S0*S2' S1*S3' Z1*Z2' T1*T2') = (S8 S9 S10 S11)
        tmp = &F51x4Reduced::from(tmp) * &other.0;

        // tmp = (S8 S9 S11 S10)
        tmp = tmp.shuffle(Shuffle::ABDC);

        // tmp = (S9-S8 S9+S8 S10-S11 S10+S11) = (S12 S13 S14 S15)
        let tmp = F51x4Reduced::from(tmp.diff_sum());

        // t0 = (S12 S15 S15 S12)
        let t0 = tmp.shuffle(Shuffle::ADDA);
        // t1 = (S14 S13 S14 S13)
        let t1 = tmp.shuffle(Shuffle::CBCB);

        // (S12*S14 S15*S13 S15*S14 S12*S13) = (X3 Y3 Z3 T3)
        ExtendedPoint(&t0 * &t1)
    }
}

#[unsafe_target_feature("avx512ifma,avx512vl")]
impl Default for CachedPoint {
    fn default() -> CachedPoint {
        CachedPoint::identity()
    }
}

#[unsafe_target_feature("avx512ifma,avx512vl")]
impl Identity for CachedPoint {
    fn identity() -> CachedPoint {
        constants::CACHEDPOINT_IDENTITY
    }
}

#[unsafe_target_feature("avx512ifma,avx512vl")]
impl ConditionallySelectable for CachedPoint {
    fn conditional_select(a: &Self, b: &Self, choice: Choice) -> Self {
        CachedPoint(F51x4Reduced::conditional_select(&a.0, &b.0, choice))
    }

    fn conditional_assign(&mut self, other: &Self, choice: Choice) {
        self.0.conditional_assign(&other.0, choice);
    }
}

#[unsafe_target_feature("avx512ifma,avx512vl")]
impl Neg for &CachedPoint {
    type Output = CachedPoint;

    fn neg(self) -> CachedPoint {
        let swapped = self.0.shuffle(Shuffle::BACD);
        CachedPoint(swapped.blend(&(-self.0), Lanes::D))
    }
}

#[unsafe_target_feature("avx512ifma,avx512vl")]
impl Sub<&CachedPoint> for &ExtendedPoint {
    type Output = ExtendedPoint;

    /// Subtract a cached point by adding its negation
    fn sub(self, other: &CachedPoint) -> ExtendedPoint {
        self + &(-other)
    }
}

#[unsafe_target_feature("avx512ifma,avx512vl")]
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

#[unsafe_target_feature("avx512ifma,avx512vl")]
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

#[unsafe_target_feature("avx512ifma,avx512vl")]
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::{ED25519_BASEPOINT_POINT, ED25519_BASEPOINT_TABLE};
    use crate::scalar::Scalar;

    /// Whether this CPU runs the backend under test, so other hosts skip it
    fn is_supported() -> bool {
        std::is_x86_feature_detected!("avx512ifma") && std::is_x86_feature_detected!("avx512vl")
    }

    const MULTIPLIER: u64 = 8475983829;

    fn sample_points() -> [edwards::EdwardsPoint; 3] {
        [
            edwards::EdwardsPoint::identity(),
            ED25519_BASEPOINT_POINT,
            ED25519_BASEPOINT_TABLE * &Scalar::from(MULTIPLIER),
        ]
    }

    fn check_addition(left: edwards::EdwardsPoint, right: edwards::EdwardsPoint) {
        let cached = CachedPoint::from(ExtendedPoint::from(right));
        let sum: edwards::EdwardsPoint = (&ExtendedPoint::from(left) + &cached).into();
        let difference: edwards::EdwardsPoint = (&ExtendedPoint::from(left) - &cached).into();

        assert_eq!(sum.compress(), (left + right).compress());
        assert_eq!(difference.compress(), (left - right).compress());
    }

    fn check_doubling(point: edwards::EdwardsPoint) {
        let doubled: edwards::EdwardsPoint = ExtendedPoint::from(point).double().into();

        assert_eq!(doubled.compress(), (point + point).compress());
    }

    // vector addition and subtraction match the serial formulas on every pair of sample points
    #[test]
    fn addition() {
        if !is_supported() {
            return;
        }
        for left in sample_points() {
            for right in sample_points() {
                check_addition(left, right);
            }
        }
    }

    // vector doubling matches serial addition of a point to itself
    #[test]
    fn doubling() {
        if !is_supported() {
            return;
        }
        for point in sample_points() {
            check_doubling(point);
        }
    }
}
