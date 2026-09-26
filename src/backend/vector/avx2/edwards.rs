//! AVX2 extended and cached points on the parallel Edwards formulas

// Vendored from curve25519-dalek, BSD-3-Clause

#![allow(non_snake_case)]

use core::ops::{Add, Neg, Sub};

use subtle::Choice;
use subtle::ConditionallySelectable;

use curve25519_dalek_derive::unsafe_target_feature;

use crate::edwards;
use crate::window::{LookupTable, NafLookupTable5};

use crate::window::NafLookupTable8;

use crate::traits::Identity;

use super::constants;
use super::field::{FieldElement2625x4, Lanes, Shuffle};

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
mod test {
    use super::*;

    #[rustfmt::skip]
    fn serial_add(P: edwards::EdwardsPoint, Q: edwards::EdwardsPoint) -> edwards::EdwardsPoint {
        use crate::backend::serial::u64::field::FieldElement51;

        let (X1, Y1, Z1, T1) = (P.X, P.Y, P.Z, P.T);
        let (X2, Y2, Z2, T2) = (Q.X, Q.Y, Q.Z, Q.T);

        macro_rules! print_var {
            ($x:ident) => {
                println!("{} = {:?}", stringify!($x), $x.to_bytes());
            };
        }

        let S0 = &Y1 - &X1;
        let S1 = &Y1 + &X1;
        let S2 = &Y2 - &X2;
        let S3 = &Y2 + &X2;
        print_var!(S0);
        print_var!(S1);
        print_var!(S2);
        print_var!(S3);
        println!("");

        let S4 = &S0 * &S2;
        let S5 = &S1 * &S3;
        let S6 = &Z1 * &Z2;
        let S7 = &T1 * &T2;
        print_var!(S4);
        print_var!(S5);
        print_var!(S6);
        print_var!(S7);
        println!("");

        let S8  =  &S4 *    &FieldElement51([  121666,0,0,0,0]);
        let S9  =  &S5 *    &FieldElement51([  121666,0,0,0,0]);
        let S10 =  &S6 *    &FieldElement51([2*121666,0,0,0,0]);
        let S11 =  &S7 * &(-&FieldElement51([2*121665,0,0,0,0]));
        print_var!(S8);
        print_var!(S9);
        print_var!(S10);
        print_var!(S11);
        println!("");

        let S12 =  &S9 - &S8;
        let S13 =  &S9 + &S8;
        let S14 = &S10 - &S11;
        let S15 = &S10 + &S11;
        print_var!(S12);
        print_var!(S13);
        print_var!(S14);
        print_var!(S15);
        println!("");

        let X3 = &S12 * &S14;
        let Y3 = &S15 * &S13;
        let Z3 = &S15 * &S14;
        let T3 = &S12 * &S13;

        edwards::EdwardsPoint {
            X: X3,
            Y: Y3,
            Z: Z3,
            T: T3,
        }
    }

    fn addition_test_helper(P: edwards::EdwardsPoint, Q: edwards::EdwardsPoint) {
        let R_serial: edwards::EdwardsPoint = serial_add(P.into(), Q.into()).into();

        let cached_Q = CachedPoint::from(ExtendedPoint::from(Q));
        let R_vector: edwards::EdwardsPoint = (&ExtendedPoint::from(P) + &cached_Q).into();
        let S_vector: edwards::EdwardsPoint = (&ExtendedPoint::from(P) - &cached_Q).into();

        println!("Testing point addition:");
        println!("P = {:?}", P);
        println!("Q = {:?}", Q);
        println!("cached Q = {:?}", cached_Q);
        println!("R = P + Q = {:?}", &P + &Q);
        println!("R_serial = {:?}", R_serial);
        println!("R_vector = {:?}", R_vector);
        println!("S = P - Q = {:?}", &P - &Q);
        println!("S_vector = {:?}", S_vector);
        assert_eq!(R_serial.compress(), (&P + &Q).compress());
        assert_eq!(R_vector.compress(), (&P + &Q).compress());
        assert_eq!(S_vector.compress(), (&P - &Q).compress());
        println!("OK!\n");
    }

    #[test]
    fn vector_addition_vs_serial_addition_vs_edwards_extendedpoint() {
        use crate::constants;
        use crate::scalar::Scalar;

        println!("Testing id +- id");
        let P = edwards::EdwardsPoint::identity();
        let Q = edwards::EdwardsPoint::identity();
        addition_test_helper(P, Q);

        println!("Testing id +- B");
        let P = edwards::EdwardsPoint::identity();
        let Q = constants::ED25519_BASEPOINT_POINT;
        addition_test_helper(P, Q);

        println!("Testing B +- B");
        let P = constants::ED25519_BASEPOINT_POINT;
        let Q = constants::ED25519_BASEPOINT_POINT;
        addition_test_helper(P, Q);

        println!("Testing B +- kB");
        let P = constants::ED25519_BASEPOINT_POINT;
        let Q = constants::ED25519_BASEPOINT_TABLE * &Scalar::from(8475983829u64);
        addition_test_helper(P, Q);
    }

    fn serial_double(P: edwards::EdwardsPoint) -> edwards::EdwardsPoint {
        let (X1, Y1, Z1, _T1) = (P.X, P.Y, P.Z, P.T);

        macro_rules! print_var {
            ($x:ident) => {
                println!("{} = {:?}", stringify!($x), $x.to_bytes());
            };
        }

        let S0 = &X1 + &Y1;
        print_var!(S0);
        println!("");

        let S1 = X1.square();
        let S2 = Y1.square();
        let S3 = Z1.square();
        let S4 = S0.square();
        print_var!(S1);
        print_var!(S2);
        print_var!(S3);
        print_var!(S4);
        println!("");

        let S5 = &S1 + &S2;
        let S6 = &S1 - &S2;
        let S7 = &S3 + &S3;
        let S8 = &S7 + &S6;
        let S9 = &S5 - &S4;
        print_var!(S5);
        print_var!(S6);
        print_var!(S7);
        print_var!(S8);
        print_var!(S9);
        println!("");

        let X3 = &S8 * &S9;
        let Y3 = &S5 * &S6;
        let Z3 = &S8 * &S6;
        let T3 = &S5 * &S9;

        edwards::EdwardsPoint {
            X: X3,
            Y: Y3,
            Z: Z3,
            T: T3,
        }
    }

    fn doubling_test_helper(P: edwards::EdwardsPoint) {
        let R1: edwards::EdwardsPoint = serial_double(P.into()).into();
        let R2: edwards::EdwardsPoint = ExtendedPoint::from(P).double().into();
        println!("Testing point doubling:");
        println!("P = {:?}", P);
        println!("(serial) R1 = {:?}", R1);
        println!("(vector) R2 = {:?}", R2);
        println!("P + P = {:?}", &P + &P);
        assert_eq!(R1.compress(), (&P + &P).compress());
        assert_eq!(R2.compress(), (&P + &P).compress());
        println!("OK!\n");
    }

    #[test]
    fn vector_doubling_vs_serial_doubling_vs_edwards_extendedpoint() {
        use crate::constants;
        use crate::scalar::Scalar;

        println!("Testing [2]id");
        let P = edwards::EdwardsPoint::identity();
        doubling_test_helper(P);

        println!("Testing [2]B");
        let P = constants::ED25519_BASEPOINT_POINT;
        doubling_test_helper(P);

        println!("Testing [2]([k]B)");
        let P = constants::ED25519_BASEPOINT_TABLE * &Scalar::from(8475983829u64);
        doubling_test_helper(P);
    }

    #[test]
    fn basepoint_odd_lookup_table_verify() {
        use crate::backend::vector::avx2::constants::BASEPOINT_ODD_LOOKUP_TABLE;
        use crate::constants;

        let basepoint_odd_table =
            NafLookupTable8::<CachedPoint>::from(&constants::ED25519_BASEPOINT_POINT);
        println!("basepoint_odd_lookup_table = {:?}", basepoint_odd_table);

        let table_B = &BASEPOINT_ODD_LOOKUP_TABLE;
        for (b_vec, base_vec) in table_B.0.iter().zip(basepoint_odd_table.0.iter()) {
            let b_splits = b_vec.0.split();
            let base_splits = base_vec.0.split();

            assert_eq!(base_splits[0], b_splits[0]);
            assert_eq!(base_splits[1], b_splits[1]);
            assert_eq!(base_splits[2], b_splits[2]);
            assert_eq!(base_splits[3], b_splits[3]);
        }
    }
}
