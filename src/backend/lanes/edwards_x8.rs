//! Eight Edwards points side by side, one per signature lane

// Vendored from the solana-ed25519 lanes fork, BSD-3-Clause

#![allow(non_snake_case)]

use super::field_x8::{sqrt_ratio_i_x8, FieldElementX8, LaneMask, LANES};
use crate::backend::serial::u64::field::FieldElement51;
use crate::constants::{EDWARDS_D, EDWARDS_D2};
#[cfg(test)]
use crate::edwards::EdwardsPoint;

/// Each lookup table holds the multiples one through eight
const TABLE_SIZE: usize = 8;

/// Eight points in extended coordinates `(X : Y : Z : T)`, one per lane
#[derive(Clone, Copy, Debug)]
pub struct ExtendedPointX8 {
    pub X: FieldElementX8,
    pub Y: FieldElementX8,
    pub Z: FieldElementX8,
    pub T: FieldElementX8,
}

/// Eight points in completed form, as the addition and doubling formulas return them
#[derive(Clone, Copy, Debug)]
pub struct CompletedPointX8 {
    pub X: FieldElementX8,
    pub Y: FieldElementX8,
    pub Z: FieldElementX8,
    pub T: FieldElementX8,
}

/// Eight projective points `(X : Y : Z)` for the doubling chain
#[derive(Clone, Copy, Debug)]
pub struct ProjectivePointX8 {
    pub X: FieldElementX8,
    pub Y: FieldElementX8,
    pub Z: FieldElementX8,
}

/// Eight cached projective Niels points for readdition
#[derive(Clone, Copy, Debug)]
pub struct ProjectiveNielsX8 {
    pub Y_plus_X: FieldElementX8,
    pub Y_minus_X: FieldElementX8,
    pub Z: FieldElementX8,
    pub T2d: FieldElementX8,
}

impl ExtendedPointX8 {
    pub const IDENTITY: ExtendedPointX8 = ExtendedPointX8 {
        X: FieldElementX8::ZERO,
        Y: FieldElementX8::ONE,
        Z: FieldElementX8::ONE,
        T: FieldElementX8::ZERO,
    };

    /// Gather eight serial points into lanes
    #[cfg(test)]
    pub fn from_lanes(points: &[EdwardsPoint; LANES]) -> ExtendedPointX8 {
        ExtendedPointX8 {
            X: FieldElementX8::from_lanes(&core::array::from_fn(|lane| points[lane].X)),
            Y: FieldElementX8::from_lanes(&core::array::from_fn(|lane| points[lane].Y)),
            Z: FieldElementX8::from_lanes(&core::array::from_fn(|lane| points[lane].Z)),
            T: FieldElementX8::from_lanes(&core::array::from_fn(|lane| points[lane].T)),
        }
    }

    /// Extract one lane as a serial point
    #[cfg(test)]
    pub fn lane(&self, lane: usize) -> EdwardsPoint {
        EdwardsPoint {
            X: self.X.lane(lane),
            Y: self.Y.lane(lane),
            Z: self.Z.lane(lane),
            T: self.T.lane(lane),
        }
    }

    pub fn as_projective(&self) -> ProjectivePointX8 {
        ProjectivePointX8 {
            X: self.X,
            Y: self.Y,
            Z: self.Z,
        }
    }

    pub fn to_projective_niels(self) -> ProjectiveNielsX8 {
        ProjectiveNielsX8 {
            Y_plus_X: self.Y.add(&self.X),
            Y_minus_X: self.Y.sub(&self.X),
            Z: self.Z,
            T2d: self.T.mul(&FieldElementX8::splat(EDWARDS_D2)),
        }
    }

    /// Add a cached point in every lane
    pub fn add(&self, other: &ProjectiveNielsX8) -> CompletedPointX8 {
        let Y_plus_X = self.Y.add(&self.X);
        let Y_minus_X = self.Y.sub(&self.X);
        let PP = Y_plus_X.mul(&other.Y_plus_X);
        let MM = Y_minus_X.mul(&other.Y_minus_X);
        let TT2d = self.T.mul(&other.T2d);
        let ZZ = self.Z.mul(&other.Z);
        let ZZ2 = ZZ.add(&ZZ);

        CompletedPointX8 {
            X: PP.sub(&MM),
            Y: PP.add(&MM),
            Z: ZZ2.add(&TT2d),
            T: ZZ2.sub(&TT2d),
        }
    }

    /// Compute `[2^k] P` in every lane by successive doublings
    pub fn mul_by_pow_2(&self, k: u32) -> ExtendedPointX8 {
        debug_assert!(k > 0);
        let mut r: CompletedPointX8;
        let mut s = self.as_projective();
        for _ in 0..(k - 1) {
            r = s.double();
            s = r.as_projective();
        }
        s.double().as_extended()
    }

    /// Flag the lanes that hold the identity, where `X = 0` and `Y = Z`
    pub fn is_identity_lanes(&self) -> LaneMask {
        let x_zero = self.X.is_zero_lanes();
        let y_eq_z = self.Y.sub(&self.Z).is_zero_lanes();
        core::array::from_fn(|lane| x_zero[lane] && y_eq_z[lane])
    }
}

impl CompletedPointX8 {
    pub fn as_projective(&self) -> ProjectivePointX8 {
        ProjectivePointX8 {
            X: self.X.mul(&self.T),
            Y: self.Y.mul(&self.Z),
            Z: self.Z.mul(&self.T),
        }
    }

    pub fn as_extended(&self) -> ExtendedPointX8 {
        ExtendedPointX8 {
            X: self.X.mul(&self.T),
            Y: self.Y.mul(&self.Z),
            Z: self.Z.mul(&self.T),
            T: self.X.mul(&self.Y),
        }
    }
}

impl ProjectivePointX8 {
    pub fn double(&self) -> CompletedPointX8 {
        let XX = self.X.square();
        let YY = self.Y.square();
        let ZZ2 = self.Z.square2();
        let X_plus_Y = self.X.add(&self.Y);
        let X_plus_Y_sq = X_plus_Y.square();
        let YY_plus_XX = YY.add(&XX);
        let YY_minus_XX = YY.sub(&XX);

        CompletedPointX8 {
            X: X_plus_Y_sq.sub(&YY_plus_XX),
            Y: YY_plus_XX,
            Z: YY_minus_XX,
            T: ZZ2.sub(&YY_minus_XX),
        }
    }
}

impl ProjectiveNielsX8 {
    /// The identity as a cached point
    pub const IDENTITY: ProjectiveNielsX8 = ProjectiveNielsX8 {
        Y_plus_X: FieldElementX8::ONE,
        Y_minus_X: FieldElementX8::ONE,
        Z: FieldElementX8::ONE,
        T2d: FieldElementX8::ZERO,
    };

    /// Copy one lane of `source` into the same lane of `self`
    fn copy_lane(&mut self, lane: usize, source: &ProjectiveNielsX8) {
        for i in 0..5 {
            self.Y_plus_X.limbs[i][lane] = source.Y_plus_X.limbs[i][lane];
            self.Y_minus_X.limbs[i][lane] = source.Y_minus_X.limbs[i][lane];
            self.Z.limbs[i][lane] = source.Z.limbs[i][lane];
            self.T2d.limbs[i][lane] = source.T2d.limbs[i][lane];
        }
    }

    /// Negate the flagged lanes by swapping `Y + X` with `Y - X` and negating `2dT`
    fn negate_lanes(&self, negate: &LaneMask) -> ProjectiveNielsX8 {
        let neg_T2d = self.T2d.neg();
        let mut out = *self;
        for (lane, should_negate) in negate.iter().enumerate() {
            if *should_negate {
                for i in 0..5 {
                    out.Y_plus_X.limbs[i][lane] = self.Y_minus_X.limbs[i][lane];
                    out.Y_minus_X.limbs[i][lane] = self.Y_plus_X.limbs[i][lane];
                    out.T2d.limbs[i][lane] = neg_T2d.limbs[i][lane];
                }
            }
        }
        out
    }
}

/// The multiples `[P, 2P, ..., 8P]` of eight points, one table per lane
pub struct LookupTableX8(pub [ProjectiveNielsX8; TABLE_SIZE]);

impl LookupTableX8 {
    /// Build `[P, 2P, ..., 8P]` for all lanes at once
    pub fn from_extended(P: &ExtendedPointX8) -> LookupTableX8 {
        let mut points = [P.to_projective_niels(); TABLE_SIZE];
        for index in 0..TABLE_SIZE - 1 {
            points[index + 1] = P.add(&points[index]).as_extended().to_projective_niels();
        }
        LookupTableX8(points)
    }

    /// Select `[d]P` per lane for a signed radix-16 digit `d` in -8..=8
    pub fn select(&self, digits: &[i8; LANES]) -> ProjectiveNielsX8 {
        let mut out = ProjectiveNielsX8::IDENTITY;
        let mut negate: LaneMask = [false; LANES];
        for lane in 0..LANES {
            let digit = digits[lane];
            if digit != 0 {
                let index = (digit.unsigned_abs() as usize) - 1;
                debug_assert!(index < TABLE_SIZE);
                out.copy_lane(lane, &self.0[index]);
                negate[lane] = digit < 0;
            }
        }
        out.negate_lanes(&negate)
    }
}

/// Decompress eight encodings under the ZIP-215 rules, with invalid lanes flagged and set to the identity
pub fn decompress_x8(encodings: &[[u8; 32]; LANES]) -> (ExtendedPointX8, LaneMask) {
    let Y = FieldElementX8::from_lanes(&core::array::from_fn(|lane| {
        FieldElement51::from_bytes(&encodings[lane])
    }));
    let Z = FieldElementX8::ONE;
    let YY = Y.square();
    let u = YY.sub(&Z);
    let v = YY.mul(&FieldElementX8::splat(EDWARDS_D)).add(&Z);
    let (is_valid, X) = sqrt_ratio_i_x8(&u, &v);

    // The root comes back nonnegative, so the encoded sign bit picks the sign
    let sign_bits: LaneMask = core::array::from_fn(|lane| encodings[lane][31] >> 7 == 1);
    let X = X.negate_lanes(&sign_bits);

    let point = ExtendedPointX8 {
        T: X.mul(&Y),
        X,
        Y,
        Z,
    };

    let mut out = ExtendedPointX8::IDENTITY;
    let keep: LaneMask = is_valid;
    out.X = out.X.select_lanes(&point.X, &keep);
    out.Y = out.Y.select_lanes(&point.Y, &keep);
    out.Z = out.Z.select_lanes(&point.Z, &keep);
    out.T = out.T.select_lanes(&point.T, &keep);

    (out, is_valid)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::ED25519_BASEPOINT_POINT;
    use crate::edwards::CompressedEdwardsY;
    use crate::scalar::Scalar;
    use crate::traits::Identity;

    fn random_points(seed: u64) -> [EdwardsPoint; LANES] {
        core::array::from_fn(|lane| {
            let mut bytes = [0u8; 32];
            bytes[..8].copy_from_slice(&(seed.wrapping_mul(1 + lane as u64)).to_le_bytes());
            bytes[8] = 1;
            ED25519_BASEPOINT_POINT * Scalar::from_bytes_mod_order(bytes)
        })
    }

    fn assert_lanes_eq(points: &ExtendedPointX8, expected: &[EdwardsPoint; LANES], what: &str) {
        for (lane, expected) in expected.iter().enumerate() {
            assert_eq!(
                points.lane(lane).compress(),
                expected.compress(),
                "{what} lane {lane}"
            );
        }
    }

    // lane-wise add, double and mul_by_pow_2 match the serial point operations
    #[test]
    fn point_arithmetic() {
        let left_points = random_points(11);
        let right_points = random_points(23);
        let left = ExtendedPointX8::from_lanes(&left_points);
        let right = ExtendedPointX8::from_lanes(&right_points);

        let sum = left.add(&right.to_projective_niels()).as_extended();
        let expected_sum: [EdwardsPoint; LANES] =
            core::array::from_fn(|lane| left_points[lane] + right_points[lane]);
        assert_lanes_eq(&sum, &expected_sum, "add");

        let doubled = left.as_projective().double().as_extended();
        let expected_double: [EdwardsPoint; LANES] =
            core::array::from_fn(|lane| left_points[lane].double());
        assert_lanes_eq(&doubled, &expected_double, "double");

        let times_eight = left.mul_by_pow_2(3);
        let expected_times_eight: [EdwardsPoint; LANES] =
            core::array::from_fn(|lane| left_points[lane].mul_by_pow_2(3));
        assert_lanes_eq(&times_eight, &expected_times_eight, "mul_by_pow_2");
    }

    // a table select yields the signed multiple of each lane's digit
    #[test]
    fn table_select() {
        let points = random_points(37);
        let table = LookupTableX8::from_extended(&ExtendedPointX8::from_lanes(&points));

        let digits: [i8; LANES] = [0, 1, -1, 8, -8, 3, -5, 7];
        let selected = ExtendedPointX8::IDENTITY
            .add(&table.select(&digits))
            .as_extended();

        for lane in 0..LANES {
            let expected = if digits[lane] == 0 {
                EdwardsPoint::identity()
            } else {
                let multiple = points[lane] * Scalar::from(digits[lane].unsigned_abs() as u64);
                if digits[lane] < 0 {
                    -multiple
                } else {
                    multiple
                }
            };
            assert_eq!(
                selected.lane(lane).compress(),
                expected.compress(),
                "digit {} lane {lane}",
                digits[lane]
            );
        }
    }

    // lane-wise decompression matches the serial one, invalid lanes included
    #[test]
    fn decompress() {
        let originals = random_points(53);
        let mut encodings: [[u8; 32]; LANES] =
            core::array::from_fn(|lane| originals[lane].compress().to_bytes());

        // lane 2 flips the sign bit, lane 4 is off the curve, lane 6 is the identity
        encodings[2][31] ^= 1 << 7;
        encodings[4] = [0x05; 32];
        encodings[6] = CompressedEdwardsY::identity().to_bytes();

        let (points, is_valid) = decompress_x8(&encodings);

        for lane in 0..LANES {
            match CompressedEdwardsY(encodings[lane]).decompress() {
                Some(expected) => {
                    assert!(is_valid[lane], "lane {lane} should be valid");
                    assert_eq!(
                        points.lane(lane).compress(),
                        expected.compress(),
                        "decompress lane {lane}"
                    );
                }
                None => assert!(!is_valid[lane], "lane {lane} should be invalid"),
            }
        }

        let identity_lanes = points.is_identity_lanes();
        assert!(identity_lanes[6], "identity lane detected");
        assert!(!identity_lanes[0], "non-identity lane not flagged");
    }
}
