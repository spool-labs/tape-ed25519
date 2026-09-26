//! Eight field elements side by side, one per signature lane

// Vendored from the solana-ed25519 lanes fork, BSD-3-Clause

#![allow(non_snake_case)]

use crate::backend::serial::u64::field::FieldElement51;

#[cfg(curve25519_backend = "simd")]
use super::ifma_x8;

/// Report whether this host runs the AVX-512 IFMA kernels
#[cfg(curve25519_backend = "simd")]
#[inline]
fn use_ifma() -> bool {
    ifma_x8::available()
}

/// Independent field elements held side by side
pub const LANES: usize = 8;

/// One flag per lane, `true` for a live lane
pub type LaneMask = [bool; LANES];

/// Eight radix 2^51 field elements stored limb-major, so `limbs[i][lane]` is limb `i` of a lane
#[derive(Clone, Copy, Debug)]
// The alignment keeps each limb row inside one cache line
#[repr(align(64))]
pub struct FieldElementX8 {
    pub limbs: [[u64; LANES]; 5],
}

impl FieldElementX8 {
    pub const ZERO: FieldElementX8 = FieldElementX8::splat(FieldElement51::ZERO);
    pub const ONE: FieldElementX8 = FieldElementX8::splat(FieldElement51::ONE);

    /// Broadcast one field element into all lanes
    pub const fn splat(fe: FieldElement51) -> FieldElementX8 {
        FieldElementX8 {
            limbs: [
                [fe.0[0]; LANES],
                [fe.0[1]; LANES],
                [fe.0[2]; LANES],
                [fe.0[3]; LANES],
                [fe.0[4]; LANES],
            ],
        }
    }

    /// Gather eight independent elements into lane order
    pub fn from_lanes(lanes: &[FieldElement51; LANES]) -> FieldElementX8 {
        let mut out = FieldElementX8::ZERO;
        for (l, fe) in lanes.iter().enumerate() {
            out.set_lane(l, fe);
        }
        out
    }

    /// Extract lane `l` as a serial field element
    pub fn lane(&self, l: usize) -> FieldElement51 {
        FieldElement51::from_limbs([
            self.limbs[0][l],
            self.limbs[1][l],
            self.limbs[2][l],
            self.limbs[3][l],
            self.limbs[4][l],
        ])
    }

    /// Overwrite lane `l`
    pub fn set_lane(&mut self, l: usize, fe: &FieldElement51) {
        for i in 0..5 {
            self.limbs[i][l] = fe.0[i];
        }
    }

    /// Scatter back to eight serial elements
    #[cfg(test)]
    pub fn to_lanes(self) -> [FieldElement51; LANES] {
        core::array::from_fn(|l| self.lane(l))
    }

    /// Apply a binary serial operation lane by lane
    #[inline]
    fn zip_lanes(
        &self,
        rhs: &FieldElementX8,
        op: impl Fn(&FieldElement51, &FieldElement51) -> FieldElement51,
    ) -> FieldElementX8 {
        let mut out = FieldElementX8::ZERO;
        for l in 0..LANES {
            out.set_lane(l, &op(&self.lane(l), &rhs.lane(l)));
        }
        out
    }

    /// Apply a unary serial operation lane by lane
    #[inline]
    fn map_lanes(&self, op: impl Fn(&FieldElement51) -> FieldElement51) -> FieldElementX8 {
        let mut out = FieldElementX8::ZERO;
        for l in 0..LANES {
            out.set_lane(l, &op(&self.lane(l)));
        }
        out
    }

    pub fn add(&self, rhs: &FieldElementX8) -> FieldElementX8 {
        #[cfg(curve25519_backend = "simd")]
        if use_ifma() {
            return unsafe { ifma_x8::add_x8(self, rhs) };
        }
        self.zip_lanes(rhs, |a, b| a + b)
    }

    pub fn sub(&self, rhs: &FieldElementX8) -> FieldElementX8 {
        #[cfg(curve25519_backend = "simd")]
        if use_ifma() {
            return unsafe { ifma_x8::sub_x8(self, rhs) };
        }
        self.zip_lanes(rhs, |a, b| a - b)
    }

    pub fn mul(&self, rhs: &FieldElementX8) -> FieldElementX8 {
        #[cfg(curve25519_backend = "simd")]
        if use_ifma() {
            return unsafe { ifma_x8::mul_x8(self, rhs) };
        }
        self.zip_lanes(rhs, |a, b| a * b)
    }

    pub fn neg(&self) -> FieldElementX8 {
        #[cfg(curve25519_backend = "simd")]
        if use_ifma() {
            return unsafe { ifma_x8::neg_x8(self) };
        }
        self.map_lanes(|a| -a)
    }

    pub fn square(&self) -> FieldElementX8 {
        #[cfg(curve25519_backend = "simd")]
        if use_ifma() {
            return unsafe { ifma_x8::square_x8(self) };
        }
        self.map_lanes(|a| a.square())
    }

    /// Compute `2 a^2` in every lane
    pub fn square2(&self) -> FieldElementX8 {
        #[cfg(curve25519_backend = "simd")]
        if use_ifma() {
            return unsafe { ifma_x8::square2_x8(self) };
        }
        self.map_lanes(|a| a.square2())
    }

    /// Square each lane `k` times
    pub fn pow2k(&self, k: u32) -> FieldElementX8 {
        debug_assert!(k > 0);
        #[cfg(curve25519_backend = "simd")]
        if use_ifma() {
            return unsafe { ifma_x8::pow2k_x8(self, k) };
        }
        self.map_lanes(|a| a.pow2k(k))
    }

    /// Return `(self^(2^250 - 1), self^11)` with the serial `pow22501` chain
    fn pow22501(&self) -> (FieldElementX8, FieldElementX8) {
        let t0 = self.square(); // 1
        let t1 = t0.square().square(); // 3
        let t2 = self.mul(&t1); // 3,0
        let t3 = t0.mul(&t2); // 3,1,0
        let t4 = t3.square(); // 4,2,1
        let t5 = t2.mul(&t4); // 4,3,2,1,0
        let t6 = t5.pow2k(5); // 9,8,7,6,5
        let t7 = t6.mul(&t5); // 9,8,7,6,5,4,3,2,1,0
        let t8 = t7.pow2k(10); // 19..10
        let t9 = t8.mul(&t7); // 19..0
        let t10 = t9.pow2k(20); // 39..20
        let t11 = t10.mul(&t9); // 39..0
        let t12 = t11.pow2k(10); // 49..10
        let t13 = t12.mul(&t7); // 49..0
        let t14 = t13.pow2k(50); // 99..50
        let t15 = t14.mul(&t13); // 99..0
        let t16 = t15.pow2k(100); // 199..100
        let t17 = t16.mul(&t15); // 199..0
        let t18 = t17.pow2k(50); // 249..50
        let t19 = t18.mul(&t13); // 249..0

        (t19, t3)
    }

    /// Raise every lane to the power `(p - 5) / 8 = 2^252 - 3`
    pub fn pow_p58(&self) -> FieldElementX8 {
        let (t19, _) = self.pow22501(); // 249..0
        let t20 = t19.pow2k(2); // 251..2
        self.mul(&t20) // 251..2,0
    }

    /// Canonical little-endian encodings, one per lane
    pub fn to_bytes_lanes(self) -> [[u8; 32]; LANES] {
        core::array::from_fn(|l| self.lane(l).to_bytes())
    }

    /// Flag the lanes that are zero mod p
    pub fn is_zero_lanes(&self) -> LaneMask {
        let bytes = self.to_bytes_lanes();
        core::array::from_fn(|l| bytes[l] == [0u8; 32])
    }

    /// Flag the lanes whose canonical encoding is odd
    pub fn is_negative_lanes(&self) -> LaneMask {
        let bytes = self.to_bytes_lanes();
        core::array::from_fn(|l| bytes[l][0] & 1 == 1)
    }

    /// Take `other` in the lanes the mask sets and `self` in the rest
    pub fn select_lanes(&self, other: &FieldElementX8, take_other: &LaneMask) -> Self {
        let mut out = *self;
        for (l, take) in take_other.iter().enumerate() {
            if *take {
                for i in 0..5 {
                    out.limbs[i][l] = other.limbs[i][l];
                }
            }
        }
        out
    }

    /// Negate the lanes the mask sets
    pub fn negate_lanes(&self, negate: &LaneMask) -> Self {
        self.select_lanes(&self.neg(), negate)
    }
}

/// Compute the nonnegative `sqrt(u / v)` per lane with a mask of square lanes, as `FieldElement::sqrt_ratio_i` does
pub fn sqrt_ratio_i_x8(u: &FieldElementX8, v: &FieldElementX8) -> (LaneMask, FieldElementX8) {
    use crate::constants::SQRT_M1;

    let sqrt_m1 = FieldElementX8::splat(SQRT_M1);

    let v3 = v.square().mul(v);
    let v7 = v3.square().mul(v);
    let r = u.mul(&v3).mul(&u.mul(&v7).pow_p58());
    let check = v.mul(&r.square());

    let u_neg = u.neg();
    let u_neg_i = u_neg.mul(&sqrt_m1);

    let check_bytes = check.to_bytes_lanes();
    let u_bytes = u.to_bytes_lanes();
    let u_neg_bytes = u_neg.to_bytes_lanes();
    let u_neg_i_bytes = u_neg_i.to_bytes_lanes();

    let correct_sign: LaneMask = core::array::from_fn(|l| check_bytes[l] == u_bytes[l]);
    let flipped_sign: LaneMask = core::array::from_fn(|l| check_bytes[l] == u_neg_bytes[l]);
    let flipped_sign_i: LaneMask = core::array::from_fn(|l| check_bytes[l] == u_neg_i_bytes[l]);

    let r_prime = r.mul(&sqrt_m1);
    let take_prime: LaneMask = core::array::from_fn(|l| flipped_sign[l] || flipped_sign_i[l]);
    let mut r = r.select_lanes(&r_prime, &take_prime);

    let r_is_negative = r.is_negative_lanes();
    r = r.negate_lanes(&r_is_negative);

    let was_square: LaneMask = core::array::from_fn(|l| correct_sign[l] || flipped_sign[l]);

    (was_square, r)
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::field::FieldElement;

    // deterministic canonical pseudorandom elements
    fn test_elements(seed: u64) -> [FieldElement51; LANES] {
        let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        core::array::from_fn(|_| {
            let mut bytes = [0u8; 32];
            for chunk in bytes.chunks_mut(8) {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                chunk.copy_from_slice(&state.to_le_bytes());
            }
            bytes[31] &= 0x7f;
            FieldElement51::from_bytes(&bytes)
        })
    }

    // lane-wise add, sub, mul, square and pow_p58 match the serial element
    #[test]
    fn lane_ops_match_serial() {
        let a_lanes = test_elements(1);
        let b_lanes = test_elements(2);
        let a = FieldElementX8::from_lanes(&a_lanes);
        let b = FieldElementX8::from_lanes(&b_lanes);

        let sum = a.add(&b).to_lanes();
        let diff = a.sub(&b).to_lanes();
        let prod = a.mul(&b).to_lanes();
        let sq = a.square().to_lanes();
        let p58 = a.pow_p58().to_lanes();

        for l in 0..LANES {
            assert_eq!(sum[l].to_bytes(), (&a_lanes[l] + &b_lanes[l]).to_bytes());
            assert_eq!(diff[l].to_bytes(), (&a_lanes[l] - &b_lanes[l]).to_bytes());
            assert_eq!(prod[l].to_bytes(), (&a_lanes[l] * &b_lanes[l]).to_bytes());
            assert_eq!(sq[l].to_bytes(), a_lanes[l].square().to_bytes());
            assert_eq!(p58[l].to_bytes(), a_lanes[l].pow_p58().to_bytes());
        }
    }

    // lane-wise sqrt_ratio_i matches the serial one, zero inputs included
    #[test]
    fn sqrt_ratio_matches_serial() {
        let u_lanes = test_elements(3);
        let v_lanes = test_elements(4);

        // lanes 0 and 1 cover u = 0 and v = 0
        let mut u_lanes = u_lanes;
        let mut v_lanes = v_lanes;
        u_lanes[0] = FieldElement51::ZERO;
        v_lanes[1] = FieldElement51::ZERO;
        // lane 2 holds u / v = w^2, which is always square
        u_lanes[2] = &u_lanes[2].square() * &v_lanes[2];

        let (mask, root) = sqrt_ratio_i_x8(
            &FieldElementX8::from_lanes(&u_lanes),
            &FieldElementX8::from_lanes(&v_lanes),
        );
        let roots = root.to_lanes();

        for l in 0..LANES {
            let (serial_ok, serial_root) = FieldElement::sqrt_ratio_i(&u_lanes[l], &v_lanes[l]);
            assert_eq!(mask[l], bool::from(serial_ok), "square mask lane {l}");
            assert_eq!(roots[l].to_bytes(), serial_root.to_bytes(), "root lane {l}");
        }
    }
}
