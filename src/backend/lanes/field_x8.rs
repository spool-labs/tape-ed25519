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
    pub const fn splat(element: FieldElement51) -> FieldElementX8 {
        FieldElementX8 {
            limbs: [
                [element.0[0]; LANES],
                [element.0[1]; LANES],
                [element.0[2]; LANES],
                [element.0[3]; LANES],
                [element.0[4]; LANES],
            ],
        }
    }

    /// Gather eight independent elements into lane order
    pub fn from_lanes(lanes: &[FieldElement51; LANES]) -> FieldElementX8 {
        let mut out = FieldElementX8::ZERO;
        for (lane, element) in lanes.iter().enumerate() {
            out.set_lane(lane, element);
        }
        out
    }

    /// Extract one lane as a serial field element
    pub fn lane(&self, lane: usize) -> FieldElement51 {
        FieldElement51::from_limbs([
            self.limbs[0][lane],
            self.limbs[1][lane],
            self.limbs[2][lane],
            self.limbs[3][lane],
            self.limbs[4][lane],
        ])
    }

    /// Overwrite one lane
    pub fn set_lane(&mut self, lane: usize, element: &FieldElement51) {
        for i in 0..5 {
            self.limbs[i][lane] = element.0[i];
        }
    }

    /// Scatter back to eight serial elements
    #[cfg(test)]
    pub fn to_lanes(self) -> [FieldElement51; LANES] {
        core::array::from_fn(|lane| self.lane(lane))
    }

    /// Apply a binary serial operation lane by lane
    #[inline]
    fn zip_lanes(
        &self,
        rhs: &FieldElementX8,
        operation: impl Fn(&FieldElement51, &FieldElement51) -> FieldElement51,
    ) -> FieldElementX8 {
        let mut out = FieldElementX8::ZERO;
        for lane in 0..LANES {
            out.set_lane(lane, &operation(&self.lane(lane), &rhs.lane(lane)));
        }
        out
    }

    /// Apply a unary serial operation lane by lane
    #[inline]
    fn map_lanes(&self, operation: impl Fn(&FieldElement51) -> FieldElement51) -> FieldElementX8 {
        let mut out = FieldElementX8::ZERO;
        for lane in 0..LANES {
            out.set_lane(lane, &operation(&self.lane(lane)));
        }
        out
    }

    pub fn add(&self, rhs: &FieldElementX8) -> FieldElementX8 {
        #[cfg(curve25519_backend = "simd")]
        if use_ifma() {
            // SAFETY: use_ifma checked the CPU for every feature the kernel needs
            return unsafe { ifma_x8::add_x8(self, rhs) };
        }
        self.zip_lanes(rhs, |a, b| a + b)
    }

    pub fn sub(&self, rhs: &FieldElementX8) -> FieldElementX8 {
        #[cfg(curve25519_backend = "simd")]
        if use_ifma() {
            // SAFETY: use_ifma checked the CPU for every feature the kernel needs
            return unsafe { ifma_x8::sub_x8(self, rhs) };
        }
        self.zip_lanes(rhs, |a, b| a - b)
    }

    pub fn mul(&self, rhs: &FieldElementX8) -> FieldElementX8 {
        #[cfg(curve25519_backend = "simd")]
        if use_ifma() {
            // SAFETY: use_ifma checked the CPU for every feature the kernel needs
            return unsafe { ifma_x8::mul_x8(self, rhs) };
        }
        self.zip_lanes(rhs, |a, b| a * b)
    }

    pub fn neg(&self) -> FieldElementX8 {
        #[cfg(curve25519_backend = "simd")]
        if use_ifma() {
            // SAFETY: use_ifma checked the CPU for every feature the kernel needs
            return unsafe { ifma_x8::neg_x8(self) };
        }
        self.map_lanes(|a| -a)
    }

    pub fn square(&self) -> FieldElementX8 {
        #[cfg(curve25519_backend = "simd")]
        if use_ifma() {
            // SAFETY: use_ifma checked the CPU for every feature the kernel needs
            return unsafe { ifma_x8::square_x8(self) };
        }
        self.map_lanes(|a| a.square())
    }

    /// Compute `2 a^2` in every lane
    pub fn square2(&self) -> FieldElementX8 {
        #[cfg(curve25519_backend = "simd")]
        if use_ifma() {
            // SAFETY: use_ifma checked the CPU for every feature the kernel needs
            return unsafe { ifma_x8::square2_x8(self) };
        }
        self.map_lanes(|a| a.square2())
    }

    /// Square each lane `k` times
    pub fn pow2k(&self, k: u32) -> FieldElementX8 {
        debug_assert!(k > 0);
        #[cfg(curve25519_backend = "simd")]
        if use_ifma() {
            // SAFETY: use_ifma checked the CPU for every feature the kernel needs
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
        core::array::from_fn(|lane| self.lane(lane).to_bytes())
    }

    /// Flag the lanes that are zero mod p
    pub fn is_zero_lanes(&self) -> LaneMask {
        let bytes = self.to_bytes_lanes();
        core::array::from_fn(|lane| bytes[lane] == [0u8; 32])
    }

    /// Flag the lanes whose canonical encoding is odd
    pub fn is_negative_lanes(&self) -> LaneMask {
        let bytes = self.to_bytes_lanes();
        core::array::from_fn(|lane| bytes[lane][0] & 1 == 1)
    }

    /// Take `other` where the mask is true and `self` elsewhere
    pub fn select_lanes(&self, other: &FieldElementX8, take_other: &LaneMask) -> Self {
        let mut out = *self;
        for (lane, should_take) in take_other.iter().enumerate() {
            if *should_take {
                for i in 0..5 {
                    out.limbs[i][lane] = other.limbs[i][lane];
                }
            }
        }
        out
    }

    /// Negate every lane whose mask flag is true
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

    let correct_sign: LaneMask = core::array::from_fn(|lane| check_bytes[lane] == u_bytes[lane]);
    let flipped_sign: LaneMask =
        core::array::from_fn(|lane| check_bytes[lane] == u_neg_bytes[lane]);
    let flipped_sign_i: LaneMask =
        core::array::from_fn(|lane| check_bytes[lane] == u_neg_i_bytes[lane]);

    let r_prime = r.mul(&sqrt_m1);
    let take_prime: LaneMask =
        core::array::from_fn(|lane| flipped_sign[lane] || flipped_sign_i[lane]);
    let mut r = r.select_lanes(&r_prime, &take_prime);

    let r_is_negative = r.is_negative_lanes();
    r = r.negate_lanes(&r_is_negative);

    let was_square: LaneMask =
        core::array::from_fn(|lane| correct_sign[lane] || flipped_sign[lane]);

    (was_square, r)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::FieldElement;

    // deterministic canonical pseudorandom elements
    fn random_elements(seed: u64) -> [FieldElement51; LANES] {
        let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        core::array::from_fn(|_| {
            let mut bytes = [0u8; 32];
            for word in bytes.chunks_mut(8) {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                word.copy_from_slice(&state.to_le_bytes());
            }
            bytes[31] &= 0x7f;
            FieldElement51::from_bytes(&bytes)
        })
    }

    // lane-wise add, sub, mul, square and pow_p58 match the serial element
    #[test]
    fn lane_arithmetic() {
        let a_lanes = random_elements(1);
        let b_lanes = random_elements(2);
        let a = FieldElementX8::from_lanes(&a_lanes);
        let b = FieldElementX8::from_lanes(&b_lanes);

        let sum = a.add(&b).to_lanes();
        let difference = a.sub(&b).to_lanes();
        let product = a.mul(&b).to_lanes();
        let squared = a.square().to_lanes();
        let power = a.pow_p58().to_lanes();

        for lane in 0..LANES {
            assert_eq!(
                sum[lane].to_bytes(),
                (&a_lanes[lane] + &b_lanes[lane]).to_bytes()
            );
            assert_eq!(
                difference[lane].to_bytes(),
                (&a_lanes[lane] - &b_lanes[lane]).to_bytes()
            );
            assert_eq!(
                product[lane].to_bytes(),
                (&a_lanes[lane] * &b_lanes[lane]).to_bytes()
            );
            assert_eq!(squared[lane].to_bytes(), a_lanes[lane].square().to_bytes());
            assert_eq!(power[lane].to_bytes(), a_lanes[lane].pow_p58().to_bytes());
        }
    }

    // lane-wise sqrt_ratio_i matches the serial one, zero inputs included
    #[test]
    fn sqrt_ratio() {
        let u_lanes = random_elements(3);
        let v_lanes = random_elements(4);

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

        for lane in 0..LANES {
            let (serial_is_square, serial_root) =
                FieldElement::sqrt_ratio_i(&u_lanes[lane], &v_lanes[lane]);
            assert_eq!(
                mask[lane],
                bool::from(serial_is_square),
                "square mask lane {lane}"
            );
            assert_eq!(
                roots[lane].to_bytes(),
                serial_root.to_bytes(),
                "root lane {lane}"
            );
        }
    }
}
