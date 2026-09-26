//! Split the challenge scalar into half-size rho and tau with the half extended Euclidean algorithm

// Vendored from the solana-ed25519 lanes fork, BSD-3-Clause

use core::cmp::Ordering;
use core::ops::{Neg, Shl};

use crate::constants;

/// A signed 256-bit integer as a sign and four little-endian u64 limbs of magnitude
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct I256 {
    limbs: [u64; 4],
    is_negative: bool,
}

/// Every digit of a half-size scalar sits at or below this bit position
pub const HEEA_MAX_INDEX: usize = 129;

/// The group order l is this many bits long
const ORDER_BITS: u32 = 253;

/// The loop stops once the remainder fits in this many bits
const HALF_SIZE_BITS: u32 = 127;

/// A tau longer than this sends the batched result to the exact algorithm
const TAU_LIMIT_BITS: u32 = 130;

/// Return signed `(rho, tau)` with `rho = tau v mod l`, one exact step at a time
pub fn curve25519_heea_vartime(v: I256) -> (I256, I256) {
    let mut r0: I256 = (&constants::BASEPOINT_ORDER).into();
    let mut r1 = v;

    debug_assert!(r1.abs() < r0, "Input v must be less than the group order L");

    let mut t0: I256 = I256::ZERO;
    let mut t1: I256 = I256::ONE;

    let mut bl_r0 = ORDER_BITS;
    let mut bl_r1 = bit_length_i256(r1);

    while bl_r1 > HALF_SIZE_BITS {
        let s = bl_r0 - bl_r1;

        let sign_r0 = r0.is_below_zero();
        let sign_r1 = r1.is_below_zero();

        let (r, t) = if sign_r0 == sign_r1 {
            (r0.wrapping_sub(r1 << s), t0.wrapping_sub(t1 << s))
        } else {
            (r0.wrapping_add(r1 << s), t0.wrapping_add(t1 << s))
        };

        let bl_r = bit_length_i256(r);

        // A remainder longer than r1 replaces r0, a shorter one rotates the pair
        if bl_r > bl_r1 {
            r0 = r;
            t0 = t;
            bl_r0 = bl_r;
        } else {
            r0 = r1;
            r1 = r;
            t0 = t1;
            t1 = t;
            bl_r0 = bl_r1;
            bl_r1 = bl_r;
        }
    }

    (r1, t1)
}

/// Radix-2^64 limbs of a two's-complement 256-bit value
type Limbs = [u64; 4];

/// Bits of approximant kept above the shift headroom
const WINDOW: u32 = 54;
/// Bits a batch matrix may grow to before it is applied
const MAT_CAP_BITS: u32 = 34;
/// Largest shift the approximant absorbs, and a larger one takes an exact step
const MAX_SHIFT: u32 = 8;

/// The group order l in two's-complement limbs
const ORDER_LIMBS: Limbs = [
    0x5812631a5cf5d3ed,
    0x14def9dea2f79cd6,
    0,
    0x1000000000000000,
];

#[inline(always)]
fn negate_limbs(v: &Limbs) -> Limbs {
    let mut out = [0u64; 4];
    let mut carry = 1u64;
    for (o, &x) in out.iter_mut().zip(v.iter()) {
        let (r, c) = (!x).overflowing_add(carry);
        *o = r;
        carry = c as u64;
    }
    out
}

/// Split a two's-complement value into magnitude and sign without a branch
#[inline(always)]
fn magnitude(v: &Limbs) -> (Limbs, bool) {
    let neg = (v[3] as i64) < 0;
    let mask = (neg as u64).wrapping_neg();
    let mut out = [0u64; 4];
    let mut carry = neg as u64;
    for (o, &x) in out.iter_mut().zip(v.iter()) {
        let (r, c) = (x ^ mask).overflowing_add(carry);
        *o = r;
        carry = c as u64;
    }
    (out, neg)
}

#[inline(always)]
fn bit_length(mag: &Limbs) -> u32 {
    let mut bl = 0u32;
    for (i, &limb) in mag.iter().enumerate() {
        let here = 64 * (i as u32) + 64 - limb.leading_zeros();
        bl = if limb != 0 { here } else { bl };
    }
    bl
}

/// Take the bits of a magnitude from bit `e` up as a signed 64-bit approximant
#[inline(always)]
fn approx(mag: &Limbs, neg: bool, e: u32) -> i64 {
    let limb = (e / 64) as usize;
    let bit = e % 64;
    let get = |i: usize| -> u64 {
        if i < 4 {
            mag[i]
        } else {
            0
        }
    };
    let lo = get(limb);
    let hi = get(limb + 1);
    // The double shift keeps a zero bit offset from shifting by 64
    let m = ((lo >> bit) | ((hi << (63 - bit)) << 1)) as i64;
    if neg {
        -m
    } else {
        m
    }
}

/// Multiply a two's-complement value by a signed word mod 2^256
#[inline(always)]
fn scale(c: i64, v: &Limbs) -> Limbs {
    let cu = c as u64;
    let mut out = [0u64; 4];
    let mut carry = 0u64;
    for (o, &x) in out.iter_mut().zip(v.iter()) {
        let p = (x as u128) * (cu as u128) + carry as u128;
        *o = p as u64;
        carry = (p >> 64) as u64;
    }
    // A negative c read as u64 adds an extra 2^64 v, which this subtracts
    let mask = ((c < 0) as u64).wrapping_neg();
    let shifted = [0u64, v[0] & mask, v[1] & mask, v[2] & mask];
    let mut borrow = 0u64;
    for (o, &x) in out.iter_mut().zip(shifted.iter()) {
        let (r, b1) = o.overflowing_sub(x);
        let (r, b2) = r.overflowing_sub(borrow);
        *o = r;
        borrow = (b1 | b2) as u64;
    }
    out
}

#[inline(always)]
fn add_wrapping(a: &Limbs, b: &Limbs) -> Limbs {
    let mut out = [0u64; 4];
    let mut carry = 0u64;
    for i in 0..4 {
        let (r, c1) = a[i].overflowing_add(b[i]);
        let (r, c2) = r.overflowing_add(carry);
        out[i] = r;
        carry = (c1 | c2) as u64;
    }
    out
}

/// Compute `c0 a + c1 b`, which every call site keeps below 2^255
#[inline(always)]
fn linear(c0: i64, a: &Limbs, c1: i64, b: &Limbs) -> Limbs {
    add_wrapping(&scale(c0, a), &scale(c1, b))
}

/// Shift left by any amount below 256
fn shl_limbs(v: &Limbs, s: u32) -> Limbs {
    let words = (s / 64) as usize;
    let bits = s % 64;
    let mut out = [0u64; 4];
    for i in words..4 {
        out[i] = v[i - words] << bits;
        if bits != 0 && i > words {
            out[i] |= v[i - words - 1] >> (64 - bits);
        }
    }
    out
}

fn limbs_from_i256(v: I256) -> Limbs {
    let bytes = v.to_le_bytes();
    let mut out = [0u64; 4];
    for (i, limb) in out.iter_mut().enumerate() {
        let mut word = [0u8; 8];
        word.copy_from_slice(&bytes[8 * i..8 * i + 8]);
        *limb = u64::from_le_bytes(word);
    }
    if v < I256::ZERO {
        negate_limbs(&out)
    } else {
        out
    }
}

fn limbs_to_i256(v: &Limbs) -> I256 {
    let (mag, neg) = magnitude(v);
    let mut bytes = [0u8; 32];
    for (i, limb) in mag.iter().enumerate() {
        bytes[8 * i..8 * i + 8].copy_from_slice(&limb.to_le_bytes());
    }
    let out = I256::from_le_bytes(bytes);
    if neg {
        -out
    } else {
        out
    }
}

/// Run the same algorithm with 64-bit approximants batching steps into a 2x2 matrix over the exact values
pub fn curve25519_heea_vartime_lehmer(v: I256) -> (I256, I256) {
    let mut r0 = ORDER_LIMBS;
    let mut r1 = limbs_from_i256(v);
    let mut t0: Limbs = [0; 4];
    let mut t1: Limbs = [1, 0, 0, 0];

    let (mut m0, mut n0) = magnitude(&r0);
    let (mut m1, mut n1) = magnitude(&r1);
    let mut bl_r0 = bit_length(&m0);
    let mut bl_r1 = bit_length(&m1);

    'outer: while bl_r1 > HALF_SIZE_BITS {
        let e = bl_r0.saturating_sub(WINDOW);
        let mut a0 = approx(&m0, n0, e);
        let mut a1 = approx(&m1, n1, e);
        let mut ab0 = bl_r0;
        let mut ab1 = bl_r1;
        let mut m = [1i64, 0, 0, 1];
        let mut mat_bits: u32 = 1;
        let mut has_progressed = false;

        loop {
            if ab1 <= HALF_SIZE_BITS {
                break;
            }
            let s = ab0 - ab1;
            if s > MAX_SHIFT || mat_bits + s + 1 > MAT_CAP_BITS {
                break;
            }
            // The mask is all ones when the signs differ, so the shifted term flips sign
            let flip = (a0 ^ a1) >> 63;
            let shifted = a1 << s;
            let a = a0 - ((shifted ^ flip) - flip);
            let bl_a = 64 - a.unsigned_abs().leading_zeros();
            if bl_a <= mat_bits + 4 {
                break;
            }
            let bl_new = e + bl_a;
            let row0 = m[0] - (((m[2] << s) ^ flip) - flip);
            let row1 = m[1] - (((m[3] << s) ^ flip) - flip);
            mat_bits += s + 1;
            has_progressed = true;

            // A result longer than the smaller term replaces a0, a shorter one rotates the pair
            let has_grown = bl_new > ab1;
            let na0 = if has_grown { a } else { a1 };
            let na1 = if has_grown { a1 } else { a };
            let nb0 = if has_grown { bl_new } else { ab1 };
            let nb1 = if has_grown { ab1 } else { bl_new };
            m = if has_grown {
                [row0, row1, m[2], m[3]]
            } else {
                [m[2], m[3], row0, row1]
            };
            a0 = na0;
            a1 = na1;
            ab0 = nb0;
            ab1 = nb1;
        }

        if has_progressed {
            let nr0 = linear(m[0], &r0, m[1], &r1);
            let nr1 = linear(m[2], &r0, m[3], &r1);
            let nt0 = linear(m[0], &t0, m[1], &t1);
            let nt1 = linear(m[2], &t0, m[3], &t1);
            r0 = nr0;
            r1 = nr1;
            t0 = nt0;
            t1 = nt1;
            (m0, n0) = magnitude(&r0);
            (m1, n1) = magnitude(&r1);
            bl_r0 = bit_length(&m0);
            bl_r1 = bit_length(&m1);
            if bl_r1 > bl_r0 {
                core::mem::swap(&mut r0, &mut r1);
                core::mem::swap(&mut t0, &mut t1);
                core::mem::swap(&mut m0, &mut m1);
                core::mem::swap(&mut n0, &mut n1);
                core::mem::swap(&mut bl_r0, &mut bl_r1);
            }
            continue 'outer;
        }

        // The approximation cannot decide here, so take one exact step
        let s = bl_r0 - bl_r1;
        let sr1 = shl_limbs(&r1, s);
        let st1 = shl_limbs(&t1, s);
        let (r, t) = if n0 == n1 {
            (
                add_wrapping(&r0, &negate_limbs(&sr1)),
                add_wrapping(&t0, &negate_limbs(&st1)),
            )
        } else {
            (add_wrapping(&r0, &sr1), add_wrapping(&t0, &st1))
        };
        let (mr, nr) = magnitude(&r);
        let bl_r = bit_length(&mr);
        if bl_r > bl_r1 {
            r0 = r;
            t0 = t;
            m0 = mr;
            n0 = nr;
            bl_r0 = bl_r;
        } else {
            r0 = r1;
            t0 = t1;
            m0 = m1;
            n0 = n1;
            bl_r0 = bl_r1;
            r1 = r;
            t1 = t;
            m1 = mr;
            n1 = nr;
            bl_r1 = bl_r;
        }
    }

    let (rho, tau) = (limbs_to_i256(&r1), limbs_to_i256(&t1));
    // A result outside the half-size shape falls back to the exact algorithm
    if bit_length_i256(rho) > HALF_SIZE_BITS
        || bit_length_i256(tau) > TAU_LIMIT_BITS
        || tau.is_zero()
    {
        return curve25519_heea_vartime(v);
    }
    (rho, tau)
}

/// Count the bits in the magnitude of an I256
#[inline(always)]
fn bit_length_i256(v: I256) -> u32 {
    for i in (0..4).rev() {
        let limb = v.limbs[i];
        if limb != 0 {
            return (i as u32) * 64 + (64 - limb.leading_zeros());
        }
    }

    0
}

impl I256 {
    pub const ZERO: Self = I256 {
        limbs: [0, 0, 0, 0],
        is_negative: false,
    };

    const ONE: Self = I256 {
        limbs: [1, 0, 0, 0],
        is_negative: false,
    };

    #[inline(always)]
    const fn abs(&self) -> Self {
        I256 {
            limbs: self.limbs,
            is_negative: false,
        }
    }

    #[cfg(test)]
    pub fn new(value: i128) -> Self {
        let mag: u128 = value.unsigned_abs();
        I256 {
            limbs: [mag as u64, (mag >> 64) as u64, 0, 0],
            is_negative: value.is_negative(),
        }
    }

    /// Read a non-negative value from 32 little-endian bytes
    pub fn from_le_bytes(bytes: [u8; 32]) -> Self {
        I256 {
            limbs: [
                u64::from_le_bytes([
                    bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
                ]),
                u64::from_le_bytes([
                    bytes[8], bytes[9], bytes[10], bytes[11], bytes[12], bytes[13], bytes[14],
                    bytes[15],
                ]),
                u64::from_le_bytes([
                    bytes[16], bytes[17], bytes[18], bytes[19], bytes[20], bytes[21], bytes[22],
                    bytes[23],
                ]),
                u64::from_le_bytes([
                    bytes[24], bytes[25], bytes[26], bytes[27], bytes[28], bytes[29], bytes[30],
                    bytes[31],
                ]),
            ],
            is_negative: false,
        }
    }

    /// Write the magnitude as 32 little-endian bytes
    pub fn to_le_bytes(self) -> [u8; 32] {
        let l = self.limbs;
        [
            l[0] as u8,
            (l[0] >> 8) as u8,
            (l[0] >> 16) as u8,
            (l[0] >> 24) as u8,
            (l[0] >> 32) as u8,
            (l[0] >> 40) as u8,
            (l[0] >> 48) as u8,
            (l[0] >> 56) as u8,
            l[1] as u8,
            (l[1] >> 8) as u8,
            (l[1] >> 16) as u8,
            (l[1] >> 24) as u8,
            (l[1] >> 32) as u8,
            (l[1] >> 40) as u8,
            (l[1] >> 48) as u8,
            (l[1] >> 56) as u8,
            l[2] as u8,
            (l[2] >> 8) as u8,
            (l[2] >> 16) as u8,
            (l[2] >> 24) as u8,
            (l[2] >> 32) as u8,
            (l[2] >> 40) as u8,
            (l[2] >> 48) as u8,
            (l[2] >> 56) as u8,
            l[3] as u8,
            (l[3] >> 8) as u8,
            (l[3] >> 16) as u8,
            (l[3] >> 24) as u8,
            (l[3] >> 32) as u8,
            (l[3] >> 40) as u8,
            (l[3] >> 48) as u8,
            (l[3] >> 56) as u8,
        ]
    }

    /// Check whether the value is zero
    #[inline(always)]
    fn is_zero(&self) -> bool {
        self.limbs[0] == 0 && self.limbs[1] == 0 && self.limbs[2] == 0 && self.limbs[3] == 0
    }

    /// Check whether the value is below zero
    #[inline(always)]
    fn is_below_zero(&self) -> bool {
        self.is_negative && !self.is_zero()
    }

    /// Flip the sign, keeping zero non-negative
    #[inline(always)]
    fn wrapping_neg(self) -> Self {
        if self.is_zero() {
            return Self::ZERO;
        }
        I256 {
            limbs: self.limbs,
            is_negative: !self.is_negative,
        }
    }

    /// Add two signed values
    #[inline(always)]
    fn wrapping_add(self, rhs: Self) -> Self {
        if self.is_negative == rhs.is_negative {
            let (limbs, _overflow) = add_limbs(&self.limbs, &rhs.limbs);
            I256 {
                limbs,
                is_negative: self.is_negative,
            }
        } else {
            let cmp = cmp_magnitude(&self.limbs, &rhs.limbs);
            match cmp {
                Ordering::Greater => {
                    let (limbs, _underflow) = sub_limbs(&self.limbs, &rhs.limbs);
                    I256 {
                        limbs,
                        is_negative: self.is_negative,
                    }
                }
                Ordering::Less => {
                    let (limbs, _underflow) = sub_limbs(&rhs.limbs, &self.limbs);
                    I256 {
                        limbs,
                        is_negative: rhs.is_negative,
                    }
                }
                Ordering::Equal => Self::ZERO,
            }
        }
    }

    /// Subtract two signed values
    #[inline(always)]
    fn wrapping_sub(self, rhs: Self) -> Self {
        self.wrapping_add(rhs.wrapping_neg())
    }

    /// Shift the magnitude left, keeping the sign
    #[inline(always)]
    fn wrapping_shl(self, shift: u32) -> Self {
        if shift == 0 {
            return self;
        }
        if shift >= 256 {
            return Self::ZERO;
        }

        let limb_shift = (shift / 64) as usize;
        let bit_shift = shift & 63;
        let l = self.limbs;
        let mut result = [0u64; 4];

        if bit_shift == 0 {
            match limb_shift {
                0 => return self,
                1 => {
                    result[1] = l[0];
                    result[2] = l[1];
                    result[3] = l[2];
                }
                2 => {
                    result[2] = l[0];
                    result[3] = l[1];
                }
                3 => {
                    result[3] = l[0];
                }
                4.. => {}
            }
        } else {
            let carry = 64 - bit_shift;
            match limb_shift {
                0 => {
                    result[0] = l[0] << bit_shift;
                    result[1] = (l[1] << bit_shift) | (l[0] >> carry);
                    result[2] = (l[2] << bit_shift) | (l[1] >> carry);
                    result[3] = (l[3] << bit_shift) | (l[2] >> carry);
                }
                1 => {
                    result[1] = l[0] << bit_shift;
                    result[2] = (l[1] << bit_shift) | (l[0] >> carry);
                    result[3] = (l[2] << bit_shift) | (l[1] >> carry);
                }
                2 => {
                    result[2] = l[0] << bit_shift;
                    result[3] = (l[1] << bit_shift) | (l[0] >> carry);
                }
                3 => {
                    result[3] = l[0] << bit_shift;
                }
                4.. => {}
            }
        }

        I256 {
            limbs: result,
            is_negative: self.is_negative,
        }
    }
}

impl Neg for I256 {
    type Output = Self;
    fn neg(self) -> <Self as Neg>::Output {
        self.wrapping_neg()
    }
}

/// Add two magnitudes and report the overflow out of the top limb
#[inline(always)]
fn add_limbs(a: &[u64; 4], b: &[u64; 4]) -> ([u64; 4], bool) {
    let mut result = [0u64; 4];
    let (r0, c0) = a[0].overflowing_add(b[0]);

    let (r1_base, c1a) = a[1].overflowing_add(b[1]);
    let (r1, c1b) = r1_base.overflowing_add(c0 as u64);
    let c1 = c1a || c1b;

    let (r2_base, c2a) = a[2].overflowing_add(b[2]);
    let (r2, c2b) = r2_base.overflowing_add(c1 as u64);
    let c2 = c2a || c2b;

    let (r3_base, c3a) = a[3].overflowing_add(b[3]);
    let (r3, c3b) = r3_base.overflowing_add(c2 as u64);
    let c3 = c3a || c3b;

    result[0] = r0;
    result[1] = r1;
    result[2] = r2;
    result[3] = r3;

    (result, c3)
}

/// Subtract magnitude b from a and report the borrow out of the top limb
#[inline(always)]
fn sub_limbs(a: &[u64; 4], b: &[u64; 4]) -> ([u64; 4], bool) {
    let mut result = [0u64; 4];
    let (r0, b0) = a[0].overflowing_sub(b[0]);

    let (r1_base, b1a) = a[1].overflowing_sub(b[1]);
    let (r1, b1b) = r1_base.overflowing_sub(b0 as u64);
    let b1 = b1a || b1b;

    let (r2_base, b2a) = a[2].overflowing_sub(b[2]);
    let (r2, b2b) = r2_base.overflowing_sub(b1 as u64);
    let b2 = b2a || b2b;

    let (r3_base, b3a) = a[3].overflowing_sub(b[3]);
    let (r3, b3b) = r3_base.overflowing_sub(b2 as u64);
    let b3 = b3a || b3b;

    result[0] = r0;
    result[1] = r1;
    result[2] = r2;
    result[3] = r3;

    (result, b3)
}

/// Compare two magnitudes
#[inline(always)]
fn cmp_magnitude(a: &[u64; 4], b: &[u64; 4]) -> Ordering {
    for i in (0..4).rev() {
        match a[i].cmp(&b[i]) {
            Ordering::Equal => continue,
            ordering @ (Ordering::Less | Ordering::Greater) => return ordering,
        }
    }
    Ordering::Equal
}

impl PartialOrd for I256 {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for I256 {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self.is_negative, other.is_negative) {
            (true, false) => Ordering::Less,
            (false, true) => Ordering::Greater,
            (false, false) => cmp_magnitude(&self.limbs, &other.limbs),
            (true, true) => cmp_magnitude(&other.limbs, &self.limbs),
        }
    }
}

impl Shl<u32> for I256 {
    type Output = Self;
    fn shl(self, rhs: u32) -> Self {
        self.wrapping_shl(rhs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Lehmer batching matches the exact iteration on the degenerate inputs
    #[test]
    fn lehmer_edges() {
        let cases = [
            I256::ZERO,
            I256::ONE,
            -I256::ONE,
            I256::new(i128::MAX),
            I256::new(1 << 100),
            I256::ONE << 252,
            I256::ONE << 127,
            I256::ONE << 128,
        ];
        for (index, value) in cases.into_iter().enumerate() {
            assert_eq!(
                curve25519_heea_vartime_lehmer(value),
                curve25519_heea_vartime(value),
                "edge case {index}"
            );
        }
    }

    // the exact iteration returns the expected pairs at zero, one, minus one and large inputs
    #[test]
    fn exact_edges() {
        // Zero skips the loop and returns the starting pair
        let v_zero = I256::ZERO;
        let (rho, tau) = curve25519_heea_vartime(v_zero);
        assert_eq!(rho, I256::ZERO, "rho should be zero when v is zero");
        assert_eq!(tau, I256::ONE, "tau should be 1 when v is zero");

        let v_one = I256::ONE;
        let (rho, tau) = curve25519_heea_vartime(v_one);
        assert_eq!(rho, I256::ONE, "rho should be 1 when v is 1");
        assert_eq!(tau, I256::ONE, "tau should be 1 when v is 1");

        let v_minus_one = -I256::ONE;
        let (rho, tau) = curve25519_heea_vartime(v_minus_one);
        assert_eq!(rho, -I256::ONE, "rho should be -1 when v is -1");
        assert_eq!(tau, I256::ONE, "tau should be 1 when v is -1");

        let v_max_i128 = I256::new(i128::MAX);
        let (rho, tau) = curve25519_heea_vartime(v_max_i128);
        let rho_bits = bit_length_i256(rho);
        assert!(
            rho_bits <= 128,
            "rho should be half-size for 2^127-1, got {} bits",
            rho_bits
        );

        let tau_bits = bit_length_i256(tau);
        assert!(
            tau_bits <= 128,
            "tau should be approximately half-size, got {} bits",
            tau_bits
        );

        let v_252 = I256::ONE << 252;
        let (rho, tau) = curve25519_heea_vartime(v_252);
        let rho_bits = bit_length_i256(rho);
        assert!(
            rho_bits <= 128,
            "rho should be half-size for 2^252, got {} bits",
            rho_bits
        );
        assert_eq!(tau, -I256::ONE, "tau should be -1 for 2^252");
    }
}
