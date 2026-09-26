//! Scalars modulo the group order l, with the encodings and digit forms the verifier needs

// Vendored from curve25519-dalek, BSD-3-Clause

use alloc::vec;

use core::borrow::Borrow;
use core::fmt::Debug;
use core::iter::{Product, Sum};
use core::ops::Index;
use core::ops::Neg;
use core::ops::{Add, AddAssign};
use core::ops::{Mul, MulAssign};
use core::ops::{Sub, SubAssign};

use subtle::Choice;
use subtle::ConditionallySelectable;
use subtle::ConstantTimeEq;
use subtle::CtOption;

use crate::backend;
use crate::constants;
use crate::heea::{curve25519_heea_vartime_lehmer, HEEA_MAX_INDEX, I256};
use crate::traits::HEEADecomposition;

pub const NAF_128_SIZE: usize = HEEA_MAX_INDEX + 1;

/// A scalar held in 52-bit limbs for arithmetic
type UnpackedScalar = backend::serial::u64::scalar::Scalar52;

/// An integer modulo the group order l, stored as 32 little-endian bytes
#[allow(clippy::derived_hash_with_manual_eq)]
#[derive(Clone, Copy, Hash)]
pub struct Scalar {
    /// Below 2^255 always, and below l except in clamped multiplication
    pub bytes: [u8; 32],
}

/// Limb width of the wide reduction, which splits 252 bits into nine limbs
const WIDE_BITS: u32 = 28;

const WIDE_MASK: i64 = (1 << WIDE_BITS) - 1;

/// The value `l - 2^252` in 28-bit limbs, low limb first
const WIDE_C: [i64; 5] = [0xcf5d3ed, 0x12631a5, 0x79cd658, 0xf9dea2f, 0x14de];

/// Read 28-bit limb `i` of a 64-byte value, where limb 18 is the last byte alone
#[inline(always)]
fn wide_limb(input: &[u8; 64], i: usize) -> i64 {
    if i == 18 {
        return input[63] as i64;
    }
    let bit = WIDE_BITS as usize * i;
    let byte = bit / 8;
    let word = u32::from_le_bytes([
        input[byte],
        input[byte + 1],
        input[byte + 2],
        input[byte + 3],
    ]);
    ((word >> (bit % 8)) as i64) & WIDE_MASK
}

/// Reduce a 64-byte value mod l by folding 28-bit limbs, with no data-dependent branch
#[inline]
fn reduce_wide(input: &[u8; 64]) -> [u8; 32] {
    let mut s: [i64; 19] = core::array::from_fn(|i| wide_limb(input, i));

    macro_rules! fold {
        ($i:expr) => {{
            let x = s[$i];
            s[$i - 9] -= x * WIDE_C[0];
            s[$i - 8] -= x * WIDE_C[1];
            s[$i - 7] -= x * WIDE_C[2];
            s[$i - 6] -= x * WIDE_C[3];
            s[$i - 5] -= x * WIDE_C[4];
            s[$i] = 0;
        }};
    }
    macro_rules! carry_round {
        ($($i:expr),*) => {$({
            let carry = (s[$i] + (1 << (WIDE_BITS - 1))) >> WIDE_BITS;
            s[$i + 1] += carry;
            s[$i] -= carry << WIDE_BITS;
        })*};
    }
    macro_rules! carry_floor {
        ($($i:expr),*) => {$({
            let carry = s[$i] >> WIDE_BITS;
            s[$i + 1] += carry;
            s[$i] -= carry << WIDE_BITS;
        })*};
    }

    // Fold the top five limbs into limbs 5 to 13, then bring those back to 28 bits
    fold!(18);
    fold!(17);
    fold!(16);
    fold!(15);
    fold!(14);
    carry_round!(5, 7, 9, 11, 13, 6, 8, 10, 12);

    // Limb 8 propagates last so it sees the overflow from limb 7 first
    fold!(14);
    fold!(13);
    fold!(12);
    fold!(11);
    fold!(10);
    fold!(9);
    carry_round!(0, 2, 4, 6, 1, 3, 5, 7, 8);

    // Limb 9 holds the sign now, and the second fold adds l exactly when the value is negative
    fold!(9);
    carry_floor!(0, 1, 2, 3, 4, 5, 6, 7, 8);
    fold!(9);
    carry_floor!(0, 1, 2, 3, 4, 5, 6, 7);

    let limb = |i: usize| s[i] as u64;
    let words = [
        limb(0) | limb(1) << 28 | limb(2) << 56,
        limb(2) >> 8 | limb(3) << 20 | limb(4) << 48,
        limb(4) >> 16 | limb(5) << 12 | limb(6) << 40,
        limb(6) >> 24 | limb(7) << 4 | limb(8) << 32,
    ];
    let mut out = [0u8; 32];
    for (i, word) in words.iter().enumerate() {
        out[8 * i..8 * i + 8].copy_from_slice(&word.to_le_bytes());
    }
    out
}

impl Scalar {
    /// Reduce a 256-bit little-endian integer modulo l
    pub fn from_bytes_mod_order(bytes: [u8; 32]) -> Scalar {
        let s_unreduced = Scalar { bytes };

        let s = s_unreduced.reduce();
        debug_assert_eq!(0u8, s[31] >> 7);

        s
    }

    /// Reduce a 512-bit little-endian integer modulo l
    pub fn from_bytes_mod_order_wide(input: &[u8; 64]) -> Scalar {
        Scalar {
            bytes: reduce_wide(input),
        }
    }

    /// Decode 32 bytes as a scalar, which is none unless the bytes are below l
    pub fn from_canonical_bytes(bytes: [u8; 32]) -> CtOption<Scalar> {
        let high_bit_unset = (bytes[31] >> 7).ct_eq(&0);
        let candidate = Scalar { bytes };
        CtOption::new(candidate, high_bit_unset & candidate.is_canonical())
    }

    /// Wrap bytes the caller knows are below l
    #[inline]
    pub const fn from_canonical_bytes_unchecked(bytes: [u8; 32]) -> Scalar {
        Scalar { bytes }
    }
}

impl Debug for Scalar {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("Scalar{..}")
    }
}

impl Eq for Scalar {}
impl PartialEq for Scalar {
    fn eq(&self, other: &Self) -> bool {
        self.ct_eq(other).into()
    }
}

impl ConstantTimeEq for Scalar {
    fn ct_eq(&self, other: &Self) -> Choice {
        self.bytes.ct_eq(&other.bytes)
    }
}

impl Index<usize> for Scalar {
    type Output = u8;

    /// Read one byte of the encoding
    fn index(&self, _index: usize) -> &u8 {
        &(self.bytes[_index])
    }
}

impl MulAssign<&Scalar> for Scalar {
    fn mul_assign(&mut self, _rhs: &Scalar) {
        *self = UnpackedScalar::mul(&self.unpack(), &_rhs.unpack()).pack();
    }
}

define_mul_assign_variants!(LHS = Scalar, RHS = Scalar);

impl Mul<&Scalar> for &Scalar {
    type Output = Scalar;
    fn mul(self, _rhs: &Scalar) -> Scalar {
        UnpackedScalar::mul(&self.unpack(), &_rhs.unpack()).pack()
    }
}

define_mul_variants!(LHS = Scalar, RHS = Scalar, Output = Scalar);

impl AddAssign<&Scalar> for Scalar {
    fn add_assign(&mut self, _rhs: &Scalar) {
        *self = *self + _rhs;
    }
}

define_add_assign_variants!(LHS = Scalar, RHS = Scalar);

impl Add<&Scalar> for &Scalar {
    type Output = Scalar;
    #[allow(non_snake_case)]
    fn add(self, _rhs: &Scalar) -> Scalar {
        UnpackedScalar::add(&self.unpack(), &_rhs.unpack()).pack()
    }
}

define_add_variants!(LHS = Scalar, RHS = Scalar, Output = Scalar);

impl SubAssign<&Scalar> for Scalar {
    fn sub_assign(&mut self, _rhs: &Scalar) {
        *self = *self - _rhs;
    }
}

define_sub_assign_variants!(LHS = Scalar, RHS = Scalar);

impl Sub<&Scalar> for &Scalar {
    type Output = Scalar;
    #[allow(non_snake_case)]
    fn sub(self, rhs: &Scalar) -> Scalar {
        UnpackedScalar::sub(&self.unpack(), &rhs.unpack()).pack()
    }
}

define_sub_variants!(LHS = Scalar, RHS = Scalar, Output = Scalar);

impl Neg for &Scalar {
    type Output = Scalar;
    #[allow(non_snake_case)]
    fn neg(self) -> Scalar {
        #[allow(unused_mut)]
        let mut self_unpacked = self.unpack();
        #[allow(unused_mut)]
        let mut self_R = UnpackedScalar::mul_internal(&self_unpacked, &constants::R);
        #[allow(unused_mut)]
        let mut self_mod_l = UnpackedScalar::montgomery_reduce(&self_R);

        UnpackedScalar::sub(&UnpackedScalar::ZERO, &self_mod_l).pack()
    }
}

impl Neg for Scalar {
    type Output = Scalar;
    fn neg(self) -> Scalar {
        -&self
    }
}

impl ConditionallySelectable for Scalar {
    fn conditional_select(a: &Self, b: &Self, choice: Choice) -> Self {
        let mut bytes = [0u8; 32];
        #[allow(clippy::needless_range_loop)]
        for i in 0..32 {
            bytes[i] = u8::conditional_select(&a.bytes[i], &b.bytes[i], choice);
        }
        Scalar { bytes }
    }
}

impl<Term> Product<Term> for Scalar
where
    Term: Borrow<Scalar>,
{
    fn product<Items>(iter: Items) -> Self
    where
        Items: Iterator<Item = Term>,
    {
        iter.fold(Scalar::ONE, |acc, item| acc * item.borrow())
    }
}

impl<Term> Sum<Term> for Scalar
where
    Term: Borrow<Scalar>,
{
    fn sum<Items>(iter: Items) -> Self
    where
        Items: Iterator<Item = Term>,
    {
        iter.fold(Scalar::ZERO, |acc, item| acc + item.borrow())
    }
}

impl Default for Scalar {
    fn default() -> Scalar {
        Scalar::ZERO
    }
}

impl From<u8> for Scalar {
    fn from(x: u8) -> Scalar {
        let mut s_bytes = [0u8; 32];
        s_bytes[0] = x;
        Scalar { bytes: s_bytes }
    }
}

impl From<u16> for Scalar {
    fn from(x: u16) -> Scalar {
        let mut s_bytes = [0u8; 32];
        let x_bytes = x.to_le_bytes();
        s_bytes[0..x_bytes.len()].copy_from_slice(&x_bytes);
        Scalar { bytes: s_bytes }
    }
}

impl From<u32> for Scalar {
    fn from(x: u32) -> Scalar {
        let mut s_bytes = [0u8; 32];
        let x_bytes = x.to_le_bytes();
        s_bytes[0..x_bytes.len()].copy_from_slice(&x_bytes);
        Scalar { bytes: s_bytes }
    }
}

impl From<u64> for Scalar {
    fn from(x: u64) -> Scalar {
        let mut s_bytes = [0u8; 32];
        let x_bytes = x.to_le_bytes();
        s_bytes[0..x_bytes.len()].copy_from_slice(&x_bytes);
        Scalar { bytes: s_bytes }
    }
}

impl From<u128> for Scalar {
    fn from(x: u128) -> Scalar {
        let mut s_bytes = [0u8; 32];
        let x_bytes = x.to_le_bytes();
        s_bytes[0..x_bytes.len()].copy_from_slice(&x_bytes);
        Scalar { bytes: s_bytes }
    }
}

impl From<&Scalar> for I256 {
    fn from(s: &Scalar) -> I256 {
        I256::from_le_bytes(s.to_bytes())
    }
}

impl HEEADecomposition for Scalar {
    /// Split the challenge into half-size `rho` and `tau` with `rho = tau h` when the flag is false and `rho = -tau h` when true
    fn heea_decompose(&self) -> (Scalar, Scalar, bool) {
        let v = self.into();

        let (mut rho_i, mut tau_i) = curve25519_heea_vartime_lehmer(v);

        // Both values go back as magnitudes and each negative sign flips h once
        let mut flip_h = false;
        if rho_i < I256::ZERO {
            rho_i = -rho_i;
            flip_h = !flip_h;
        }
        if tau_i < I256::ZERO {
            tau_i = -tau_i;
            flip_h = !flip_h;
        }

        let rho = Scalar::from_positive_i256(&rho_i);
        let tau = Scalar::from_positive_i256(&tau_i);

        (rho, tau, flip_h)
    }
}

impl Scalar {
    /// The scalar zero
    pub const ZERO: Self = Self { bytes: [0u8; 32] };

    /// The scalar one
    pub const ONE: Self = Self {
        bytes: [
            1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0,
        ],
    };

    /// Copy out the 32-byte little-endian encoding
    pub const fn to_bytes(self) -> [u8; 32] {
        self.bytes
    }

    /// Borrow the 32-byte little-endian encoding
    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.bytes
    }

    /// Compute the multiplicative inverse of a nonzero scalar
    pub fn invert(&self) -> Scalar {
        #[allow(unused_mut)]
        let mut unpacked = self.unpack();
        #[allow(unused_mut)]
        let mut inverse = unpacked.invert();

        inverse.pack()
    }

    /// Invert every nonzero scalar in place and return the inverse of their product
    pub fn invert_batch<const COUNT: usize>(inputs: &mut [Scalar; COUNT]) -> Scalar {
        let one: UnpackedScalar = Scalar::ONE.unpack().as_montgomery();

        let mut scratch = [one; COUNT];

        Self::invert_batch_internal(inputs, &mut scratch)
    }

    /// Invert a slice of nonzero scalars in place and return the inverse of their product
    pub fn invert_batch_alloc(inputs: &mut [Scalar]) -> Scalar {
        let n = inputs.len();
        let one: UnpackedScalar = Scalar::ONE.unpack().as_montgomery();

        let mut scratch = vec![one; n];

        Self::invert_batch_internal(inputs, &mut scratch)
    }

    fn invert_batch_internal(inputs: &mut [Scalar], scratch: &mut [UnpackedScalar]) -> Scalar {
        // Inputs stay in Montgomery form so the backward pass needs no conversion
        let mut acc = Scalar::ONE.unpack().as_montgomery();

        for (input, scratch) in inputs.iter_mut().zip(scratch.iter_mut()) {
            *scratch = acc;

            let tmp = input.unpack().as_montgomery();
            *input = tmp.pack();
            acc = UnpackedScalar::montgomery_mul(&acc, &tmp);
        }

        debug_assert!(acc.pack() != Scalar::ZERO);

        acc = acc.montgomery_invert().from_montgomery();

        let ret = acc.pack();

        for (input, scratch) in inputs.iter_mut().rev().zip(scratch.iter().rev()) {
            let tmp = UnpackedScalar::montgomery_mul(&acc, &input.unpack());
            *input = UnpackedScalar::montgomery_mul(&acc, scratch).pack();
            acc = tmp;
        }

        ret
    }

    /// Compute `b` with `b + b = a mod l`
    pub fn div_by_2(&self) -> Self {
        // An odd value gains l first so the halving is exact
        let is_odd = Choice::from(self.as_bytes()[0] & 1);
        let mut scalar = self.unpack();
        scalar.conditional_add_l(is_odd);

        let carry = scalar.shr1_assign();
        debug_assert_eq!(carry, 0);

        scalar.pack()
    }

    /// Iterate the 256 bits of the scalar, low bit first
    pub fn bits_le(&self) -> impl DoubleEndedIterator<Item = bool> + '_ {
        (0..256).map(|i| ((self.bytes[i >> 3] >> (i & 7)) & 1u8) == 1)
    }

    /// Compute the width-`w` non-adjacent form, odd digits below `2^(w-1)` in magnitude with at most one nonzero in any `w` in a row
    pub fn non_adjacent_form(&self, w: usize) -> [i8; 256] {
        debug_assert!(w >= 2);
        // Wider windows give digits that do not fit in i8
        debug_assert!(w <= 8);

        let mut naf = [0i8; 256];

        let mut x_u64 = [0u64; 5];
        read_le_u64_into(&self.bytes, &mut x_u64[0..4]);

        let width = 1 << w;
        let window_mask = width - 1;

        let mut pos = 0;
        let mut carry = 0;
        while pos < 256 {
            let u64_idx = pos / 64;
            let bit_idx = pos % 64;
            let bit_buf: u64 = if bit_idx < 64 - w {
                x_u64[u64_idx] >> bit_idx
            } else {
                (x_u64[u64_idx] >> bit_idx) | (x_u64[1 + u64_idx] << (64 - bit_idx))
            };

            let window = carry + (bit_buf & window_mask);

            if window & 1 == 0 {
                // An even window leaves the pending overflow bit for the next position
                pos += 1;
                continue;
            }

            if window < width / 2 {
                carry = 0;
                naf[pos] = window as i8;
            } else {
                carry = 1;
                naf[pos] = (window as i8).wrapping_sub(width as i8);
            }

            pos += w;
        }

        naf
    }

    /// Compute the width-`w` non-adjacent form of a scalar below 2^128
    pub fn non_adjacent_form_128(&self, w: usize) -> [i8; NAF_128_SIZE] {
        debug_assert!(w >= 2);
        debug_assert!(w <= 8);
        debug_assert!(self.bytes[16..32].iter().all(|&b| b == 0));

        let mut naf = [0i8; NAF_128_SIZE];

        let mut x_u64 = [0u64; 3];
        read_le_u64_into(&self.bytes[..16], &mut x_u64[0..2]);

        let width = 1u64 << w;
        let window_mask = width - 1;

        let mut pos = 0;
        let mut carry = 0;
        while pos < HEEA_MAX_INDEX {
            let u64_idx = pos / 64;
            let bit_idx = pos % 64;
            let bit_buf: u64 = if bit_idx < 64 - w {
                x_u64[u64_idx] >> bit_idx
            } else {
                (x_u64[u64_idx] >> bit_idx) | (x_u64[1 + u64_idx] << (64 - bit_idx))
            };

            let window = carry + (bit_buf & window_mask);

            if window & 1 == 0 {
                pos += 1;
                continue;
            }

            if window < width / 2 {
                carry = 0;
                naf[pos] = window as i8;
            } else {
                carry = 1;
                naf[pos] = (window as i8).wrapping_sub(width as i8);
            }

            pos += w;
        }

        debug_assert_eq!(carry, 0);
        naf
    }

    /// Write the scalar as 64 signed radix-16 digits in -8..8, the top digit up to 8, for a scalar below 2^255
    pub fn as_radix_16(&self) -> [i8; 64] {
        debug_assert!(self[31] <= 127);
        let mut output = [0i8; 64];

        #[allow(clippy::identity_op)]
        #[inline(always)]
        fn bot_half(x: u8) -> u8 {
            (x >> 0) & 15
        }
        #[inline(always)]
        fn top_half(x: u8) -> u8 {
            (x >> 4) & 15
        }

        for i in 0..32 {
            output[2 * i] = bot_half(self[i]) as i8;
            output[2 * i + 1] = top_half(self[i]) as i8;
        }

        // Recenter each digit from 0..16 to -8..8
        for i in 0..63 {
            let carry = (output[i] + 8) >> 4;
            output[i] -= carry << 4;
            output[i + 1] += carry;
        }
        // The top digit starts at 7 or less and gains at most one, so it stays at 8 or less

        output
    }

    /// Count the digits `as_radix_2w` can make nonzero for width `w`
    #[cfg(test)]
    pub fn to_radix_2w_size_hint(w: usize) -> usize {
        debug_assert!(w >= 4);
        debug_assert!(w <= 8);

        let digits_count = match w {
            4..=7 => 256_usize.div_ceil(w),
            8 => 256_usize.div_ceil(w) + 1_usize,
            0..=3 | 9.. => panic!("invalid radix parameter"),
        };

        debug_assert!(digits_count <= 64);
        digits_count
    }

    /// Write the scalar in signed radix `2^w` digits for `w` from 4 to 8
    pub fn as_radix_2w(&self, w: usize) -> [i8; 64] {
        debug_assert!(w >= 4);
        debug_assert!(w <= 8);

        if w == 4 {
            return self.as_radix_16();
        }

        let mut scalar64x4 = [0u64; 4];
        read_le_u64_into(&self.bytes, &mut scalar64x4[0..4]);

        let radix: u64 = 1 << w;
        let window_mask: u64 = radix - 1;

        let mut carry = 0u64;
        let mut digits = [0i8; 64];
        let digits_count = 256_usize.div_ceil(w);
        #[allow(clippy::needless_range_loop)]
        for i in 0..digits_count {
            let bit_offset = i * w;
            let u64_idx = bit_offset / 64;
            let bit_idx = bit_offset % 64;

            let bit_buf: u64 = if bit_idx < 64 - w || u64_idx == 3 {
                scalar64x4[u64_idx] >> bit_idx
            } else {
                (scalar64x4[u64_idx] >> bit_idx) | (scalar64x4[1 + u64_idx] << (64 - bit_idx))
            };

            let coef = carry + (bit_buf & window_mask);

            // Recenter the digit from 0..2^w to -2^(w-1)..2^(w-1)
            carry = (coef + (radix / 2)) >> w;
            digits[i] = ((coef as i64) - (carry << w) as i64) as i8;
        }

        // An i8 cannot hold the top overflow shifted by 8, so width 8 puts it in an extra digit
        match w {
            8 => digits[digits_count] += carry as i8,
            0..=7 | 9.. => digits[digits_count - 1] += (carry << w) as i8,
        }

        digits
    }

    /// Unpack the scalar into 52-bit limbs
    pub fn unpack(&self) -> UnpackedScalar {
        UnpackedScalar::from_bytes(&self.bytes)
    }

    /// Convert a non-negative `I256` to a scalar
    pub fn from_positive_i256(val: &I256) -> Self {
        debug_assert!(val >= &I256::ZERO);
        Scalar {
            bytes: val.to_le_bytes(),
        }
    }

    /// Reduce the scalar modulo l
    #[allow(non_snake_case)]
    fn reduce(&self) -> Scalar {
        #[allow(unused_mut)]
        let mut x = self.unpack();
        #[allow(unused_mut)]
        let mut xR = UnpackedScalar::mul_internal(&x, &constants::R);
        #[allow(unused_mut)]
        let mut x_mod_l = UnpackedScalar::montgomery_reduce(&xR);

        x_mod_l.pack()
    }

    /// Check whether the scalar is already below l
    fn is_canonical(&self) -> Choice {
        self.ct_eq(&self.reduce())
    }
}

impl UnpackedScalar {
    /// Pack the limbs back into a 32-byte scalar
    fn pack(&self) -> Scalar {
        Scalar {
            bytes: self.to_bytes(),
        }
    }

    /// Invert a scalar in Montgomery form with a fixed addition chain
    #[rustfmt::skip]
    #[allow(unused_mut)]
    #[allow(clippy::just_underscores_and_digits)]
    pub fn montgomery_invert(&self) -> UnpackedScalar {
        let mut    _1 = *self;
        let mut   _10 = _1.montgomery_square();
        let mut  _100 = _10.montgomery_square();
        let mut   _11 = UnpackedScalar::montgomery_mul(&_10,     &_1);
        let mut  _101 = UnpackedScalar::montgomery_mul(&_10,    &_11);
        let mut  _111 = UnpackedScalar::montgomery_mul(&_10,   &_101);
        let mut _1001 = UnpackedScalar::montgomery_mul(&_10,   &_111);
        let mut _1011 = UnpackedScalar::montgomery_mul(&_10,  &_1001);
        let mut _1111 = UnpackedScalar::montgomery_mul(&_100, &_1011);

        let mut y = UnpackedScalar::montgomery_mul(&_1111, &_1);

        #[inline]
        fn square_multiply(y: &mut UnpackedScalar, squarings: usize, x: &UnpackedScalar) {
            for _ in 0..squarings {
                *y = y.montgomery_square();
            }
            *y = UnpackedScalar::montgomery_mul(y, x);
        }

        square_multiply(&mut y, 123 + 3, &_101);
        square_multiply(&mut y,   2 + 2, &_11);
        square_multiply(&mut y,   1 + 4, &_1111);
        square_multiply(&mut y,   1 + 4, &_1111);
        square_multiply(&mut y,       4, &_1001);
        square_multiply(&mut y,       2, &_11);
        square_multiply(&mut y,   1 + 4, &_1111);
        square_multiply(&mut y,   1 + 3, &_101);
        square_multiply(&mut y,   3 + 3, &_101);
        square_multiply(&mut y,       3, &_111);
        square_multiply(&mut y,   1 + 4, &_1111);
        square_multiply(&mut y,   2 + 3, &_111);
        square_multiply(&mut y,   2 + 2, &_11);
        square_multiply(&mut y,   1 + 4, &_1011);
        square_multiply(&mut y,   2 + 4, &_1011);
        square_multiply(&mut y,   6 + 4, &_1001);
        square_multiply(&mut y,   2 + 2, &_11);
        square_multiply(&mut y,   3 + 2, &_11);
        square_multiply(&mut y,   3 + 2, &_11);
        square_multiply(&mut y,   1 + 4, &_1001);
        square_multiply(&mut y,   1 + 3, &_111);
        square_multiply(&mut y,   2 + 4, &_1111);
        square_multiply(&mut y,   1 + 4, &_1011);
        square_multiply(&mut y,       3, &_101);
        square_multiply(&mut y,   2 + 4, &_1111);
        square_multiply(&mut y,       3, &_101);
        square_multiply(&mut y,   1 + 2, &_11);

        y
    }

    /// Invert a scalar in plain form
    pub fn invert(&self) -> UnpackedScalar {
        #[allow(unused_mut)]
        let mut montgomery = self.as_montgomery();
        #[allow(unused_mut)]
        let mut inverse = montgomery.montgomery_invert();

        inverse.from_montgomery()
    }
}

/// Read little-endian u64 words from bytes, eight bytes per word
fn read_le_u64_into(source: &[u8], destination: &mut [u64]) {
    debug_assert_eq!(source.len(), 8 * destination.len());
    let (words, _) = source.as_chunks::<8>();
    for (bytes, word) in words.iter().zip(destination.iter_mut()) {
        *word = u64::from_le_bytes(*bytes);
    }
}

/// Clamp 32 bytes to a multiple of the cofactor with bit 254 set and bit 255 clear
#[must_use]
pub const fn clamp_integer(mut bytes: [u8; 32]) -> [u8; 32] {
    bytes[0] &= 0b1111_1000;
    bytes[31] &= 0b0111_1111;
    bytes[31] |= 0b0100_0000;
    bytes
}

#[cfg(test)]
pub mod test {
    use super::*;

    use alloc::vec::Vec;
    use rand::Rng;

    /// x = 2238329342913194256032495932344128051776374960164957527413114840482143558222
    pub static X: Scalar = Scalar {
        bytes: [
            0x4e, 0x5a, 0xb4, 0x34, 0x5d, 0x47, 0x08, 0x84, 0x59, 0x13, 0xb4, 0x64, 0x1b, 0xc2,
            0x7d, 0x52, 0x52, 0xa5, 0x85, 0x10, 0x1b, 0xcc, 0x42, 0x44, 0xd4, 0x49, 0xf4, 0xa8,
            0x79, 0xd9, 0xf2, 0x04,
        ],
    };
    /// 1/x = 6859937278830797291664592131120606308688036382723378951768035303146619657244
    pub static XINV: Scalar = Scalar {
        bytes: [
            0x1c, 0xdc, 0x17, 0xfc, 0xe0, 0xe9, 0xa5, 0xbb, 0xd9, 0x24, 0x7e, 0x56, 0xbb, 0x01,
            0x63, 0x47, 0xbb, 0xba, 0x31, 0xed, 0xd5, 0xa9, 0xbb, 0x96, 0xd5, 0x0b, 0xcd, 0x7a,
            0x3f, 0x96, 0x2a, 0x0f,
        ],
    };
    /// y = 2592331292931086675770238855846338635550719849568364935475441891787804997264
    pub static Y: Scalar = Scalar {
        bytes: [
            0x90, 0x76, 0x33, 0xfe, 0x1c, 0x4b, 0x66, 0xa4, 0xa2, 0x8d, 0x2d, 0xd7, 0x67, 0x83,
            0x86, 0xc3, 0x53, 0xd0, 0xde, 0x54, 0x55, 0xd4, 0xfc, 0x9d, 0xe8, 0xef, 0x7a, 0xc3,
            0x1f, 0x35, 0xbb, 0x05,
        ],
    };

    /// The largest scalar with the top bit clear, which is above l and only fit for point multiplication
    pub static LARGEST_UNREDUCED_SCALAR: Scalar = Scalar {
        bytes: [
            0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
            0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
            0xff, 0xff, 0xff, 0x7f,
        ],
    };

    /// x*y = 5690045403673944803228348699031245560686958845067437804563560795922180092780
    static X_TIMES_Y: Scalar = Scalar {
        bytes: [
            0x6c, 0x33, 0x74, 0xa1, 0x89, 0x4f, 0x62, 0x21, 0x0a, 0xaa, 0x2f, 0xe1, 0x86, 0xa6,
            0xf9, 0x2c, 0xe0, 0xaa, 0x75, 0xc2, 0x77, 0x95, 0x81, 0xc2, 0x95, 0xfc, 0x08, 0x17,
            0x9a, 0x73, 0x94, 0x0c,
        ],
    };

    /// The value 2^256 - 1 reduced mod l
    static CANONICAL_2_256_MINUS_1: Scalar = Scalar {
        bytes: [
            28, 149, 152, 141, 116, 49, 236, 214, 112, 207, 125, 115, 244, 91, 239, 198, 254, 255,
            255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 15,
        ],
    };

    static A_SCALAR: Scalar = Scalar {
        bytes: [
            0x1a, 0x0e, 0x97, 0x8a, 0x90, 0xf6, 0x62, 0x2d, 0x37, 0x47, 0x02, 0x3f, 0x8a, 0xd8,
            0x26, 0x4d, 0xa7, 0x58, 0xaa, 0x1b, 0x88, 0xe0, 0x40, 0xd1, 0x58, 0x9e, 0x7b, 0x7f,
            0x23, 0x76, 0xef, 0x09,
        ],
    };

    static A_NAF: [i8; 256] = [
        0, 13, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, -9, 0, 0, 0, 0, -11, 0, 0, 0, 0, 3, 0, 0,
        0, 0, 1, 0, 0, 0, 0, 9, 0, 0, 0, 0, -5, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 11, 0, 0, 0, 0,
        11, 0, 0, 0, 0, 0, -9, 0, 0, 0, 0, 0, -3, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
        0, -1, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0, -15, 0, 0, 0, 0, -7, 0, 0, 0, 0, -9, 0, 0, 0, 0, 0, 5,
        0, 0, 0, 0, 13, 0, 0, 0, 0, 0, -3, 0, 0, 0, 0, -11, 0, 0, 0, 0, -7, 0, 0, 0, 0, -13, 0, 0,
        0, 0, 11, 0, 0, 0, 0, -9, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, -15, 0, 0, 0, 0, 1, 0, 0, 0, 0,
        7, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 13, 0, 0, 0, 0, 0, 0, 11, 0, 0, 0, 0, 0, 15,
        0, 0, 0, 0, 0, -9, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, -15, 0,
        0, 0, 0, 0, 15, 0, 0, 0, 0, 15, 0, 0, 0, 0, 15, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
    ];

    const BASEPOINT_ORDER_MINUS_ONE: Scalar = Scalar {
        bytes: [
            0xec, 0xd3, 0xf5, 0x5c, 0x1a, 0x63, 0x12, 0x58, 0xd6, 0x9c, 0xf7, 0xa2, 0xde, 0xf9,
            0xde, 0x14, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x10,
        ],
    };

    /// The largest clamped integer
    static LARGEST_CLAMPED_INTEGER: [u8; 32] = clamp_integer(LARGEST_UNREDUCED_SCALAR.bytes);

    #[test]
    fn fuzzer_testcase_reduction() {
        // Little-endian bytes of 24519928653854221733733552434404946937899825954937634815
        let a_bytes = [
            255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
            255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ];
        // Little-endian bytes of 4975441334397345751130612518500927154628011511324180036903450236863266160640
        let b_bytes = [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 210, 210,
            210, 255, 255, 255, 255, 10,
        ];
        // Little-endian bytes of 6432735165214683820902750800207468552549813371247423777071615116673864412038
        let c_bytes = [
            134, 171, 119, 216, 180, 128, 178, 62, 171, 132, 32, 62, 34, 119, 104, 193, 47, 215,
            181, 250, 14, 207, 172, 93, 75, 207, 211, 103, 144, 204, 56, 14,
        ];

        let a = Scalar::from_bytes_mod_order(a_bytes);
        let b = Scalar::from_bytes_mod_order(b_bytes);
        let c = Scalar::from_bytes_mod_order(c_bytes);

        let mut tmp = [0u8; 64];

        tmp[0..32].copy_from_slice(&a_bytes[..]);
        let also_a = Scalar::from_bytes_mod_order_wide(&tmp);

        tmp[0..32].copy_from_slice(&b_bytes[..]);
        let also_b = Scalar::from_bytes_mod_order_wide(&tmp);

        let expected_c = a * b;
        let also_expected_c = also_a * also_b;

        assert_eq!(c, expected_c);
        assert_eq!(c, also_expected_c);
    }

    #[test]
    fn non_adjacent_form_test_vector() {
        let naf = A_SCALAR.non_adjacent_form(5);
        for i in 0..256 {
            assert_eq!(naf[i], A_NAF[i]);
        }
    }

    #[test]
    fn non_adjacent_form_128_matches_generic() {
        let mut high_bit = [0u8; 32];
        high_bit[15] = 0x80;

        let cases = [
            Scalar::ZERO,
            Scalar::ONE,
            Scalar::from(u64::MAX),
            Scalar::from(0xfedc_ba98_7654_3210_0123_4567_89ab_cdefu128),
            Scalar::from(u128::MAX),
            Scalar { bytes: high_bit },
        ];

        for scalar in cases {
            for w in 2..=8 {
                let generic = scalar.non_adjacent_form(w);
                let naf_128 = scalar.non_adjacent_form_128(w);

                for i in 0..=HEEA_MAX_INDEX {
                    assert_eq!(
                        naf_128[i], generic[i],
                        "NAF mismatch at index {i} for width {w}"
                    );
                }

                assert!(
                    generic[NAF_128_SIZE..].iter().all(|&digit| digit == 0),
                    "generic NAF has non-zero digits above 128-bit range for width {w}"
                );
            }
        }
    }

    #[test]
    fn from_u64() {
        let val: u64 = 0xdeadbeefdeadbeef;
        let s = Scalar::from(val);
        assert_eq!(s[7], 0xde);
        assert_eq!(s[6], 0xad);
        assert_eq!(s[5], 0xbe);
        assert_eq!(s[4], 0xef);
        assert_eq!(s[3], 0xde);
        assert_eq!(s[2], 0xad);
        assert_eq!(s[1], 0xbe);
        assert_eq!(s[0], 0xef);
    }

    #[test]
    fn scalar_mul_by_one() {
        let test_scalar = X * Scalar::ONE;
        for i in 0..32 {
            assert!(test_scalar[i] == X[i]);
        }
    }

    #[test]
    fn add_reduces() {
        assert_eq!(BASEPOINT_ORDER_MINUS_ONE + Scalar::ONE, Scalar::ZERO);
    }

    #[test]
    fn sub_reduces() {
        assert_eq!(Scalar::ZERO - Scalar::ONE, BASEPOINT_ORDER_MINUS_ONE);
    }

    #[test]
    fn impl_add() {
        let two = Scalar::from(2u64);
        let one = Scalar::ONE;
        let should_be_two = one + one;
        assert_eq!(should_be_two, two);
    }

    #[allow(non_snake_case)]
    #[test]
    fn impl_mul() {
        let should_be_X_times_Y = X * Y;
        assert_eq!(should_be_X_times_Y, X_TIMES_Y);
    }

    #[allow(non_snake_case)]
    #[test]
    fn impl_product() {
        let X_Y_vector = [X, Y];
        let should_be_X_times_Y: Scalar = X_Y_vector.iter().product();
        assert_eq!(should_be_X_times_Y, X_TIMES_Y);

        let one = Scalar::ONE;
        let empty_vector = [];
        let should_be_one: Scalar = empty_vector.iter().product();
        assert_eq!(should_be_one, one);

        let xs = [Scalar::from(2u64); 10];
        let ys = [Scalar::from(3u64); 10];
        let zs = xs.iter().zip(ys.iter()).map(|(x, y)| x * y);

        let x_prod: Scalar = xs.iter().product();
        let y_prod: Scalar = ys.iter().product();
        let z_prod: Scalar = zs.product();

        assert_eq!(x_prod, Scalar::from(1024u64));
        assert_eq!(y_prod, Scalar::from(59049u64));
        assert_eq!(z_prod, Scalar::from(60466176u64));
        assert_eq!(x_prod * y_prod, z_prod);
    }

    #[test]
    fn impl_sum() {
        let two = Scalar::from(2u64);
        let one_vector = [Scalar::ONE, Scalar::ONE];
        let should_be_two: Scalar = one_vector.iter().sum();
        assert_eq!(should_be_two, two);

        let zero = Scalar::ZERO;
        let empty_vector = [];
        let should_be_zero: Scalar = empty_vector.iter().sum();
        assert_eq!(should_be_zero, zero);

        let xs = [Scalar::from(1u64); 10];
        let ys = [Scalar::from(2u64); 10];
        let zs = xs.iter().zip(ys.iter()).map(|(x, y)| x + y);

        let x_sum: Scalar = xs.iter().sum();
        let y_sum: Scalar = ys.iter().sum();
        let z_sum: Scalar = zs.sum();

        assert_eq!(x_sum, Scalar::from(10u64));
        assert_eq!(y_sum, Scalar::from(20u64));
        assert_eq!(z_sum, Scalar::from(30u64));
        assert_eq!(x_sum + y_sum, z_sum);
    }

    #[test]
    fn square() {
        let expected = X * X;
        let actual = X.unpack().square().pack();
        for i in 0..32 {
            assert!(expected[i] == actual[i]);
        }
    }

    #[test]
    fn div_by_2() {
        for i in 0u64..32 {
            let scalar = Scalar::from(i);
            let dividend = scalar.div_by_2();
            assert_eq!(scalar, dividend + dividend);
        }

        for i in 0u64..32 {
            let scalar = Scalar::ZERO - Scalar::from(i);
            let dividend = scalar.div_by_2();
            assert_eq!(scalar, dividend + dividend);
        }
    }

    #[test]
    fn reduce() {
        let biggest = Scalar::from_bytes_mod_order([0xff; 32]);
        assert_eq!(biggest, CANONICAL_2_256_MINUS_1);
    }

    // the fold reduction agrees with the Montgomery path at every edge
    #[test]
    fn wide_reduction_matches_montgomery() {
        let l = constants::BASEPOINT_ORDER.to_bytes();
        let mut vectors: Vec<[u8; 64]> = vec![[0u8; 64], [0xffu8; 64]];
        for shift in [252usize, 253, 255, 256, 260, 300, 400, 511] {
            let mut v = [0u8; 64];
            v[shift / 8] = 1 << (shift % 8);
            vectors.push(v);
        }
        for delta in [-2i64, -1, 0, 1, 2] {
            let mut v = [0u8; 64];
            v[..32].copy_from_slice(&l);
            let mut carry = delta;
            for b in v.iter_mut() {
                let x = *b as i64 + carry;
                *b = x.rem_euclid(256) as u8;
                carry = x.div_euclid(256);
            }
            vectors.push(v);
        }
        let mut state = 0x5EED_1EE5_u64;
        for _ in 0..4096 {
            let mut v = [0u8; 64];
            for chunk in v.as_chunks_mut::<8>().0 {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                chunk.copy_from_slice(&state.to_le_bytes());
            }
            vectors.push(v);
        }
        for v in &vectors {
            let want = UnpackedScalar::from_bytes_wide(v).pack();
            assert_eq!(Scalar::from_bytes_mod_order_wide(v), want, "{v:02x?}");
        }
    }

    #[test]
    fn from_bytes_mod_order_wide() {
        let mut bignum = [0u8; 64];
        for i in 0..32 {
            bignum[i] = X[i];
            bignum[32 + i] = X[i];
        }
        // The value x + 2^256 x mod l
        let reduced = Scalar {
            bytes: [
                216, 154, 179, 139, 210, 121, 2, 71, 69, 99, 158, 216, 23, 173, 63, 100, 204, 0,
                91, 50, 219, 153, 57, 249, 28, 82, 31, 197, 100, 165, 192, 8,
            ],
        };
        let test_red = Scalar::from_bytes_mod_order_wide(&bignum);
        for i in 0..32 {
            assert!(test_red[i] == reduced[i]);
        }
    }

    #[allow(non_snake_case)]
    #[test]
    fn invert() {
        let inv_X = X.invert();
        assert_eq!(inv_X, XINV);
        let should_be_one = inv_X * X;
        assert_eq!(should_be_one, Scalar::ONE);
    }

    // negating a scalar twice gives the scalar back
    #[allow(non_snake_case)]
    #[test]
    fn neg_twice_is_identity() {
        let negative_X = -&X;
        let should_be_X = -&negative_X;

        assert_eq!(should_be_X, X);
    }

    #[test]
    fn to_bytes_from_bytes_roundtrips() {
        let unpacked = X.unpack();
        let bytes = unpacked.to_bytes();
        let should_be_unpacked = UnpackedScalar::from_bytes(&bytes);

        assert_eq!(should_be_unpacked.0, unpacked.0);
    }

    #[test]
    fn montgomery_reduce_matches_from_bytes_mod_order_wide() {
        let mut bignum = [0u8; 64];

        for i in 0..32 {
            bignum[i] = X[i];
            bignum[32 + i] = X[i];
        }
        // The value x + 2^256 x mod l
        let expected = Scalar {
            bytes: [
                216, 154, 179, 139, 210, 121, 2, 71, 69, 99, 158, 216, 23, 173, 63, 100, 204, 0,
                91, 50, 219, 153, 57, 249, 28, 82, 31, 197, 100, 165, 192, 8,
            ],
        };
        let reduced = Scalar::from_bytes_mod_order_wide(&bignum);

        assert_eq!(reduced.bytes, expected.bytes);

        let interim =
            UnpackedScalar::mul_internal(&UnpackedScalar::from_bytes_wide(&bignum), &constants::R);
        let montgomery_reduced = UnpackedScalar::montgomery_reduce(&interim);

        assert_eq!(montgomery_reduced.0, reduced.unpack().0);
        assert_eq!(montgomery_reduced.0, expected.unpack().0)
    }

    #[test]
    fn canonical_decoding() {
        // The canonical encoding of 1667457891
        let canonical_bytes = [
            99, 99, 99, 99, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0,
        ];

        // Every byte 16 encodes a value at or above l
        let non_canonical_bytes_because_unreduced = [16; 32];

        // The decoder must refuse a set high bit without masking it off
        let non_canonical_bytes_because_highbit = [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 128,
        ];

        assert!(bool::from(
            Scalar::from_canonical_bytes(canonical_bytes).is_some()
        ));
        assert!(bool::from(
            Scalar::from_canonical_bytes(non_canonical_bytes_because_unreduced).is_none()
        ));
        assert!(bool::from(
            Scalar::from_canonical_bytes(non_canonical_bytes_because_highbit).is_none()
        ));
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic]
    fn invert_batch_with_a_zero_input_panics() {
        let mut xs = [Scalar::ONE; 16];
        xs[3] = Scalar::ZERO;
        Scalar::invert_batch(&mut xs);
    }

    #[test]
    fn invert_batch_empty() {
        assert_eq!(Scalar::ONE, Scalar::invert_batch(&mut []));
    }

    #[test]
    fn invert_batch_consistency() {
        let mut x = Scalar::from(1u64);
        let mut v1: [Scalar; 16] = core::array::from_fn(|_| {
            let tmp = x;
            x = x + x;
            tmp
        });
        let v2 = v1;

        let expected: Scalar = v1.iter().product();
        let expected = expected.invert();
        let ret = Scalar::invert_batch(&mut v1);
        assert_eq!(ret, expected);

        for (a, b) in v1.iter().zip(v2.iter()) {
            assert_eq!(a * b, Scalar::ONE);
        }
    }

    #[test]
    fn batch_vec_invert_consistency() {
        let mut x = Scalar::from(1u64);
        let mut v1: Vec<_> = (0..16)
            .map(|_| {
                let tmp = x;
                x = x + x;
                tmp
            })
            .collect();
        let v2 = v1.clone();

        let expected: Scalar = v1.iter().product();
        let expected = expected.invert();
        let ret = Scalar::invert_batch_alloc(&mut v1);
        assert_eq!(ret, expected);

        for (a, b) in v1.iter().zip(v2.iter()) {
            assert_eq!(a * b, Scalar::ONE);
        }
    }

    fn test_pippenger_radix_iter(scalar: Scalar, w: usize) {
        let digits_count = Scalar::to_radix_2w_size_hint(w);
        let digits = scalar.as_radix_2w(w);

        let radix = Scalar::from((1 << w) as u64);
        let mut term = Scalar::ONE;
        let mut recovered_scalar = Scalar::ZERO;
        for digit in &digits[0..digits_count] {
            let digit = *digit;
            if digit != 0 {
                let sdigit = if digit < 0 {
                    -Scalar::from((-(digit as i64)) as u64)
                } else {
                    Scalar::from(digit as u64)
                };
                recovered_scalar += term * sdigit;
            }
            term *= radix;
        }
        // An unreduced input comes back only mod l
        assert_eq!(recovered_scalar, scalar.reduce());
    }

    #[test]
    fn test_pippenger_radix() {
        use core::iter;
        let cases = (2..100)
            .map(|s| Scalar::from(s as u64).invert())
            .chain(iter::once(LARGEST_UNREDUCED_SCALAR));

        for scalar in cases {
            test_pippenger_radix_iter(scalar, 6);
            test_pippenger_radix_iter(scalar, 7);
            test_pippenger_radix_iter(scalar, 8);
        }
    }

    #[test]
    fn test_read_le_u64_into() {
        let cases: &[(&[u8], &[u64])] = &[
            (
                &[0xFE, 0xEF, 0x10, 0x01, 0x1F, 0xF1, 0x0F, 0xF0],
                &[0xF00F_F11F_0110_EFFE],
            ),
            (
                &[
                    0xFE, 0xEF, 0x10, 0x01, 0x1F, 0xF1, 0x0F, 0xF0, 0x12, 0x34, 0x56, 0x78, 0x9A,
                    0xBC, 0xDE, 0xF0,
                ],
                &[0xF00F_F11F_0110_EFFE, 0xF0DE_BC9A_7856_3412],
            ),
        ];

        for (src, expected) in cases {
            let mut dst = vec![0; expected.len()];
            read_le_u64_into(src, &mut dst);

            assert_eq!(&dst, expected, "Expected {:x?} got {:x?}", expected, dst);
        }
    }

    // every integer conversion agrees with scalar addition
    #[test]
    fn test_scalar_from_int() {
        let s1 = Scalar::ONE;

        let x = 0x23u8;
        let sx = Scalar::from(x);
        assert_eq!(sx + s1, Scalar::from(x + 1));

        let x = 0x2323u16;
        let sx = Scalar::from(x);
        assert_eq!(sx + s1, Scalar::from(x + 1));

        let x = 0x2323_2323u32;
        let sx = Scalar::from(x);
        assert_eq!(sx + s1, Scalar::from(x + 1));

        let x = 0x2323_2323_2323_2323u64;
        let sx = Scalar::from(x);
        assert_eq!(sx + s1, Scalar::from(x + 1));

        let x = 0x2323_2323_2323_2323_2323_2323_2323_2323u128;
        let sx = Scalar::from(x);
        assert_eq!(sx + s1, Scalar::from(x + 1));
    }

    #[test]
    fn test_scalar_clamp() {
        let input = A_SCALAR.bytes;
        let expected = [
            0x18, 0x0e, 0x97, 0x8a, 0x90, 0xf6, 0x62, 0x2d, 0x37, 0x47, 0x02, 0x3f, 0x8a, 0xd8,
            0x26, 0x4d, 0xa7, 0x58, 0xaa, 0x1b, 0x88, 0xe0, 0x40, 0xd1, 0x58, 0x9e, 0x7b, 0x7f,
            0x23, 0x76, 0xef, 0x49,
        ];
        let actual = clamp_integer(input);
        assert_eq!(actual, expected);

        let expected = [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0x40,
        ];
        let actual = clamp_integer([0; 32]);
        assert_eq!(expected, actual);
        let expected = [
            0xf8, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
            0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
            0xff, 0xff, 0xff, 0x7f,
        ];
        let actual = clamp_integer([0xff; 32]);
        assert_eq!(actual, expected);

        assert_eq!(
            LARGEST_CLAMPED_INTEGER,
            clamp_integer(LARGEST_CLAMPED_INTEGER)
        );
    }

    // multiplication gives the same product whether or not either input is reduced first
    #[test]
    fn test_mul_reduction_invariance() {
        let mut rng = rand::rng();

        for _ in 0..10 {
            let (a, b, c) = {
                let mut a_bytes = [0u8; 32];
                let mut b_bytes = [0u8; 32];
                let mut c_bytes = [0u8; 32];
                rng.fill_bytes(&mut a_bytes);
                rng.fill_bytes(&mut b_bytes);
                rng.fill_bytes(&mut c_bytes);
                (
                    Scalar { bytes: a_bytes },
                    Scalar { bytes: b_bytes },
                    Scalar {
                        bytes: clamp_integer(c_bytes),
                    },
                )
            };

            let reduced_mul_ab = a.reduce() * b.reduce();
            let reduced_mul_ac = a.reduce() * c.reduce();
            assert_eq!(a * b, reduced_mul_ab);
            assert_eq!(a.reduce() * b, reduced_mul_ab);
            assert_eq!(a * b.reduce(), reduced_mul_ab);
            assert_eq!(a * c, reduced_mul_ac);
            assert_eq!(a.reduce() * c, reduced_mul_ac);
            assert_eq!(a * c.reduce(), reduced_mul_ac);
        }
    }
}
