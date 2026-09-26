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
    fn index(&self, index: usize) -> &u8 {
        &(self.bytes[index])
    }
}

impl MulAssign<&Scalar> for Scalar {
    fn mul_assign(&mut self, rhs: &Scalar) {
        *self = UnpackedScalar::mul(&self.unpack(), &rhs.unpack()).pack();
    }
}

define_mul_assign_variants!(LHS = Scalar, RHS = Scalar);

impl Mul<&Scalar> for &Scalar {
    type Output = Scalar;
    fn mul(self, rhs: &Scalar) -> Scalar {
        UnpackedScalar::mul(&self.unpack(), &rhs.unpack()).pack()
    }
}

define_mul_variants!(LHS = Scalar, RHS = Scalar, Output = Scalar);

impl AddAssign<&Scalar> for Scalar {
    fn add_assign(&mut self, rhs: &Scalar) {
        *self = *self + rhs;
    }
}

define_add_assign_variants!(LHS = Scalar, RHS = Scalar);

impl Add<&Scalar> for &Scalar {
    type Output = Scalar;
    fn add(self, rhs: &Scalar) -> Scalar {
        UnpackedScalar::add(&self.unpack(), &rhs.unpack()).pack()
    }
}

define_add_variants!(LHS = Scalar, RHS = Scalar, Output = Scalar);

impl SubAssign<&Scalar> for Scalar {
    fn sub_assign(&mut self, rhs: &Scalar) {
        *self = *self - rhs;
    }
}

define_sub_assign_variants!(LHS = Scalar, RHS = Scalar);

impl Sub<&Scalar> for &Scalar {
    type Output = Scalar;
    fn sub(self, rhs: &Scalar) -> Scalar {
        UnpackedScalar::sub(&self.unpack(), &rhs.unpack()).pack()
    }
}

define_sub_variants!(LHS = Scalar, RHS = Scalar, Output = Scalar);

impl Neg for &Scalar {
    type Output = Scalar;
    #[allow(non_snake_case)]
    fn neg(self) -> Scalar {
        let self_unpacked = self.unpack();
        let self_R = UnpackedScalar::mul_internal(&self_unpacked, &constants::R);
        let self_mod_l = UnpackedScalar::montgomery_reduce(&self_R);

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
    fn from(value: u8) -> Scalar {
        let mut bytes = [0u8; 32];
        bytes[0] = value;
        Scalar { bytes }
    }
}

impl From<u16> for Scalar {
    fn from(value: u16) -> Scalar {
        let mut bytes = [0u8; 32];
        let value_bytes = value.to_le_bytes();
        bytes[0..value_bytes.len()].copy_from_slice(&value_bytes);
        Scalar { bytes }
    }
}

impl From<u32> for Scalar {
    fn from(value: u32) -> Scalar {
        let mut bytes = [0u8; 32];
        let value_bytes = value.to_le_bytes();
        bytes[0..value_bytes.len()].copy_from_slice(&value_bytes);
        Scalar { bytes }
    }
}

impl From<u64> for Scalar {
    fn from(value: u64) -> Scalar {
        let mut bytes = [0u8; 32];
        let value_bytes = value.to_le_bytes();
        bytes[0..value_bytes.len()].copy_from_slice(&value_bytes);
        Scalar { bytes }
    }
}

impl From<u128> for Scalar {
    fn from(value: u128) -> Scalar {
        let mut bytes = [0u8; 32];
        let value_bytes = value.to_le_bytes();
        bytes[0..value_bytes.len()].copy_from_slice(&value_bytes);
        Scalar { bytes }
    }
}

impl From<&Scalar> for I256 {
    fn from(scalar: &Scalar) -> I256 {
        I256::from_le_bytes(scalar.to_bytes())
    }
}

impl HEEADecomposition for Scalar {
    /// Split the challenge into half-size `rho` and `tau` with `rho = tau h` when the flag is false and `rho = -tau h` when true
    fn heea_decompose(&self) -> (Scalar, Scalar, bool) {
        let challenge = self.into();

        let (mut rho_i, mut tau_i) = curve25519_heea_vartime_lehmer(challenge);

        // Both values go back as magnitudes and each negative sign flips h once
        let mut is_flipped = false;
        if rho_i < I256::ZERO {
            rho_i = -rho_i;
            is_flipped = !is_flipped;
        }
        if tau_i < I256::ZERO {
            tau_i = -tau_i;
            is_flipped = !is_flipped;
        }

        let rho = Scalar::from_positive_i256(&rho_i);
        let tau = Scalar::from_positive_i256(&tau_i);

        (rho, tau, is_flipped)
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
        let unpacked = self.unpack();
        let inverse = unpacked.invert();

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
        let one: UnpackedScalar = Scalar::ONE.unpack().as_montgomery();

        let mut scratch = vec![one; inputs.len()];

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

        let product_inverse = acc.pack();

        for (input, scratch) in inputs.iter_mut().rev().zip(scratch.iter().rev()) {
            let tmp = UnpackedScalar::montgomery_mul(&acc, &input.unpack());
            *input = UnpackedScalar::montgomery_mul(&acc, scratch).pack();
            acc = tmp;
        }

        product_inverse
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
    pub fn from_positive_i256(value: &I256) -> Self {
        debug_assert!(value >= &I256::ZERO);
        Scalar {
            bytes: value.to_le_bytes(),
        }
    }

    /// Reduce the scalar modulo l
    #[allow(non_snake_case)]
    fn reduce(&self) -> Scalar {
        let x = self.unpack();
        let xR = UnpackedScalar::mul_internal(&x, &constants::R);
        let x_mod_l = UnpackedScalar::montgomery_reduce(&xR);

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
        let montgomery = self.as_montgomery();
        let inverse = montgomery.montgomery_invert();

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
pub mod tests {
    use alloc::vec::Vec;

    use rand::Rng;

    use super::*;

    /// The value 2238329342913194256032495932344128051776374960164957527413114840482143558222
    static FIRST: Scalar = Scalar {
        bytes: [
            0x4e, 0x5a, 0xb4, 0x34, 0x5d, 0x47, 0x08, 0x84, 0x59, 0x13, 0xb4, 0x64, 0x1b, 0xc2,
            0x7d, 0x52, 0x52, 0xa5, 0x85, 0x10, 0x1b, 0xcc, 0x42, 0x44, 0xd4, 0x49, 0xf4, 0xa8,
            0x79, 0xd9, 0xf2, 0x04,
        ],
    };
    /// The inverse of the first test scalar
    static FIRST_INVERSE: Scalar = Scalar {
        bytes: [
            0x1c, 0xdc, 0x17, 0xfc, 0xe0, 0xe9, 0xa5, 0xbb, 0xd9, 0x24, 0x7e, 0x56, 0xbb, 0x01,
            0x63, 0x47, 0xbb, 0xba, 0x31, 0xed, 0xd5, 0xa9, 0xbb, 0x96, 0xd5, 0x0b, 0xcd, 0x7a,
            0x3f, 0x96, 0x2a, 0x0f,
        ],
    };
    /// The value 2592331292931086675770238855846338635550719849568364935475441891787804997264
    static SECOND: Scalar = Scalar {
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

    /// The product of the two test scalars mod l
    static FIRST_TIMES_SECOND: Scalar = Scalar {
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

    /// A fixed test scalar
    static A_SCALAR: Scalar = Scalar {
        bytes: [
            0x1a, 0x0e, 0x97, 0x8a, 0x90, 0xf6, 0x62, 0x2d, 0x37, 0x47, 0x02, 0x3f, 0x8a, 0xd8,
            0x26, 0x4d, 0xa7, 0x58, 0xaa, 0x1b, 0x88, 0xe0, 0x40, 0xd1, 0x58, 0x9e, 0x7b, 0x7f,
            0x23, 0x76, 0xef, 0x09,
        ],
    };

    /// The width-5 non-adjacent form of the A test scalar
    static A_NONADJACENT_FORM: [i8; 256] = [
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

    /// The largest scalar below l
    const BASEPOINT_ORDER_MINUS_ONE: Scalar = Scalar {
        bytes: [
            0xec, 0xd3, 0xf5, 0x5c, 0x1a, 0x63, 0x12, 0x58, 0xd6, 0x9c, 0xf7, 0xa2, 0xde, 0xf9,
            0xde, 0x14, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x10,
        ],
    };

    /// The largest clamped value
    static LARGEST_CLAMPED_INTEGER: [u8; 32] = clamp_integer(LARGEST_UNREDUCED_SCALAR.bytes);

    fn check_radix_digits(scalar: Scalar, width: usize) {
        let digits_count = Scalar::to_radix_2w_size_hint(width);
        let digits = scalar.as_radix_2w(width);

        let radix = Scalar::from((1 << width) as u64);
        let mut term = Scalar::ONE;
        let mut recovered_scalar = Scalar::ZERO;
        for digit in &digits[0..digits_count] {
            let digit = *digit;
            if digit != 0 {
                let signed_digit = match digit < 0 {
                    true => -Scalar::from((-(digit as i64)) as u64),
                    false => Scalar::from(digit as u64),
                };
                recovered_scalar += term * signed_digit;
            }
            term *= radix;
        }
        // An unreduced input comes back only mod l
        assert_eq!(recovered_scalar, scalar.reduce());
    }

    // a product the fuzzer found reduces the same through both reductions
    #[test]
    fn fuzzer_reduction() {
        // Little-endian bytes of 24519928653854221733733552434404946937899825954937634815
        let first_bytes = [
            255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
            255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ];
        // Little-endian bytes of 4975441334397345751130612518500927154628011511324180036903450236863266160640
        let second_bytes = [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 210, 210,
            210, 255, 255, 255, 255, 10,
        ];
        // Little-endian bytes of 6432735165214683820902750800207468552549813371247423777071615116673864412038
        let product_bytes = [
            134, 171, 119, 216, 180, 128, 178, 62, 171, 132, 32, 62, 34, 119, 104, 193, 47, 215,
            181, 250, 14, 207, 172, 93, 75, 207, 211, 103, 144, 204, 56, 14,
        ];

        let first = Scalar::from_bytes_mod_order(first_bytes);
        let second = Scalar::from_bytes_mod_order(second_bytes);
        let product = Scalar::from_bytes_mod_order(product_bytes);

        let mut wide = [0u8; 64];
        wide[0..32].copy_from_slice(&first_bytes[..]);
        let wide_first = Scalar::from_bytes_mod_order_wide(&wide);
        wide[0..32].copy_from_slice(&second_bytes[..]);
        let wide_second = Scalar::from_bytes_mod_order_wide(&wide);

        assert_eq!(product, first * second);
        assert_eq!(product, wide_first * wide_second);
    }

    // the width-5 form of a known scalar matches the reference digits
    #[test]
    fn nonadjacent_form() {
        let digits = A_SCALAR.non_adjacent_form(5);
        for index in 0..256 {
            assert_eq!(digits[index], A_NONADJACENT_FORM[index]);
        }
    }

    // the 128-bit form matches the full form at every width
    #[test]
    fn nonadjacent_narrow() {
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
            for width in 2..=8 {
                let full = scalar.non_adjacent_form(width);
                let narrow = scalar.non_adjacent_form_128(width);

                for index in 0..=HEEA_MAX_INDEX {
                    assert_eq!(
                        narrow[index], full[index],
                        "digit {index} differs at width {width}"
                    );
                }

                assert!(
                    full[NAF_128_SIZE..].iter().all(|&digit| digit == 0),
                    "the full form has digits above bit 128 at width {width}"
                );
            }
        }
    }

    // a u64 lands in the low eight bytes in little-endian order
    #[test]
    fn from_word() {
        let value: u64 = 0xdeadbeefdeadbeef;
        let scalar = Scalar::from(value);
        assert_eq!(scalar[7], 0xde);
        assert_eq!(scalar[6], 0xad);
        assert_eq!(scalar[5], 0xbe);
        assert_eq!(scalar[4], 0xef);
        assert_eq!(scalar[3], 0xde);
        assert_eq!(scalar[2], 0xad);
        assert_eq!(scalar[1], 0xbe);
        assert_eq!(scalar[0], 0xef);
    }

    // multiplying by one leaves every byte alone
    #[test]
    fn multiply_one() {
        let product = FIRST * Scalar::ONE;
        for index in 0..32 {
            assert!(product[index] == FIRST[index]);
        }
    }

    // addition wraps around the group order
    #[test]
    fn add_wraps() {
        assert_eq!(BASEPOINT_ORDER_MINUS_ONE + Scalar::ONE, Scalar::ZERO);
    }

    // subtraction wraps around the group order
    #[test]
    fn subtract_wraps() {
        assert_eq!(Scalar::ZERO - Scalar::ONE, BASEPOINT_ORDER_MINUS_ONE);
    }

    // one plus one is two
    #[test]
    fn addition() {
        let two = Scalar::from(2u64);
        let one = Scalar::ONE;
        assert_eq!(one + one, two);
    }

    // the product of the two test scalars matches the reference value
    #[test]
    fn multiplication() {
        assert_eq!(FIRST * SECOND, FIRST_TIMES_SECOND);
    }

    // the product iterator handles borrowed, empty and owned inputs
    #[test]
    fn product() {
        let pair = [FIRST, SECOND];
        let pair_product: Scalar = pair.iter().product();
        assert_eq!(pair_product, FIRST_TIMES_SECOND);

        let empty: [Scalar; 0] = [];
        let empty_product: Scalar = empty.iter().product();
        assert_eq!(empty_product, Scalar::ONE);

        let twos = [Scalar::from(2u64); 10];
        let threes = [Scalar::from(3u64); 10];
        let sixes = twos
            .iter()
            .zip(threes.iter())
            .map(|(two, three)| two * three);

        let twos_product: Scalar = twos.iter().product();
        let threes_product: Scalar = threes.iter().product();
        let sixes_product: Scalar = sixes.product();

        assert_eq!(twos_product, Scalar::from(1024u64));
        assert_eq!(threes_product, Scalar::from(59049u64));
        assert_eq!(sixes_product, Scalar::from(60466176u64));
        assert_eq!(twos_product * threes_product, sixes_product);
    }

    // the sum iterator handles borrowed, empty and owned inputs
    #[test]
    fn sum() {
        let two = Scalar::from(2u64);
        let ones = [Scalar::ONE, Scalar::ONE];
        let ones_sum: Scalar = ones.iter().sum();
        assert_eq!(ones_sum, two);

        let empty: [Scalar; 0] = [];
        let empty_sum: Scalar = empty.iter().sum();
        assert_eq!(empty_sum, Scalar::ZERO);

        let ones = [Scalar::from(1u64); 10];
        let twos = [Scalar::from(2u64); 10];
        let threes = ones.iter().zip(twos.iter()).map(|(one, two)| one + two);

        let ones_sum: Scalar = ones.iter().sum();
        let twos_sum: Scalar = twos.iter().sum();
        let threes_sum: Scalar = threes.sum();

        assert_eq!(ones_sum, Scalar::from(10u64));
        assert_eq!(twos_sum, Scalar::from(20u64));
        assert_eq!(threes_sum, Scalar::from(30u64));
        assert_eq!(ones_sum + twos_sum, threes_sum);
    }

    // squaring matches multiplying a scalar by itself
    #[test]
    fn square() {
        let expected = FIRST * FIRST;
        let actual = FIRST.unpack().square().pack();
        for index in 0..32 {
            assert!(expected[index] == actual[index]);
        }
    }

    // halving then doubling gives the scalar back near zero and near the order
    #[test]
    fn halving() {
        for value in 0u64..32 {
            let scalar = Scalar::from(value);
            let half = scalar.div_by_2();
            assert_eq!(scalar, half + half);
        }

        for value in 0u64..32 {
            let scalar = Scalar::ZERO - Scalar::from(value);
            let half = scalar.div_by_2();
            assert_eq!(scalar, half + half);
        }
    }

    // the largest 256-bit value reduces to its reference residue
    #[test]
    fn reduce() {
        let biggest = Scalar::from_bytes_mod_order([0xff; 32]);
        assert_eq!(biggest, CANONICAL_2_256_MINUS_1);
    }

    // the fold reduction agrees with the Montgomery path at every edge
    #[test]
    fn wide_reduction() {
        let order = constants::BASEPOINT_ORDER.to_bytes();
        let mut vectors: Vec<[u8; 64]> = vec![[0u8; 64], [0xffu8; 64]];
        for shift in [252usize, 253, 255, 256, 260, 300, 400, 511] {
            let mut vector = [0u8; 64];
            vector[shift / 8] = 1 << (shift % 8);
            vectors.push(vector);
        }
        for delta in [-2i64, -1, 0, 1, 2] {
            let mut vector = [0u8; 64];
            vector[..32].copy_from_slice(&order);
            let mut overflow = delta;
            for byte in vector.iter_mut() {
                let sum = *byte as i64 + overflow;
                *byte = sum.rem_euclid(256) as u8;
                overflow = sum.div_euclid(256);
            }
            vectors.push(vector);
        }
        let mut state = 0x5EED_1EE5_u64;
        for _ in 0..4096 {
            let mut vector = [0u8; 64];
            for word in vector.as_chunks_mut::<8>().0 {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                word.copy_from_slice(&state.to_le_bytes());
            }
            vectors.push(vector);
        }
        for vector in &vectors {
            let expected = UnpackedScalar::from_bytes_wide(vector).pack();
            assert_eq!(
                Scalar::from_bytes_mod_order_wide(vector),
                expected,
                "{vector:02x?}"
            );
        }
    }

    // a 512-bit value reduces to its reference residue
    #[test]
    fn wide_decode() {
        let mut wide = [0u8; 64];
        for index in 0..32 {
            wide[index] = FIRST[index];
            wide[32 + index] = FIRST[index];
        }
        // The value x + 2^256 x mod l
        let reduced = Scalar {
            bytes: [
                216, 154, 179, 139, 210, 121, 2, 71, 69, 99, 158, 216, 23, 173, 63, 100, 204, 0,
                91, 50, 219, 153, 57, 249, 28, 82, 31, 197, 100, 165, 192, 8,
            ],
        };
        let decoded = Scalar::from_bytes_mod_order_wide(&wide);
        for index in 0..32 {
            assert!(decoded[index] == reduced[index]);
        }
    }

    // the inverse matches the reference and multiplies back to one
    #[test]
    fn invert() {
        let inverse = FIRST.invert();
        assert_eq!(inverse, FIRST_INVERSE);
        assert_eq!(inverse * FIRST, Scalar::ONE);
    }

    // negating a scalar twice gives the scalar back
    #[test]
    fn double_negation() {
        let negated = -&FIRST;
        assert_eq!(-&negated, FIRST);
    }

    // packing and unpacking round-trips the limbs
    #[test]
    fn bytes_roundtrip() {
        let unpacked = FIRST.unpack();
        let bytes = unpacked.to_bytes();
        let repacked = UnpackedScalar::from_bytes(&bytes);

        assert_eq!(repacked.0, unpacked.0);
    }

    // Montgomery reduction of a wide value matches the wide decoder
    #[test]
    fn montgomery_reduce() {
        let mut wide = [0u8; 64];
        for index in 0..32 {
            wide[index] = FIRST[index];
            wide[32 + index] = FIRST[index];
        }
        // The value x + 2^256 x mod l
        let expected = Scalar {
            bytes: [
                216, 154, 179, 139, 210, 121, 2, 71, 69, 99, 158, 216, 23, 173, 63, 100, 204, 0,
                91, 50, 219, 153, 57, 249, 28, 82, 31, 197, 100, 165, 192, 8,
            ],
        };
        let reduced = Scalar::from_bytes_mod_order_wide(&wide);
        assert_eq!(reduced.bytes, expected.bytes);

        let interim =
            UnpackedScalar::mul_internal(&UnpackedScalar::from_bytes_wide(&wide), &constants::R);
        let montgomery_reduced = UnpackedScalar::montgomery_reduce(&interim);

        assert_eq!(montgomery_reduced.0, reduced.unpack().0);
        assert_eq!(montgomery_reduced.0, expected.unpack().0)
    }

    // the decoder accepts values below l and refuses the rest
    #[test]
    fn canonical_decoding() {
        // The canonical encoding of 1667457891
        let canonical_bytes = [
            99, 99, 99, 99, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0,
        ];

        // Every byte 16 encodes a value at or above l
        let unreduced_bytes = [16; 32];

        // The decoder must refuse a set high bit without masking it off
        let high_bit_bytes = [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 128,
        ];

        assert!(bool::from(
            Scalar::from_canonical_bytes(canonical_bytes).is_some()
        ));
        assert!(bool::from(
            Scalar::from_canonical_bytes(unreduced_bytes).is_none()
        ));
        assert!(bool::from(
            Scalar::from_canonical_bytes(high_bit_bytes).is_none()
        ));
    }

    // a zero input trips the debug check in batch inversion
    #[cfg(debug_assertions)]
    #[test]
    #[should_panic]
    fn batch_zero() {
        let mut scalars = [Scalar::ONE; 16];
        scalars[3] = Scalar::ZERO;
        Scalar::invert_batch(&mut scalars);
    }

    // inverting an empty batch returns one
    #[test]
    fn batch_empty() {
        assert_eq!(Scalar::ONE, Scalar::invert_batch(&mut []));
    }

    // batch inversion inverts every entry and returns the inverse product
    #[test]
    fn batch_inverse() {
        let mut power = Scalar::from(1u64);
        let mut inverted: [Scalar; 16] = core::array::from_fn(|_| {
            let current = power;
            power = power + power;
            current
        });
        let originals = inverted;

        let product: Scalar = inverted.iter().product();
        let expected = product.invert();
        let returned = Scalar::invert_batch(&mut inverted);
        assert_eq!(returned, expected);

        for (inverse, original) in inverted.iter().zip(originals.iter()) {
            assert_eq!(inverse * original, Scalar::ONE);
        }
    }

    // the allocating batch inversion matches the fixed-size one
    #[test]
    fn batch_allocating() {
        let mut power = Scalar::from(1u64);
        let mut inverted: Vec<_> = (0..16)
            .map(|_| {
                let current = power;
                power = power + power;
                current
            })
            .collect();
        let originals = inverted.clone();

        let product: Scalar = inverted.iter().product();
        let expected = product.invert();
        let returned = Scalar::invert_batch_alloc(&mut inverted);
        assert_eq!(returned, expected);

        for (inverse, original) in inverted.iter().zip(originals.iter()) {
            assert_eq!(inverse * original, Scalar::ONE);
        }
    }

    // radix digits at widths 6, 7 and 8 recover the scalar they came from
    #[test]
    fn radix_digits() {
        let cases = (2..100)
            .map(|value| Scalar::from(value as u64).invert())
            .chain(core::iter::once(LARGEST_UNREDUCED_SCALAR));

        for scalar in cases {
            check_radix_digits(scalar, 6);
            check_radix_digits(scalar, 7);
            check_radix_digits(scalar, 8);
        }
    }

    // bytes read back as little-endian words
    #[test]
    fn read_words() {
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

        for (source, expected) in cases {
            let mut destination = vec![0; expected.len()];
            read_le_u64_into(source, &mut destination);

            assert_eq!(&destination, expected);
        }
    }

    // every integer conversion agrees with scalar addition
    #[test]
    fn from_integers() {
        let one = Scalar::ONE;

        let value = 0x23u8;
        assert_eq!(Scalar::from(value) + one, Scalar::from(value + 1));

        let value = 0x2323u16;
        assert_eq!(Scalar::from(value) + one, Scalar::from(value + 1));

        let value = 0x2323_2323u32;
        assert_eq!(Scalar::from(value) + one, Scalar::from(value + 1));

        let value = 0x2323_2323_2323_2323u64;
        assert_eq!(Scalar::from(value) + one, Scalar::from(value + 1));

        let value = 0x2323_2323_2323_2323_2323_2323_2323_2323u128;
        assert_eq!(Scalar::from(value) + one, Scalar::from(value + 1));
    }

    // clamping sets and clears the expected bits and is idempotent
    #[test]
    fn clamp() {
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
    fn unreduced_product() {
        let mut rng = rand::rng();

        for _ in 0..10 {
            let mut first_bytes = [0u8; 32];
            let mut second_bytes = [0u8; 32];
            let mut clamped_bytes = [0u8; 32];
            rng.fill_bytes(&mut first_bytes);
            rng.fill_bytes(&mut second_bytes);
            rng.fill_bytes(&mut clamped_bytes);
            let first = Scalar { bytes: first_bytes };
            let second = Scalar {
                bytes: second_bytes,
            };
            let clamped = Scalar {
                bytes: clamp_integer(clamped_bytes),
            };

            let reduced_second = first.reduce() * second.reduce();
            let reduced_clamped = first.reduce() * clamped.reduce();
            assert_eq!(first * second, reduced_second);
            assert_eq!(first.reduce() * second, reduced_second);
            assert_eq!(first * second.reduce(), reduced_second);
            assert_eq!(first * clamped, reduced_clamped);
            assert_eq!(first.reduce() * clamped, reduced_clamped);
            assert_eq!(first * clamped.reduce(), reduced_clamped);
        }
    }
}
