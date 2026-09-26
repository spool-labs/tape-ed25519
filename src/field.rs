//! Field arithmetic modulo p = 2^255 - 19 built from the backend multiplication and squaring

// Vendored from curve25519-dalek, BSD-3-Clause

#![allow(unused_qualifications)]

use alloc::vec;

use subtle::Choice;
use subtle::ConditionallyNegatable;
use subtle::ConditionallySelectable;
use subtle::ConstantTimeEq;

use crate::backend;
use crate::constants;

/// An element of the field of integers mod 2^255 - 19, on the 64-bit serial backend
pub type FieldElement = backend::serial::u64::field::FieldElement51;

impl Eq for FieldElement {}

impl PartialEq for FieldElement {
    fn eq(&self, other: &FieldElement) -> bool {
        self.ct_eq(other).into()
    }
}

impl ConstantTimeEq for FieldElement {
    /// Compare canonical encodings, since the limb form is not unique
    fn ct_eq(&self, other: &FieldElement) -> Choice {
        self.to_bytes().ct_eq(&other.to_bytes())
    }
}

impl Default for FieldElement {
    fn default() -> Self {
        FieldElement::ZERO
    }
}

impl FieldElement {
    /// Return true when the low bit of the canonical encoding is set, which Ed25519 calls negative
    pub fn is_negative(&self) -> Choice {
        let bytes = self.to_bytes();
        (bytes[0] & 1).into()
    }

    /// Return true when this element is zero
    pub fn is_zero(&self) -> Choice {
        let zero = [0u8; 32];
        let bytes = self.to_bytes();

        bytes.ct_eq(&zero)
    }

    /// Compute (self^(2^250 - 1), self^11), the shared prefix of `invert` and `pow_p58`
    #[rustfmt::skip]
    fn pow22501(&self) -> (FieldElement, FieldElement) {
        // Each trailing note lists the set bits in the exponent of t_i
        let t0  = self.square();           // 1
        let t1  = t0.square().square();    // 3
        let t2  = self * &t1;              // 3,0
        let t3  = &t0 * &t2;               // 3,1,0
        let t4  = t3.square();             // 4,2,1
        let t5  = &t2 * &t4;               // 4,3,2,1,0
        let t6  = t5.pow2k(5);             // 9,8,7,6,5
        let t7  = &t6 * &t5;               // 9,8,7,6,5,4,3,2,1,0
        let t8  = t7.pow2k(10);            // 19..10
        let t9  = &t8 * &t7;               // 19..0
        let t10 = t9.pow2k(20);            // 39..20
        let t11 = &t10 * &t9;              // 39..0
        let t12 = t11.pow2k(10);           // 49..10
        let t13 = &t12 * &t7;              // 49..0
        let t14 = t13.pow2k(50);           // 99..50
        let t15 = &t14 * &t13;             // 99..0
        let t16 = t15.pow2k(100);          // 199..100
        let t17 = &t16 * &t15;             // 199..0
        let t18 = t17.pow2k(50);           // 249..50
        let t19 = &t18 * &t13;             // 249..0

        (t19, t3)
    }

    /// Replace each public element with its inverse, leaving zeros unchanged
    pub fn invert_batch_alloc(inputs: &mut [FieldElement]) {
        let n = inputs.len();
        let mut scratch = vec![FieldElement::ONE; n];

        Self::internal_invert_batch(inputs, &mut scratch);
    }

    /// Invert every element with one inversion by Montgomery's trick, with `scratch` as long as `inputs`
    fn internal_invert_batch(inputs: &mut [FieldElement], scratch: &mut [FieldElement]) {
        debug_assert_eq!(inputs.len(), scratch.len());

        let mut acc = FieldElement::ONE;

        // Each scratch slot keeps the product of every earlier input
        for (input, scratch) in inputs.iter().zip(scratch.iter_mut()) {
            *scratch = acc;
            // Zeros are skipped in constant time so one zero cannot spoil the others
            #[allow(unused_mut)]
            let mut product = &acc * input;
            acc.conditional_assign(&product, !input.is_zero());
        }

        assert!(bool::from(!acc.is_zero()));

        acc = acc.invert();

        for (input, scratch) in inputs.iter_mut().rev().zip(scratch.iter_mut().rev()) {
            #[allow(unused_mut)]
            let mut tmp = &acc * input;
            #[allow(unused_mut)]
            let mut inverse = &acc * scratch;
            let nz = !input.is_zero();
            input.conditional_assign(&inverse, nz);
            acc.conditional_assign(&tmp, nz);
        }
    }

    /// Compute the inverse as self^(p - 2), which maps zero to zero
    #[rustfmt::skip]
    #[allow(clippy::let_and_return)]
    pub fn invert(&self) -> FieldElement {
        // Each trailing note lists the set bits of the exponent so far
        let (t19, t3) = self.pow22501();   // t19 249..0 and t3 3,1,0
        let t20 = t19.pow2k(5);            // 254..5
        let t21 = &t20 * &t3;              // 254..5,3,1,0

        t21
    }

    /// Raise this element to the power (p - 5) / 8 = 2^252 - 3
    #[rustfmt::skip]
    #[allow(clippy::let_and_return)]
    pub fn pow_p58(&self) -> FieldElement {
        // Each trailing note lists the set bits of the exponent so far
        let (t19, _) = self.pow22501();    // 249..0
        let t20 = t19.pow2k(2);            // 251..2
        let t21 = self * &t20;             // 251..2,0

        t21
    }

    /// Return the nonnegative sqrt(u/v) and true, or sqrt(i u/v) and false when u/v is not square
    pub fn sqrt_ratio_i(u: &FieldElement, v: &FieldElement) -> (Choice, FieldElement) {
        // r = (u v^3)(u v^7)^((p - 5) / 8) merges the inversion and square root, then v r^2 fixes the sign
        let v3 = &v.square() * v;
        let v7 = &v3.square() * v;
        let mut r = &(u * &v3) * &(u * &v7).pow_p58();
        let check = v * &r.square();

        let i = &constants::SQRT_M1;

        let correct_sign_sqrt = check.ct_eq(u);
        let flipped_sign_sqrt = check.ct_eq(&(-u));
        let flipped_sign_sqrt_i = check.ct_eq(&(&(-u) * i));

        let r_prime = &constants::SQRT_M1 * &r;
        r.conditional_assign(&r_prime, flipped_sign_sqrt | flipped_sign_sqrt_i);

        let r_is_negative = r.is_negative();
        r.conditional_negate(r_is_negative);

        let was_nonzero_square = correct_sign_sqrt | flipped_sign_sqrt;

        (was_nonzero_square, r)
    }

    /// Return the nonnegative sqrt(1/self) and whether self is a nonzero square
    pub fn invsqrt(&self) -> (Choice, FieldElement) {
        FieldElement::sqrt_ratio_i(&FieldElement::ONE, self)
    }
}

#[cfg(test)]
mod test {
    use crate::field::*;

    /// A random field element
    static A_BYTES: [u8; 32] = [
        0x04, 0xfe, 0xdf, 0x98, 0xa7, 0xfa, 0x0a, 0x68, 0x84, 0x92, 0xbd, 0x59, 0x08, 0x07, 0xa7,
        0x03, 0x9e, 0xd1, 0xf6, 0xf2, 0xe1, 0xd9, 0xe2, 0xa4, 0xa4, 0x51, 0x47, 0x36, 0xf3, 0xc3,
        0xa9, 0x17,
    ];

    /// The encoding of a^2
    static ASQ_BYTES: [u8; 32] = [
        0x75, 0x97, 0x24, 0x9e, 0xe6, 0x06, 0xfe, 0xab, 0x24, 0x04, 0x56, 0x68, 0x07, 0x91, 0x2d,
        0x5d, 0x0b, 0x0f, 0x3f, 0x1c, 0xb2, 0x6e, 0xf2, 0xe2, 0x63, 0x9c, 0x12, 0xba, 0x73, 0x0b,
        0xe3, 0x62,
    ];

    /// The encoding of 1/a
    static AINV_BYTES: [u8; 32] = [
        0x96, 0x1b, 0xcd, 0x8d, 0x4d, 0x5e, 0xa2, 0x3a, 0xe9, 0x36, 0x37, 0x93, 0xdb, 0x7b, 0x4d,
        0x70, 0xb8, 0x0d, 0xc0, 0x55, 0xd0, 0x4c, 0x1d, 0x7b, 0x90, 0x71, 0xd8, 0xe9, 0xb6, 0x18,
        0xe6, 0x30,
    ];

    /// The encoding of a^((p - 5) / 8)
    static AP58_BYTES: [u8; 32] = [
        0x6a, 0x4f, 0x24, 0x89, 0x1f, 0x57, 0x60, 0x36, 0xd0, 0xbe, 0x12, 0x3c, 0x8f, 0xf5, 0xb1,
        0x59, 0xe0, 0xf0, 0xb8, 0x1b, 0x20, 0xd2, 0xb5, 0x1f, 0x15, 0x21, 0xf9, 0xe3, 0xe1, 0x61,
        0x21, 0x55,
    ];

    #[test]
    fn a_mul_a_vs_a_squared_constant() {
        let a = FieldElement::from_bytes(&A_BYTES);
        let asq = FieldElement::from_bytes(&ASQ_BYTES);
        assert_eq!(asq, &a * &a);
    }

    #[test]
    fn a_square_vs_a_squared_constant() {
        let a = FieldElement::from_bytes(&A_BYTES);
        let asq = FieldElement::from_bytes(&ASQ_BYTES);
        assert_eq!(asq, a.square());
    }

    #[test]
    fn a_square2_vs_a_squared_constant() {
        let a = FieldElement::from_bytes(&A_BYTES);
        let asq = FieldElement::from_bytes(&ASQ_BYTES);
        assert_eq!(a.square2(), &asq + &asq);
    }

    #[test]
    fn a_invert_vs_inverse_of_a_constant() {
        let a = FieldElement::from_bytes(&A_BYTES);
        let ainv = FieldElement::from_bytes(&AINV_BYTES);
        let should_be_inverse = a.invert();
        assert_eq!(ainv, should_be_inverse);
        assert_eq!(FieldElement::ONE, &a * &should_be_inverse);
    }

    #[test]
    fn invert_batch_a_matches_nonbatched() {
        let a = FieldElement::from_bytes(&A_BYTES);
        let ap58 = FieldElement::from_bytes(&AP58_BYTES);
        let asq = FieldElement::from_bytes(&ASQ_BYTES);
        let ainv = FieldElement::from_bytes(&AINV_BYTES);
        let a0 = &a - &a;
        let a2 = &a + &a;
        let a_list = vec![a, ap58, asq, ainv, a0, a2];
        let mut ainv_list = a_list.clone();
        FieldElement::invert_batch_alloc(&mut ainv_list[..]);
        for i in 0..6 {
            assert_eq!(a_list[i].invert(), ainv_list[i]);
        }
    }

    #[test]
    fn sqrt_ratio_behavior() {
        let zero = FieldElement::ZERO;
        let one = FieldElement::ONE;
        let i = constants::SQRT_M1;
        let two = &one + &one;
        let four = &two + &two;

        // A zero u gives true and zero
        let (choice, sqrt) = FieldElement::sqrt_ratio_i(&zero, &zero);
        assert!(bool::from(choice));
        assert_eq!(sqrt, zero);
        assert!(bool::from(!sqrt.is_negative()));

        // A zero v with a nonzero u gives false and zero
        let (choice, sqrt) = FieldElement::sqrt_ratio_i(&one, &zero);
        assert!(bool::from(!choice));
        assert_eq!(sqrt, zero);
        assert!(bool::from(!sqrt.is_negative()));

        // Two is not a square mod p, so the result is sqrt(2i)
        let (choice, sqrt) = FieldElement::sqrt_ratio_i(&two, &one);
        assert!(bool::from(!choice));
        assert_eq!(sqrt.square(), &two * &i);
        assert!(bool::from(!sqrt.is_negative()));

        // Four is a square mod p
        let (choice, sqrt) = FieldElement::sqrt_ratio_i(&four, &one);
        assert!(bool::from(choice));
        assert_eq!(sqrt.square(), four);
        assert!(bool::from(!sqrt.is_negative()));

        // A quarter is a square mod p
        let (choice, sqrt) = FieldElement::sqrt_ratio_i(&one, &four);
        assert!(bool::from(choice));
        assert_eq!(&sqrt.square() * &four, one);
        assert!(bool::from(!sqrt.is_negative()));
    }

    #[test]
    fn a_p58_vs_ap58_constant() {
        let a = FieldElement::from_bytes(&A_BYTES);
        let ap58 = FieldElement::from_bytes(&AP58_BYTES);
        assert_eq!(ap58, a.pow_p58());
    }

    #[test]
    fn equality() {
        let a = FieldElement::from_bytes(&A_BYTES);
        let ainv = FieldElement::from_bytes(&AINV_BYTES);
        assert!(a == a);
        assert!(a != ainv);
    }

    /// Bytes with the top bit set, which decoding ignores
    static B_BYTES: [u8; 32] = [
        113, 191, 169, 143, 91, 234, 121, 15, 241, 131, 217, 36, 230, 101, 92, 234, 8, 208, 170,
        251, 97, 127, 70, 210, 58, 23, 166, 87, 240, 169, 184, 178,
    ];

    #[test]
    fn from_bytes_highbit_is_ignored() {
        let mut cleared_bytes = B_BYTES;
        cleared_bytes[31] &= 127u8;
        let with_highbit_set = FieldElement::from_bytes(&B_BYTES);
        let without_highbit_set = FieldElement::from_bytes(&cleared_bytes);
        assert_eq!(without_highbit_set, with_highbit_set);
    }

    #[test]
    fn conditional_negate() {
        let one = FieldElement::ONE;
        let minus_one = FieldElement::MINUS_ONE;
        let mut x = one;
        x.conditional_negate(Choice::from(1));
        assert_eq!(x, minus_one);
        x.conditional_negate(Choice::from(0));
        assert_eq!(x, minus_one);
        x.conditional_negate(Choice::from(1));
        assert_eq!(x, one);
    }

    #[test]
    fn encoding_is_canonical() {
        // One plus p encodes one non-canonically
        let one_encoded_wrongly_bytes: [u8; 32] = [
            0xee, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
            0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
            0xff, 0xff, 0xff, 0x7f,
        ];
        let one = FieldElement::from_bytes(&one_encoded_wrongly_bytes);
        let one_bytes = one.to_bytes();
        assert_eq!(one_bytes[0], 1);
        for byte in &one_bytes[1..] {
            assert_eq!(*byte, 0);
        }
    }

    #[test]
    fn invert_batch_empty() {
        FieldElement::invert_batch_alloc(&mut []);
    }
}
