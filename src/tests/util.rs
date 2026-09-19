//! Signing, encodings and edge-case vectors the test modules share

use alloc::vec::Vec;
use std::string::String;

use ed25519_dalek::Signer;

use crate::constants::EIGHT_TORSION;
use crate::edwards::CompressedEdwardsY;
use crate::scalar::Scalar;
use crate::signature::Signature;

/// Byte strings that decode to a point of order dividing the cofactor 8
pub const LOW_ORDER_COUNT: usize = 14;

/// The message the ZIP-215 vectors are signed over
pub const ZIP215_MESSAGE: &[u8] = b"Zcash";

/// One encoded key and signature, with what each rule should say about it
pub struct TestCase {
    /// The encoded verifying key
    pub key: [u8; 32],

    /// The encoded signature
    pub signature: [u8; 64],

    /// Whether the pre-ZIP-215 rules accept it, which is what `verify_strict` applies
    pub is_valid_strict: bool,

    /// Whether the ZIP-215 rules accept it
    pub is_valid_zip215: bool,
}

/// A signing key from one seed byte, deterministic across runs
pub fn signing_key(seed: u8) -> ed25519_dalek::SigningKey {
    ed25519_dalek::SigningKey::from_bytes(&[seed; 32])
}

/// Sign `message` and hand back the item the batch verifier takes
pub fn signed(seed: u8, message: &[u8]) -> ([u8; 32], Signature) {
    let key = signing_key(seed);
    let signature = key.sign(message);
    (
        key.verifying_key().to_bytes(),
        Signature::from_bytes(&signature.to_bytes()),
    )
}

/// Lowercase hexadecimal, for the message a failing assertion prints
pub fn hex(bytes: &[u8]) -> String {
    hex::encode(bytes)
}

/// The 19 field elements that also encode non-canonically as `x + p` inside 255 bits
pub fn non_canonical_field_encodings() -> Vec<[u8; 32]> {
    const FIRST: u8 = 237;
    const COUNT: u8 = 19;

    let mut bytes = [
        237, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 127,
    ];
    let mut encodings = Vec::new();
    for offset in 0..COUNT {
        bytes[0] = FIRST + offset;
        encodings.push(bytes);
    }
    encodings
}

/// All 25 non-canonical point encodings, the first 6 of them of low order
///
/// A point is encoded by its y coordinate and the sign of x, so an encoding is non-canonical
/// either because y is encoded as `y + p`, or because both sign choices give the same x, which
/// happens only at `x = 0`, that is `y = 1` or `y = -1`.
pub fn non_canonical_point_encodings() -> Vec<[u8; 32]> {
    const SIGN_BIT: u8 = 128;

    let mut encodings = Vec::new();

    // Canonical y with the sign bit set where x is zero.
    encodings.push([
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 128,
    ]);
    encodings.push([
        236, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    ]);

    // Not every field element is the y coordinate of a curve point, so decompression decides.
    for mut encoding in non_canonical_field_encodings() {
        if CompressedEdwardsY(encoding).decompress().is_some() {
            encodings.push(encoding);
        }
        encoding[31] |= SIGN_BIT;
        if CompressedEdwardsY(encoding).decompress().is_some() {
            encodings.push(encoding);
        }
    }

    for encoding in &encodings {
        let point = CompressedEdwardsY(*encoding)
            .decompress()
            .expect("a non-canonical encoding still decodes");
        assert_ne!(*encoding, point.compress().to_bytes());
    }

    encodings
}

/// Every encoding of a point of order dividing 8, canonical and not
///
/// Each 8-torsion point has both a sign-bit-clear and a sign-bit-set encoding, so a blacklist
/// holding only one of the two, or only the canonical encodings, leaves low-order keys reachable.
pub fn low_order_encodings() -> Vec<[u8; 32]> {
    let mut encodings = Vec::new();
    for point in EIGHT_TORSION.iter() {
        encodings.push(point.compress().to_bytes());
    }
    for encoding in non_canonical_point_encodings().into_iter().take(6) {
        encodings.push(encoding);
    }

    assert_eq!(encodings.len(), LOW_ORDER_COUNT);
    for encoding in &encodings {
        let point = CompressedEdwardsY(*encoding).decompress().expect("decodes");
        assert!(point.is_small_order(), "{} is not small order", hex(encoding));
    }
    encodings
}

/// Every pair of low-order `A` and `R` with `s = 0`
///
/// The verification equation is `[8][s]B = [8]R + [8][k]A`. With `R` and `A` of low order the
/// right side is zero, and `s = 0` makes the left side zero, so ZIP-215 accepts all of them. The
/// pre-ZIP-215 rules refuse a low-order `A` or `R` before the equation is reached, so none of
/// these is valid there.
pub fn small_order_cases() -> Vec<TestCase> {
    let mut cases = Vec::new();
    let encodings = low_order_encodings();

    for key in &encodings {
        for r_bytes in &encodings {
            let mut signature = [0u8; 64];
            signature[..32].copy_from_slice(&r_bytes[..]);
            signature[32..].copy_from_slice(Scalar::ZERO.as_bytes());

            cases.push(TestCase {
                key: *key,
                signature,
                is_valid_strict: false,
                is_valid_zip215: true,
            });
        }
    }
    cases
}
