//! Shared signing helpers, encodings and edge-case vectors for the tests

use alloc::vec::Vec;
use std::string::String;

use ed25519_dalek::Signer;

use crate::constants::EIGHT_TORSION;
use crate::edwards::CompressedEdwardsY;
use crate::scalar::Scalar;
use crate::signature::Signature;

/// Count of encodings that decode to a point of order dividing the cofactor 8
pub const LOW_ORDER_COUNT: usize = 14;

/// The first non-canonical point encodings up to this count have low order
const LOW_ORDER_NON_CANONICAL_COUNT: usize = 6;

/// The ZIP-215 vectors sign this message
pub const ZIP215_MESSAGE: &[u8] = b"Zcash";

/// One encoded key and signature with the expected verdict under each rule
pub struct TestCase {
    /// The encoded verifying key
    pub key: [u8; 32],

    /// The encoded signature
    pub signature: [u8; 64],

    /// Whether the pre-ZIP-215 rules of `verify_strict` accept it
    pub is_valid_strict: bool,

    /// Whether the ZIP-215 rules accept it
    pub is_valid_zip215: bool,
}

/// A signing key from one seed byte, deterministic across runs
pub fn signing_key(seed: u8) -> ed25519_dalek::SigningKey {
    ed25519_dalek::SigningKey::from_bytes(&[seed; 32])
}

/// Sign `message` and return the encoded key and signature
pub fn signed(seed: u8, message: &[u8]) -> ([u8; 32], Signature) {
    let key = signing_key(seed);
    let signature = key.sign(message);
    (
        key.verifying_key().to_bytes(),
        Signature::from_bytes(&signature.to_bytes()),
    )
}

/// Encode bytes as lowercase hexadecimal for failure messages
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
pub fn non_canonical_point_encodings() -> Vec<[u8; 32]> {
    const SIGN_BIT: u8 = 128;

    let mut encodings = Vec::new();

    // Canonical y with the sign bit set where x is zero
    encodings.push([
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 128,
    ]);
    encodings.push([
        236, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    ]);

    // Decompression decides which field elements are curve y coordinates
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

/// Every canonical and non-canonical encoding of a point of order dividing 8
pub fn low_order_encodings() -> Vec<[u8; 32]> {
    let mut encodings = Vec::new();
    for point in EIGHT_TORSION.iter() {
        encodings.push(point.compress().to_bytes());
    }
    for encoding in non_canonical_point_encodings()
        .into_iter()
        .take(LOW_ORDER_NON_CANONICAL_COUNT)
    {
        encodings.push(encoding);
    }

    assert_eq!(encodings.len(), LOW_ORDER_COUNT);
    for encoding in &encodings {
        let point = CompressedEdwardsY(*encoding)
            .decompress()
            .expect("a low-order encoding decodes");
        assert!(
            point.is_small_order(),
            "{} is not small order",
            hex(encoding)
        );
    }
    encodings
}

/// Every pair of low-order `A` and `R` with `s = 0`, which ZIP-215 accepts and the strict rules refuse
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
