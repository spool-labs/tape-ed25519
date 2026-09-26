//! The ZIP-215 edge cases for low-order points and non-canonical encodings under each rule

use alloc::vec::Vec;
use std::format;

use crate::batch::verify_batch;
use crate::challenge::challenge;
use crate::constants::ED25519_BASEPOINT_POINT;
use crate::edwards::{CompressedEdwardsY, EdwardsPoint};
use crate::lanes::Item;
use crate::scalar::Scalar;
use crate::signature::{Signature, HALF_LENGTH, SIGNATURE_LENGTH};
use crate::tests::util::{
    hex, low_order_encodings, small_order_cases, LOW_ORDER_COUNT, ZIP215_MESSAGE,
};
use crate::verifying_key::{verify, VerifyingKey, VERIFYING_KEY_LENGTH};

/// A forgery attempt tries this many messages, eight times the expected need at `ord(A) = 8`
const GRIND_ATTEMPTS: u32 = 64;

/// This crate's verdict on one case
fn zip215_verdict(
    key: &[u8; VERIFYING_KEY_LENGTH],
    signature: &[u8; SIGNATURE_LENGTH],
    message: &[u8],
) -> bool {
    match VerifyingKey::from_bytes(key) {
        Ok(key) => verify(&key, &Signature::from_bytes(signature), message).is_ok(),
        Err(_) => false,
    }
}

/// The pre-ZIP-215 verdict of `ed25519-dalek` on one case
fn strict_verdict(
    key: &[u8; VERIFYING_KEY_LENGTH],
    signature: &[u8; SIGNATURE_LENGTH],
    message: &[u8],
) -> bool {
    ed25519_dalek::VerifyingKey::from_bytes(key)
        .and_then(|key| {
            key.verify_strict(message, &ed25519_dalek::Signature::from_bytes(signature))
        })
        .is_ok()
}

// every low-order pair with s = 0 is accepted under ZIP-215 and refused under the strict rules
#[test]
fn low_order() {
    let cases = small_order_cases();
    assert_eq!(cases.len(), LOW_ORDER_COUNT * LOW_ORDER_COUNT);

    for case in &cases {
        assert_eq!(
            zip215_verdict(&case.key, &case.signature, ZIP215_MESSAGE),
            case.is_valid_zip215,
            "zip215 verdict for key={} signature={}",
            hex(&case.key),
            hex(&case.signature),
        );
        assert_eq!(
            strict_verdict(&case.key, &case.signature, ZIP215_MESSAGE),
            case.is_valid_strict,
            "strict verdict for key={} signature={}",
            hex(&case.key),
            hex(&case.signature),
        );
    }
}

// the batch verifier agrees with the single one on every small-order case
#[test]
fn batch_order() {
    let cases = small_order_cases();
    let items: Vec<Item<'_>> = cases
        .iter()
        .map(|case| {
            (
                case.key,
                Signature::from_bytes(&case.signature),
                ZIP215_MESSAGE,
            )
        })
        .collect();

    let verdicts = verify_batch(&items);

    for (index, case) in cases.iter().enumerate() {
        assert_eq!(
            verdicts[index],
            zip215_verdict(&case.key, &case.signature, ZIP215_MESSAGE),
            "batch and single disagree at {index}"
        );
    }
}

// a low-order key admits a satisfied equation, so the strict rules must test the order
#[test]
fn forgeries() {
    let s = Scalar::from(1u64);
    let r_bytes = (ED25519_BASEPOINT_POINT * s).compress().to_bytes();
    let mut signature = [0u8; SIGNATURE_LENGTH];
    signature[..HALF_LENGTH].copy_from_slice(&r_bytes);
    signature[HALF_LENGTH..].copy_from_slice(s.as_bytes());

    let mut forgeable = 0usize;
    for key_bytes in low_order_encodings() {
        let key = VerifyingKey::from_bytes(&key_bytes).expect("low-order keys decode");
        let mut is_found = false;

        for attempt in 0..GRIND_ATTEMPTS {
            let message = format!("pay attacker {attempt}").into_bytes();

            // Check the bare equation without the order tests
            let challenge_scalar = challenge(&r_bytes, &key_bytes, &message);
            let expected_r = EdwardsPoint::vartime_double_scalar_mul_basepoint(
                &challenge_scalar,
                &negated(&key),
                &s,
            )
            .compress();
            if expected_r.as_bytes() != &r_bytes {
                continue;
            }
            is_found = true;

            assert!(
                !strict_verdict(&key_bytes, &signature, &message),
                "verify_strict accepted a forgery for key={}",
                hex(&key_bytes),
            );
        }

        if is_found {
            forgeable += 1;
        }
    }

    assert_eq!(
        forgeable, LOW_ORDER_COUNT,
        "every low-order encoding admits a forgery"
    );
}

/// Recover the negated key point from the encoded key
fn negated(key: &VerifyingKey) -> EdwardsPoint {
    -CompressedEdwardsY(*key.as_bytes())
        .decompress()
        .expect("a decoded key decompresses")
}
