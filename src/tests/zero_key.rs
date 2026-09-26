//! The all-zero key of the system program address and every other small-order key are refused

use crate::batch::verify_batch;
use crate::edwards::CompressedEdwardsY;
use crate::error::Error;
use crate::signature::{Signature, SIGNATURE_LENGTH};
use crate::small_order::is_small_order_encoding;
use crate::tests::util::{low_order_encodings, signed, small_order_cases};
use crate::verifying_key::{verify_encoded, VerifyingKey, VERIFYING_KEY_LENGTH};

/// The system program address 11111111111111111111111111111111 decodes to these bytes
const SYSTEM_PROGRAM: [u8; VERIFYING_KEY_LENGTH] = [0; VERIFYING_KEY_LENGTH];

/// The forged message
const MESSAGE: &[u8] = b"move every lamport out";

/// Random encodings the table is checked against
const RANDOM_ENCODINGS: usize = 100_000;

/// Fill `bytes` from a xorshift state so a failing encoding replays
fn fill(state: &mut u64, bytes: &mut [u8]) {
    for byte in bytes.iter_mut() {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *byte = *state as u8;
    }
}

/// A signature with R at the given encoding and s zero, which ZIP-215 accepts under a small-order key
fn forged(r: &[u8; VERIFYING_KEY_LENGTH]) -> Signature {
    let mut bytes = [0u8; SIGNATURE_LENGTH];
    bytes[..VERIFYING_KEY_LENGTH].copy_from_slice(r);
    Signature::from_bytes(&bytes)
}

// the system program key does not decode as a verifying key
#[test]
fn zero_key_refused() {
    assert_eq!(
        VerifyingKey::from_bytes(&SYSTEM_PROGRAM),
        Err(Error::SmallOrderPublicKey)
    );
}

// no forged signature verifies for the system program key on any path
#[test]
fn zero_key_forgeries() {
    for r in low_order_encodings() {
        let signature = forged(&r);
        assert!(!verify_encoded(&SYSTEM_PROGRAM, &signature, MESSAGE));
        assert_eq!(
            verify_batch(&[(SYSTEM_PROGRAM, signature, MESSAGE)]),
            [false]
        );
    }
    let zeros = Signature::from_bytes(&[0; SIGNATURE_LENGTH]);
    assert!(!verify_encoded(&SYSTEM_PROGRAM, &zeros, MESSAGE));
}

// a full batch of zero-key forgeries mixed with honest items keeps every honest verdict
#[test]
fn zero_key_batch() {
    let mut items = std::vec::Vec::new();
    for seed in 0..16u8 {
        let (key, signature) = signed(seed, MESSAGE);
        items.push((key, signature, MESSAGE));
        items.push((
            SYSTEM_PROGRAM,
            forged(&low_order_encodings()[seed as usize % 14]),
            MESSAGE,
        ));
    }
    let verdicts = verify_batch(&items);
    for (index, verdict) in verdicts.iter().enumerate() {
        assert_eq!(*verdict, index % 2 == 0, "item {index}");
    }
}

// every small-order case is refused on both paths
#[test]
fn small_order_refused() {
    for case in small_order_cases() {
        let signature = Signature::from_bytes(&case.signature);
        assert!(!verify_encoded(&case.key, &signature, MESSAGE));
        assert!(!verify_encoded(&case.key, &signature, b"Zcash"));
        assert_eq!(
            verify_batch(&[(case.key, signature, b"Zcash".as_slice())]),
            [false]
        );
    }
}

// the table flags exactly the encodings that decode to a small-order point
#[test]
fn table_matches_decoding() {
    let mut encodings = low_order_encodings();
    let mut state = 0x5eed_u64;
    for _ in 0..RANDOM_ENCODINGS {
        let mut bytes = [0u8; VERIFYING_KEY_LENGTH];
        fill(&mut state, &mut bytes);
        encodings.push(bytes);
    }
    for low in low_order_encodings() {
        let mut flipped = low;
        flipped[VERIFYING_KEY_LENGTH - 1] ^= 0x80;
        encodings.push(flipped);
    }
    for bytes in encodings {
        let decodes_small = CompressedEdwardsY(bytes)
            .decompress()
            .is_some_and(|point| point.is_small_order());
        let is_flagged = is_small_order_encoding(&bytes);
        assert_eq!(decodes_small, is_flagged, "{bytes:?}");
    }
}
