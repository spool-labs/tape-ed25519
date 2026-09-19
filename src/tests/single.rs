//! The single-signature verifier on its own

use std::format;

use crate::constants;
use crate::edwards::CompressedEdwardsY;
use crate::error::Error;
use crate::signature::Signature;
use crate::tests::util::signed;
use crate::verifying_key::{verify, VerifyingKey};

/// A key from `seed` and its signature over `message`, both decoded
fn pair(seed: u8, message: &[u8]) -> (VerifyingKey, Signature) {
    let (encoded, signature) = signed(seed, message);
    (
        VerifyingKey::from_bytes(&encoded).expect("a key from the signer decodes"),
        signature,
    )
}

/// The first encoding that does not decompress, for the refusal tests
fn undecodable_point() -> [u8; 32] {
    for candidate in 0u16..=u16::MAX {
        let mut bytes = [0u8; 32];
        bytes[..2].copy_from_slice(&candidate.to_le_bytes());
        if CompressedEdwardsY(bytes).decompress().is_none() {
            return bytes;
        }
    }

    unreachable!("some encoding fails to decompress")
}

// a hundred honest signatures verify
#[test]
fn honest_signatures() {
    for seed in 0..100u8 {
        let message = format!("message number {seed}");
        let (key, signature) = pair(seed, message.as_bytes());

        let checked = verify(&key, &signature, message.as_bytes());

        assert!(checked.is_ok(), "honest signature {seed} refused");
    }
}

// a signature over one message is refused over another
#[test]
fn wrong_message() {
    let (key, signature) = pair(1, b"the original message");

    let checked = verify(&key, &signature, b"a different message");

    assert_eq!(checked, Err(Error::InvalidSignature));
}

// an s at or above the group order is refused
#[test]
fn non_canonical_scalar() {
    let (key, signature) = pair(4, b"non-canonical s");
    let mut bytes = signature.to_bytes();
    bytes[32..].copy_from_slice(&constants::BASEPOINT_ORDER.to_bytes());

    let checked = verify(&key, &Signature::from_bytes(&bytes), b"non-canonical s");

    assert_eq!(checked, Err(Error::InvalidSignature));
}

// an R that does not decompress is refused
#[test]
fn undecodable_r() {
    let (key, signature) = pair(5, b"undecodable r");
    let mut bytes = signature.to_bytes();
    bytes[..32].copy_from_slice(&undecodable_point());

    let checked = verify(&key, &Signature::from_bytes(&bytes), b"undecodable r");

    assert_eq!(checked, Err(Error::InvalidSignature));
}

// a key that does not decompress never becomes a key
#[test]
fn undecodable_key() {
    let checked = VerifyingKey::from_bytes(&undecodable_point());

    assert_eq!(checked, Err(Error::MalformedPublicKey));
}

// a slice of the wrong length is refused with the length it was given
#[test]
fn wrong_lengths() {
    assert_eq!(
        VerifyingKey::from_slice(&[0u8; 31]),
        Err(Error::WrongLength {
            given: 31,
            expected: 32
        })
    );
    assert_eq!(
        Signature::from_slice(&[0u8; 65]),
        Err(Error::WrongLength {
            given: 65,
            expected: 64
        })
    );
}

// the two halves survive a round trip through the encoded form
#[test]
fn signature_round_trip() {
    let (_, signature) = signed(9, b"round trip");

    let bytes = signature.to_bytes();

    assert_eq!(Signature::from_bytes(&bytes), signature);
    assert_eq!(&bytes[..32], &signature.r_bytes()[..]);
    assert_eq!(&bytes[32..], &signature.s_bytes()[..]);
}
