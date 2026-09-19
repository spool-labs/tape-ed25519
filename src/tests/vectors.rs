//! The RFC 8032 known-answer vectors, through the verifier

use crate::signature::Signature;
use crate::tests::util::hex;
use crate::verifying_key::{verify, VerifyingKey};

/// Decode a hexadecimal test vector
fn bytes(text: &str) -> alloc::vec::Vec<u8> {
    let digits: alloc::vec::Vec<u8> = text
        .bytes()
        .map(|c| (c as char).to_digit(16).expect("a hex digit") as u8)
        .collect();
    digits
        .chunks(2)
        .map(|pair| pair[0] << 4 | pair[1])
        .collect()
}

/// One vector: the key accepts its own signature over the message
fn accepts(key_text: &str, signature_text: &str, message_text: &str) {
    let key = VerifyingKey::from_slice(&bytes(key_text)).expect("the vector's key decodes");
    let signature =
        Signature::from_slice(&bytes(signature_text)).expect("the vector's signature decodes");
    let message = bytes(message_text);

    let checked = verify(&key, &signature, &message);

    assert!(
        checked.is_ok(),
        "RFC 8032 vector refused: {}",
        hex(&message)
    );
}

// the empty-message vector verifies
#[test]
fn empty_message() {
    accepts(
        "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a",
        "e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e065224901555fb8821590a33bacc61e39701cf9b46bd25bf5f0595bbe24655141438e7a100b",
        "",
    );
}

// the one-byte-message vector verifies
#[test]
fn one_byte() {
    accepts(
        "3d4017c3e843895a92b70aa74d1b7ebc9c982ccf2ec4968cc0cd55f12af4660c",
        "92a009a9f0d4cab8720e820b5f642540a2b27b5416503f8fb3762223ebdb69da085ac1e43e15996e458f3613d0f11d8c387b2eaeb4302aeeb00d291612bb0c00",
        "72",
    );
}

// the two-byte-message vector verifies
#[test]
fn two_bytes() {
    accepts(
        "fc51cd8e6218a1a38da47ed00230f0580816ed13ba3303ac5deb911548908025",
        "6291d657deec24024827e69c3abe01a30ce548a284743a445e3680d7db5ac3ac18ff9b538d16f290ae67f760984dc6594a7c15e9716ed28dc027beceea1ec40a",
        "af82",
    );
}
