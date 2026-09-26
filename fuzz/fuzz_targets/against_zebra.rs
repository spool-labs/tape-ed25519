//! Damaged honest signatures get the verify_zebra verdict here, with small-order keys refused

#![no_main]

use ed25519_dalek::Signer;
use libfuzzer_sys::fuzz_target;

use tape_ed25519::batch::verify_batch;
use tape_ed25519::signature::{Signature, SIGNATURE_LENGTH};
use tape_ed25519::small_order::is_small_order_encoding;
use tape_ed25519::verifying_key::{verify_encoded, VERIFYING_KEY_LENGTH};

/// Bytes the input spends on the signing seed
const SEED_LENGTH: usize = 32;

/// Bytes the input spends on one damage instruction
const DAMAGE_LENGTH: usize = 3;

/// Bits in the key and signature together, the range a damage instruction flips within
const DAMAGED_BITS: usize = (VERIFYING_KEY_LENGTH + SIGNATURE_LENGTH) * 8;

/// Batches repeat the case up to this many times so the lane path fills
const LANE_WIDTH: usize = 8;

/// The zebra verdict the fork of curve25519-dalek gives
fn zebra(key: &[u8; VERIFYING_KEY_LENGTH], signature: &[u8; SIGNATURE_LENGTH], message: &[u8]) -> bool {
    let signature = solana_ed25519::ed_sigs::Signature::from_bytes(signature);
    solana_ed25519::ed_sigs::VerificationKey::try_from(*key)
        .and_then(|key| key.verify_zebra(&signature, message))
        .is_ok()
}

fuzz_target!(|data: &[u8]| {
    let Some((seed, rest)) = data.split_first_chunk::<SEED_LENGTH>() else {
        return;
    };
    let Some((&damage_count, rest)) = rest.split_first() else {
        return;
    };
    let damage_bytes = usize::from(damage_count % 4) * DAMAGE_LENGTH;
    if rest.len() < damage_bytes {
        return;
    }
    let (damage, message) = rest.split_at(damage_bytes);

    let signing_key = ed25519_dalek::SigningKey::from_bytes(seed);
    let mut encoded = [0u8; VERIFYING_KEY_LENGTH + SIGNATURE_LENGTH];
    encoded[..VERIFYING_KEY_LENGTH].copy_from_slice(&signing_key.verifying_key().to_bytes());
    encoded[VERIFYING_KEY_LENGTH..].copy_from_slice(&signing_key.sign(message).to_bytes());

    for step in damage.chunks_exact(DAMAGE_LENGTH) {
        let position = usize::from(u16::from_le_bytes([step[1], step[2]])) % DAMAGED_BITS;
        let is_flip = step[0] % 2 == 0;
        if is_flip {
            encoded[position / 8] ^= 1 << (position % 8);
        } else {
            encoded[position / 8] = step[0];
        }
    }

    let (key, signature) = encoded.split_at(VERIFYING_KEY_LENGTH);
    let key: [u8; VERIFYING_KEY_LENGTH] = key.try_into().expect("key width");
    let signature: [u8; SIGNATURE_LENGTH] = signature.try_into().expect("signature width");

    let expected = zebra(&key, &signature, message) && !is_small_order_encoding(&key);
    let parsed = Signature::from_bytes(&signature);
    assert_eq!(verify_encoded(&key, &parsed, message), expected);
    let batch = verify_batch(&vec![(key, parsed, message); LANE_WIDTH]);
    assert!(batch.iter().all(|&verdict| verdict == expected));
});
