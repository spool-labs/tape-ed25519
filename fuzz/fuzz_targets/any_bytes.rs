//! Arbitrary keys, signatures and messages never panic, and every verifier agrees on them

#![no_main]

use libfuzzer_sys::fuzz_target;

use tape_ed25519::batch::verify_batch;
use tape_ed25519::signature::{Signature, SIGNATURE_LENGTH};
use tape_ed25519::verifying_key::{verify_encoded, VERIFYING_KEY_LENGTH};

/// Batches repeat the case up to this many times so the lane path fills
const LANE_WIDTH: usize = 8;

fuzz_target!(|data: &[u8]| {
    let Some((key, rest)) = data.split_first_chunk::<VERIFYING_KEY_LENGTH>() else {
        return;
    };
    let Some((signature, message)) = rest.split_first_chunk::<SIGNATURE_LENGTH>() else {
        return;
    };
    let signature = Signature::from_bytes(signature);

    let single = verify_encoded(key, &signature, message);
    let batch = verify_batch(&vec![(*key, signature, message); LANE_WIDTH]);
    assert!(batch.iter().all(|&verdict| verdict == single));
});
