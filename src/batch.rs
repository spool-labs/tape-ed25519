//! Many signatures in one call, one verdict per item

use alloc::vec::Vec;

use crate::lanes;
use crate::signature::Signature;
use crate::verifying_key::{verify_encoded, VERIFYING_KEY_LENGTH};

/// Verify every item and return one independent verdict per item in the caller's order
pub fn verify_batch(items: &[([u8; VERIFYING_KEY_LENGTH], Signature, &[u8])]) -> Vec<bool> {
    if lanes::is_lane_engine_available() {
        return lanes::verify_batch(items);
    }

    let mut verdicts = Vec::with_capacity(items.len());
    for (encoded, signature, message) in items {
        verdicts.push(verify_encoded(encoded, signature, message));
    }
    verdicts
}
