//! Many signatures in one call, one verdict per item

use alloc::vec::Vec;

use crate::lanes;
use crate::signature::Signature;
use crate::verifying_key::{verify_encoded, VERIFYING_KEY_LENGTH};

/// Verify every item, handing back one verdict each in the caller's order
///
/// Verdicts are independent: a refused signature never costs another item its accept. On a host
/// with AVX-512 IFMA the items go through the eight-lane verifier, which is where the batch is
/// cheaper than the same signatures one at a time. Everywhere else each item goes through the
/// single verifier, which is what that host is fastest at.
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
