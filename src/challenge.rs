//! The challenge scalar `k = H(R || A || M) mod l`, hashed the same way by both verifiers

use sha2::{digest::Update, Digest, Sha512};

use crate::scalar::Scalar;
use crate::signature::HALF_LENGTH;
use crate::verifying_key::VERIFYING_KEY_LENGTH;

/// Length of a SHA-512 digest
const DIGEST_LENGTH: usize = 64;

/// Hash `R || A || message` and reduce the wide digest to a scalar
pub fn challenge(
    r_bytes: &[u8; HALF_LENGTH],
    a_bytes: &[u8; VERIFYING_KEY_LENGTH],
    message: &[u8],
) -> Scalar {
    let digest = Sha512::default()
        .chain(&r_bytes[..])
        .chain(&a_bytes[..])
        .chain(message)
        .finalize();

    let mut wide = [0u8; DIGEST_LENGTH];
    wide.copy_from_slice(digest.as_slice());
    Scalar::from_bytes_mod_order_wide(&wide)
}
