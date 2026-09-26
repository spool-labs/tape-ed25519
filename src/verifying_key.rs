//! A verifying key, decompressed once, and the single-signature check over it

use crate::backend::dispatch;
use crate::challenge::challenge;
use crate::edwards::{CompressedEdwardsY, EdwardsPoint};
use crate::error::Error;
use crate::scalar::Scalar;
use crate::signature::Signature;
use crate::small_order::is_small_order_encoding;
use crate::traits::{HEEADecomposition, IsIdentity};

/// Bytes in an encoded verifying key
pub const VERIFYING_KEY_LENGTH: usize = 32;

/// A decompressed verifying key, stored negated for the verification equation
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VerifyingKey {
    encoded: [u8; VERIFYING_KEY_LENGTH],
    minus_a: EdwardsPoint,
}

impl VerifyingKey {
    /// Decompress 32 bytes, non-canonical encodings included and small-order points refused
    pub fn from_bytes(bytes: &[u8; VERIFYING_KEY_LENGTH]) -> Result<Self, Error> {
        if is_small_order_encoding(bytes) {
            return Err(Error::SmallOrderPublicKey);
        }
        let point = CompressedEdwardsY(*bytes)
            .decompress()
            .ok_or(Error::MalformedPublicKey)?;

        Ok(Self {
            encoded: *bytes,
            minus_a: -point,
        })
    }

    /// Decompress a slice of unknown length
    pub fn from_slice(slice: &[u8]) -> Result<Self, Error> {
        let bytes: &[u8; VERIFYING_KEY_LENGTH] =
            slice.try_into().map_err(|_| Error::WrongLength {
                given: slice.len(),
                expected: VERIFYING_KEY_LENGTH,
            })?;
        Self::from_bytes(bytes)
    }

    /// The key bytes in their original encoding
    pub fn as_bytes(&self) -> &[u8; VERIFYING_KEY_LENGTH] {
        &self.encoded
    }
}

/// Verify `signature` over `message` under `key` with the ZIP-215 cofactored equation
pub fn verify(key: &VerifyingKey, signature: &Signature, message: &[u8]) -> Result<(), Error> {
    let k = challenge(signature.r_bytes(), &key.encoded, message);
    let (rho, tau, is_k_flipped) = k.heea_decompose();

    let s = Option::<Scalar>::from(Scalar::from_canonical_bytes(*signature.s_bytes()))
        .ok_or(Error::InvalidSignature)?;

    let minus_r = -CompressedEdwardsY(*signature.r_bytes())
        .decompress()
        .ok_or(Error::InvalidSignature)?;

    // The A term is -A when rho == tau*k and A when rho == -tau*k
    let a = match is_k_flipped {
        true => -key.minus_a,
        false => key.minus_a,
    };
    let result = dispatch::vartime_triple_base_mul_128_128_256_prechecked(
        &tau,
        &minus_r,
        &rho,
        &a,
        &(tau * s),
    );

    // An honest signature meets the exact equation and skips the cofactor doublings
    match result.is_identity_vartime() || result.mul_by_cofactor().is_identity() {
        true => Ok(()),
        false => Err(Error::InvalidSignature),
    }
}

/// Verify one batch item and refuse a key that does not decode
pub fn verify_encoded(
    encoded: &[u8; VERIFYING_KEY_LENGTH],
    signature: &Signature,
    message: &[u8],
) -> bool {
    match VerifyingKey::from_bytes(encoded) {
        Ok(key) => verify(&key, signature, message).is_ok(),
        Err(_) => false,
    }
}
