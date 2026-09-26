//! A verifying key, decompressed once, and the single-signature check over it

use crate::backend::dispatch;
use crate::challenge::challenge;
use crate::edwards::{CompressedEdwardsY, EdwardsPoint};
use crate::error::Error;
use crate::scalar::Scalar;
use crate::signature::Signature;
use crate::traits::{HEEADecomposition, IsIdentity};

/// Bytes in an encoded verifying key
pub const VERIFYING_KEY_LENGTH: usize = 32;

/// A key that decompressed to a point, held as the `-A` the verification equation wants
///
/// A non-canonical encoding decompresses and is accepted, which is what ZIP-215 asks for, and the
/// bytes are kept as they arrived because the challenge hash covers them.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VerifyingKey {
    encoded: [u8; VERIFYING_KEY_LENGTH],
    minus_a: EdwardsPoint,
}

impl VerifyingKey {
    /// Decompress 32 bytes, non-canonical encodings included
    pub fn from_bytes(bytes: &[u8; VERIFYING_KEY_LENGTH]) -> Result<Self, Error> {
        let point = CompressedEdwardsY(*bytes)
            .decompress()
            .ok_or(Error::MalformedPublicKey)?;

        Ok(Self {
            encoded: *bytes,
            minus_a: -point,
        })
    }

    /// The same from a slice of unknown length
    pub fn from_slice(slice: &[u8]) -> Result<Self, Error> {
        let bytes: &[u8; VERIFYING_KEY_LENGTH] =
            slice.try_into().map_err(|_| Error::WrongLength {
                given: slice.len(),
                expected: VERIFYING_KEY_LENGTH,
            })?;
        Self::from_bytes(bytes)
    }

    /// The key as it arrived
    pub fn as_bytes(&self) -> &[u8; VERIFYING_KEY_LENGTH] {
        &self.encoded
    }
}

/// Verify `signature` over `message` under `key`, with the ZIP-215 rules
///
/// `A` and `R` must decode to points on the twisted Edwards curve, and a non-canonical encoding of
/// either is accepted. `s` must be less than the group order. The cofactored equation
/// `[8][s]B = [8]R + [8][k]A` must hold, and the RFC 8032 alternative `[s]B = R + [k]A` is not
/// enough on its own.
///
/// The equation is checked in the HEEA-transformed form of "Accelerating EdDSA Signature
/// Verification with Faster Scalar Size Halving" (TCHES 2025): the decomposition gives half-size
/// `rho` and `tau` with `rho == tau*k (mod l)` up to a sign, so a four-variable multiplication with
/// half-size scalars decides the same accept or reject.
pub fn verify(key: &VerifyingKey, signature: &Signature, message: &[u8]) -> Result<(), Error> {
    let k = challenge(signature.r_bytes(), &key.encoded, message);
    let (rho, tau, is_k_flipped) = k.heea_decompose();

    let s = Option::<Scalar>::from(Scalar::from_canonical_bytes(*signature.s_bytes()))
        .ok_or(Error::InvalidSignature)?;

    let minus_r = -CompressedEdwardsY(*signature.r_bytes())
        .decompress()
        .ok_or(Error::InvalidSignature)?;

    // The A term is -A when rho == tau*k, and A when rho == -tau*k.
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

    // An honest signature satisfies the equation exactly, so testing that first skips the cofactor
    // doublings. They stay for the ZIP-215 signatures that hold torsion.
    match result.is_identity_vartime() || result.mul_by_cofactor().is_identity() {
        true => Ok(()),
        false => Err(Error::InvalidSignature),
    }
}

/// Verify one item of a batch, a key that does not decode refused like a signature that does not
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
