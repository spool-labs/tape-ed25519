//! The 64 bytes of an Ed25519 signature, the R and s halves still encoded

use crate::error::Error;

/// Bytes in an encoded signature
pub const SIGNATURE_LENGTH: usize = 64;

/// Bytes in each half of a signature
pub const HALF_LENGTH: usize = SIGNATURE_LENGTH / 2;

/// An Ed25519 signature with both halves still encoded
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Signature {
    r: [u8; HALF_LENGTH],
    s: [u8; HALF_LENGTH],
}

impl Signature {
    /// Split 64 bytes into the two halves
    pub fn from_bytes(bytes: &[u8; SIGNATURE_LENGTH]) -> Self {
        let mut r = [0u8; HALF_LENGTH];
        let mut s = [0u8; HALF_LENGTH];
        r.copy_from_slice(&bytes[..HALF_LENGTH]);
        s.copy_from_slice(&bytes[HALF_LENGTH..]);
        Self { r, s }
    }

    /// Split a slice of unknown length into the two halves
    pub fn from_slice(slice: &[u8]) -> Result<Self, Error> {
        let bytes: &[u8; SIGNATURE_LENGTH] = slice.try_into().map_err(|_| Error::WrongLength {
            given: slice.len(),
            expected: SIGNATURE_LENGTH,
        })?;
        Ok(Self::from_bytes(bytes))
    }

    /// Encode the signature back into 64 bytes
    pub fn to_bytes(&self) -> [u8; SIGNATURE_LENGTH] {
        let mut bytes = [0u8; SIGNATURE_LENGTH];
        bytes[..HALF_LENGTH].copy_from_slice(&self.r);
        bytes[HALF_LENGTH..].copy_from_slice(&self.s);
        bytes
    }

    /// The encoded `R` point
    pub fn r_bytes(&self) -> &[u8; HALF_LENGTH] {
        &self.r
    }

    /// The encoded `s` scalar
    pub fn s_bytes(&self) -> &[u8; HALF_LENGTH] {
        &self.s
    }
}
