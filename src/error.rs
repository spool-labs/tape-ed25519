//! What a verification call says when it does not accept

use thiserror::Error;

/// Why a key, a signature or a check failed
#[derive(Clone, Copy, Debug, Eq, Error, PartialEq)]
pub enum Error {
    /// The 32 bytes of the key do not decode to a point on the curve
    #[error("malformed public key")]
    MalformedPublicKey,

    /// The signature does not verify over this message under this key
    #[error("invalid signature")]
    InvalidSignature,

    /// A decoder received a slice of the wrong length
    #[error("slice is {given} bytes, expected {expected}")]
    WrongLength {
        /// Length of the slice the caller passed
        given: usize,

        /// Length the decoder expects
        expected: usize,
    },
}
