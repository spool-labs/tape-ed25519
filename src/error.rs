//! What a verification call says when it does not accept

use thiserror::Error;

/// Why a key, a signature or a check did not stand
#[derive(Clone, Copy, Debug, Eq, Error, PartialEq)]
pub enum Error {
    /// The 32 bytes of the key do not decode to a point on the curve
    #[error("malformed public key")]
    MalformedPublicKey,

    /// The signature does not verify over this message under this key
    #[error("invalid signature")]
    InvalidSignature,

    /// A slice of the wrong length was handed to a decoder
    #[error("slice is {given} bytes, expected {expected}")]
    WrongLength {
        /// Bytes the caller handed over
        given: usize,

        /// Bytes the decoder wanted
        expected: usize,
    },
}
