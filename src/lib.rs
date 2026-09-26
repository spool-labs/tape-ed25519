//! Ed25519 verification under the ZIP-215 rules, one signature at a time or eight lanes wide
//!
//! The single verifier and the batch verifier decide the same way on every input, so a signature
//! accepted by one is accepted by the other. The batch verifier picks its path per host: eight
//! lanes where AVX-512 IFMA is available, the single verifier everywhere else.

#![no_std]
#![warn(missing_docs)]

extern crate alloc;

#[cfg(test)]
#[macro_use]
extern crate std;

#[macro_use]
mod macros;

mod affine;
mod backend;
mod challenge;
mod constants;
mod edwards;
mod field;
mod heea;
mod lanes;
mod scalar;
mod traits;
mod window;

#[cfg(test)]
mod tests;

pub mod batch;
pub mod error;
pub mod signature;
pub mod verifying_key;
