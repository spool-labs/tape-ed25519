//! Ed25519 verification under the ZIP-215 rules, one signature at a time or eight lanes wide

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
