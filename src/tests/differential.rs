//! Random signatures and the ZIP-215 edge vectors checked against two outside verifiers

use alloc::vec::Vec;

use crate::batch::verify_batch;
use crate::lanes::{self, Item};
use crate::signature::{Signature, HALF_LENGTH, SIGNATURE_LENGTH};
use crate::small_order::is_small_order_encoding;
use crate::tests::util::{
    hex, low_order_encodings, non_canonical_point_encodings, signed, small_order_cases,
    ZIP215_MESSAGE,
};
use crate::verifying_key::{verify_encoded, VERIFYING_KEY_LENGTH};

/// Each test draws this many random cases
const ROUNDS: usize = 10_000;

/// Number of `Shape` variants
const SHAPE_COUNT: usize = 7;

/// Message lengths cycle through this many kinds
const LENGTH_KINDS: usize = 5;

/// Longest random message length
const RANDOM_LENGTH_BOUND: usize = 300;

/// Top byte of an `s` above the group order and below 2^253
const NON_CANONICAL_S_TOP: u8 = 0x1f;

/// Seed of the honest signature over the ZIP-215 message
const EDGE_SEED: u8 = 7;

/// How a case was built, for the failure message
#[derive(Clone, Copy, Debug)]
enum Shape {
    /// A signature over the message it was made for
    Honest,

    /// One flipped bit of `s`
    BrokenScalar,

    /// One flipped bit of `R`
    BrokenPoint,

    /// An honest signature under somebody else's key
    WrongKey,

    /// An `s` at or above the group order
    NonCanonicalScalar,

    /// A key that encodes a point non-canonically
    NonCanonicalKey,

    /// A key and an `R` of order dividing the cofactor, with `s` zero
    LowOrder,
}

impl Shape {
    /// Spread the shapes evenly over the draws
    fn of(draw: usize) -> Self {
        match draw % SHAPE_COUNT {
            0 => Shape::Honest,
            1 => Shape::BrokenScalar,
            2 => Shape::BrokenPoint,
            3 => Shape::WrongKey,
            4 => Shape::NonCanonicalScalar,
            5 => Shape::NonCanonicalKey,
            6.. => Shape::LowOrder,
        }
    }
}

/// One case and its signed message
struct Case {
    key: [u8; VERIFYING_KEY_LENGTH],
    signature: [u8; SIGNATURE_LENGTH],
    message: Vec<u8>,
    shape: Shape,
}

/// A deterministic xorshift generator that replays a failing round from its seed
struct Generator(u64);

impl Generator {
    /// Advance the state and return it
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// Draw a value below `bound`
    fn below(&mut self, bound: usize) -> usize {
        (self.next() % bound as u64) as usize
    }
}

/// The crates.io `verify_zebra` verdict on one case
fn zebra_verdict(case: &Case) -> bool {
    let signature = solana_ed25519::ed_sigs::Signature::from_bytes(&case.signature);
    solana_ed25519::ed_sigs::VerificationKey::try_from(case.key)
        .and_then(|key| key.verify_zebra(&signature, &case.message))
        .is_ok()
}

/// The pre-ZIP-215 verdict of `ed25519-dalek` on one case
fn strict_verdict(case: &Case) -> bool {
    ed25519_dalek::VerifyingKey::from_bytes(&case.key)
        .and_then(|key| {
            key.verify_strict(
                &case.message,
                &ed25519_dalek::Signature::from_bytes(&case.signature),
            )
        })
        .is_ok()
}

/// Draw one case of the shape for `draw`
fn draw_case(generator: &mut Generator, draw: usize) -> Case {
    let shape = Shape::of(draw);
    let length = match draw % LENGTH_KINDS {
        0 => 0,
        1 => 1,
        2 => 32,
        3 => 1232,
        4.. => generator.below(RANDOM_LENGTH_BOUND),
    };
    let message: Vec<u8> = (0..length).map(|_| generator.next() as u8).collect();
    let seed = generator.next() as u8;
    let (mut key, signature) = signed(seed, &message);
    let mut bytes = signature.to_bytes();

    match shape {
        Shape::Honest => {}
        Shape::BrokenScalar => {
            let bit = generator.below(u8::BITS as usize);
            bytes[HALF_LENGTH + generator.below(HALF_LENGTH - 1)] ^= 1 << bit;
        }
        Shape::BrokenPoint => {
            let bit = generator.below(u8::BITS as usize);
            bytes[generator.below(HALF_LENGTH)] ^= 1 << bit;
        }
        Shape::WrongKey => key = signed(seed.wrapping_add(1), &message).0,
        Shape::NonCanonicalScalar => {
            bytes[HALF_LENGTH..].copy_from_slice(&[0xff; HALF_LENGTH]);
            bytes[SIGNATURE_LENGTH - 1] &= NON_CANONICAL_S_TOP;
        }
        Shape::NonCanonicalKey => {
            let encodings = non_canonical_point_encodings();
            key = encodings[generator.below(encodings.len())];
        }
        Shape::LowOrder => {
            let encodings = low_order_encodings();
            key = encodings[generator.below(encodings.len())];
            bytes[..HALF_LENGTH].copy_from_slice(&encodings[generator.below(encodings.len())]);
            bytes[HALF_LENGTH..].copy_from_slice(&[0u8; HALF_LENGTH]);
        }
    }

    Case {
        key,
        signature: bytes,
        message,
        shape,
    }
}

/// Build cases from the ZIP-215 edge vectors and each low-order key under an honest signature
fn edge_cases() -> Vec<Case> {
    let mut cases = Vec::new();
    for case in small_order_cases() {
        cases.push(Case {
            key: case.key,
            signature: case.signature,
            message: ZIP215_MESSAGE.to_vec(),
            shape: Shape::LowOrder,
        });
    }
    for key in low_order_encodings() {
        let (_, signature) = signed(EDGE_SEED, ZIP215_MESSAGE);
        cases.push(Case {
            key,
            signature: signature.to_bytes(),
            message: ZIP215_MESSAGE.to_vec(),
            shape: Shape::LowOrder,
        });
    }
    cases
}

// verify, verify_batch and the lane pipeline agree with verify_zebra except on small-order keys
#[test]
fn matches_zebra() {
    let mut generator = Generator(0x2026_0918_1200_0001);
    let mut cases = edge_cases();
    for draw in 0..ROUNDS {
        cases.push(draw_case(&mut generator, draw));
    }

    let items: Vec<Item<'_>> = cases
        .iter()
        .map(|case| {
            (
                case.key,
                Signature::from_bytes(&case.signature),
                case.message.as_slice(),
            )
        })
        .collect();
    let batched = verify_batch(&items);
    let laned = lanes::verify_batch(&items);

    let mut accepted = 0usize;
    for (index, case) in cases.iter().enumerate() {
        let expected = zebra_verdict(case) && !is_small_order_encoding(&case.key);
        let single = verify_encoded(&case.key, &items[index].1, &case.message);

        assert_eq!(
            single,
            expected,
            "verify disagrees with verify_zebra on a {:?} case: key={} signature={}",
            case.shape,
            hex(&case.key),
            hex(&case.signature),
        );
        assert_eq!(
            batched[index], expected,
            "verify_batch disagrees at {index}"
        );
        assert_eq!(
            laned[index], expected,
            "the lane pipeline disagrees at {index}"
        );
        accepted += usize::from(expected);
    }

    // The run must see both verdicts to prove anything
    assert!(accepted > 0 && accepted < cases.len());
}

// verify_strict differs from ZIP-215 only where a point is low order or encoded non-canonically
#[test]
fn strict_differs() {
    let mut generator = Generator(0x2026_0918_1200_0002);
    let mut cases = edge_cases();
    for draw in 0..ROUNDS {
        cases.push(draw_case(&mut generator, draw));
    }

    let mut divergences = 0usize;
    for case in &cases {
        let zip215 = zebra_verdict(case);
        let strict = strict_verdict(case);
        if zip215 == strict {
            continue;
        }

        divergences += 1;
        assert!(
            zip215 && !strict,
            "verify_strict accepted what ZIP-215 refused: key={} signature={}",
            hex(&case.key),
            hex(&case.signature),
        );
        assert!(
            matches!(case.shape, Shape::LowOrder | Shape::NonCanonicalKey),
            "unexpected divergence on a {:?} case: key={} signature={}",
            case.shape,
            hex(&case.key),
            hex(&case.signature),
        );
    }

    // The low-order vectors always diverge, so zero divergences means an idle oracle
    assert!(divergences > 0);
}
