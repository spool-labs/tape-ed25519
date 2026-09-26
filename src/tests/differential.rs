//! Ten thousand random signatures and the ZIP-215 edge vectors, against two outside verifiers
//!
//! The oracle is `solana-ed25519`'s `verify_zebra`, the verifier tapesvm called before this crate
//! existed: `verify` and `verify_batch` must agree with it on every input. `verify_strict` from
//! `ed25519-dalek` applies the pre-ZIP-215 rules and refuses a low-order or non-canonical point
//! that ZIP-215 accepts, so the test asserts where the two are the same and where they differ.

use alloc::vec::Vec;

use crate::batch::verify_batch;
use crate::lanes;
use crate::signature::Signature;
use crate::tests::util::{
    hex, low_order_encodings, non_canonical_point_encodings, signed, small_order_cases,
    ZIP215_MESSAGE,
};
use crate::verifying_key::verify_encoded;

/// Random cases the harness draws
const ROUNDS: usize = 10_000;

/// How a case was built, which is what a failure prints
#[derive(Clone, Copy, Debug)]
enum Shape {
    /// A signature over the message it was made for
    Honest,

    /// One bit of `s` turned over
    BrokenScalar,

    /// One bit of `R` turned over
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
    /// The shape for one draw, which spreads the seven kinds evenly
    fn of(draw: usize) -> Self {
        match draw % 7 {
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

/// One case and the message it covers
struct Case {
    key: [u8; 32],
    signature: [u8; 64],
    message: Vec<u8>,
    shape: Shape,
}

/// A deterministic generator, so a failing round replays from the seed
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, bound: usize) -> usize {
        (self.next() % bound as u64) as usize
    }
}

/// What the crates.io verifier tapesvm used before says about one case
fn zebra_verdict(case: &Case) -> bool {
    let signature = solana_ed25519::ed_sigs::Signature::from_bytes(&case.signature);
    solana_ed25519::ed_sigs::VerificationKey::try_from(case.key)
        .and_then(|key| key.verify_zebra(&signature, &case.message))
        .is_ok()
}

/// What `ed25519-dalek`'s pre-ZIP-215 rules say about one case
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

/// Draw one case of the shape `draw` calls for
fn draw_case(rng: &mut Rng, draw: usize) -> Case {
    let shape = Shape::of(draw);
    let length = match draw % 5 {
        0 => 0,
        1 => 1,
        2 => 32,
        3 => 1232,
        4.. => rng.below(300),
    };
    let message: Vec<u8> = (0..length).map(|_| (rng.next() & 0xff) as u8).collect();
    let seed = (rng.next() & 0xff) as u8;
    let (mut key, signature) = signed(seed, &message);
    let mut bytes = signature.to_bytes();

    match shape {
        Shape::Honest => {}
        Shape::BrokenScalar => {
            let bit = rng.below(8);
            bytes[32 + rng.below(31)] ^= 1 << bit;
        }
        Shape::BrokenPoint => {
            let bit = rng.below(8);
            bytes[rng.below(32)] ^= 1 << bit;
        }
        Shape::WrongKey => key = signed(seed.wrapping_add(1), &message).0,
        Shape::NonCanonicalScalar => {
            bytes[32..].copy_from_slice(&[0xff; 32]);
            bytes[63] &= 0x1f;
        }
        Shape::NonCanonicalKey => {
            let encodings = non_canonical_point_encodings();
            key = encodings[rng.below(encodings.len())];
        }
        Shape::LowOrder => {
            let encodings = low_order_encodings();
            key = encodings[rng.below(encodings.len())];
            bytes[..32].copy_from_slice(&encodings[rng.below(encodings.len())]);
            bytes[32..].copy_from_slice(&[0u8; 32]);
        }
    }

    Case {
        key,
        signature: bytes,
        message,
        shape,
    }
}

/// The ZIP-215 edge vectors as cases, with the low-order encodings under an ordinary message
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
        let (_, signature) = signed(7, ZIP215_MESSAGE);
        cases.push(Case {
            key,
            signature: signature.to_bytes(),
            message: ZIP215_MESSAGE.to_vec(),
            shape: Shape::LowOrder,
        });
    }
    cases
}

// verify, verify_batch and the lane pipeline all agree with verify_zebra on every case
#[test]
fn matches_zebra() {
    let mut rng = Rng(0x2026_0918_1200_0001);
    let mut cases = edge_cases();
    for draw in 0..ROUNDS {
        cases.push(draw_case(&mut rng, draw));
    }

    let items: Vec<([u8; 32], Signature, &[u8])> = cases
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
        let expected = zebra_verdict(case);
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

    // Both verdicts have to be present, or the run proves nothing.
    assert!(accepted > 0 && accepted < cases.len());
}

// verify_strict differs from ZIP-215 only where a point is low order or encoded non-canonically
#[test]
fn strict_differs_where_documented() {
    let mut rng = Rng(0x2026_0918_1200_0002);
    let mut cases = edge_cases();
    for draw in 0..ROUNDS {
        cases.push(draw_case(&mut rng, draw));
    }

    let mut divergences = 0usize;
    for case in &cases {
        let ours = zebra_verdict(case);
        let strict = strict_verdict(case);
        if ours == strict {
            continue;
        }

        divergences += 1;
        assert!(
            ours && !strict,
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

    // The low-order vectors alone guarantee divergences, so none would mean the oracle is idle.
    assert!(divergences > 0);
}
