//! The lane verifier checked against the single verifier at every group and tail size

use alloc::vec;
use alloc::vec::Vec;

use crate::batch::verify_batch;
use crate::lanes::{self, Item};
use crate::signature::{Signature, HALF_LENGTH, SIGNATURE_LENGTH};
use crate::tests::util::{signed, small_order_cases, ZIP215_MESSAGE};
use crate::verifying_key::verify_encoded;

/// Message lengths across every SHA-512 padding boundary and the sizes the grouping sorts on
const RAGGED_LENGTHS: [usize; 11] = [0, 1, 63, 64, 65, 127, 128, 129, 200, 1232, 4096];

/// Multiplier of the linear congruential generator
const GENERATOR_MULTIPLIER: u32 = 1_664_525;

/// Increment of the linear congruential generator
const GENERATOR_INCREMENT: u32 = 1_013_904_223;

/// The generator drops this many low bits of its state
const GENERATOR_SHIFT: u32 = 8;

/// Each signing seed is the item index mixed with this mask
const SEED_MASK: u8 = 0xC3;

/// Every item at a multiple of this period stays honest
const BREAK_PERIOD: usize = 3;

/// Top byte of an `s` above the group order and below 2^253
const NON_CANONICAL_S_TOP: u8 = 0x1f;

/// Replay a ragged arrival order from a deterministic generator
fn next(state: &mut u32) -> usize {
    *state = state
        .wrapping_mul(GENERATOR_MULTIPLIER)
        .wrapping_add(GENERATOR_INCREMENT);
    (*state >> GENERATOR_SHIFT) as usize
}

/// Sign every message under its own key
fn sign_all(messages: &[Vec<u8>]) -> Vec<Item<'_>> {
    let mut items = Vec::with_capacity(messages.len());
    for (index, message) in messages.iter().enumerate() {
        let (key, signature) = signed(index as u8 ^ SEED_MASK, message);
        items.push((key, signature, message.as_slice()));
    }
    items
}

/// Replace `s` with a value at or above the group order
fn break_scalar(bytes: &mut [u8; SIGNATURE_LENGTH]) {
    bytes[HALF_LENGTH..].copy_from_slice(&[0xff; HALF_LENGTH]);
    bytes[SIGNATURE_LENGTH - 1] &= NON_CANONICAL_S_TOP;
}

/// Assert the lane verdicts equal the single verifier's item for item
fn assert_matches_single(items: &[Item<'_>]) {
    let verdicts = lanes::verify_batch(items);
    assert_eq!(verdicts.len(), items.len());

    for (index, (encoded, signature, message)) in items.iter().enumerate() {
        assert_eq!(
            verdicts[index],
            verify_encoded(encoded, signature, message),
            "verdict mismatch at index {index}"
        );
    }
}

// honest signatures verify at every group and tail size
#[test]
fn honest_signatures() {
    let messages: Vec<Vec<u8>> = (0..21u8)
        .map(|index| vec![index ^ 0x5A; 3 + 11 * index as usize])
        .collect();
    let items = sign_all(&messages);

    let verdicts = lanes::verify_batch(&items);

    assert!(
        verdicts.iter().all(|&is_accepted| is_accepted),
        "honest signature refused"
    );
    for count in [0, 1, 3, 4, 7, 8, 9, 11, 12, 15, 16, 20, 21] {
        assert_matches_single(&items[..count]);
    }
}

// the order the lengths arrive in does not change a verdict
#[test]
fn ragged_lengths() {
    let mut state = 0x1234_5678u32;

    for round in 0..8usize {
        let count = 4 + round * 3;
        let messages: Vec<Vec<u8>> = (0..count)
            .map(|index| {
                let length = match round % 2 == 0 {
                    true => RAGGED_LENGTHS[next(&mut state) % RAGGED_LENGTHS.len()],
                    false => next(&mut state) % 4200,
                };
                vec![(index as u8) ^ 0x3C; length]
            })
            .collect();
        let items = sign_all(&messages);

        let verdicts = lanes::verify_batch(&items);

        assert!(
            verdicts.iter().all(|&is_accepted| is_accepted),
            "refused in round {round}"
        );
        assert_matches_single(&items);
    }
}

// every verdict returns to the item it belongs to, at anti-sorted lengths
#[test]
fn verdict_order() {
    let lengths = [
        4096usize, 0, 1232, 1, 200, 63, 129, 64, 128, 65, 127, 1232, 0, 4096, 200,
    ];
    let messages: Vec<Vec<u8>> = lengths
        .iter()
        .enumerate()
        .map(|(index, &length)| vec![(index as u8) ^ 0x9E; length])
        .collect();
    let mut items = sign_all(&messages);

    // Break every third item in a different way each time
    for (index, item) in items.iter_mut().enumerate() {
        let mut bytes = item.1.to_bytes();
        match index % BREAK_PERIOD {
            0 => continue,
            1 => bytes[index % SIGNATURE_LENGTH] ^= 0x08,
            2.. => break_scalar(&mut bytes),
        }
        item.1 = Signature::from_bytes(&bytes);
    }

    let verdicts = lanes::verify_batch(&items);

    for (index, &is_accepted) in verdicts.iter().enumerate() {
        assert_eq!(
            is_accepted,
            index % BREAK_PERIOD == 0,
            "verdict landed on the wrong item"
        );
    }
    assert_matches_single(&items);
}

// batches either side of the point where the group ordering leaves the stack
#[test]
fn stack_limit() {
    let mut state = 0xFEED_0BADu32;

    for count in [127usize, 128, 129, 200] {
        let messages: Vec<Vec<u8>> = (0..count)
            .map(|index| vec![index as u8; next(&mut state) % 400])
            .collect();
        let mut items = sign_all(&messages);

        for index in (3..count).step_by(17) {
            let mut bytes = items[index].1.to_bytes();
            bytes[index % SIGNATURE_LENGTH] ^= 0x20;
            items[index].1 = Signature::from_bytes(&bytes);
        }

        let verdicts = lanes::verify_batch(&items);

        for (index, &is_accepted) in verdicts.iter().enumerate() {
            let is_broken = index >= 3 && (index - 3) % 17 == 0;
            assert_eq!(is_accepted, !is_broken, "count={count}, index {index}");
        }
        assert_matches_single(&items);
    }
}

// a refused item of any kind costs its own lane and no other
#[test]
fn independent_lanes() {
    let message: &[u8] = b"wide lanes";
    let mut items: Vec<Item<'_>> = (0..8u8)
        .map(|index| {
            let (key, signature) = signed(40 + index, message);
            (key, signature, message)
        })
        .collect();

    let mut corrupted_r = items[1].1.to_bytes();
    corrupted_r[0] ^= 0x40;
    items[1].1 = Signature::from_bytes(&corrupted_r);

    let mut non_canonical_s = items[2].1.to_bytes();
    break_scalar(&mut non_canonical_s);
    items[2].1 = Signature::from_bytes(&non_canonical_s);

    items[3].0 = [0x05; 32];

    items[4].1 = signed(99, message).1;

    let verdicts = lanes::verify_batch(&items);

    assert_eq!(
        verdicts,
        vec![true, false, false, false, false, true, true, true]
    );
    assert_matches_single(&items);
}

// the lane verdicts match the single verifier on every ZIP-215 small-order case
#[test]
fn small_order() {
    let cases = small_order_cases();
    let items: Vec<Item<'_>> = cases
        .iter()
        .map(|case| {
            (
                case.key,
                Signature::from_bytes(&case.signature),
                ZIP215_MESSAGE,
            )
        })
        .collect();

    assert_matches_single(&items);
}

// the public entry point agrees with the lane pipeline on any host
#[test]
fn dispatch() {
    let message: &[u8] = b"dispatch";
    let items: Vec<Item<'_>> = (0..9u8)
        .map(|index| {
            let (key, signature) = signed(70 + index, message);
            (key, signature, message)
        })
        .collect();

    assert_eq!(verify_batch(&items), lanes::verify_batch(&items));
}
