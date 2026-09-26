//! Eight signatures at a time, one per SIMD lane, each with its own verdict

use alloc::vec::Vec;

use crate::backend::lanes::digits::GroupDigits;
use crate::backend::lanes::digits::{joint_digits, negate, signed_digits, HEEA_DIGITS};
use crate::backend::lanes::edwards_x8::{
    decompress_x8, CompletedPointX8, ExtendedPointX8, LookupTableX8, ProjectivePointX8,
};
use crate::backend::lanes::field_x8::{LaneMask, LANES};
use crate::backend::lanes::joint_data::JOINT_TABLE_B;
use crate::challenge::challenge;
use crate::scalar::Scalar;
use crate::signature::{Signature, HALF_LENGTH};
use crate::traits::HEEADecomposition;
use crate::verifying_key::{verify_encoded, VERIFYING_KEY_LENGTH};

#[cfg(curve25519_backend = "simd")]
use crate::backend::lanes::ifma_x8;
#[cfg(curve25519_backend = "simd")]
use crate::backend::lanes::sha512_x8::{challenge_digests_x16, challenge_digests_x8};

/// One batch item: the encoded key, the signature and the signed message
pub type Item<'message> = ([u8; VERIFYING_KEY_LENGTH], Signature, &'message [u8]);

/// The group ordering needs no allocation up to this batch length
const ORDER_STACK_CAPACITY: usize = 128;

/// Bytes in a SHA-512 block
const BLOCK_LENGTH: usize = 128;

/// Length of SHA-512 padding, one marker byte and a 16-byte length
const PADDING_LENGTH: usize = 1 + 16;

/// Length of the `R || A` prefix in front of every hashed message
const PREFIX_LENGTH: usize = HALF_LENGTH + VERIFYING_KEY_LENGTH;

/// Shortest tail that is padded into a group
const PADDED_TAIL_FLOOR: usize = 3;

/// The ZIP-215 equation ends with this many cofactor doublings
const COFACTOR_DOUBLINGS: u32 = 3;

/// Doublings between two radix-16 digit positions
const DIGIT_DOUBLINGS: usize = 4;

/// Groups in one sixteen-wide hash pass
const GROUPS_PER_PAIR: usize = 2;

/// Whether this host runs the lane path, which only AVX-512 IFMA makes faster
pub fn is_lane_engine_available() -> bool {
    #[cfg(curve25519_backend = "simd")]
    {
        ifma_x8::available()
    }
    #[cfg(not(curve25519_backend = "simd"))]
    {
        false
    }
}

/// Verify `items` through the lane pipeline, returning verdicts in the caller's order
pub fn verify_batch(items: &[Item<'_>]) -> Vec<bool> {
    let count = items.len();
    let mut verdicts = alloc::vec![false; count];

    let mut stack = [0u32; ORDER_STACK_CAPACITY];
    let mut heap: Vec<u32>;
    let order: &mut [u32] = match count <= ORDER_STACK_CAPACITY {
        true => {
            for (index, slot) in stack[..count].iter_mut().enumerate() {
                *slot = index as u32;
            }
            &mut stack[..count]
        }
        false => {
            heap = (0..count as u32).collect();
            &mut heap[..]
        }
    };
    // Sort by hash block count so a long message does not slow a group of short ones
    order.sort_unstable_by_key(|&index| (hash_blocks(items[index as usize].2.len()), index));

    // Pair two groups while sixteen items remain so one hash pass serves both
    let mut done = 0;
    while done + GROUPS_PER_PAIR * LANES <= count {
        let pair: [[Item<'_>; LANES]; GROUPS_PER_PAIR] = core::array::from_fn(|group| {
            core::array::from_fn(|lane| items[order[done + group * LANES + lane] as usize])
        });
        let pair_verdicts = verify_group_pair(&pair);
        for group in 0..GROUPS_PER_PAIR {
            for lane in 0..LANES {
                verdicts[order[done + group * LANES + lane] as usize] = pair_verdicts[group][lane];
            }
        }
        done += GROUPS_PER_PAIR * LANES;
    }
    while done + LANES <= count {
        let group: [Item<'_>; LANES] =
            core::array::from_fn(|lane| items[order[done + lane] as usize]);
        let group_verdicts = verify_group(&group);
        for lane in 0..LANES {
            verdicts[order[done + lane] as usize] = group_verdicts[lane];
        }
        done += LANES;
    }

    // Pad a tail of three or more into one group by repeating its first item
    let tail = &order[done..];
    if tail.len() >= PADDED_TAIL_FLOOR {
        let group: [Item<'_>; LANES] =
            core::array::from_fn(|lane| items[tail[lane % tail.len()] as usize]);
        let group_verdicts = verify_group(&group);
        for (lane, &index) in tail.iter().enumerate() {
            verdicts[index as usize] = group_verdicts[lane];
        }
    } else {
        for &index in tail.iter() {
            let (encoded, signature, message) = &items[index as usize];
            verdicts[index as usize] = verify_encoded(encoded, signature, message);
        }
    }

    verdicts
}

/// Count the SHA-512 blocks of one message with its prefix and padding
fn hash_blocks(message_length: usize) -> u32 {
    (PREFIX_LENGTH + message_length + PADDING_LENGTH).div_ceil(BLOCK_LENGTH) as u32
}

/// Verify one full group of eight signatures in lockstep
fn verify_group(group: &[Item<'_>; LANES]) -> LaneMask {
    let a_bytes: [[u8; VERIFYING_KEY_LENGTH]; LANES] = core::array::from_fn(|lane| group[lane].0);
    let r_bytes: [[u8; HALF_LENGTH]; LANES] = core::array::from_fn(|lane| *group[lane].1.r_bytes());
    let s_bytes: [[u8; HALF_LENGTH]; LANES] = core::array::from_fn(|lane| *group[lane].1.s_bytes());
    let messages: [&[u8]; LANES] = core::array::from_fn(|lane| group[lane].2);

    let challenges = challenge_scalars(&a_bytes, &r_bytes, &messages);
    let (digits, alive) = prepare_group(&challenges, &s_bytes);
    run_curve_stage(&a_bytes, &r_bytes, &digits, &alive)
}

/// Verify two full groups with one sixteen-wide challenge hash pass
fn verify_group_pair(pair: &[[Item<'_>; LANES]; GROUPS_PER_PAIR]) -> [LaneMask; GROUPS_PER_PAIR] {
    let a_bytes: [[[u8; VERIFYING_KEY_LENGTH]; LANES]; GROUPS_PER_PAIR] =
        core::array::from_fn(|group| core::array::from_fn(|lane| pair[group][lane].0));
    let r_bytes: [[[u8; HALF_LENGTH]; LANES]; GROUPS_PER_PAIR] =
        core::array::from_fn(|group| core::array::from_fn(|lane| *pair[group][lane].1.r_bytes()));
    let s_bytes: [[[u8; HALF_LENGTH]; LANES]; GROUPS_PER_PAIR] =
        core::array::from_fn(|group| core::array::from_fn(|lane| *pair[group][lane].1.s_bytes()));
    let messages: [[&[u8]; LANES]; GROUPS_PER_PAIR] =
        core::array::from_fn(|group| core::array::from_fn(|lane| pair[group][lane].2));

    let challenges = challenge_scalars_pair(&a_bytes, &r_bytes, &messages);
    core::array::from_fn(|group| {
        let (digits, alive) = prepare_group(&challenges[group], &s_bytes[group]);
        run_curve_stage(&a_bytes[group], &r_bytes[group], &digits, &alive)
    })
}

/// Compute one group's challenge scalars, eight-wide where the vector unit allows
fn challenge_scalars(
    a_bytes: &[[u8; VERIFYING_KEY_LENGTH]; LANES],
    r_bytes: &[[u8; HALF_LENGTH]; LANES],
    messages: &[&[u8]; LANES],
) -> [Scalar; LANES] {
    #[cfg(curve25519_backend = "simd")]
    if ifma_x8::available() {
        // SAFETY: the cpuid check above guards the kernel
        let digests = unsafe { challenge_digests_x8(r_bytes, a_bytes, messages) };
        return core::array::from_fn(|lane| Scalar::from_bytes_mod_order_wide(&digests[lane]));
    }

    core::array::from_fn(|lane| challenge(&r_bytes[lane], &a_bytes[lane], messages[lane]))
}

/// Compute two groups' challenge scalars, sixteen-wide where the vector unit allows
fn challenge_scalars_pair(
    a_bytes: &[[[u8; VERIFYING_KEY_LENGTH]; LANES]; GROUPS_PER_PAIR],
    r_bytes: &[[[u8; HALF_LENGTH]; LANES]; GROUPS_PER_PAIR],
    messages: &[[&[u8]; LANES]; GROUPS_PER_PAIR],
) -> [[Scalar; LANES]; GROUPS_PER_PAIR] {
    #[cfg(curve25519_backend = "simd")]
    if ifma_x8::available() {
        // SAFETY: the cpuid check above guards the kernel
        let digests = unsafe { challenge_digests_x16(r_bytes, a_bytes, messages) };
        return core::array::from_fn(|group| {
            core::array::from_fn(|lane| Scalar::from_bytes_mod_order_wide(&digests[group][lane]))
        });
    }

    core::array::from_fn(|group| {
        core::array::from_fn(|lane| {
            challenge(
                &r_bytes[group][lane],
                &a_bytes[group][lane],
                messages[group][lane],
            )
        })
    })
}

/// Range-check `s` per lane, decompose the challenge, and lay out the digit schedules
fn prepare_group(
    challenges: &[Scalar; LANES],
    s_bytes: &[[u8; HALF_LENGTH]; LANES],
) -> (GroupDigits, LaneMask) {
    // A non-canonical s invalidates only its own lane
    let mut alive: LaneMask = [true; LANES];
    let s: [Scalar; LANES] = core::array::from_fn(|lane| {
        match Option::<Scalar>::from(Scalar::from_canonical_bytes(s_bytes[lane])) {
            Some(s) => s,
            None => {
                alive[lane] = false;
                Scalar::ZERO
            }
        }
    });

    // Each lane checks tau*s B + rho A' - tau R with the HEEA sign folded into the rho digits
    let mut b = [[0u64; LANES]; HEEA_DIGITS];
    let mut a = [[0i8; HEEA_DIGITS]; LANES];
    let mut r = [[0i8; HEEA_DIGITS]; LANES];

    for lane in 0..LANES {
        let (rho, tau, is_k_flipped) = challenges[lane].heea_decompose();
        let tau_s = tau * s[lane];
        for (position, &digit) in joint_digits(tau_s.as_bytes()).iter().enumerate() {
            b[position][lane] = digit as u64;
        }
        a[lane] = signed_digits(&rho);
        if !is_k_flipped {
            negate(&mut a[lane]);
        }
        r[lane] = signed_digits(&tau);
        negate(&mut r[lane]);
    }

    // Find the highest position with any nonzero digit
    let mut start = 0;
    for position in (0..HEEA_DIGITS).rev() {
        let is_live = (0..LANES)
            .any(|lane| b[position][lane] != 0 || a[lane][position] != 0 || r[lane][position] != 0);
        if is_live {
            start = position;
            break;
        }
    }

    (GroupDigits { b, a, r, start }, alive)
}

/// Run one prepared group through the fused or the portable curve stage
fn run_curve_stage(
    a_encodings: &[[u8; VERIFYING_KEY_LENGTH]; LANES],
    r_encodings: &[[u8; HALF_LENGTH]; LANES],
    digits: &GroupDigits,
    alive: &LaneMask,
) -> LaneMask {
    #[cfg(curve25519_backend = "simd")]
    if ifma_x8::available() {
        // SAFETY: the cpuid check above guards the kernel
        return unsafe {
            ifma_x8::fused::verify_curve_x8(a_encodings, r_encodings, None, digits, alive)
        };
    }

    verify_curve_portable(a_encodings, r_encodings, digits, alive)
}

/// Run the curve stage without IFMA in the same structure as the fused path
fn verify_curve_portable(
    a_encodings: &[[u8; VERIFYING_KEY_LENGTH]; LANES],
    r_encodings: &[[u8; HALF_LENGTH]; LANES],
    digits: &GroupDigits,
    alive_in: &LaneMask,
) -> LaneMask {
    let mut alive = *alive_in;

    let (a_point, a_valid) = decompress_x8(a_encodings);
    let (r_point, r_valid) = decompress_x8(r_encodings);
    for lane in 0..LANES {
        alive[lane] &= a_valid[lane] & r_valid[lane];
    }

    let a_table = LookupTableX8::from_extended(&a_point);
    let r_table = LookupTableX8::from_extended(&r_point);

    let mut accumulator: Option<ProjectivePointX8> = None;
    let mut last: Option<CompletedPointX8> = None;
    for position in (0..=digits.start).rev() {
        let mut point = match accumulator {
            Some(held) => {
                let mut held = held;
                for _ in 0..DIGIT_DOUBLINGS - 1 {
                    held = held.double().as_projective();
                }
                held.double().as_extended()
            }
            None => ExtendedPointX8::IDENTITY,
        };

        point = point
            .add(&JOINT_TABLE_B.select(&digits.b[position]))
            .as_extended();
        point = point
            .add(&a_table.select(&core::array::from_fn(|lane| digits.a[lane][position])))
            .as_extended();
        let completed =
            point.add(&r_table.select(&core::array::from_fn(|lane| digits.r[lane][position])));
        accumulator = Some(completed.as_projective());
        last = Some(completed);
    }

    let checked = match last {
        Some(completed) => completed.as_extended().mul_by_pow_2(COFACTOR_DOUBLINGS),
        None => ExtendedPointX8::IDENTITY,
    };

    let is_identity = checked.is_identity_lanes();
    core::array::from_fn(|lane| alive[lane] && is_identity[lane])
}
