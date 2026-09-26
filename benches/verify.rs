//! Benchmark the verifier against the crates.io `solana-ed25519` verifier under the same ZIP-215 rules

use criterion::{criterion_group, criterion_main, Criterion};
use ed25519_dalek::Signer;

use tape_ed25519::batch::verify_batch;
use tape_ed25519::signature::Signature;
use tape_ed25519::verifying_key::{verify, VerifyingKey};

/// Each row measures messages of this length, a full Solana packet
const MESSAGE_LENGTH: usize = 1232;

/// Batch sizes for the batch rows
const BATCH_SIZES: [usize; 3] = [1, 8, 64];

/// Each key seed byte is the item index mixed with this mask
const KEY_SEED_MASK: u8 = 0x3B;

/// Sign `count` items, each under its own key
fn signed(count: usize) -> Vec<([u8; 32], [u8; 64], Vec<u8>)> {
    let mut items = Vec::with_capacity(count);
    for index in 0..count {
        let key = ed25519_dalek::SigningKey::from_bytes(&[index as u8 ^ KEY_SEED_MASK; 32]);
        let message: Vec<u8> = (0..MESSAGE_LENGTH)
            .map(|position| (position as u8) ^ (index as u8))
            .collect();
        let signature = key.sign(&message);
        items.push((
            key.verifying_key().to_bytes(),
            signature.to_bytes(),
            message,
        ));
    }
    items
}

/// Verify one item with the crates.io verifier and decode the key inside the call
fn zebra_verify(key: &[u8; 32], signature: &[u8; 64], message: &[u8]) -> bool {
    let signature = solana_ed25519::ed_sigs::Signature::from_bytes(signature);
    solana_ed25519::ed_sigs::VerificationKey::try_from(*key)
        .and_then(|key| key.verify_zebra(&signature, message))
        .is_ok()
}

fn single(criterion: &mut Criterion) {
    let items = signed(1);
    let (encoded, signature, message) = &items[0];
    let key = VerifyingKey::from_bytes(encoded).expect("a key from the signer decodes");
    let parsed = Signature::from_bytes(signature);

    // Decompress both keys up front so the row times only the equation
    let zebra_key = solana_ed25519::ed_sigs::VerificationKey::try_from(*encoded)
        .expect("the same key decodes for the crates.io verifier");
    let zebra_signature = solana_ed25519::ed_sigs::Signature::from_bytes(signature);

    let mut group = criterion.benchmark_group("verify");
    group.bench_function("tape-ed25519", |bencher| {
        bencher.iter(|| verify(&key, &parsed, message).is_ok())
    });
    group.bench_function("solana-ed25519", |bencher| {
        bencher.iter(|| zebra_key.verify_zebra(&zebra_signature, message).is_ok())
    });
    group.finish();
}

fn batched(criterion: &mut Criterion) {
    for count in BATCH_SIZES {
        let items = signed(count);
        let prepared: Vec<([u8; 32], Signature, &[u8])> = items
            .iter()
            .map(|(encoded, signature, message)| {
                (
                    *encoded,
                    Signature::from_bytes(signature),
                    message.as_slice(),
                )
            })
            .collect();

        let mut group = criterion.benchmark_group(format!("verify_batch/{count}"));
        group.throughput(criterion::Throughput::Elements(count as u64));
        group.bench_function("tape-ed25519", |bencher| {
            bencher.iter(|| verify_batch(&prepared))
        });
        group.bench_function("solana-ed25519", |bencher| {
            bencher.iter(|| {
                let mut verdicts = Vec::with_capacity(items.len());
                for (encoded, signature, message) in &items {
                    verdicts.push(zebra_verify(encoded, signature, message));
                }
                verdicts
            })
        });
        group.finish();
    }
}

criterion_group!(benches, single, batched);
criterion_main!(benches);
