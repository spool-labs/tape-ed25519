# tape-ed25519

Ed25519 verification under the ZIP-215 rules, one signature at a time or eight lanes wide.

- `verifying_key::verify` checks one signature and accepts what `ed25519-zebra` accepts.
- `batch::verify_batch` checks many and returns one verdict per signature. It runs eight SIMD lanes on a host with AVX-512 IFMA and the single verifier everywhere else. `cpuid` picks the path at run time, so there are no features to set.

Both verifiers decide the same way on every input. A differential test holds them and the lane pipeline against `verify_zebra` on ten thousand random signatures plus the ZIP-215 edge vectors.

| Host | Batch of 64, 1232-byte messages | One at a time |
|------|------|------|
| EPYC 9275F (IFMA lanes) | 243 us | 927 us |
| Apple M4 Max (single) | 19.8 us per signature | 21.8 us with `solana-ed25519` |

```sh
cargo test --release
cargo bench
```

Verification only: no signing and no key generation.

## License

BSD-3-Clause, from curve25519-dalek by way of the solana-ed25519 lanes fork. See [LICENSE](LICENSE).
