//! Pick the field arithmetic per host and route each scalar multiplication to it

use crate::backend::serial;
#[cfg(curve25519_backend = "simd")]
use crate::backend::vector;
use crate::edwards::EdwardsPoint;
use crate::scalar::Scalar;

/// A scalar below 2^128 has zero bytes from this index up
const HALF_SCALAR_LENGTH: usize = 16;

/// A field arithmetic backend for the running host
#[derive(Clone, Copy)]
enum BackendKind {
    #[cfg(curve25519_backend = "simd")]
    Avx2,
    #[cfg(curve25519_backend = "simd")]
    Avx512,
    Serial,
}

#[inline]
fn get_selected_backend() -> BackendKind {
    #[cfg(curve25519_backend = "simd")]
    {
        cpufeatures::new!(cpuid_avx512, "avx512ifma", "avx512vl");
        let token_avx512: cpuid_avx512::InitToken = cpuid_avx512::init();
        if token_avx512.get() {
            return BackendKind::Avx512;
        }

        cpufeatures::new!(cpuid_avx2, "avx2");
        let token_avx2: cpuid_avx2::InitToken = cpuid_avx2::init();
        if token_avx2.get() {
            return BackendKind::Avx2;
        }
    }

    BackendKind::Serial
}

/// Multiply `point` by `scalar` in constant time
pub fn variable_base_mul(point: &EdwardsPoint, scalar: &Scalar) -> EdwardsPoint {
    match get_selected_backend() {
        #[cfg(curve25519_backend = "simd")]
        BackendKind::Avx2 => vector::scalar_mul::variable_base::spec_avx2::mul(point, scalar),
        #[cfg(curve25519_backend = "simd")]
        BackendKind::Avx512 => {
            vector::scalar_mul::variable_base::spec_avx512ifma_avx512vl::mul(point, scalar)
        }
        BackendKind::Serial => serial::scalar_mul::variable_base::mul(point, scalar),
    }
}

/// Compute aA + bB in variable time, with B the Ed25519 basepoint
#[allow(non_snake_case)]
pub fn vartime_double_base_mul(a: &Scalar, A: &EdwardsPoint, b: &Scalar) -> EdwardsPoint {
    match get_selected_backend() {
        #[cfg(curve25519_backend = "simd")]
        BackendKind::Avx2 => vector::scalar_mul::vartime_double_base::spec_avx2::mul(a, A, b),
        #[cfg(curve25519_backend = "simd")]
        BackendKind::Avx512 => {
            vector::scalar_mul::vartime_double_base::spec_avx512ifma_avx512vl::mul(a, A, b)
        }
        BackendKind::Serial => serial::scalar_mul::vartime_double_base::mul(a, A, b),
    }
}

/// Compute a1 A1 + a2 A2 + bB in variable time, for a1 and a2 below 2^128
#[allow(non_snake_case)]
pub fn vartime_triple_base_mul_128_128_256_prechecked(
    a1: &Scalar,
    A1: &EdwardsPoint,
    a2: &Scalar,
    A2: &EdwardsPoint,
    b: &Scalar,
) -> EdwardsPoint {
    debug_assert!(scalar_fits_in_128_bits(a1) && scalar_fits_in_128_bits(a2));

    match get_selected_backend() {
        #[cfg(curve25519_backend = "simd")]
        BackendKind::Avx2 => {
            vector::scalar_mul::vartime_triple_base::spec_avx2::mul_128_128_256_prechecked(
                a1, A1, a2, A2, b,
            )
        }
        #[cfg(curve25519_backend = "simd")]
        BackendKind::Avx512 => {
            vector::scalar_mul::vartime_triple_base::spec_avx512ifma_avx512vl::mul_128_128_256_prechecked(
                a1, A1, a2, A2, b,
            )
        }
        BackendKind::Serial => {
            serial::scalar_mul::vartime_triple_base::mul_128_128_256_prechecked(a1, A1, a2, A2, b)
        }
    }
}

#[inline]
fn scalar_fits_in_128_bits(scalar: &Scalar) -> bool {
    scalar.as_bytes()[HALF_SCALAR_LENGTH..]
        .iter()
        .all(|&byte| byte == 0)
}
