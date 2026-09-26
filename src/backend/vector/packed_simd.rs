// Vendored from curve25519-dalek, BSD-3-Clause. See LICENSE.

// Nightly and stable currently disagree on the requirement of unsafe blocks when `unsafe_target_feature`
// gets used.
// See: https://github.com/rust-lang/rust/issues/132856
#![allow(unused_unsafe)]

//! This module defines wrappers over platform-specific SIMD types to make them
//! more convenient to use.
//!
//! UNSAFETY: Everything in this module assumes that we're running on hardware
//!           which supports at least AVX2. This invariant *must* be enforced
//!           by the callers of this code.
use core::ops::{Add, AddAssign, BitAnd, BitAndAssign, BitXor, BitXorAssign, Sub};

use curve25519_dalek_derive::unsafe_target_feature;

// We define SIMD functionality over packed unsigned integer types. However, all the integer
// intrinsics deal with signed integers. So we cast unsigned to signed, pack it into SIMD, do
// add/sub/shl/shr arithmetic, and finally cast back to unsigned at the end. Why is this equivalent
// to doing the same thing on unsigned integers? Shl/shr is clear, because casting does not change
// the bits of the integer. But what about add/sub? This is due to the following:
//
//     1) Rust uses two's complement to represent signed integers. So we're assured that the values
//        we cast into SIMD and extract out at the end are two's complement.
//
//        https://doc.rust-lang.org/reference/types/numeric.html
//
//     2) Wrapping add/sub is compatible between two's complement signed and unsigned integers.
//        That is, for all x,y: u64 (or any unsigned integer type),
//
//            x.wrapping_add(y) == (x as i64).wrapping_add(y as i64) as u64, and
//            x.wrapping_sub(y) == (x as i64).wrapping_sub(y as i64) as u64
//
//        https://julesjacobs.com/2019/03/20/why-twos-complement-works.html
//
//     3) The add/sub functions we use for SIMD are indeed wrapping. The docs indicate that
//        __mm256_add/sub compile to vpaddX/vpsubX instructions where X = w, d, or q depending on
//        the bitwidth. From x86 docs:
//
//            When an individual result is too large to be represented in X bits (overflow), the
//            result is wrapped around and the low X bits are written to the destination operand
//            (that is, the carry is ignored).
//
//        https://www.felixcloutier.com/x86/paddb:paddw:paddd:paddq
//        https://www.felixcloutier.com/x86/psubb:psubw:psubd
//        https://www.felixcloutier.com/x86/psubq

impl_shared!(
    u64x4,
    u64,
    _mm256_add_epi64,
    _mm256_sub_epi64,
    _mm256_slli_epi64,
    _mm256_srli_epi64,
    _mm256_extract_epi64
);
impl_shared!(
    u32x8,
    u32,
    _mm256_add_epi32,
    _mm256_sub_epi32,
    _mm256_slli_epi32,
    _mm256_srli_epi32,
    _mm256_extract_epi32
);

impl_conv!(u64x4 => u32x8);

impl u64x4 {
    /// A constified variant of `new`.
    ///
    /// Should only be called from `const` contexts. At runtime `new` is going to be faster.
    #[inline]
    pub const fn new_const(x0: u64, x1: u64, x2: u64, x3: u64) -> Self {
        // SAFETY: Transmuting between an array and a SIMD type is safe
        // https://rust-lang.github.io/unsafe-code-guidelines/layout/packed-simd-vectors.html
        unsafe {
            Self(core::mem::transmute::<[u64; 4], core::arch::x86_64::__m256i>([x0, x1, x2, x3]))
        }
    }

    /// A constified variant of `splat`.
    ///
    /// Should only be called from `const` contexts. At runtime `splat` is going to be faster.
    #[inline]
    pub const fn splat_const<const N: u64>() -> Self {
        Self::new_const(N, N, N, N)
    }

    /// Constructs a new instance.
    #[unsafe_target_feature("avx2")]
    #[inline]
    pub fn new(x0: u64, x1: u64, x2: u64, x3: u64) -> u64x4 {
        unsafe {
            // _mm256_set_epi64 sets the underlying vector in reverse order of the args
            u64x4(core::arch::x86_64::_mm256_set_epi64x(
                x3 as i64, x2 as i64, x1 as i64, x0 as i64,
            ))
        }
    }

    /// Constructs a new instance with all of the elements initialized to the given value.
    #[unsafe_target_feature("avx2")]
    #[inline]
    pub fn splat(x: u64) -> u64x4 {
        unsafe { u64x4(core::arch::x86_64::_mm256_set1_epi64x(x as i64)) }
    }
}

impl u32x8 {
    /// A constified variant of `new`.
    ///
    /// Should only be called from `const` contexts. At runtime `new` is going to be faster.
    #[allow(clippy::too_many_arguments)]
    #[inline]
    pub const fn new_const(
        x0: u32,
        x1: u32,
        x2: u32,
        x3: u32,
        x4: u32,
        x5: u32,
        x6: u32,
        x7: u32,
    ) -> Self {
        // SAFETY: Transmuting between an array and a SIMD type is safe
        // https://rust-lang.github.io/unsafe-code-guidelines/layout/packed-simd-vectors.html
        unsafe {
            Self(
                core::mem::transmute::<[u32; 8], core::arch::x86_64::__m256i>([
                    x0, x1, x2, x3, x4, x5, x6, x7,
                ]),
            )
        }
    }

    /// A constified variant of `splat`.
    ///
    /// Should only be called from `const` contexts. At runtime `splat` is going to be faster.
    #[inline]
    pub const fn splat_const<const N: u32>() -> Self {
        Self::new_const(N, N, N, N, N, N, N, N)
    }

    /// Constructs a new instance.
    #[allow(clippy::too_many_arguments)]
    #[unsafe_target_feature("avx2")]
    #[inline]
    pub fn new(x0: u32, x1: u32, x2: u32, x3: u32, x4: u32, x5: u32, x6: u32, x7: u32) -> u32x8 {
        unsafe {
            // _mm256_set_epi32 sets the underlying vector in reverse order of the args
            u32x8(core::arch::x86_64::_mm256_set_epi32(
                x7 as i32, x6 as i32, x5 as i32, x4 as i32, x3 as i32, x2 as i32, x1 as i32,
                x0 as i32,
            ))
        }
    }

    /// Constructs a new instance with all of the elements initialized to the given value.
    #[unsafe_target_feature("avx2")]
    #[inline]
    pub fn splat(x: u32) -> u32x8 {
        unsafe { u32x8(core::arch::x86_64::_mm256_set1_epi32(x as i32)) }
    }
}

#[unsafe_target_feature("avx2")]
impl u32x8 {
    /// Multiplies the low unsigned 32-bits from each packed 64-bit element
    /// and returns the unsigned 64-bit results.
    ///
    /// (This ignores the upper 32-bits from each packed 64-bits!)
    #[inline]
    pub fn mul32(self, rhs: u32x8) -> u64x4 {
        // NOTE: This ignores the upper 32-bits from each packed 64-bits.
        unsafe { core::arch::x86_64::_mm256_mul_epu32(self.0, rhs.0).into() }
    }
}
