//! Packed u64x4 and u32x8 wrappers over AVX2 vectors, which callers only use after an AVX2 check

// Vendored from curve25519-dalek, BSD-3-Clause

#![allow(unused_unsafe)]

use core::ops::{Add, AddAssign, BitAnd, BitAndAssign, BitXor, BitXorAssign, Sub};

use curve25519_dalek_derive::unsafe_target_feature;

// Wrapping add and sub give the same bits on signed and unsigned lanes, so signed intrinsics serve both
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
    /// Build a vector in a `const` context
    #[inline]
    pub const fn new_const(lane0: u64, lane1: u64, lane2: u64, lane3: u64) -> Self {
        // SAFETY: the array and the vector have the same size and every bit pattern is valid
        unsafe {
            Self(
                core::mem::transmute::<[u64; 4], core::arch::x86_64::__m256i>([
                    lane0, lane1, lane2, lane3,
                ]),
            )
        }
    }

    /// Broadcast one value in a `const` context
    #[inline]
    pub const fn splat_const<const VALUE: u64>() -> Self {
        Self::new_const(VALUE, VALUE, VALUE, VALUE)
    }

    /// Build a vector from its lanes, low lane first
    #[unsafe_target_feature("avx2")]
    #[inline]
    pub fn new(lane0: u64, lane1: u64, lane2: u64, lane3: u64) -> u64x4 {
        unsafe {
            // The intrinsic takes its lanes high first
            u64x4(core::arch::x86_64::_mm256_set_epi64x(
                lane3 as i64,
                lane2 as i64,
                lane1 as i64,
                lane0 as i64,
            ))
        }
    }

    /// Broadcast one value to every lane
    #[unsafe_target_feature("avx2")]
    #[inline]
    pub fn splat(value: u64) -> u64x4 {
        unsafe { u64x4(core::arch::x86_64::_mm256_set1_epi64x(value as i64)) }
    }
}

impl u32x8 {
    /// Build a vector in a `const` context
    #[allow(clippy::too_many_arguments)]
    #[inline]
    pub const fn new_const(
        lane0: u32,
        lane1: u32,
        lane2: u32,
        lane3: u32,
        lane4: u32,
        lane5: u32,
        lane6: u32,
        lane7: u32,
    ) -> Self {
        // SAFETY: the array and the vector have the same size and every bit pattern is valid
        unsafe {
            Self(
                core::mem::transmute::<[u32; 8], core::arch::x86_64::__m256i>([
                    lane0, lane1, lane2, lane3, lane4, lane5, lane6, lane7,
                ]),
            )
        }
    }

    /// Broadcast one value in a `const` context
    #[inline]
    pub const fn splat_const<const VALUE: u32>() -> Self {
        Self::new_const(VALUE, VALUE, VALUE, VALUE, VALUE, VALUE, VALUE, VALUE)
    }

    /// Build a vector from its lanes, low lane first
    #[allow(clippy::too_many_arguments)]
    #[unsafe_target_feature("avx2")]
    #[inline]
    pub fn new(
        lane0: u32,
        lane1: u32,
        lane2: u32,
        lane3: u32,
        lane4: u32,
        lane5: u32,
        lane6: u32,
        lane7: u32,
    ) -> u32x8 {
        unsafe {
            // The intrinsic takes its lanes high first
            u32x8(core::arch::x86_64::_mm256_set_epi32(
                lane7 as i32,
                lane6 as i32,
                lane5 as i32,
                lane4 as i32,
                lane3 as i32,
                lane2 as i32,
                lane1 as i32,
                lane0 as i32,
            ))
        }
    }

    /// Broadcast one value to every lane
    #[unsafe_target_feature("avx2")]
    #[inline]
    pub fn splat(value: u32) -> u32x8 {
        unsafe { u32x8(core::arch::x86_64::_mm256_set1_epi32(value as i32)) }
    }
}

#[unsafe_target_feature("avx2")]
impl u32x8 {
    /// Multiply the low 32 bits of each 64-bit lane into a 64-bit product
    #[inline]
    pub fn mul32(self, rhs: u32x8) -> u64x4 {
        unsafe { core::arch::x86_64::_mm256_mul_epu32(self.0, rhs.0).into() }
    }
}
