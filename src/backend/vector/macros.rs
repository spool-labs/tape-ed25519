//! Macros for the packed SIMD wrapper types

macro_rules! impl_shared {
    (
        $ty:ident,
        $lane_ty:ident,
        $add_intrinsic:ident,
        $sub_intrinsic:ident,
        $shl_intrinsic:ident,
        $shr_intrinsic:ident,
        $extract_intrinsic:ident
    ) => {
        #[allow(non_camel_case_types)]
        #[derive(Copy, Clone, Debug)]
        #[repr(transparent)]
        pub struct $ty(core::arch::x86_64::__m256i);

        #[unsafe_target_feature("avx2")]
        impl From<$ty> for core::arch::x86_64::__m256i {
            #[inline]
            fn from(value: $ty) -> core::arch::x86_64::__m256i {
                value.0
            }
        }

        #[unsafe_target_feature("avx2")]
        impl From<core::arch::x86_64::__m256i> for $ty {
            #[inline]
            fn from(value: core::arch::x86_64::__m256i) -> $ty {
                $ty(value)
            }
        }

        #[unsafe_target_feature("avx2")]
        impl PartialEq for $ty {
            #[inline]
            fn eq(&self, rhs: &$ty) -> bool {
                unsafe {
                    // This compares each pair of 8-bit packed integers and returns either 0xFF or
                    // 0x00 depending on whether they're equal.
                    //
                    // So the values are equal if (and only if) this returns a value that's filled
                    // with only 0xFF.
                    //
                    // Pseudocode of what this does:
                    //     self.0
                    //         .bytes()
                    //         .zip(rhs.0.bytes())
                    //         .map(|a, b| if a == b { 0xFF } else { 0x00 })
                    //         .join();
                    let m = core::arch::x86_64::_mm256_cmpeq_epi8(self.0, rhs.0);

                    // Now we need to reduce the 256-bit value to something on which we can branch.
                    //
                    // This will just take the most significant bit of every 8-bit packed integer
                    // and build an `i32` out of it. If the values we previously compared were
                    // equal then all off the most significant bits will be equal to 1, which means
                    // that this will return 0xFFFFFFFF, which is equal to -1 when represented as
                    // an `i32`.
                    core::arch::x86_64::_mm256_movemask_epi8(m) == -1
                }
            }
        }

        impl Eq for $ty {}

        #[unsafe_target_feature("avx2")]
        impl Add for $ty {
            type Output = Self;

            #[inline]
            fn add(self, rhs: $ty) -> Self {
                unsafe { core::arch::x86_64::$add_intrinsic(self.0, rhs.0).into() }
            }
        }

        #[allow(clippy::assign_op_pattern)]
        #[unsafe_target_feature("avx2")]
        impl AddAssign for $ty {
            #[inline]
            fn add_assign(&mut self, rhs: $ty) {
                *self = *self + rhs
            }
        }

        #[unsafe_target_feature("avx2")]
        impl Sub for $ty {
            type Output = Self;

            #[inline]
            fn sub(self, rhs: $ty) -> Self {
                unsafe { core::arch::x86_64::$sub_intrinsic(self.0, rhs.0).into() }
            }
        }

        #[unsafe_target_feature("avx2")]
        impl BitAnd for $ty {
            type Output = Self;

            #[inline]
            fn bitand(self, rhs: $ty) -> Self {
                unsafe { core::arch::x86_64::_mm256_and_si256(self.0, rhs.0).into() }
            }
        }

        #[unsafe_target_feature("avx2")]
        impl BitXor for $ty {
            type Output = Self;

            #[inline]
            fn bitxor(self, rhs: $ty) -> Self {
                unsafe { core::arch::x86_64::_mm256_xor_si256(self.0, rhs.0).into() }
            }
        }

        #[allow(clippy::assign_op_pattern)]
        #[unsafe_target_feature("avx2")]
        impl BitAndAssign for $ty {
            #[inline]
            fn bitand_assign(&mut self, rhs: $ty) {
                *self = *self & rhs;
            }
        }

        #[allow(clippy::assign_op_pattern)]
        #[unsafe_target_feature("avx2")]
        impl BitXorAssign for $ty {
            #[inline]
            fn bitxor_assign(&mut self, rhs: $ty) {
                *self = *self ^ rhs;
            }
        }

        #[unsafe_target_feature("avx2")]
        impl $ty {
            #[inline]
            pub fn shl<const N: i32>(self) -> Self {
                unsafe { core::arch::x86_64::$shl_intrinsic(self.0, N).into() }
            }

            #[inline]
            pub fn shr<const N: i32>(self) -> Self {
                unsafe { core::arch::x86_64::$shr_intrinsic(self.0, N).into() }
            }

            #[inline]
            pub fn extract<const N: i32>(self) -> $lane_ty {
                unsafe { core::arch::x86_64::$extract_intrinsic(self.0, N) as $lane_ty }
            }
        }
    };
}

macro_rules! impl_conv {
    ($src:ident => $($dst:ident),+) => {
        $(
            #[unsafe_target_feature("avx2")]
            impl From<$src> for $dst {
                #[inline]
                fn from(value: $src) -> $dst {
                    $dst(value.0)
                }
            }
        )+
    }
}
