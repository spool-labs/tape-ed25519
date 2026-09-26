//! Macros for the packed SIMD wrapper types

/// Define a packed integer type with its arithmetic and shift operators
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
        #[derive(Clone, Copy, Debug)]
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
                    let mask = core::arch::x86_64::_mm256_cmpeq_epi8(self.0, rhs.0);

                    // Every byte matched when all 32 byte masks are set, which reads as -1
                    core::arch::x86_64::_mm256_movemask_epi8(mask) == -1
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
            pub fn shl<const SHIFT: i32>(self) -> Self {
                unsafe { core::arch::x86_64::$shl_intrinsic(self.0, SHIFT).into() }
            }

            #[inline]
            pub fn shr<const SHIFT: i32>(self) -> Self {
                unsafe { core::arch::x86_64::$shr_intrinsic(self.0, SHIFT).into() }
            }

            #[inline]
            pub fn extract<const INDEX: i32>(self) -> $lane_ty {
                unsafe { core::arch::x86_64::$extract_intrinsic(self.0, INDEX) as $lane_ty }
            }
        }
    };
}

/// Define free bit casts from one packed type to others
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
