//! Four field elements per vector in radix 2^51 on the AVX-512 IFMA backend

// Vendored from curve25519-dalek, BSD-3-Clause

// Nightly and stable disagree on whether intrinsics inside unsafe_target_feature need unsafe blocks
#![allow(unused_unsafe)]
#![allow(non_snake_case)]

use core::ops::{Add, Mul, Neg};

use curve25519_dalek_derive::unsafe_target_feature;
use subtle::{Choice, ConditionallySelectable};

use crate::backend::serial::u64::field::FieldElement51;
use crate::backend::vector::packed_simd::u64x4;

/// Run `vpmadd52luq` on `u64x4` values
#[unsafe_target_feature("avx512ifma,avx512vl")]
#[inline]
unsafe fn madd52lo(z: u64x4, x: u64x4, y: u64x4) -> u64x4 {
    use core::arch::x86_64::_mm256_madd52lo_epu64;
    _mm256_madd52lo_epu64(z.into(), x.into(), y.into()).into()
}

/// Run `vpmadd52huq` on `u64x4` values
#[unsafe_target_feature("avx512ifma,avx512vl")]
#[inline]
unsafe fn madd52hi(z: u64x4, x: u64x4, y: u64x4) -> u64x4 {
    use core::arch::x86_64::_mm256_madd52hi_epu64;
    _mm256_madd52hi_epu64(z.into(), x.into(), y.into()).into()
}

/// Four field elements in radix 2^51 with unreduced limbs
#[derive(Clone, Copy, Debug)]
pub struct F51x4Unreduced(pub [u64x4; 5]);

/// Four field elements in radix 2^51 with reduced limbs
#[derive(Clone, Copy, Debug)]
pub struct F51x4Reduced(pub [u64x4; 5]);

#[allow(clippy::upper_case_acronyms)]
#[derive(Clone, Copy)]
pub enum Shuffle {
    AAAA,
    BBBB,
    BADC,
    BACD,
    ADDA,
    CBCB,
    ABDC,
    ABAB,
    DBBD,
    CACA,
}

#[unsafe_target_feature("avx512ifma,avx512vl")]
#[inline(always)]
fn shuffle_lanes(x: u64x4, control: Shuffle) -> u64x4 {
    unsafe {
        use core::arch::x86_64::_mm256_permute4x64_epi64 as perm;

        match control {
            Shuffle::AAAA => perm(x.into(), 0b00_00_00_00).into(),
            Shuffle::BBBB => perm(x.into(), 0b01_01_01_01).into(),
            Shuffle::BADC => perm(x.into(), 0b10_11_00_01).into(),
            Shuffle::BACD => perm(x.into(), 0b11_10_00_01).into(),
            Shuffle::ADDA => perm(x.into(), 0b00_11_11_00).into(),
            Shuffle::CBCB => perm(x.into(), 0b01_10_01_10).into(),
            Shuffle::ABDC => perm(x.into(), 0b10_11_01_00).into(),
            Shuffle::ABAB => perm(x.into(), 0b01_00_01_00).into(),
            Shuffle::DBBD => perm(x.into(), 0b11_01_01_11).into(),
            Shuffle::CACA => perm(x.into(), 0b00_10_00_10).into(),
        }
    }
}

#[allow(clippy::upper_case_acronyms)]
#[derive(Clone, Copy)]
pub enum Lanes {
    D,
    C,
    AB,
    AC,
    AD,
    BCD,
}

#[unsafe_target_feature("avx512ifma,avx512vl")]
#[inline]
fn blend_lanes(x: u64x4, y: u64x4, control: Lanes) -> u64x4 {
    unsafe {
        use core::arch::x86_64::_mm256_blend_epi32 as blend;

        match control {
            Lanes::D => blend(x.into(), y.into(), 0b11_00_00_00).into(),
            Lanes::C => blend(x.into(), y.into(), 0b00_11_00_00).into(),
            Lanes::AB => blend(x.into(), y.into(), 0b00_00_11_11).into(),
            Lanes::AC => blend(x.into(), y.into(), 0b00_11_00_11).into(),
            Lanes::AD => blend(x.into(), y.into(), 0b11_00_00_11).into(),
            Lanes::BCD => blend(x.into(), y.into(), 0b11_11_11_00).into(),
        }
    }
}

#[unsafe_target_feature("avx512ifma,avx512vl")]
impl F51x4Unreduced {
    pub const ZERO: F51x4Unreduced = F51x4Unreduced([u64x4::splat_const::<0>(); 5]);

    pub fn new(
        x0: &FieldElement51,
        x1: &FieldElement51,
        x2: &FieldElement51,
        x3: &FieldElement51,
    ) -> F51x4Unreduced {
        F51x4Unreduced([
            u64x4::new(x0.0[0], x1.0[0], x2.0[0], x3.0[0]),
            u64x4::new(x0.0[1], x1.0[1], x2.0[1], x3.0[1]),
            u64x4::new(x0.0[2], x1.0[2], x2.0[2], x3.0[2]),
            u64x4::new(x0.0[3], x1.0[3], x2.0[3], x3.0[3]),
            u64x4::new(x0.0[4], x1.0[4], x2.0[4], x3.0[4]),
        ])
    }

    pub fn split(&self) -> [FieldElement51; 4] {
        let x = &self.0;
        [
            FieldElement51([
                x[0].extract::<0>(),
                x[1].extract::<0>(),
                x[2].extract::<0>(),
                x[3].extract::<0>(),
                x[4].extract::<0>(),
            ]),
            FieldElement51([
                x[0].extract::<1>(),
                x[1].extract::<1>(),
                x[2].extract::<1>(),
                x[3].extract::<1>(),
                x[4].extract::<1>(),
            ]),
            FieldElement51([
                x[0].extract::<2>(),
                x[1].extract::<2>(),
                x[2].extract::<2>(),
                x[3].extract::<2>(),
                x[4].extract::<2>(),
            ]),
            FieldElement51([
                x[0].extract::<3>(),
                x[1].extract::<3>(),
                x[2].extract::<3>(),
                x[3].extract::<3>(),
                x[4].extract::<3>(),
            ]),
        ]
    }

    #[inline]
    pub fn diff_sum(&self) -> F51x4Unreduced {
        // tmp1 = (B, A, D, C)
        let tmp1 = self.shuffle(Shuffle::BADC);
        // tmp2 = (-A, B, -C, D)
        let tmp2 = self.blend(&self.negate_lazy(), Lanes::AC);
        // (B - A, B + A, D - C, D + C)
        tmp1 + tmp2
    }

    #[inline]
    pub fn negate_lazy(&self) -> F51x4Unreduced {
        let lo = u64x4::splat(36028797018963664u64);
        let hi = u64x4::splat(36028797018963952u64);
        F51x4Unreduced([
            lo - self.0[0],
            hi - self.0[1],
            hi - self.0[2],
            hi - self.0[3],
            hi - self.0[4],
        ])
    }

    #[inline]
    pub fn shuffle(&self, control: Shuffle) -> F51x4Unreduced {
        F51x4Unreduced([
            shuffle_lanes(self.0[0], control),
            shuffle_lanes(self.0[1], control),
            shuffle_lanes(self.0[2], control),
            shuffle_lanes(self.0[3], control),
            shuffle_lanes(self.0[4], control),
        ])
    }

    #[inline]
    pub fn blend(&self, other: &F51x4Unreduced, control: Lanes) -> F51x4Unreduced {
        F51x4Unreduced([
            blend_lanes(self.0[0], other.0[0], control),
            blend_lanes(self.0[1], other.0[1], control),
            blend_lanes(self.0[2], other.0[2], control),
            blend_lanes(self.0[3], other.0[3], control),
            blend_lanes(self.0[4], other.0[4], control),
        ])
    }
}

#[unsafe_target_feature("avx512ifma,avx512vl")]
impl Neg for F51x4Reduced {
    type Output = F51x4Reduced;

    fn neg(self) -> F51x4Reduced {
        F51x4Unreduced::from(self).negate_lazy().into()
    }
}

#[unsafe_target_feature("avx512ifma,avx512vl")]
impl ConditionallySelectable for F51x4Reduced {
    #[inline]
    fn conditional_select(a: &F51x4Reduced, b: &F51x4Reduced, choice: Choice) -> F51x4Reduced {
        let mask = (-(choice.unwrap_u8() as i64)) as u64;
        let mask_vec = u64x4::splat(mask);
        F51x4Reduced([
            a.0[0] ^ (mask_vec & (a.0[0] ^ b.0[0])),
            a.0[1] ^ (mask_vec & (a.0[1] ^ b.0[1])),
            a.0[2] ^ (mask_vec & (a.0[2] ^ b.0[2])),
            a.0[3] ^ (mask_vec & (a.0[3] ^ b.0[3])),
            a.0[4] ^ (mask_vec & (a.0[4] ^ b.0[4])),
        ])
    }

    #[inline]
    fn conditional_assign(&mut self, other: &F51x4Reduced, choice: Choice) {
        let mask = (-(choice.unwrap_u8() as i64)) as u64;
        let mask_vec = u64x4::splat(mask);
        self.0[0] ^= mask_vec & (self.0[0] ^ other.0[0]);
        self.0[1] ^= mask_vec & (self.0[1] ^ other.0[1]);
        self.0[2] ^= mask_vec & (self.0[2] ^ other.0[2]);
        self.0[3] ^= mask_vec & (self.0[3] ^ other.0[3]);
        self.0[4] ^= mask_vec & (self.0[4] ^ other.0[4]);
    }
}

#[unsafe_target_feature("avx512ifma,avx512vl")]
impl F51x4Reduced {
    #[inline]
    pub fn shuffle(&self, control: Shuffle) -> F51x4Reduced {
        F51x4Reduced([
            shuffle_lanes(self.0[0], control),
            shuffle_lanes(self.0[1], control),
            shuffle_lanes(self.0[2], control),
            shuffle_lanes(self.0[3], control),
            shuffle_lanes(self.0[4], control),
        ])
    }

    #[inline]
    pub fn blend(&self, other: &F51x4Reduced, control: Lanes) -> F51x4Reduced {
        F51x4Reduced([
            blend_lanes(self.0[0], other.0[0], control),
            blend_lanes(self.0[1], other.0[1], control),
            blend_lanes(self.0[2], other.0[2], control),
            blend_lanes(self.0[3], other.0[3], control),
            blend_lanes(self.0[4], other.0[4], control),
        ])
    }

    #[inline]
    pub fn square(&self) -> F51x4Unreduced {
        unsafe {
            let x = &self.0;

            // Accumulators for terms with coefficient 2
            let mut z0_2 = u64x4::splat(0);
            let mut z1_2 = u64x4::splat(0);
            let mut z2_2 = u64x4::splat(0);
            let mut z3_2 = u64x4::splat(0);
            let mut z4_2 = u64x4::splat(0);
            let mut z5_2 = u64x4::splat(0);
            let mut z6_2 = u64x4::splat(0);
            let mut z7_2 = u64x4::splat(0);
            let mut z9_2 = u64x4::splat(0);

            // Accumulators for terms with coefficient 4
            let mut z2_4 = u64x4::splat(0);
            let mut z3_4 = u64x4::splat(0);
            let mut z4_4 = u64x4::splat(0);
            let mut z5_4 = u64x4::splat(0);
            let mut z6_4 = u64x4::splat(0);
            let mut z7_4 = u64x4::splat(0);
            let mut z8_4 = u64x4::splat(0);

            let mut z0_1 = u64x4::splat(0);
            z0_1 = madd52lo(z0_1, x[0], x[0]);

            let mut z1_1 = u64x4::splat(0);
            z1_2 = madd52lo(z1_2, x[0], x[1]);
            z1_2 = madd52hi(z1_2, x[0], x[0]);

            z2_4 = madd52hi(z2_4, x[0], x[1]);
            let mut z2_1 = z2_4.shl::<2>();
            z2_2 = madd52lo(z2_2, x[0], x[2]);
            z2_1 = madd52lo(z2_1, x[1], x[1]);

            z3_4 = madd52hi(z3_4, x[0], x[2]);
            let mut z3_1 = z3_4.shl::<2>();
            z3_2 = madd52lo(z3_2, x[1], x[2]);
            z3_2 = madd52lo(z3_2, x[0], x[3]);
            z3_2 = madd52hi(z3_2, x[1], x[1]);

            z4_4 = madd52hi(z4_4, x[1], x[2]);
            z4_4 = madd52hi(z4_4, x[0], x[3]);
            let mut z4_1 = z4_4.shl::<2>();
            z4_2 = madd52lo(z4_2, x[1], x[3]);
            z4_2 = madd52lo(z4_2, x[0], x[4]);
            z4_1 = madd52lo(z4_1, x[2], x[2]);

            z5_4 = madd52hi(z5_4, x[1], x[3]);
            z5_4 = madd52hi(z5_4, x[0], x[4]);
            let mut z5_1 = z5_4.shl::<2>();
            z5_2 = madd52lo(z5_2, x[2], x[3]);
            z5_2 = madd52lo(z5_2, x[1], x[4]);
            z5_2 = madd52hi(z5_2, x[2], x[2]);

            z6_4 = madd52hi(z6_4, x[2], x[3]);
            z6_4 = madd52hi(z6_4, x[1], x[4]);
            let mut z6_1 = z6_4.shl::<2>();
            z6_2 = madd52lo(z6_2, x[2], x[4]);
            z6_1 = madd52lo(z6_1, x[3], x[3]);

            z7_4 = madd52hi(z7_4, x[2], x[4]);
            let mut z7_1 = z7_4.shl::<2>();
            z7_2 = madd52lo(z7_2, x[3], x[4]);
            z7_2 = madd52hi(z7_2, x[3], x[3]);

            z8_4 = madd52hi(z8_4, x[3], x[4]);
            let mut z8_1 = z8_4.shl::<2>();
            z8_1 = madd52lo(z8_1, x[4], x[4]);

            let mut z9_1 = u64x4::splat(0);
            z9_2 = madd52hi(z9_2, x[4], x[4]);

            z5_1 += z5_2.shl::<1>();
            z6_1 += z6_2.shl::<1>();
            z7_1 += z7_2.shl::<1>();
            z9_1 += z9_2.shl::<1>();

            let mut t0 = u64x4::splat(0);
            let mut t1 = u64x4::splat(0);
            let r19 = u64x4::splat(19);

            t0 = madd52hi(t0, r19, z9_1);
            t1 = madd52lo(t1, r19, z9_1.shr::<52>());

            z4_2 = madd52lo(z4_2, r19, z8_1.shr::<52>());
            z3_2 = madd52lo(z3_2, r19, z7_1.shr::<52>());
            z2_2 = madd52lo(z2_2, r19, z6_1.shr::<52>());
            z1_2 = madd52lo(z1_2, r19, z5_1.shr::<52>());

            z0_2 = madd52lo(z0_2, r19, t0 + t1);
            z1_2 = madd52hi(z1_2, r19, z5_1);
            z2_2 = madd52hi(z2_2, r19, z6_1);
            z3_2 = madd52hi(z3_2, r19, z7_1);
            z4_2 = madd52hi(z4_2, r19, z8_1);

            z0_1 = madd52lo(z0_1, r19, z5_1);
            z1_1 = madd52lo(z1_1, r19, z6_1);
            z2_1 = madd52lo(z2_1, r19, z7_1);
            z3_1 = madd52lo(z3_1, r19, z8_1);
            z4_1 = madd52lo(z4_1, r19, z9_1);

            F51x4Unreduced([
                z0_1 + z0_2 + z0_2,
                z1_1 + z1_2 + z1_2,
                z2_1 + z2_2 + z2_2,
                z3_1 + z3_2 + z3_2,
                z4_1 + z4_2 + z4_2,
            ])
        }
    }
}

#[unsafe_target_feature("avx512ifma,avx512vl")]
impl From<F51x4Reduced> for F51x4Unreduced {
    #[inline]
    fn from(x: F51x4Reduced) -> F51x4Unreduced {
        F51x4Unreduced(x.0)
    }
}

#[unsafe_target_feature("avx512ifma,avx512vl")]
impl From<F51x4Unreduced> for F51x4Reduced {
    #[inline]
    fn from(x: F51x4Unreduced) -> F51x4Reduced {
        let mask = u64x4::splat((1 << 51) - 1);
        let r19 = u64x4::splat(19);

        let c0 = x.0[0].shr::<51>();
        let c1 = x.0[1].shr::<51>();
        let c2 = x.0[2].shr::<51>();
        let c3 = x.0[3].shr::<51>();
        let c4 = x.0[4].shr::<51>();

        unsafe {
            F51x4Reduced([
                madd52lo(x.0[0] & mask, c4, r19),
                (x.0[1] & mask) + c0,
                (x.0[2] & mask) + c1,
                (x.0[3] & mask) + c2,
                (x.0[4] & mask) + c3,
            ])
        }
    }
}

#[unsafe_target_feature("avx512ifma,avx512vl")]
impl Add<F51x4Unreduced> for F51x4Unreduced {
    type Output = F51x4Unreduced;
    #[inline]
    fn add(self, rhs: F51x4Unreduced) -> F51x4Unreduced {
        F51x4Unreduced([
            self.0[0] + rhs.0[0],
            self.0[1] + rhs.0[1],
            self.0[2] + rhs.0[2],
            self.0[3] + rhs.0[3],
            self.0[4] + rhs.0[4],
        ])
    }
}

#[unsafe_target_feature("avx512ifma,avx512vl")]
impl Mul<(u32, u32, u32, u32)> for &F51x4Reduced {
    type Output = F51x4Unreduced;
    #[inline]
    fn mul(self, scalars: (u32, u32, u32, u32)) -> F51x4Unreduced {
        unsafe {
            let x = &self.0;
            let y = u64x4::new(
                scalars.0 as u64,
                scalars.1 as u64,
                scalars.2 as u64,
                scalars.3 as u64,
            );
            let r19 = u64x4::splat(19);

            let mut z0_1 = u64x4::splat(0);
            let mut z1_1 = u64x4::splat(0);
            let mut z2_1 = u64x4::splat(0);
            let mut z3_1 = u64x4::splat(0);
            let mut z4_1 = u64x4::splat(0);
            let mut z1_2 = u64x4::splat(0);
            let mut z2_2 = u64x4::splat(0);
            let mut z3_2 = u64x4::splat(0);
            let mut z4_2 = u64x4::splat(0);
            let mut z5_2 = u64x4::splat(0);

            z4_2 = madd52hi(z4_2, y, x[3]);
            z5_2 = madd52hi(z5_2, y, x[4]);
            z4_1 = madd52lo(z4_1, y, x[4]);
            z0_1 = madd52lo(z0_1, y, x[0]);
            z3_1 = madd52lo(z3_1, y, x[3]);
            z2_1 = madd52lo(z2_1, y, x[2]);
            z1_1 = madd52lo(z1_1, y, x[1]);
            z3_2 = madd52hi(z3_2, y, x[2]);

            z2_2 = madd52hi(z2_2, y, x[1]);
            z1_2 = madd52hi(z1_2, y, x[0]);
            z0_1 = madd52lo(z0_1, z5_2 + z5_2, r19);

            F51x4Unreduced([
                z0_1,
                z1_1 + z1_2 + z1_2,
                z2_1 + z2_2 + z2_2,
                z3_1 + z3_2 + z3_2,
                z4_1 + z4_2 + z4_2,
            ])
        }
    }
}

#[unsafe_target_feature("avx512ifma,avx512vl")]
impl Mul<&F51x4Reduced> for &F51x4Reduced {
    type Output = F51x4Unreduced;
    #[inline]
    fn mul(self, rhs: &F51x4Reduced) -> F51x4Unreduced {
        unsafe {
            let x = &self.0;
            let y = &rhs.0;

            // Accumulators for terms with coefficient 1
            let mut z0_1 = u64x4::splat(0);
            let mut z1_1 = u64x4::splat(0);
            let mut z2_1 = u64x4::splat(0);
            let mut z3_1 = u64x4::splat(0);
            let mut z4_1 = u64x4::splat(0);
            let mut z5_1 = u64x4::splat(0);
            let mut z6_1 = u64x4::splat(0);
            let mut z7_1 = u64x4::splat(0);
            let mut z8_1 = u64x4::splat(0);

            // Accumulators for terms with coefficient 2
            let mut z0_2 = u64x4::splat(0);
            let mut z1_2 = u64x4::splat(0);
            let mut z2_2 = u64x4::splat(0);
            let mut z3_2 = u64x4::splat(0);
            let mut z4_2 = u64x4::splat(0);
            let mut z5_2 = u64x4::splat(0);
            let mut z6_2 = u64x4::splat(0);
            let mut z7_2 = u64x4::splat(0);
            let mut z8_2 = u64x4::splat(0);
            let mut z9_2 = u64x4::splat(0);

            // Multiply-adds come in waves of eight independent operations since LLVM keeps their order
            z4_1 = madd52lo(z4_1, x[2], y[2]);
            z5_2 = madd52hi(z5_2, x[2], y[2]);
            z5_1 = madd52lo(z5_1, x[4], y[1]);
            z6_2 = madd52hi(z6_2, x[4], y[1]);
            z6_1 = madd52lo(z6_1, x[4], y[2]);
            z7_2 = madd52hi(z7_2, x[4], y[2]);
            z7_1 = madd52lo(z7_1, x[4], y[3]);
            z8_2 = madd52hi(z8_2, x[4], y[3]);

            z4_1 = madd52lo(z4_1, x[3], y[1]);
            z5_2 = madd52hi(z5_2, x[3], y[1]);
            z5_1 = madd52lo(z5_1, x[3], y[2]);
            z6_2 = madd52hi(z6_2, x[3], y[2]);
            z6_1 = madd52lo(z6_1, x[3], y[3]);
            z7_2 = madd52hi(z7_2, x[3], y[3]);
            z7_1 = madd52lo(z7_1, x[3], y[4]);
            z8_2 = madd52hi(z8_2, x[3], y[4]);

            z8_1 = madd52lo(z8_1, x[4], y[4]);
            z9_2 = madd52hi(z9_2, x[4], y[4]);
            z4_1 = madd52lo(z4_1, x[4], y[0]);
            z5_2 = madd52hi(z5_2, x[4], y[0]);
            z5_1 = madd52lo(z5_1, x[2], y[3]);
            z6_2 = madd52hi(z6_2, x[2], y[3]);
            z6_1 = madd52lo(z6_1, x[2], y[4]);
            z7_2 = madd52hi(z7_2, x[2], y[4]);

            let z8 = z8_1 + z8_2 + z8_2;
            let z9 = z9_2 + z9_2;

            z3_1 = madd52lo(z3_1, x[3], y[0]);
            z4_2 = madd52hi(z4_2, x[3], y[0]);
            z4_1 = madd52lo(z4_1, x[1], y[3]);
            z5_2 = madd52hi(z5_2, x[1], y[3]);
            z5_1 = madd52lo(z5_1, x[1], y[4]);
            z6_2 = madd52hi(z6_2, x[1], y[4]);
            z2_1 = madd52lo(z2_1, x[2], y[0]);
            z3_2 = madd52hi(z3_2, x[2], y[0]);

            let z6 = z6_1 + z6_2 + z6_2;
            let z7 = z7_1 + z7_2 + z7_2;

            z3_1 = madd52lo(z3_1, x[2], y[1]);
            z4_2 = madd52hi(z4_2, x[2], y[1]);
            z4_1 = madd52lo(z4_1, x[0], y[4]);
            z5_2 = madd52hi(z5_2, x[0], y[4]);
            z1_1 = madd52lo(z1_1, x[1], y[0]);
            z2_2 = madd52hi(z2_2, x[1], y[0]);
            z2_1 = madd52lo(z2_1, x[1], y[1]);
            z3_2 = madd52hi(z3_2, x[1], y[1]);

            let z5 = z5_1 + z5_2 + z5_2;

            z3_1 = madd52lo(z3_1, x[1], y[2]);
            z4_2 = madd52hi(z4_2, x[1], y[2]);
            z0_1 = madd52lo(z0_1, x[0], y[0]);
            z1_2 = madd52hi(z1_2, x[0], y[0]);
            z1_1 = madd52lo(z1_1, x[0], y[1]);
            z2_1 = madd52lo(z2_1, x[0], y[2]);
            z2_2 = madd52hi(z2_2, x[0], y[1]);
            z3_2 = madd52hi(z3_2, x[0], y[2]);

            let mut t0 = u64x4::splat(0);
            let mut t1 = u64x4::splat(0);
            let r19 = u64x4::splat(19);

            t0 = madd52hi(t0, r19, z9);
            t1 = madd52lo(t1, r19, z9.shr::<52>());
            z3_1 = madd52lo(z3_1, x[0], y[3]);
            z4_2 = madd52hi(z4_2, x[0], y[3]);
            z1_2 = madd52lo(z1_2, r19, z5.shr::<52>());
            z2_2 = madd52lo(z2_2, r19, z6.shr::<52>());
            z3_2 = madd52lo(z3_2, r19, z7.shr::<52>());
            z0_1 = madd52lo(z0_1, r19, z5);

            z4_1 = madd52lo(z4_1, r19, z9);
            z1_1 = madd52lo(z1_1, r19, z6);
            z0_2 = madd52lo(z0_2, r19, t0 + t1);
            z4_2 = madd52hi(z4_2, r19, z8);
            z2_1 = madd52lo(z2_1, r19, z7);
            z1_2 = madd52hi(z1_2, r19, z5);
            z2_2 = madd52hi(z2_2, r19, z6);
            z3_2 = madd52hi(z3_2, r19, z7);

            z3_1 = madd52lo(z3_1, r19, z8);
            z4_2 = madd52lo(z4_2, r19, z8.shr::<52>());

            F51x4Unreduced([
                z0_1 + z0_2 + z0_2,
                z1_1 + z1_2 + z1_2,
                z2_1 + z2_2 + z2_2,
                z3_1 + z3_2 + z3_2,
                z4_1 + z4_2 + z4_2,
            ])
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Whether this CPU runs the backend under test, so other hosts skip it
    fn is_supported() -> bool {
        std::is_x86_feature_detected!("avx512ifma") && std::is_x86_feature_detected!("avx512vl")
    }

    const ROUNDS: usize = 1024;
    const LANE_COUNT: usize = 4;
    const SCALE_SHIFT: u32 = 4;
    const SMALL_CONSTANT: u32 = 121665;

    // Inverting a small element gives one with full-size limbs
    fn big_element() -> FieldElement51 {
        FieldElement51([2438, 24, 243, 0, 0]).invert()
    }

    fn other_big_element() -> FieldElement51 {
        FieldElement51([98098, 87987897, 0, 1, 0]).invert()
    }

    fn scaled_unreduced(element: &FieldElement51) -> FieldElement51 {
        FieldElement51([
            element.0[0] << SCALE_SHIFT,
            element.0[1] << SCALE_SHIFT,
            element.0[2] << SCALE_SHIFT,
            element.0[3] << SCALE_SHIFT,
            element.0[4] << SCALE_SHIFT,
        ])
    }

    fn splat(element: &FieldElement51) -> F51x4Unreduced {
        F51x4Unreduced::new(element, element, element, element)
    }

    fn distinct_elements() -> [FieldElement51; LANE_COUNT] {
        [
            FieldElement51::from_bytes(&[0x10; 32]),
            FieldElement51::from_bytes(&[0x11; 32]),
            FieldElement51::from_bytes(&[0x12; 32]),
            FieldElement51::from_bytes(&[0x13; 32]),
        ]
    }

    // the vpmadd52luq wrapper adds the low product to the accumulator
    #[test]
    fn madd_low() {
        if !is_supported() {
            return;
        }
        let left = u64x4::splat(2);
        let right = u64x4::splat(3);
        let accumulator = u64x4::splat(5);

        // SAFETY: the test module only compiles with avx512ifma and avx512vl enabled
        let result = unsafe { madd52lo(accumulator, left, right) };

        assert_eq!(result, u64x4::splat(5 + 2 * 3));
    }

    // packing and splitting reduced limbs gives the inputs back
    #[test]
    fn split_reduced() {
        if !is_supported() {
            return;
        }
        let element = big_element();

        let lanes = splat(&element).split();

        for lane in lanes {
            assert_eq!(element, lane);
        }
    }

    // packing and splitting unreduced limbs gives the inputs back
    #[test]
    fn split_unreduced() {
        if !is_supported() {
            return;
        }
        let element = scaled_unreduced(&big_element());

        let lanes = splat(&element).split();

        for lane in lanes {
            assert_eq!(element, lane);
        }
    }

    // reduction keeps the value of scaled unreduced limbs
    #[test]
    fn reduction() {
        if !is_supported() {
            return;
        }
        let element = big_element();
        let scaled = scaled_unreduced(&element);

        let reduced: F51x4Reduced = splat(&scaled).into();
        let lanes = F51x4Unreduced::from(reduced).split();
        let expected = &element * &FieldElement51([1 << SCALE_SHIFT, 0, 0, 0, 0]);

        for lane in lanes {
            assert_eq!(expected, lane);
        }
    }

    // one vector multiplication matches the serial product
    #[test]
    fn multiplication() {
        if !is_supported() {
            return;
        }
        let left = big_element();
        let right = other_big_element();
        let expected = &left * &right;

        let left_vector: F51x4Reduced = splat(&left).into();
        let right_vector: F51x4Reduced = splat(&right).into();
        let lanes = (&left_vector * &right_vector).split();

        for lane in lanes {
            assert_eq!(expected, lane);
        }
    }

    // a long chain of vector multiplications matches the serial chain
    #[test]
    fn iterated_multiplication() {
        if !is_supported() {
            return;
        }
        let left = big_element();
        let right = other_big_element();
        let mut expected = &left * &right;
        for _ in 0..ROUNDS {
            expected = &left * &expected;
            expected = &right * &expected;
        }

        let left_vector: F51x4Reduced = splat(&left).into();
        let right_vector: F51x4Reduced = splat(&right).into();
        let mut product = &left_vector * &right_vector;
        for _ in 0..ROUNDS {
            product = &left_vector * &F51x4Reduced::from(product);
            product = &right_vector * &F51x4Reduced::from(product);
        }

        for lane in product.split() {
            assert_eq!(expected, lane);
        }
    }

    // vector squaring matches vector multiplication by itself
    #[test]
    fn square() {
        if !is_supported() {
            return;
        }
        let vector: F51x4Reduced = splat(&big_element()).into();

        let product = (&vector * &vector).split();
        let squared = vector.square().split();

        for lane in 0..LANE_COUNT {
            assert_eq!(squared[lane], product[lane]);
        }
    }

    // a long chain of vector squarings matches the serial chain
    #[test]
    fn iterated_square() {
        if !is_supported() {
            return;
        }
        let mut element = big_element();
        let mut vector = splat(&element);
        for _ in 0..ROUNDS {
            element = element.square();
            vector = F51x4Reduced::from(vector).square();

            for lane in vector.split() {
                assert_eq!(element, lane);
            }
        }
    }

    // a long chain of multiplications by a small constant matches the serial chain
    #[test]
    fn small_multiplication() {
        if !is_supported() {
            return;
        }
        let element = big_element();
        let constant = FieldElement51([SMALL_CONSTANT as u64, 0, 0, 0, 0]);
        let mut expected = &element * &constant;
        for _ in 0..ROUNDS {
            expected = &constant * &expected;
        }

        let constants = (
            SMALL_CONSTANT,
            SMALL_CONSTANT,
            SMALL_CONSTANT,
            SMALL_CONSTANT,
        );
        let mut product = &F51x4Reduced::from(splat(&element)) * constants;
        for _ in 0..ROUNDS {
            product = &F51x4Reduced::from(product) * constants;
        }

        for lane in product.split() {
            assert_eq!(expected, lane);
        }
    }

    // the AAAA shuffle copies lane A into every lane
    #[test]
    fn shuffle() {
        if !is_supported() {
            return;
        }
        let elements = distinct_elements();
        let vector = F51x4Unreduced::new(&elements[0], &elements[1], &elements[2], &elements[3]);

        let lanes = vector.shuffle(Shuffle::AAAA).split();

        for lane in lanes {
            assert_eq!(lane, elements[0]);
        }
    }

    // the AB blend takes lanes A and B from the second vector
    #[test]
    fn blend() {
        if !is_supported() {
            return;
        }
        let elements = distinct_elements();
        let first = F51x4Unreduced::new(&elements[0], &elements[1], &elements[2], &elements[3]);
        let second = F51x4Unreduced::new(&elements[3], &elements[2], &elements[1], &elements[0]);

        let lanes = first.blend(&second, Lanes::AB).split();

        assert_eq!(lanes[0], elements[3]);
        assert_eq!(lanes[1], elements[2]);
        assert_eq!(lanes[2], elements[2]);
        assert_eq!(lanes[3], elements[3]);
    }
}
