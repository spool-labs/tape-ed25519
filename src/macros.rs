//! Macros for operator variants, lookup tables and basepoint tables

// Vendored from curve25519-dalek, BSD-3-Clause

/// Define borrow and non-borrow variants of `Add`
macro_rules! define_add_variants {
    (LHS = $lhs:ty, RHS = $rhs:ty, Output = $out:ty) => {
        impl Add<&$rhs> for $lhs {
            type Output = $out;
            fn add(self, rhs: &$rhs) -> $out {
                &self + rhs
            }
        }

        impl Add<$rhs> for &$lhs {
            type Output = $out;
            fn add(self, rhs: $rhs) -> $out {
                self + &rhs
            }
        }

        impl Add<$rhs> for $lhs {
            type Output = $out;
            fn add(self, rhs: $rhs) -> $out {
                &self + &rhs
            }
        }
    };
}

/// Define non-borrow variants of `AddAssign`
macro_rules! define_add_assign_variants {
    (LHS = $lhs:ty, RHS = $rhs:ty) => {
        impl AddAssign<$rhs> for $lhs {
            fn add_assign(&mut self, rhs: $rhs) {
                *self += &rhs;
            }
        }
    };
}

/// Define borrow and non-borrow variants of `Sub`
macro_rules! define_sub_variants {
    (LHS = $lhs:ty, RHS = $rhs:ty, Output = $out:ty) => {
        impl Sub<&$rhs> for $lhs {
            type Output = $out;
            fn sub(self, rhs: &$rhs) -> $out {
                &self - rhs
            }
        }

        impl Sub<$rhs> for &$lhs {
            type Output = $out;
            fn sub(self, rhs: $rhs) -> $out {
                self - &rhs
            }
        }

        impl Sub<$rhs> for $lhs {
            type Output = $out;
            fn sub(self, rhs: $rhs) -> $out {
                &self - &rhs
            }
        }
    };
}

/// Define non-borrow variants of `SubAssign`
macro_rules! define_sub_assign_variants {
    (LHS = $lhs:ty, RHS = $rhs:ty) => {
        impl SubAssign<$rhs> for $lhs {
            fn sub_assign(&mut self, rhs: $rhs) {
                *self -= &rhs;
            }
        }
    };
}

/// Define borrow and non-borrow variants of `Mul`
macro_rules! define_mul_variants {
    (LHS = $lhs:ty, RHS = $rhs:ty, Output = $out:ty) => {
        impl Mul<&$rhs> for $lhs {
            type Output = $out;
            fn mul(self, rhs: &$rhs) -> $out {
                &self * rhs
            }
        }

        impl Mul<$rhs> for &$lhs {
            type Output = $out;
            fn mul(self, rhs: $rhs) -> $out {
                self * &rhs
            }
        }

        impl Mul<$rhs> for $lhs {
            type Output = $out;
            fn mul(self, rhs: $rhs) -> $out {
                &self * &rhs
            }
        }
    };
}

/// Define non-borrow variants of `MulAssign`
macro_rules! define_mul_assign_variants {
    (LHS = $lhs:ty, RHS = $rhs:ty) => {
        impl MulAssign<$rhs> for $lhs {
            fn mul_assign(&mut self, rhs: $rhs) {
                *self *= &rhs;
            }
        }
    };
}

macro_rules! impl_lookup_table {
    (Name = $name:ident, Size = $size:expr, SizeNeg = $neg:expr, SizeRange = $range:expr, ConversionRange = $conv_range:expr) => {
        /// The multiples P, 2P, ..., of a point P for a constant-time signed digit lookup
        #[derive(Clone, Copy)]
        pub struct $name<Point>(pub [Point; $size]);

        impl<Point> $name<Point>
        where
            Point: Identity + ConditionallySelectable + ConditionallyNegatable,
        {
            /// Return xP in constant time for a signed digit x within the table size
            pub fn select(&self, x: i8) -> Point {
                debug_assert!(x >= $neg);
                debug_assert!(x as i16 <= $size as i16);

                let xmask = x as i16 >> 7;
                let xabs = (x as i16 + xmask) ^ xmask;

                let mut t = Point::identity();
                for j in $range {
                    // Every entry is visited so the access pattern does not depend on x
                    let c = (xabs as u16).ct_eq(&(j as u16));
                    t.conditional_assign(&self.0[j - 1], c);
                }

                let neg_mask = Choice::from((xmask & 1) as u8);
                t.conditional_negate(neg_mask);

                t
            }
        }

        impl<Point> Default for $name<Point>
        where
            Point: Copy + Default,
        {
            fn default() -> $name<Point> {
                $name([Point::default(); $size])
            }
        }

        impl<Point: Debug> Debug for $name<Point> {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                write!(f, "{:?}(", stringify!($name))?;

                for x in self.0.iter() {
                    write!(f, "{:?}", x)?;
                }

                write!(f, ")")
            }
        }

        impl From<&EdwardsPoint> for $name<ProjectiveNielsPoint> {
            fn from(P: &EdwardsPoint) -> Self {
                let mut points = [P.as_projective_niels(); $size];
                for j in $conv_range {
                    points[j + 1] = (P + &points[j]).as_extended().as_projective_niels();
                }
                $name(points)
            }
        }

        impl From<&EdwardsPoint> for $name<AffineNielsPoint> {
            fn from(P: &EdwardsPoint) -> Self {
                let mut points = [P.as_affine_niels(); $size];
                for j in $conv_range {
                    points[j + 1] = (P + &points[j]).as_extended().as_affine_niels()
                }
                $name(points)
            }
        }
    };
}

macro_rules! impl_basepoint_table {
    (Name = $name:ident, LookupTable = $table:ident, Point = $point:ty, Radix = $radix:expr, Additions = $adds:expr) => {
        /// Precomputed multiples of a basepoint for constant-time fixed-base multiplication
        #[derive(Clone)]
        #[repr(transparent)]
        pub struct $name(pub [$table<AffineNielsPoint>; 32]);

        impl BasepointTable for $name {
            type Point = $point;

            /// Build the table of multiples of `basepoint`
            fn create(basepoint: &$point) -> $name {
                let mut table = $name([$table::default(); 32]);
                let mut P = *basepoint;
                for i in 0..32 {
                    // Row i holds multiples of (2^w)^(2i) B
                    table.0[i] = $table::from(&P);
                    P = P.mul_by_pow_2($radix + $radix);
                }
                table
            }

            /// Recover the basepoint this table was built from
            fn basepoint(&self) -> $point {
                // Adding the affine Niels entry for 1B to the identity converts it to extended form
                (&<$point>::identity() + &self.0[0].select(1)).as_extended()
            }

            /// Sum the odd digits of `scalar`, scale by 2^w, then sum the even digits
            fn mul_base(&self, scalar: &Scalar) -> $point {
                let a = scalar.as_radix_2w($radix);

                let tables = &self.0;
                let mut P = <$point>::identity();

                for i in (0..$adds).filter(|x| x % 2 == 1) {
                    P = (&P + &tables[i / 2].select(a[i])).as_extended();
                }

                P = P.mul_by_pow_2($radix);

                for i in (0..$adds).filter(|x| x % 2 == 0) {
                    P = (&P + &tables[i / 2].select(a[i])).as_extended();
                }

                P
            }
        }

        impl Mul<&Scalar> for &$name {
            type Output = $point;

            /// Multiply the basepoint of this table by `scalar`
            fn mul(self, scalar: &Scalar) -> $point {
                self.mul_base(scalar)
            }
        }

        impl Mul<&$name> for &Scalar {
            type Output = $point;

            /// Multiply the basepoint of `basepoint_table` by this scalar
            fn mul(self, basepoint_table: &$name) -> $point {
                basepoint_table * self
            }
        }

        impl Debug for $name {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                write!(f, "{:?}([\n", stringify!($name))?;
                for i in 0..32 {
                    write!(f, "\t{:?},\n", self.0[i])?;
                }
                write!(f, "])")
            }
        }
    };
}

macro_rules! impl_basepoint_table_conversions {
    (LHS = $lhs:ty, RHS = $rhs:ty) => {
        impl From<&$lhs> for $rhs {
            fn from(table: &$lhs) -> $rhs {
                <$rhs>::create(&table.basepoint())
            }
        }

        impl From<&$rhs> for $lhs {
            fn from(table: &$rhs) -> $lhs {
                <$lhs>::create(&table.basepoint())
            }
        }
    };
}
