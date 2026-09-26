//! Internal macros.

/// Define borrow and non-borrow variants of `Add`.
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

/// Define non-borrow variants of `AddAssign`.
macro_rules! define_add_assign_variants {
    (LHS = $lhs:ty, RHS = $rhs:ty) => {
        impl AddAssign<$rhs> for $lhs {
            fn add_assign(&mut self, rhs: $rhs) {
                *self += &rhs;
            }
        }
    };
}

/// Define borrow and non-borrow variants of `Sub`.
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

/// Define non-borrow variants of `SubAssign`.
macro_rules! define_sub_assign_variants {
    (LHS = $lhs:ty, RHS = $rhs:ty) => {
        impl SubAssign<$rhs> for $lhs {
            fn sub_assign(&mut self, rhs: $rhs) {
                *self -= &rhs;
            }
        }
    };
}

/// Define borrow and non-borrow variants of `Mul`.
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

/// Define non-borrow variants of `MulAssign`.
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
        /// A lookup table of precomputed multiples of a point \\(P\\), used to
        /// compute \\( xP \\) for \\( -8 \leq x \leq 8 \\).
        ///
        /// The computation of \\( xP \\) is done in constant time by the `select` function.
        ///
        /// Since `LookupTable` does not implement `Index`, it's more difficult
        /// to accidentally use the table directly.  Unfortunately the table is
        /// only `pub` so that we can write hardcoded constants, so it's
        /// still technically possible.  It would be nice to prevent direct
        /// access to the table.
        #[derive(Clone, Copy)]
        pub struct $name<Point>(pub [Point; $size]);

        impl<Point> $name<Point>
        where
            Point: Identity + ConditionallySelectable + ConditionallyNegatable,
        {
            /// Given \\(-8 \leq x \leq 8\\), return \\(xP\\) in constant time.
            pub fn select(&self, x: i8) -> Point {
                debug_assert!(x >= $neg);
                debug_assert!(x as i16 <= $size as i16); // XXX We have to convert to i16s here for the radix-256 case.. this is wrong.

                // Compute xabs = |x|
                let xmask = x as i16 >> 7;
                let xabs = (x as i16 + xmask) ^ xmask;

                // Set t = 0 * P = identity
                let mut t = Point::identity();
                for j in $range {
                    // Copy `points[j-1] == j*P` onto `t` in constant time if `|x| == j`.
                    let c = (xabs as u16).ct_eq(&(j as u16));
                    t.conditional_assign(&self.0[j - 1], c);
                }
                // Now t == |x| * P.

                let neg_mask = Choice::from((xmask & 1) as u8);
                t.conditional_negate(neg_mask);
                // Now t == x * P.

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
                // XXX batch inversion would be good if perf mattered here
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
        /// A precomputed table of multiples of a basepoint, for accelerating
        /// fixed-base scalar multiplication.  One table, for the Ed25519
        /// basepoint, is provided in the [`constants`] module.
        ///
        /// The basepoint tables are reasonably large, so they should probably be boxed.
        ///
        /// The sizes for the tables and the number of additions required for one scalar
        /// multiplication are as follows:
        ///
        /// * [`EdwardsBasepointTableRadix16`]: 30KB, 64A
        ///   (this is the default size, and is used for
        ///   [`constants::ED25519_BASEPOINT_TABLE`])
        /// * [`EdwardsBasepointTableRadix64`]: 120KB, 43A
        /// * [`EdwardsBasepointTableRadix128`]: 240KB, 37A
        /// * [`EdwardsBasepointTableRadix256`]: 480KB, 33A
        ///
        /// # Why 33 additions for radix-256?
        ///
        /// Normally, the radix-256 tables would allow for only 32 additions per scalar
        /// multiplication.  However, due to the fact that standardised definitions of
        /// legacy protocols—such as x25519—require allowing unreduced 255-bit scalars
        /// invariants, when converting such an unreduced scalar's representation to
        /// radix-\\(2^{8}\\), we cannot guarantee the carry bit will fit in the last
        /// coefficient (the coefficients are `i8`s).  When, \\(w\\), the power-of-2 of
        /// the radix, is \\(w < 8\\), we can fold the final carry onto the last
        /// coefficient, \\(d\\), because \\(d < 2^{w/2}\\), so
        /// $$
        ///     d + carry \cdot 2^{w} = d + 1 \cdot 2^{w} < 2^{w+1} < 2^{8}
        /// $$
        /// When \\(w = 8\\), we can't fit \\(carry \cdot 2^{w}\\) into an `i8`, so we
        /// add the carry bit onto an additional coefficient.
        #[derive(Clone)]
        #[repr(transparent)]
        pub struct $name(pub [$table<AffineNielsPoint>; 32]);

        impl BasepointTable for $name {
            type Point = $point;

            /// Create a table of precomputed multiples of `basepoint`.
            fn create(basepoint: &$point) -> $name {
                // XXX use init_with
                let mut table = $name([$table::default(); 32]);
                let mut P = *basepoint;
                for i in 0..32 {
                    // P = (2w)^i * B
                    table.0[i] = $table::from(&P);
                    P = P.mul_by_pow_2($radix + $radix);
                }
                table
            }

            /// Get the basepoint for this table as an `EdwardsPoint`.
            fn basepoint(&self) -> $point {
                // self.0[0].select(1) = 1*(16^2)^0*B
                // but as an `AffineNielsPoint`, so add identity to convert to extended.
                (&<$point>::identity() + &self.0[0].select(1)).as_extended()
            }

            /// The computation uses Pippeneger's algorithm, as described for the
            /// specific case of radix-16 on page 13 of the Ed25519 paper.
            ///
            /// # Piggenger's Algorithm Generalised
            ///
            /// Write the scalar \\(a\\) in radix-\\(w\\), where \\(w\\) is a power of
            /// 2, with coefficients in \\([\frac{-w}{2},\frac{w}{2})\\), i.e.,
            /// $$
            ///     a = a\_0 + a\_1 w\^1 + \cdots + a\_{x} w\^{x},
            /// $$
            /// with
            /// $$
            /// \begin{aligned}
            ///     \frac{-w}{2} \leq a_i < \frac{w}{2}
            ///     &&\cdots&&
            ///     \frac{-w}{2} \leq a\_{x} \leq \frac{w}{2}
            /// \end{aligned}
            /// $$
            /// and the number of additions, \\(x\\), is given by
            /// \\(x = \lceil \frac{256}{w} \rceil\\). Then
            /// $$
            ///     a B = a\_0 B + a\_1 w\^1 B + \cdots + a\_{x-1} w\^{x-1} B.
            /// $$
            /// Grouping even and odd coefficients gives
            /// $$
            /// \begin{aligned}
            ///     a B = \quad a\_0 w\^0 B +& a\_2 w\^2 B + \cdots + a\_{x-2} w\^{x-2} B    \\\\
            ///               + a\_1 w\^1 B +& a\_3 w\^3 B + \cdots + a\_{x-1} w\^{x-1} B    \\\\
            ///         = \quad(a\_0 w\^0 B +& a\_2 w\^2 B + \cdots + a\_{x-2} w\^{x-2} B)   \\\\
            ///             + w(a\_1 w\^0 B +& a\_3 w\^2 B + \cdots + a\_{x-1} w\^{x-2} B).  \\\\
            /// \end{aligned}
            /// $$
            /// For each \\(i = 0 \ldots 31\\), we create a lookup table of
            /// $$
            /// [w\^{2i} B, \ldots, \frac{w}{2}\cdot w\^{2i} B],
            /// $$
            /// and use it to select \\( y \cdot w\^{2i} \cdot B \\) in constant time.
            ///
            /// The radix-\\(w\\) representation requires that the scalar is bounded
            /// by \\(2\^{255}\\), which is always the case.
            ///
            /// The above algorithm is trivially generalised to other powers-of-2 radices.
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

            /// Construct an `EdwardsPoint` from a `Scalar` \\(a\\) by
            /// computing the multiple \\(aB\\) of this basepoint \\(B\\).
            fn mul(self, scalar: &Scalar) -> $point {
                // delegate to a private function so that its documentation appears in internal docs
                self.mul_base(scalar)
            }
        }

        impl Mul<&$name> for &Scalar {
            type Output = $point;

            /// Construct an `EdwardsPoint` from a `Scalar` \\(a\\) by
            /// computing the multiple \\(aB\\) of this basepoint \\(B\\).
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
