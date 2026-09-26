//! The projective, completed, extended and Niels point models of the curve

// Vendored from curve25519-dalek, BSD-3-Clause

#![allow(non_snake_case)]

use core::fmt::Debug;
use core::ops::{Add, Neg, Sub};

use subtle::Choice;
use subtle::ConditionallySelectable;

#[cfg(test)]
use crate::constants;

use crate::edwards::EdwardsPoint;
use crate::field::FieldElement;
#[cfg(test)]
use crate::traits::ValidityCheck;

/// A point `(X:Y:Z)` on the projective P^2 model, where affine `(x, y)` is `(x:y:1)`
#[allow(missing_docs)]
#[derive(Clone, Copy)]
pub struct ProjectivePoint {
    pub X: FieldElement,
    pub Y: FieldElement,
    pub Z: FieldElement,
}

/// A point `((X:Z), (Y:T))` on the completed P^1 x P^1 model, where affine `(x, y)` is `((x:1), (y:1))`
#[derive(Clone, Copy)]
#[allow(missing_docs)]
pub struct CompletedPoint {
    pub X: FieldElement,
    pub Y: FieldElement,
    pub Z: FieldElement,
    pub T: FieldElement,
}

/// A precomputed affine point in Niels coordinates `(y+x, y-x, 2dxy)`
// Affine coordinates are unique, so the derived Eq is sound
#[derive(Clone, Copy, Eq, PartialEq)]
#[allow(missing_docs)]
pub struct AffineNielsPoint {
    pub y_plus_x: FieldElement,
    pub y_minus_x: FieldElement,
    pub xy2d: FieldElement,
}

/// A precomputed extended point in Niels coordinates `(Y+X, Y-X, Z, 2dXY)`
#[derive(Clone, Copy)]
#[allow(missing_docs)]
pub struct ProjectiveNielsPoint {
    pub Y_plus_X: FieldElement,
    pub Y_minus_X: FieldElement,
    pub Z: FieldElement,
    pub T2d: FieldElement,
}

use crate::traits::Identity;

impl Identity for ProjectivePoint {
    fn identity() -> ProjectivePoint {
        ProjectivePoint {
            X: FieldElement::ZERO,
            Y: FieldElement::ONE,
            Z: FieldElement::ONE,
        }
    }
}

impl Identity for ProjectiveNielsPoint {
    fn identity() -> ProjectiveNielsPoint {
        ProjectiveNielsPoint {
            Y_plus_X: FieldElement::ONE,
            Y_minus_X: FieldElement::ONE,
            Z: FieldElement::ONE,
            T2d: FieldElement::ZERO,
        }
    }
}

impl Default for ProjectiveNielsPoint {
    fn default() -> ProjectiveNielsPoint {
        ProjectiveNielsPoint::identity()
    }
}

impl Identity for AffineNielsPoint {
    fn identity() -> AffineNielsPoint {
        AffineNielsPoint {
            y_plus_x: FieldElement::ONE,
            y_minus_x: FieldElement::ONE,
            xy2d: FieldElement::ZERO,
        }
    }
}

impl Default for AffineNielsPoint {
    fn default() -> AffineNielsPoint {
        AffineNielsPoint::identity()
    }
}

#[cfg(test)]
impl ValidityCheck for ProjectivePoint {
    fn is_valid(&self) -> bool {
        // Check -x^2 + y^2 = 1 + d x^2 y^2 in homogeneous form
        let XX = self.X.square();
        let YY = self.Y.square();
        let ZZ = self.Z.square();
        let ZZZZ = ZZ.square();
        let lhs = &(&YY - &XX) * &ZZ;
        let rhs = &ZZZZ + &(&constants::EDWARDS_D * &(&XX * &YY));

        lhs == rhs
    }
}

impl ConditionallySelectable for ProjectiveNielsPoint {
    fn conditional_select(a: &Self, b: &Self, choice: Choice) -> Self {
        ProjectiveNielsPoint {
            Y_plus_X: FieldElement::conditional_select(&a.Y_plus_X, &b.Y_plus_X, choice),
            Y_minus_X: FieldElement::conditional_select(&a.Y_minus_X, &b.Y_minus_X, choice),
            Z: FieldElement::conditional_select(&a.Z, &b.Z, choice),
            T2d: FieldElement::conditional_select(&a.T2d, &b.T2d, choice),
        }
    }

    fn conditional_assign(&mut self, other: &Self, choice: Choice) {
        self.Y_plus_X.conditional_assign(&other.Y_plus_X, choice);
        self.Y_minus_X.conditional_assign(&other.Y_minus_X, choice);
        self.Z.conditional_assign(&other.Z, choice);
        self.T2d.conditional_assign(&other.T2d, choice);
    }
}

impl ConditionallySelectable for AffineNielsPoint {
    fn conditional_select(a: &Self, b: &Self, choice: Choice) -> Self {
        AffineNielsPoint {
            y_plus_x: FieldElement::conditional_select(&a.y_plus_x, &b.y_plus_x, choice),
            y_minus_x: FieldElement::conditional_select(&a.y_minus_x, &b.y_minus_x, choice),
            xy2d: FieldElement::conditional_select(&a.xy2d, &b.xy2d, choice),
        }
    }

    fn conditional_assign(&mut self, other: &Self, choice: Choice) {
        self.y_plus_x.conditional_assign(&other.y_plus_x, choice);
        self.y_minus_x.conditional_assign(&other.y_minus_x, choice);
        self.xy2d.conditional_assign(&other.xy2d, choice);
    }
}

impl ProjectivePoint {
    /// Convert from the projective model to the extended model
    pub fn as_extended(&self) -> EdwardsPoint {
        EdwardsPoint {
            X: &self.X * &self.Z,
            Y: &self.Y * &self.Z,
            Z: self.Z.square(),
            T: &self.X * &self.Y,
        }
    }
}

impl CompletedPoint {
    /// Convert from the completed model to the projective model
    pub fn as_projective(&self) -> ProjectivePoint {
        ProjectivePoint {
            X: &self.X * &self.T,
            Y: &self.Y * &self.Z,
            Z: &self.Z * &self.T,
        }
    }

    /// Convert from the completed model to the extended model
    pub fn as_extended(&self) -> EdwardsPoint {
        EdwardsPoint {
            X: &self.X * &self.T,
            Y: &self.Y * &self.Z,
            Z: &self.Z * &self.T,
            T: &self.X * &self.Y,
        }
    }
}

impl ProjectivePoint {
    /// Double this point
    pub fn double(&self) -> CompletedPoint {
        let XX = self.X.square();
        let YY = self.Y.square();
        let ZZ2 = self.Z.square2();
        let X_plus_Y = &self.X + &self.Y;
        let X_plus_Y_sq = X_plus_Y.square();
        let YY_plus_XX = &YY + &XX;
        let YY_minus_XX = &YY - &XX;

        CompletedPoint {
            X: &X_plus_Y_sq - &YY_plus_XX,
            Y: YY_plus_XX,
            Z: YY_minus_XX,
            T: &ZZ2 - &YY_minus_XX,
        }
    }
}

impl Add<&ProjectiveNielsPoint> for &EdwardsPoint {
    type Output = CompletedPoint;

    fn add(self, other: &ProjectiveNielsPoint) -> CompletedPoint {
        let Y_plus_X = &self.Y + &self.X;
        let Y_minus_X = &self.Y - &self.X;
        let PP = &Y_plus_X * &other.Y_plus_X;
        let MM = &Y_minus_X * &other.Y_minus_X;
        let TT2d = &self.T * &other.T2d;
        let ZZ = &self.Z * &other.Z;
        let ZZ2 = &ZZ + &ZZ;

        CompletedPoint {
            X: &PP - &MM,
            Y: &PP + &MM,
            Z: &ZZ2 + &TT2d,
            T: &ZZ2 - &TT2d,
        }
    }
}

impl Sub<&ProjectiveNielsPoint> for &EdwardsPoint {
    type Output = CompletedPoint;

    fn sub(self, other: &ProjectiveNielsPoint) -> CompletedPoint {
        let Y_plus_X = &self.Y + &self.X;
        let Y_minus_X = &self.Y - &self.X;
        let PM = &Y_plus_X * &other.Y_minus_X;
        let MP = &Y_minus_X * &other.Y_plus_X;
        let TT2d = &self.T * &other.T2d;
        let ZZ = &self.Z * &other.Z;
        let ZZ2 = &ZZ + &ZZ;

        CompletedPoint {
            X: &PM - &MP,
            Y: &PM + &MP,
            Z: &ZZ2 - &TT2d,
            T: &ZZ2 + &TT2d,
        }
    }
}

impl Add<&AffineNielsPoint> for &EdwardsPoint {
    type Output = CompletedPoint;

    fn add(self, other: &AffineNielsPoint) -> CompletedPoint {
        let Y_plus_X = &self.Y + &self.X;
        let Y_minus_X = &self.Y - &self.X;
        let PP = &Y_plus_X * &other.y_plus_x;
        let MM = &Y_minus_X * &other.y_minus_x;
        let Txy2d = &self.T * &other.xy2d;
        let Z2 = &self.Z + &self.Z;

        CompletedPoint {
            X: &PP - &MM,
            Y: &PP + &MM,
            Z: &Z2 + &Txy2d,
            T: &Z2 - &Txy2d,
        }
    }
}

impl Sub<&AffineNielsPoint> for &EdwardsPoint {
    type Output = CompletedPoint;

    fn sub(self, other: &AffineNielsPoint) -> CompletedPoint {
        let Y_plus_X = &self.Y + &self.X;
        let Y_minus_X = &self.Y - &self.X;
        let PM = &Y_plus_X * &other.y_minus_x;
        let MP = &Y_minus_X * &other.y_plus_x;
        let Txy2d = &self.T * &other.xy2d;
        let Z2 = &self.Z + &self.Z;

        CompletedPoint {
            X: &PM - &MP,
            Y: &PM + &MP,
            Z: &Z2 - &Txy2d,
            T: &Z2 + &Txy2d,
        }
    }
}

impl Neg for &ProjectiveNielsPoint {
    type Output = ProjectiveNielsPoint;

    fn neg(self) -> ProjectiveNielsPoint {
        ProjectiveNielsPoint {
            Y_plus_X: self.Y_minus_X,
            Y_minus_X: self.Y_plus_X,
            Z: self.Z,
            T2d: -(&self.T2d),
        }
    }
}

impl Neg for &AffineNielsPoint {
    type Output = AffineNielsPoint;

    fn neg(self) -> AffineNielsPoint {
        AffineNielsPoint {
            y_plus_x: self.y_minus_x,
            y_minus_x: self.y_plus_x,
            xy2d: -(&self.xy2d),
        }
    }
}

impl Debug for ProjectivePoint {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "ProjectivePoint{{\n\tX: {:?},\n\tY: {:?},\n\tZ: {:?}\n}}",
            self.X, self.Y, self.Z
        )
    }
}

impl Debug for CompletedPoint {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "CompletedPoint{{\n\tX: {:?},\n\tY: {:?},\n\tZ: {:?},\n\tT: {:?}\n}}",
            self.X, self.Y, self.Z, self.T
        )
    }
}

impl Debug for AffineNielsPoint {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "AffineNielsPoint{{\n\ty_plus_x: {:?},\n\ty_minus_x: {:?},\n\txy2d: {:?}\n}}",
            self.y_plus_x, self.y_minus_x, self.xy2d
        )
    }
}

impl Debug for ProjectiveNielsPoint {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "ProjectiveNielsPoint{{\n\tY_plus_X: {:?},\n\tY_minus_X: {:?},\n\tZ: {:?},\n\tT2d: {:?}\n}}",
            self.Y_plus_X, self.Y_minus_X, self.Z, self.T2d
        )
    }
}
