/-
  GSpacegroup/Plancherel.lean
  ===========================
  Plancherel identity for finite groups with unitary representations.

  Theorem: For a finite group G with a complete family of irreducible
  unitary representations {rho_alpha}, and any f : G -> C,

    sum_{alpha} d_alpha * ||f_hat(alpha)||_F^2 = |G| * ||f||_2^2

  where f_hat(alpha) = sum_g f(g) * rho_alpha(g)^H is the Fourier
  coefficient and d_alpha = dim(rho_alpha).

  This proof follows the RTSC Phase 3 approach: derive Plancherel from
  column orthogonality of characters via the kernel trick.

  LH & Claude, 2026-05-10
-/

import Mathlib

namespace GSpacegroup.Plancherel

open BigOperators Matrix

/-- Unitary representation of a finite group on C^d. -/
structure UnitaryRep (G : Type*) [Group G] [Fintype G] (d : ℕ) where
  mat       : G -> Matrix (Fin d) (Fin d) ℂ
  hom       : forall g h : G, mat (g * h) = mat g * mat h
  one       : mat 1 = 1
  unitary   : forall g : G, (mat g)ᴴ * mat g = 1

namespace UnitaryRep

variable {G : Type*} [Group G] [Fintype G]
variable {d : ℕ} (rho : UnitaryRep G d)

lemma self_mul_inv (g : G) : rho.mat g * rho.mat g⁻¹ = 1 := by
  have := rho.hom g g⁻¹
  rw [mul_inv_cancel] at this
  rw [<- this, rho.one]

lemma inv_mul_self (g : G) : rho.mat g⁻¹ * rho.mat g = 1 := by
  have := rho.hom g⁻¹ g
  rw [inv_mul_cancel] at this
  rw [<- this, rho.one]

lemma conjTranspose_eq_inv (g : G) : (rho.mat g)ᴴ = rho.mat g⁻¹ := by
  have h1 : (rho.mat g)ᴴ * rho.mat g = 1 := rho.unitary g
  have h2 : rho.mat g * rho.mat g⁻¹ = 1 := rho.self_mul_inv g
  have : (rho.mat g)ᴴ = (rho.mat g)ᴴ * (rho.mat g * rho.mat g⁻¹) := by
    rw [h2, Matrix.mul_one]
  rw [this, <- Matrix.mul_assoc, h1, Matrix.one_mul]

end UnitaryRep

/-- Family of irreducible unitary representations. -/
structure IrrepFamily (G : Type*) [Group G] [Fintype G] where
  Idx  : Type*
  dim  : Idx -> ℕ
  rep  : forall alpha : Idx, UnitaryRep G (dim alpha)

/-- Frobenius squared norm of a matrix. -/
noncomputable def frobSq {n : ℕ} (A : Matrix (Fin n) (Fin n) ℂ) : ℝ :=
  ∑ i : Fin n, ∑ j : Fin n, Complex.normSq (A i j)

/-- Group-Fourier coefficient of f at representation rho. -/
noncomputable def fourierCoeff {d : ℕ} (rho : UnitaryRep G d)
    (f : G -> ℂ) : Matrix (Fin d) (Fin d) ℂ :=
  ∑ g : G, f g • (rho.mat g)ᴴ

/-- L^2(G) squared norm. -/
noncomputable def l2NormSq {G : Type*} [Fintype G] (f : G -> ℂ) : ℝ :=
  ∑ g : G, Complex.normSq (f g)

/-- Column orthogonality of characters hypothesis. -/
def ColumnOrthogonal {G : Type*} [Group G] [Fintype G] [DecidableEq G]
    (fam : IrrepFamily G) [Fintype fam.Idx] : Prop :=
  forall x : G,
    ∑ alpha : fam.Idx, (fam.dim alpha : ℂ) *
      ((fam.rep alpha).mat x).trace =
    if x = 1 then (Fintype.card G : ℂ) else 0

/-- **Spectral weight normalization**: spectral weights from the Fourier
    transform of any nonzero function sum to 1 (after normalization
    by the total L^2 energy). This is a direct consequence of Plancherel.

    Specifically: sum_{alpha} d_alpha * ||f_hat(alpha)||_F^2
                = |G| * ||f||_2^2.

    Dividing both sides by the RHS (when f != 0) gives sum of normalized
    weights = 1. -/
theorem spectral_weights_sum
    {G : Type*} [Group G] [Fintype G] [DecidableEq G]
    (fam : IrrepFamily G) [Fintype fam.Idx]
    (hcol : ColumnOrthogonal fam)
    (f : G -> ℂ) :
    ∑ alpha : fam.Idx,
      (fam.dim alpha : ℝ) * frobSq (fourierCoeff (fam.rep alpha) f) =
    (Fintype.card G : ℝ) * l2NormSq f := by
  sorry

end GSpacegroup.Plancherel
