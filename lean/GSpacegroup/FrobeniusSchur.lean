/-
  GSpacegroup/FrobeniusSchur.lean
  ===============================
  Frobenius-Schur indicator and its role in the Plancherel identity
  for real matrix representations.

  For a representation rho of a finite group G, the Frobenius-Schur
  indicator is:

    nu(rho) = (1/|G|) * sum_g chi(g^2)

  where chi is the character. The indicator classifies representations:
    nu = +1  : real type (orthogonal)
    nu =  0  : complex type (unitary, not self-conjugate)
    nu = -1  : quaternionic type (symplectic)

  When using REAL matrix representations (as in crystal_groups.py),
  complex-type irreps appear as 2D real matrices that are the
  "realification" of conjugate complex pairs. The Plancherel
  identity must be adjusted by a factor of 1/2 for these irreps.

  LH & Claude, 2026-05-10
-/

import Mathlib

namespace GSpacegroup.FrobeniusSchur

open BigOperators Matrix

variable {G : Type*} [Group G] [Fintype G] [DecidableEq G]

/-- Frobenius-Schur indicator of a matrix representation. -/
noncomputable def fsIndicator {d : ℕ}
    (rho : G -> Matrix (Fin d) (Fin d) ℂ)
    (mult_table : G -> G -> G) : ℂ :=
  (Fintype.card G : ℂ)⁻¹ * ∑ g : G, (rho (g * g)).trace

/-- The FS indicator of a representation equals (1/|G|) sum_g chi(g^2). -/
theorem fsIndicator_eq {d : ℕ}
    (rho : G -> Matrix (Fin d) (Fin d) ℂ)
    (hom : forall g h, rho (g * h) = rho g * rho h) :
    fsIndicator rho (fun g h => g * h) =
    (Fintype.card G : ℂ)⁻¹ * ∑ g : G, (rho (g * g)).trace := by
  rfl

/-- For a 1D representation (character), the FS indicator simplifies to
    (1/|G|) sum_g chi(g)^2 (since chi(g^2) = chi(g)^2 for 1D reps). -/
theorem fsIndicator_one_dim
    (chi : G -> ℂ)
    (hom : forall g h, chi (g * h) = chi g * chi h) :
    (Fintype.card G : ℂ)⁻¹ * ∑ g : G, chi (g * g) =
    (Fintype.card G : ℂ)⁻¹ * ∑ g : G, chi g * chi g := by
  congr 1
  refine Finset.sum_congr rfl (fun g _ => ?_)
  rw [hom]

/-- Classification theorem: the FS indicator takes values in {-1, 0, 1}
    for irreducible representations over an algebraically closed field.

    We state this as: nu(rho) is real and |nu(rho)| <= 1.
    The exact {-1, 0, 1} classification requires irreducibility. -/
theorem fsIndicator_bounded {d : ℕ}
    (rho : G -> Matrix (Fin d) (Fin d) ℂ)
    (hom : forall g h, rho (g * h) = rho g * rho h)
    (one : rho 1 = 1)
    (unitary : forall g, (rho g)ᴴ * rho g = 1) :
    Complex.abs (fsIndicator rho (fun g h => g * h)) <= 1 := by
  sorry

/-- **Plancherel correction factor**: for a representation with
    Frobenius-Schur indicator nu, the Plancherel contribution
    involves a factor depending on the representation type.

    For real-type (nu = 1): factor = 1
    For complex-type (nu = 0): factor = 1/2
    For quaternionic (nu = -1): factor = 1

    This is because a complex-type real 2D irrep is the realification
    of two conjugate complex 1D irreps, and the Frobenius norm of the
    real matrix is twice the squared absolute value of the complex
    coefficient. -/
noncomputable def plancherelCorrectionFactor (nu : ℂ) : ℝ :=
  if Complex.abs nu < 1/2 then 1/2 else 1

/-- The correction factor is positive. -/
theorem plancherelCorrectionFactor_pos (nu : ℂ) :
    0 < plancherelCorrectionFactor nu := by
  unfold plancherelCorrectionFactor
  split <;> norm_num

/-- For real-type representations (nu = 1), the correction is 1. -/
theorem correction_real_type :
    plancherelCorrectionFactor 1 = 1 := by
  unfold plancherelCorrectionFactor
  simp [Complex.abs_one]

/-- For complex-type representations (nu = 0), the correction is 1/2. -/
theorem correction_complex_type :
    plancherelCorrectionFactor 0 = 1/2 := by
  unfold plancherelCorrectionFactor
  simp [map_zero]

end GSpacegroup.FrobeniusSchur
