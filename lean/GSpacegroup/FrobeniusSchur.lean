/-
  GSpacegroup/FrobeniusSchur.lean
  ===============================
  Frobenius-Schur indicator and its role in the Plancherel identity
  for real matrix representations.

  LH & Claude, 2026-05-10
-/

import Mathlib

namespace GSpacegroup.FrobeniusSchur

open BigOperators Matrix

variable {G : Type*} [Group G] [Fintype G] [DecidableEq G]

noncomputable def fsIndicator {d : ℕ}
    (rho : G -> Matrix (Fin d) (Fin d) ℂ) : ℂ :=
  (Fintype.card G : ℂ)⁻¹ * ∑ g : G, (rho (g * g)).trace

omit [DecidableEq G] in
theorem fsIndicator_eq {d : ℕ}
    (rho : G -> Matrix (Fin d) (Fin d) ℂ) :
    fsIndicator rho =
    (Fintype.card G : ℂ)⁻¹ * ∑ g : G, (rho (g * g)).trace := by
  rfl

omit [DecidableEq G] in
theorem fsIndicator_one_dim
    (chi : G -> ℂ)
    (hom : forall g h, chi (g * h) = chi g * chi h) :
    (Fintype.card G : ℂ)⁻¹ * ∑ g : G, chi (g * g) =
    (Fintype.card G : ℂ)⁻¹ * ∑ g : G, chi g * chi g := by
  congr 1
  refine Finset.sum_congr rfl (fun g _ => ?_)
  rw [hom]

theorem fsIndicator_bounded {d : ℕ}
    (rho : G -> Matrix (Fin d) (Fin d) ℂ)
    (hom : forall g h, rho (g * h) = rho g * rho h)
    (one : rho 1 = 1)
    (unitary : forall g, (rho g)ᴴ * rho g = 1) :
    ‖fsIndicator rho‖ <= 1 := by
  sorry

noncomputable def plancherelCorrectionFactor (nu : ℂ) : ℝ :=
  if ‖nu‖ < 1/2 then 1/2 else 1

theorem plancherelCorrectionFactor_pos (nu : ℂ) :
    0 < plancherelCorrectionFactor nu := by
  unfold plancherelCorrectionFactor
  split_ifs <;> norm_num

theorem correction_real_type :
    plancherelCorrectionFactor 1 = 1 := by
  unfold plancherelCorrectionFactor
  simp only [norm_one]
  norm_num

theorem correction_complex_type :
    plancherelCorrectionFactor 0 = 1/2 := by
  unfold plancherelCorrectionFactor
  simp only [norm_zero]
  norm_num

end GSpacegroup.FrobeniusSchur
