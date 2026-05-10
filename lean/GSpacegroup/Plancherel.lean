/-
  GSpacegroup/Plancherel.lean
  ===========================
  Plancherel identity for finite groups with unitary representations.

  Theorem: For a finite group G with a complete family of irreducible
  unitary representations {rho_alpha}, and any f : G -> C,

    sum_{alpha} d_alpha * ||f_hat(alpha)||_F^2 = |G| * ||f||_2^2

  where f_hat(alpha) = sum_g f(g) * rho_alpha(g)^H is the Fourier
  coefficient and d_alpha = dim(rho_alpha).

  Proof via kernel trick: expand ||f_hat||^2 as a double sum, apply
  column orthogonality of characters, collapse to L^2 norm.

  Adapted from the RTSC Phase 3 proof (rtsc/lean/RtscPlancherel.lean).

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

/-- Frobenius^2 as a C-valued sum. -/
lemma frobSq_eq_complex_sum {n : ℕ} (A : Matrix (Fin n) (Fin n) ℂ) :
    (frobSq A : ℂ) = ∑ i : Fin n, ∑ j : Fin n, A i j * star (A i j) := by
  unfold frobSq
  push_cast
  refine Finset.sum_congr rfl (fun i _ => ?_)
  refine Finset.sum_congr rfl (fun j _ => ?_)
  rw [Complex.normSq_eq_conj_mul_self, RCLike.star_def, mul_comm]

/-- L^2 norm^2 as a C-valued sum. -/
lemma l2NormSq_eq_complex_sum (f : G -> ℂ) :
    (l2NormSq f : ℂ) = ∑ g : G, f g * star (f g) := by
  unfold l2NormSq
  push_cast
  refine Finset.sum_congr rfl (fun g _ => ?_)
  rw [Complex.normSq_eq_conj_mul_self, RCLike.star_def, mul_comm]

/-- Frobenius^2 equals trace of A * A^H. -/
lemma frobSq_eq_trace_mul_conjTranspose {n : ℕ}
    (A : Matrix (Fin n) (Fin n) ℂ) :
    (frobSq A : ℂ) = (A * Aᴴ).trace := by
  rw [frobSq_eq_complex_sum]
  rw [Matrix.trace]
  refine Finset.sum_congr rfl (fun i _ => ?_)
  rw [Matrix.diag_apply, Matrix.mul_apply]
  refine Finset.sum_congr rfl (fun j _ => ?_)
  rw [Matrix.conjTranspose_apply]

variable {G : Type*} [Group G] [Fintype G] [DecidableEq G]

/-- Kernel form of ||f_hat(rho)||_F^2:
    ||f_hat||_F^2 = sum_{g,h} f(g) * star(f(h)) * chi(g^{-1} h). -/
lemma frobSq_fourierCoeff_kernel
    {d : ℕ} (rho : UnitaryRep G d) (f : G -> ℂ) :
    (frobSq (fourierCoeff rho f) : ℂ) =
    ∑ g : G, ∑ h : G, f g * star (f h) * (rho.mat (g⁻¹ * h)).trace := by
  rw [frobSq_eq_trace_mul_conjTranspose]
  unfold fourierCoeff
  rw [Matrix.conjTranspose_sum, Matrix.sum_mul]
  simp_rw [Matrix.mul_sum]
  rw [Matrix.trace_sum]
  refine Finset.sum_congr rfl (fun g _ => ?_)
  rw [Matrix.trace_sum]
  refine Finset.sum_congr rfl (fun h _ => ?_)
  rw [Matrix.conjTranspose_smul, Matrix.smul_mul, Matrix.mul_smul,
      Matrix.trace_smul, Matrix.trace_smul, smul_eq_mul, smul_eq_mul]
  rw [Matrix.conjTranspose_conjTranspose, rho.conjTranspose_eq_inv,
      <- rho.hom g⁻¹ h]
  ring

/-- **Plancherel identity from column orthogonality.**

    sum_{alpha} d_alpha * ||f_hat(alpha)||_F^2 = |G| * ||f||_2^2

    Proof: expand each ||f_hat||^2 via kernel form, push alpha sum
    inside, apply column orthogonality to collapse to diagonal. -/
theorem spectral_weights_sum
    (fam : IrrepFamily G) [Fintype fam.Idx]
    (hcol : ColumnOrthogonal fam)
    (f : G -> ℂ) :
    ∑ alpha : fam.Idx,
      (fam.dim alpha : ℝ) * frobSq (fourierCoeff (fam.rep alpha) f) =
    (Fintype.card G : ℝ) * l2NormSq f := by
  have hC :
      (∑ alpha : fam.Idx,
        (fam.dim alpha : ℂ) * (frobSq (fourierCoeff (fam.rep alpha) f) : ℂ)) =
      (Fintype.card G : ℂ) * (l2NormSq f : ℂ) := by
    simp_rw [frobSq_fourierCoeff_kernel]
    rw [show (∑ alpha : fam.Idx, (fam.dim alpha : ℂ) *
              ∑ g : G, ∑ h : G,
                f g * star (f h) * ((fam.rep alpha).mat (g⁻¹ * h)).trace) =
            (∑ g : G, ∑ h : G,
              f g * star (f h) *
              (∑ alpha : fam.Idx, (fam.dim alpha : ℂ) *
                ((fam.rep alpha).mat (g⁻¹ * h)).trace)) from by
      simp_rw [Finset.mul_sum]
      rw [Finset.sum_comm]
      refine Finset.sum_congr rfl (fun g _ => ?_)
      rw [Finset.sum_comm]
      refine Finset.sum_congr rfl (fun h _ => ?_)
      refine Finset.sum_congr rfl (fun alpha _ => ?_)
      ring]
    have hcol' : forall g h : G,
        (∑ alpha : fam.Idx, (fam.dim alpha : ℂ) *
          ((fam.rep alpha).mat (g⁻¹ * h)).trace) =
        if g⁻¹ * h = 1 then (Fintype.card G : ℂ) else 0 :=
      fun g h => hcol (g⁻¹ * h)
    simp_rw [hcol']
    rw [l2NormSq_eq_complex_sum]
    rw [Finset.mul_sum]
    refine Finset.sum_congr rfl (fun g _ => ?_)
    rw [Finset.sum_eq_single g]
    · rw [if_pos (by rw [inv_mul_cancel])]
      ring
    · intro h _ hne
      rw [if_neg (by intro habs
                     apply hne
                     have hgh : h = g := by rw [<- mul_one g, <- habs]; group
                     exact hgh)]
      ring
    · intro habs; exact absurd (Finset.mem_univ g) habs
  have hLHSC : (∑ alpha : fam.Idx,
        (fam.dim alpha : ℂ) * (frobSq (fourierCoeff (fam.rep alpha) f) : ℂ)) =
      ((∑ alpha : fam.Idx,
        (fam.dim alpha : ℝ) * frobSq (fourierCoeff (fam.rep alpha) f)) : ℂ) := by
    push_cast; rfl
  have hRHSC : (Fintype.card G : ℂ) * (l2NormSq f : ℂ) =
      (((Fintype.card G : ℝ) * l2NormSq f) : ℂ) := by
    push_cast; rfl
  rw [hLHSC, hRHSC] at hC
  exact_mod_cast hC

end GSpacegroup.Plancherel
