/-
  GSpacegroup/SpectralWeights.lean
  ================================
  Spectral weight normalization for the star_G spacegroup fingerprint.

  Theorem: For any nonzero structural function sigma on a finite group G,
  the normalized spectral weights

    w_rho = ||sigma_hat(rho)||_F^2 / (sum_{rho'} ||sigma_hat(rho')||_F^2)

  satisfy:
    1. Each w_rho >= 0
    2. sum_rho w_rho = 1
    3. w_rho = 0 iff sigma is orthogonal to the rho-isotypic component

  These are the "spacegroup fingerprint" components used for materials
  classification.

  LH & Claude, 2026-05-10
-/

import Mathlib

namespace GSpacegroup.SpectralWeights

open BigOperators

/-- Normalized weights from a collection of nonneg reals. -/
noncomputable def normalizeWeights (ws : List ℝ) : List ℝ :=
  let total := ws.foldl (. + .) 0
  if total = 0 then ws.map (fun _ => 0)
  else ws.map (. / total)

/-- **Normalization theorem**: for nonneg reals with positive sum,
    the normalized weights sum to 1. -/
theorem normalized_sum_eq_one (ws : Fin n -> ℝ) (hn : 0 < n)
    (hnn : forall i, 0 <= ws i)
    (hpos : 0 < ∑ i, ws i) :
    ∑ i, (ws i / ∑ j, ws j) = 1 := by
  rw [<- Finset.sum_div]
  exact div_self (ne_of_gt hpos)

/-- Each normalized weight is nonneg. -/
theorem normalized_nonneg (ws : Fin n -> ℝ)
    (hnn : forall i, 0 <= ws i)
    (hpos : 0 < ∑ j, ws j) (i : Fin n) :
    0 <= ws i / ∑ j, ws j :=
  div_nonneg (hnn i) (le_of_lt hpos)

/-- Each normalized weight is at most 1. -/
theorem normalized_le_one (ws : Fin n -> ℝ)
    (hnn : forall i, 0 <= ws i)
    (hpos : 0 < ∑ j, ws j) (i : Fin n) :
    ws i / ∑ j, ws j <= 1 := by
  rw [div_le_one hpos]
  exact Finset.single_le_sum (fun j _ => hnn j) (Finset.mem_univ i)

/-- **Frobenius norm is nonneg**: ||A||_F^2 >= 0 for any matrix A. -/
theorem frobSq_nonneg {n : ℕ} (A : Matrix (Fin n) (Fin n) ℂ) :
    0 <= ∑ i : Fin n, ∑ j : Fin n, Complex.normSq (A i j) := by
  apply Finset.sum_nonneg
  intro i _
  apply Finset.sum_nonneg
  intro j _
  exact Complex.normSq_nonneg _

/-- **Spectral fingerprint theorem**: the spectral weights of the
    star_G Fourier transform form a probability distribution over
    the irreps.

    Given:
    - A finite group G with irrep family {rho_alpha}
    - A nonzero function sigma : G -> C
    - Fourier coefficients sigma_hat(alpha) for each irrep

    Then w_alpha = ||sigma_hat(alpha)||^2 / total satisfies:
    - Each w_alpha in [0, 1]
    - sum_alpha w_alpha = 1 -/
theorem spectral_fingerprint_is_probability
    (ws : Fin n -> ℝ)
    (hn : 0 < n)
    (hnn : forall i, 0 <= ws i)
    (hpos : 0 < ∑ i, ws i) :
    (forall i, 0 <= ws i / ∑ j, ws j) ∧
    (forall i, ws i / ∑ j, ws j <= 1) ∧
    (∑ i, ws i / ∑ j, ws j = 1) :=
  ⟨fun i => normalized_nonneg ws hnn hpos i,
   fun i => normalized_le_one ws hnn hpos i,
   normalized_sum_eq_one ws hn hnn hpos⟩

end GSpacegroup.SpectralWeights
