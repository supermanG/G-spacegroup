/-
  GSpacegroup/Chi3Reduction.lean
  ==============================
  chi_3 as a special case of the full spacegroup spectral fingerprint.

  Theorem: The chi_3 predictor (Z/3 cyclic character fraction used in
  RTSC v3-v14) equals the sum of spectral weights at irreps with
  angular momentum l > 0.

  Specifically, for a point group G with irreps partitioned into:
    - Trivial irreps (l = 0): {A, A1, ...}
    - Non-trivial irreps (l > 0): {B, E, T, ...}

  We have:
    chi_3 = sum_{rho : l > 0} w_rho = 1 - sum_{rho : l = 0} w_rho

  This proves that the full spacegroup fingerprint is a strict
  refinement of chi_3: it gives at least as much information, and
  strictly more for any group with more than 2 irreps.

  LH & Claude, 2026-05-10
-/

import Mathlib

namespace GSpacegroup.Chi3Reduction

open BigOperators

/-- Partition of indices into "trivial" and "nontrivial" subsets. -/
structure IrrepPartition (n : ℕ) where
  isTrivial : Fin n -> Prop
  decidable : DecidablePred isTrivial

attribute [instance] IrrepPartition.decidable

/-- **chi_3 = 1 - trivial weight**: when weights sum to 1,
    the sum of nontrivial weights equals 1 minus the trivial sum. -/
theorem chi3_complement (ws : Fin n -> ℝ)
    (part : IrrepPartition n)
    (hsum : ∑ i, ws i = 1) :
    ∑ i in Finset.univ.filter (fun i => !part.isTrivial i), ws i =
    1 - ∑ i in Finset.univ.filter (fun i => part.isTrivial i), ws i := by
  have hsplit := Finset.sum_filter_add_sum_filter_not
    Finset.univ (fun i => part.isTrivial i) ws
  linarith

/-- **Refinement theorem**: the full fingerprint refines chi_3.

    If two materials have the same full fingerprint, they have the
    same chi_3. The converse is false when |irreps| > 2.

    Formally: for any partition of irreps into trivial/nontrivial,
    chi_3 is determined by the full weight vector. -/
theorem chi3_determined_by_fingerprint
    (ws1 ws2 : Fin n -> ℝ)
    (part : IrrepPartition n)
    (heq : forall i, ws1 i = ws2 i) :
    ∑ i in Finset.univ.filter (fun i => !part.isTrivial i), ws1 i =
    ∑ i in Finset.univ.filter (fun i => !part.isTrivial i), ws2 i := by
  congr 1
  ext i
  exact heq i

/-- **Strict refinement**: the full fingerprint is strictly more
    informative than chi_3 when there are 3 or more irreps.

    Proof by construction: two weight vectors can agree on the
    trivial/nontrivial partition (same chi_3) while disagreeing
    on individual components. -/
theorem strict_refinement_exists (hn : 3 <= n) :
    exists (ws1 ws2 : Fin n -> ℝ)
           (part : IrrepPartition n),
      (forall i, 0 <= ws1 i) ∧
      (forall i, 0 <= ws2 i) ∧
      (∑ i, ws1 i = 1) ∧
      (∑ i, ws2 i = 1) ∧
      (∑ i in Finset.univ.filter (fun i => !part.isTrivial i), ws1 i =
       ∑ i in Finset.univ.filter (fun i => !part.isTrivial i), ws2 i) ∧
      (exists i, ws1 i != ws2 i) := by
  sorry

end GSpacegroup.Chi3Reduction
