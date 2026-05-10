/-
  GSpacegroup/Chi3Reduction.lean
  ==============================
  chi_3 as a special case of the full spacegroup spectral fingerprint.

  LH & Claude, 2026-05-10
-/

import Mathlib

namespace GSpacegroup.Chi3Reduction

open BigOperators

structure IrrepPartition (n : ℕ) where
  isTrivial : Fin n -> Prop
  decidable : DecidablePred isTrivial

attribute [instance] IrrepPartition.decidable

theorem chi3_complement (ws : Fin n -> ℝ)
    (part : IrrepPartition n)
    (hsum : ∑ i, ws i = 1) :
    ∑ i ∈ Finset.univ.filter (fun i => ¬ part.isTrivial i), ws i =
    1 - ∑ i ∈ Finset.univ.filter (fun i => part.isTrivial i), ws i := by
  have hsplit := Finset.sum_filter_add_sum_filter_not
    Finset.univ (fun i => part.isTrivial i) ws
  linarith

theorem chi3_determined_by_fingerprint
    (ws1 ws2 : Fin n -> ℝ)
    (part : IrrepPartition n)
    (heq : forall i, ws1 i = ws2 i) :
    ∑ i ∈ Finset.univ.filter (fun i => ¬ part.isTrivial i), ws1 i =
    ∑ i ∈ Finset.univ.filter (fun i => ¬ part.isTrivial i), ws2 i := by
  congr 1
  ext i
  exact heq i

theorem strict_refinement_exists (hn : 3 <= n) :
    ∃ (ws1 ws2 : Fin n -> ℝ)
      (part : IrrepPartition n),
      (forall i, 0 <= ws1 i) ∧
      (forall i, 0 <= ws2 i) ∧
      (∑ i, ws1 i = 1) ∧
      (∑ i, ws2 i = 1) ∧
      (∑ i ∈ Finset.univ.filter (fun i => ¬ part.isTrivial i), ws1 i =
       ∑ i ∈ Finset.univ.filter (fun i => ¬ part.isTrivial i), ws2 i) ∧
      (∃ i, ws1 i ≠ ws2 i) := by
  sorry

end GSpacegroup.Chi3Reduction
