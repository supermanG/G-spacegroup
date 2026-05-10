# G-spacegroup

Full star_G spacegroup-Fourier spectroscopy for room-temperature
superconductivity prediction.

## Overview

This project extends the GLuon star_G group-Fourier transform from
point-group (32 classes, Z/3 character) to the FULL 230 crystallographic
spacegroups, giving ~10x more discriminating channels per material.

The chi_3 predictor used in RTSC v3-v14 is a special case: it uses the
Z/3 point-group character. The full spacegroup spectroscopy replaces this
with the complete Plancherel decomposition over the crystal's spacegroup,
capturing screw-axis and glide-plane content invisible to the point-group
projection.

## Relationship to RTSC

This is the **T1.2** task from the RTSC overnight protocol. It is a
standalone publishable extension of GLuon to materials science, targeting
npj Computational Materials, Phys Rev Materials, or Phys Rev Research.

## Structure

```
python/
  data/
    spacegroup_star.py    -- core star_G transform for spacegroups
    crystal_groups.py     -- point-group irrep tables (from RTSC)
    irrep_cache.py        -- Bilbao/spglib irrep table caching
  experiments/
    run_spectroscopy.py   -- compute star_G spectrum for all candidates
    run_orthogonality.py  -- orthogonality matrix analysis
    run_ranking.py        -- re-rank candidates with full spectrum
  tests/
    test_plancherel.py    -- verify Plancherel identity per spacegroup
    test_chi3_recovery.py -- verify chi_3 = point-group reduction
data/
  irrep_tables/           -- cached irrep character tables (JSON)
paper/
  main.tex                -- LaTeX paper draft
```

## Dependencies

- numpy, scipy
- spglib (spacegroup detection and Wyckoff positions)
- networkx (optional, for integration with RTSC knowledge graph)

## Mathematical foundation

For a spacegroup G with irreps {rho}, the star_G Plancherel decomposition
of a structural function sigma on the crystal gives:

    sigma = sum_{rho in G_hat} d_rho * tr(sigma_hat(rho) * rho)

The spectral weight vector (w_rho) is the "spacegroup fingerprint".
chi_3 equals w_rho restricted to the C_3 character irrep of the
point-group quotient.

## References

- Hoyos, Ubaru, Huh, Kalantzis, Clarkson, Kilmer, Avron, Horesh (2025).
  "Group-Algebraic Tensors." Nature submission.
- Horesh (2026). "Hierarchical Group-Algebraic Tensors."
- Horesh (2026). "The star_G luon Fusion Engine."

## License

Proprietary. Coordinate with GLuon IP team before publication.
