# CLAUDE.md for G-spacegroup

## Project context

This is the T1.2 (star_G spacegroup spectroscopy) task from the RTSC
project. It extends the GLuon star_G group-Fourier transform to the
230 crystallographic spacegroups for superconductivity prediction.

## Key files

- `python/data/spacegroup_star.py`: core star_G transform
- `python/data/crystal_groups.py`: 11 proper point groups with irreps
- `python/data/irrep_cache.py`: Bilbao/spglib irrep table caching

## Conventions

- No emdash characters (U+2014) anywhere. Use commas, colons, or periods.
- No `--` digraph used as emdash in prose.
- Numerical reproducibility: any predictor-affecting change must preserve
  the v14 axiom footprint behavior on the calibration set.
- Patent gating: coordinate with GLuon IP team (DEEPMATH-003-PROV) before
  any publication or external sharing.

## Related repos

- RTSC: `C:\Users\superman\rtsc` (parent project, v14 predictor)
- GLuon: `C:\Users\superman\MIT Dropbox\Lior Horesh\Gluon\G\` (star_G source)

## Agent coordination

This repo is maintained by the G_SPACEGROUP agent (T1.2).
The MAIN agent in RTSC handles framework integration (T2.x tasks).
Disagreements are surfaced in `rtsc/roadmap/overnight/SYNTHESIS.md`.
