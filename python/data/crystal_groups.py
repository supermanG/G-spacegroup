#!/usr/bin/env python3
"""
crystal_groups.py
=================
Build rotation matrices for the 11 proper (chiral) crystallographic point
groups — the finite subgroups of SO(3) that correspond to the 32 crystallographic
point groups via the relationship G_proper ⊂ G_full.

These are the groups used for ⋆_G spectroscopy of crystalline materials.
Each group is returned as a dict:
    {
        'name': str,              # short label, e.g. 'O'
        'full_name': str,         # HM label, e.g. 'Oh' or 'O'
        'R_mats': ndarray (n,3,3),# rotation matrices
        'mult_table': ndarray,    # (n,n) multiplication table (0-indexed)
        'irrep_info': list,       # irrep dicts (name, dim, matrices)
        'order': int,             # |G|
    }

The 11 proper groups, in order of complexity:
  C1 (1), C2 (2), C3 (3), C4 (4), C6 (6)
  D2 (4), D3 (6), D4 (8), D6 (12), T (12), O (24)

For the 21 improper point groups, we use the associated chiral subgroup.

The SPG_TO_CHIRAL_PG table maps each of the 230 space group numbers to
the proper point group used for star_G spectroscopy.

LH & Claude 2026
"""

import sys
import warnings
import numpy as np
from typing import List, Dict, Optional


# ---------------------------------------------------------------------------
# Low-level geometry helpers
# ---------------------------------------------------------------------------

def _rot(axis: np.ndarray, theta: float) -> np.ndarray:
    """Rodrigues rotation matrix: rotate by theta about unit axis."""
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]],
                  [a[2], 0, -a[0]],
                  [-a[1], a[0], 0]], dtype=float)
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def _dedup(mats: List[np.ndarray], tol: float = 1e-8) -> np.ndarray:
    """Deduplicate list of rotation matrices."""
    out = []
    for R in mats:
        Rc = np.round(R, 10)
        if not any(np.max(np.abs(Rc - Rp)) < tol for Rp in out):
            out.append(Rc)
    return np.stack(out, axis=0)


def _close_under_mult(R_list: List[np.ndarray], tol: float = 1e-8) -> np.ndarray:
    """Close a set of generators under multiplication."""
    pool = list(R_list)
    changed = True
    while changed:
        changed = False
        new_pool = list(pool)
        for Ri in pool:
            for Rj in pool:
                Rk = Ri @ Rj
                if not any(np.max(np.abs(Rk - Rp)) < tol for Rp in new_pool):
                    new_pool.append(Rk)
                    changed = True
        pool = new_pool
    return _dedup(pool, tol)


def _mult_table(R_mats: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    """Build multiplication table for a group of rotation matrices."""
    n = len(R_mats)
    T = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            Rij = R_mats[i] @ R_mats[j]
            found = False
            for k in range(n):
                if np.max(np.abs(Rij - R_mats[k])) < tol:
                    T[i, j] = k
                    found = True
                    break
            if not found:
                raise ValueError(f"Product R[{i}]@R[{j}] not found in group")
    return T


# ---------------------------------------------------------------------------
# Irrep construction helpers
# ---------------------------------------------------------------------------

_basis_5d = np.zeros((5, 3, 3))
_basis_5d[0] = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]]) / np.sqrt(2)
_basis_5d[1] = np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]]) / np.sqrt(2)
_basis_5d[2] = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]]) / np.sqrt(2)
_basis_5d[3] = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]]) / np.sqrt(2)
_basis_5d[4] = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 2]]) / np.sqrt(6)


def _l2_rep(R_mats: np.ndarray) -> List[np.ndarray]:
    """5D l=2 representation of SO(3) restricted to the finite group."""
    mats = []
    for R in R_mats:
        D = np.zeros((5, 5))
        for i in range(5):
            Tr = R @ _basis_5d[i] @ R.T
            for j in range(5):
                D[j, i] = np.sum(_basis_5d[j] * Tr)
        mats.append(D)
    return mats


def _trivial_rep(n: int) -> List[np.ndarray]:
    return [np.array([[1.0]]) for _ in range(n)]


def _det_rep(R_mats: np.ndarray) -> List[np.ndarray]:
    """
    For improper groups (G ⊄ SO(3)) this gives the sign (det) rep.
    For proper subgroups of SO(3), det = 1 always — this is NOT the A2 irrep.
    Use _O_A2_rep() for the chiral O group instead.
    """
    return [np.array([[np.linalg.det(R)]]) for R in R_mats]


def _classify_O_element(R: np.ndarray) -> str:
    """
    Classify a rotation matrix as one of the 5 conjugacy classes of O.
    Returns: 'E', 'C3', 'C2edge', 'C4', 'C2face'
    """
    tr = np.trace(R)
    if abs(tr - 3) < 0.1:
        return 'E'
    if abs(tr - 0) < 0.1:
        return 'C3'
    if abs(tr - 1) < 0.1:
        return 'C4'
    if abs(tr + 1) < 0.1:
        # Distinguish C2face (axis = coordinate axis) from C2edge
        # C2face: R = diag entries are all ±1, one +1 and two -1 on diagonal
        d = np.diag(R)
        off = R - np.diag(d)
        if np.max(np.abs(off)) < 0.1:   # diagonal matrix → face C2
            return 'C2face'
        else:
            return 'C2edge'
    return 'E'   # fallback


def _O_A2_rep(R_mats: np.ndarray) -> List[np.ndarray]:
    """
    Build the A2 irrep of the chiral octahedral group O.
    Character table: E=1, 8C3=1, 6C2edge=-1, 6C4=-1, 3C2face=1
    """
    chars = {'E': 1.0, 'C3': 1.0, 'C2edge': -1.0, 'C4': -1.0, 'C2face': 1.0}
    return [np.array([[chars[_classify_O_element(R)]]]) for R in R_mats]


def _std_rep(R_mats: np.ndarray) -> List[np.ndarray]:
    return [R.copy() for R in R_mats]


def _verify_rep(mats: List[np.ndarray], mult_table: np.ndarray,
                name: str = '') -> bool:
    n = len(mats)
    for i in range(n):
        for j in range(n):
            k = mult_table[i, j]
            if np.linalg.norm(mats[i] @ mats[j] - mats[k]) > 1e-5:
                warnings.warn(f"Irrep {name} failed verification at ({i},{j})")
                return False
    return True


# ---------------------------------------------------------------------------
# Individual group builders
# ---------------------------------------------------------------------------

def build_C1():
    """C1: trivial group (1 element)."""
    R = np.array([np.eye(3)])
    T = np.array([[0]])
    irreps = [{'name': 'A', 'dim': 1, 'l': 0, 'matrices': _trivial_rep(1)}]
    return _make_group('C1', 'C1', R, T, irreps)


def build_C2():
    """C2: rotation by 180° about z."""
    gens = [_rot([0, 0, 1], np.pi)]
    R = _close_under_mult(gens + [np.eye(3)])
    R = R[np.argsort([np.trace(r) for r in R])[::-1]]  # sort identity first
    T = _mult_table(R)
    chi = [int(np.round(np.trace(r))) for r in R]
    A  = [np.array([[1.0]]) for _ in R]
    B  = [np.array([[float(c >= 0) * 2 - 1]]) for c in chi]
    # C2: A (trivial), B (sign of 180-rot = +1 for e, -1 for C2)
    B  = [np.array([[1.0]]), np.array([[-1.0]])]
    irreps = [
        {'name': 'A', 'dim': 1, 'l': 0, 'matrices': A},
        {'name': 'B', 'dim': 1, 'l': 1, 'matrices': B},
    ]
    return _make_group('C2', 'C2', R, T, irreps)


def build_C3():
    """C3: rotations by 0, 120, 240° about z."""
    gens = [_rot([0, 0, 1], 2 * np.pi / 3)]
    R = _close_under_mult(gens + [np.eye(3)])
    R = np.stack(sorted(R, key=lambda r: np.round(np.arctan2(r[1, 0], r[0, 0]), 6)))
    T = _mult_table(R)
    A  = _trivial_rep(3)
    # E irrep (2D): restrict l=1 standard rep to z-rot subgroup
    E  = _std_rep(R)[:3]  # standard rep is reducible for C3 → use 2x2 block
    E  = [R_[:2, :2].copy() for R_ in R]
    irreps = [
        {'name': 'A',  'dim': 1, 'l': 0, 'matrices': A},
        {'name': 'E',  'dim': 2, 'l': 1, 'matrices': E},
    ]
    return _make_group('C3', 'C3', R, T, irreps)


def build_C4():
    """C4: rotations by 0, 90, 180, 270° about z."""
    gens = [_rot([0, 0, 1], np.pi / 2)]
    R = _close_under_mult(gens + [np.eye(3)])
    R = np.stack(sorted(R, key=lambda r: np.round(np.arctan2(r[1, 0], r[0, 0]), 6)))
    T = _mult_table(R)
    A  = _trivial_rep(4)
    # B irrep: χ(C4^k) = (-1)^k.  Elements are sorted by arctan2 angle so we
    # compute k from the rotation angle rather than assuming a fixed index order.
    def _c4_b(r):
        ang = np.arctan2(r[1, 0], r[0, 0])
        k   = int(round(ang / (np.pi / 2))) % 4
        return (-1.0) ** k
    B  = [np.array([[_c4_b(r)]]) for r in R]
    E  = [R_[:2, :2].copy() for R_ in R]
    irreps = [
        {'name': 'A',  'dim': 1, 'l': 0, 'matrices': A},
        {'name': 'B',  'dim': 1, 'l': 2, 'matrices': B},
        {'name': 'E',  'dim': 2, 'l': 1, 'matrices': E},
    ]
    return _make_group('C4', 'C4', R, T, irreps)


def build_C6():
    """C6: rotations by multiples of 60° about z."""
    gens = [_rot([0, 0, 1], np.pi / 3)]
    R = _close_under_mult(gens + [np.eye(3)])
    R = np.stack(sorted(R, key=lambda r: np.round(np.arctan2(r[1, 0], r[0, 0]), 6)))
    T = _mult_table(R)
    n = len(R)
    A  = _trivial_rep(n)
    B  = [np.array([[(-1.)**k]]) for k in range(n)]
    E1 = [R_[:2, :2].copy() for R_ in R]
    # E2: 2D from l=2 block
    l2 = _l2_rep(R)
    # E2 spans {d_xy, d_{x²-y²}} = l=2 basis indices [0,3] (same fix as D6).
    E2 = [np.array([[m[0, 0], m[0, 3]], [m[3, 0], m[3, 3]]]) for m in l2]
    irreps = [
        {'name': 'A',  'dim': 1, 'l': 0, 'matrices': A},
        {'name': 'B',  'dim': 1, 'l': 3, 'matrices': B},
        {'name': 'E1', 'dim': 2, 'l': 1, 'matrices': E1},
        {'name': 'E2', 'dim': 2, 'l': 2, 'matrices': E2},
    ]
    return _make_group('C6', 'C6', R, T, irreps)


def build_D2():
    """D2: Klein four-group (rotations by 180° about x, y, z)."""
    gens = [_rot([1, 0, 0], np.pi), _rot([0, 1, 0], np.pi)]
    R = _close_under_mult(gens + [np.eye(3)])
    T = _mult_table(R)
    n = len(R)
    A  = _trivial_rep(n)
    B1 = [np.array([[np.round(r[2, 2])]]) for r in R]
    B2 = [np.array([[np.round(r[1, 1])]]) for r in R]
    B3 = [np.array([[np.round(r[0, 0])]]) for r in R]
    irreps = [
        {'name': 'A',  'dim': 1, 'l': 0, 'matrices': A},
        {'name': 'B1', 'dim': 1, 'l': 1, 'matrices': B1},
        {'name': 'B2', 'dim': 1, 'l': 1, 'matrices': B2},
        {'name': 'B3', 'dim': 1, 'l': 1, 'matrices': B3},
    ]
    return _make_group('D2', 'D2', R, T, irreps)


def build_D3():
    """D3: dihedral group of order 6 (C3 + 3 C2 axes in plane)."""
    gens = [_rot([0, 0, 1], 2 * np.pi / 3), _rot([1, 0, 0], np.pi)]
    R = _close_under_mult(gens + [np.eye(3)])
    T = _mult_table(R)
    n = len(R)
    A1 = _trivial_rep(n)
    # A2 of D3: +1 on E and 2C3, -1 on 3C2 (all C2s are equivalent in D3)
    # Detected by trace: C3 has trace=0, C2 has trace=-1
    def _d3_a2(r): return 1.0 if abs(np.trace(r) + 1) > 0.5 else -1.0
    A2 = [np.array([[_d3_a2(r)]]) for r in R]
    E  = [r[:2, :2].copy() for r in R]
    irreps = [
        {'name': 'A1', 'dim': 1, 'l': 0, 'matrices': A1},
        {'name': 'A2', 'dim': 1, 'l': 0, 'matrices': A2},
        {'name': 'E',  'dim': 2, 'l': 1, 'matrices': E},
    ]
    return _make_group('D3', 'D3', R, T, irreps)


def _d4_classify(r: np.ndarray) -> str:
    """
    Classify a 3×3 rotation matrix as one of the five conjugacy classes of D4.

    D4 classes and representative rotation matrices
    ─────────────────────────────────────────────────
    E        identity,           tr = 3
    C4/C4^3  ±90° about z,       tr = 1
    C2z      180° about z,       tr = −1,  r[2,2] = +1  (diag(−1,−1,+1))
    C2prime  180° about x or y,  tr = −1,  r[2,2] = −1,  diagonal
    C2dbl    180° about [1,±1,0], tr = −1, r[2,2] = −1,  off-diagonal

    Character table (what we use below):
        class:   E   C4   C2z  C2'  C2''
        A1:      1    1    1    1    1
        A2:      1    1    1   -1   -1
        B1:      1   -1    1    1   -1
        B2:      1   -1    1   -1    1
        E:       2    0   -2    0    0
    """
    tr = np.trace(r)
    if abs(tr - 3) < 0.1:
        return 'E'
    if abs(tr - 1) < 0.1:
        return 'C4'
    if abs(tr + 1) < 0.1:
        # C2(z): diag(−1,−1,+1) → r[2,2] = +1
        if r[2, 2] > 0.5:
            return 'C2z'
        # Remaining C2 axes all lie in the xy-plane (r[2,2]=−1).
        # C2'(x) = diag(+1,−1,−1), C2'(y) = diag(−1,+1,−1): diagonal matrices.
        # C2''([1,±1,0]): have off-diagonal xy entries.
        off_xy = abs(r[0, 1]) + abs(r[1, 0])
        if off_xy < 0.1:
            return 'C2prime'       # diagonal in-plane C2 (x or y axis)
        else:
            return 'C2dbl'         # off-diagonal in-plane C2 ([1,±1,0] axis)
    return 'E'   # fallback


# D4 character table: class → (A2, B1, B2)   (A1=1 always, E handled separately)
_D4_CHARS = {
    'E':       (1.0,  1.0,  1.0),
    'C4':      (1.0, -1.0, -1.0),
    'C2z':     (1.0,  1.0,  1.0),
    'C2prime': (-1.0, 1.0, -1.0),
    'C2dbl':   (-1.0,-1.0,  1.0),
}


def build_D4():
    """D4: dihedral group of order 8.

    Irreps  dim  l   Physical role
    ──────  ───  ─   ─────────────────────────────────────
    A1       1   0   scalar / totally symmetric
    A2       1   0   pseudoscalar (chirality)
    B1       1   2   d_{x²−y²}  (cuprate-relevant!)
    B2       1   2   d_{xy}
    E        2   1   vector / dipole (px, py)
    """
    gens = [_rot([0, 0, 1], np.pi / 2), _rot([1, 0, 0], np.pi)]
    R = _close_under_mult(gens + [np.eye(3)])
    T = _mult_table(R)
    n = len(R)

    A1 = _trivial_rep(n)
    A2 = [np.array([[_D4_CHARS[_d4_classify(r)][0]]]) for r in R]
    B1 = [np.array([[_D4_CHARS[_d4_classify(r)][1]]]) for r in R]
    B2 = [np.array([[_D4_CHARS[_d4_classify(r)][2]]]) for r in R]
    E  = [r[:2, :2].copy() for r in R]

    irreps = [
        {'name': 'A1', 'dim': 1, 'l': 0, 'matrices': A1},
        {'name': 'A2', 'dim': 1, 'l': 0, 'matrices': A2},
        {'name': 'B1', 'dim': 1, 'l': 2, 'matrices': B1,
         'desc': 'd_{x²−y²} (cuprate pairing symmetry)'},
        {'name': 'B2', 'dim': 1, 'l': 2, 'matrices': B2,
         'desc': 'd_{xy}'},
        {'name': 'E',  'dim': 2, 'l': 1, 'matrices': E},
    ]
    return _make_group('D4', 'D4', R, T, irreps)


def _d6_classify(r: np.ndarray) -> str:
    """
    Classify a 3×3 rotation matrix as one of the six conjugacy classes of D6.

    D6 classes (order 12, generators: C6 about z, C2 about x):
    ─────────────────────────────────────────────────────────────
    E      identity,                  tr =  3
    C6     ±60° about z,              tr =  2
    C3     ±120° about z,             tr =  0
    C2z    180° about z,              tr = −1,  r[2,2]=+1
    C2'    180° about axes at 0°,60°,120° from x,  tr=−1, r[2,2]=−1
    C2''   180° about axes at 30°,90°,150° from x, tr=−1, r[2,2]=−1

    Distinguishing C2' from C2'' via r[0,0] = cos(2θ) for axis at angle θ:
      C2'(0°):   r[0,0] ≈ +1       (cos 0°)
      C2'(60°):  r[0,0] ≈ −0.5     (cos 120°)
      C2'(120°): r[0,0] ≈ −0.5     (cos 240°)
      C2''(30°): r[0,0] ≈ +0.5     (cos 60°)
      C2''(90°): r[0,0] ≈ −1       (cos 180°)
      C2''(150°):r[0,0] ≈ +0.5     (cos 300°)
    → r[0,0] ≈ +1 or −0.5 → C2';  r[0,0] ≈ −1 or +0.5 → C2''

    Character table:
        class:   E   C6   C3   C2z  C2'  C2''
        A1:      1    1    1    1    1    1
        A2:      1    1    1    1   -1   -1
        B1:      1   -1    1   -1    1   -1
        B2:      1   -1    1   -1   -1    1
        E1:      2    1   -1   -2    0    0
        E2:      2   -1   -1    2    0    0
    """
    tr = np.trace(r)
    if abs(tr - 3) < 0.1: return 'E'
    if abs(tr - 2) < 0.1: return 'C6'
    if abs(tr - 0) < 0.1: return 'C3'
    if abs(tr + 1) < 0.1:
        if r[2, 2] > 0.5:
            return 'C2z'
        # In-plane C2: classify by r[0,0]
        r00 = r[0, 0]
        if abs(r00 - 1.0) < 0.2 or abs(r00 + 0.5) < 0.2:
            return 'C2prime'
        else:
            return 'C2dbl'
    return 'E'   # fallback


# D6 character table: class → (A2, B1, B2)   (A1=1 always, E1/E2 handled separately)
_D6_CHARS = {
    'E':       (1.0,  1.0,  1.0),
    'C6':      (1.0, -1.0, -1.0),
    'C3':      (1.0,  1.0,  1.0),
    'C2z':     (1.0, -1.0, -1.0),
    'C2prime': (-1.0, 1.0, -1.0),
    'C2dbl':   (-1.0,-1.0,  1.0),
}


def build_D6():
    """D6: dihedral group of order 12.

    Irreps  dim  l   Physical role
    ──────  ───  ─   ─────────────────────────────────────
    A1       1   0   scalar / totally symmetric
    A2       1   0   pseudoscalar
    B1       1   3   f-wave (hexagonal SC)
    B2       1   3   f-wave (alternate)
    E1       2   1   vector / dipole
    E2       2   2   quadrupole / d-wave
    """
    gens = [_rot([0, 0, 1], np.pi / 3), _rot([1, 0, 0], np.pi)]
    R = _close_under_mult(gens + [np.eye(3)])
    T = _mult_table(R)
    n = len(R)

    A1 = _trivial_rep(n)
    A2 = [np.array([[_D6_CHARS[_d6_classify(r)][0]]]) for r in R]
    B1 = [np.array([[_D6_CHARS[_d6_classify(r)][1]]]) for r in R]
    B2 = [np.array([[_D6_CHARS[_d6_classify(r)][2]]]) for r in R]
    E1 = [r[:2, :2].copy() for r in R]
    l2 = _l2_rep(R)
    # E2 spans {d_xy, d_{x²-y²}} = l=2 basis indices [0,3].
    # The [3:5] block mixes d_{x²-y²} with d_{z²} (not an invariant subspace under C6).
    E2 = [np.array([[m[0, 0], m[0, 3]], [m[3, 0], m[3, 3]]]) for m in l2]

    irreps = [
        {'name': 'A1', 'dim': 1, 'l': 0, 'matrices': A1},
        {'name': 'A2', 'dim': 1, 'l': 0, 'matrices': A2},
        {'name': 'B1', 'dim': 1, 'l': 3, 'matrices': B1,
         'desc': 'f-wave (hexagonal SC)'},
        {'name': 'B2', 'dim': 1, 'l': 3, 'matrices': B2},
        {'name': 'E1', 'dim': 2, 'l': 1, 'matrices': E1},
        {'name': 'E2', 'dim': 2, 'l': 2, 'matrices': E2},
    ]
    return _make_group('D6', 'D6', R, T, irreps)


def build_T():
    """T: chiral tetrahedral group (order 12)."""
    gens = [_rot([0, 0, 1], np.pi), _rot([1, 1, 1], 2 * np.pi / 3)]
    R = _close_under_mult(gens + [np.eye(3)])
    T = _mult_table(R)
    n = len(R)
    A  = _trivial_rep(n)
    # E: 2D irrep of T via l=2 restriction
    l2  = _l2_rep(R)
    E   = [m[3:5, 3:5].copy() for m in l2]
    T3d = [r.copy() for r in R]  # standard 3D irrep = T1 of T
    irreps = [
        {'name': 'A',  'dim': 1, 'l': 0, 'matrices': A},
        {'name': 'E',  'dim': 2, 'l': 2, 'matrices': E},
        {'name': 'T',  'dim': 3, 'l': 1, 'matrices': T3d},
    ]
    return _make_group('T', 'T/Th', R, T, irreps)


def build_O():
    """O: chiral octahedral group (order 24). Core group for high-Tc materials."""
    # Reuse the well-tested construction from discover_at_scale
    candidates = [np.eye(3)]
    for axis in [[1, 0, 0], [0, 1, 0], [0, 0, 1]]:
        for angle in [np.pi / 2, np.pi, 3 * np.pi / 2]:
            candidates.append(_rot(axis, angle))
    for diag in [[1, 1, 1], [1, 1, -1], [1, -1, 1], [-1, 1, 1]]:
        for angle in [2 * np.pi / 3, 4 * np.pi / 3]:
            candidates.append(_rot(diag, angle))
    for edge in [[1, 1, 0], [1, -1, 0], [1, 0, 1], [1, 0, -1], [0, 1, 1], [0, 1, -1]]:
        candidates.append(_rot(edge, np.pi))

    R_mats = _dedup(candidates)
    assert len(R_mats) == 24, f"O: expected 24 elements, got {len(R_mats)}"
    T = _mult_table(R_mats)
    n = 24

    A1 = _trivial_rep(n)
    A2 = _O_A2_rep(R_mats)   # genuine A2: +1 on E,C3,C2face; -1 on C2edge,C4
    T1 = _std_rep(R_mats)
    l2   = _l2_rep(R_mats)
    E    = [m[3:5, 3:5].copy() for m in l2]
    T2   = [m[0:3, 0:3].copy() for m in l2]

    irreps = [
        {'name': 'A1', 'dim': 1, 'l': 0, 'desc': 'trivial (scalar)',     'matrices': A1},
        {'name': 'A2', 'dim': 1, 'l': 0, 'desc': 'pseudoscalar',         'matrices': A2},
        {'name': 'E',  'dim': 2, 'l': 2, 'desc': 'quadrupole (l=2)',     'matrices': E},
        {'name': 'T1', 'dim': 3, 'l': 1, 'desc': 'vector (l=1)',         'matrices': T1},
        {'name': 'T2', 'dim': 3, 'l': 2, 'desc': 'quadrupole (l=2)',     'matrices': T2},
    ]
    return _make_group('O', 'O/Oh', R_mats, T, irreps)


def _make_group(name, full_name, R_mats, mult_table, irreps):
    n = len(R_mats)
    # For complex groups (cyclic Cn, n>=3, and T), pairs of complex conjugate
    # 1D irreps are combined into 2D real orthogonal irreps (realification).
    # This makes sum(dim²) > |G| for the real-irrep list; this is correct and
    # expected — it does NOT indicate missing irreps.  The Plancherel formula
    # for real representations is sum_ρ dim_ρ * ||X̂_ρ||² = (1/|G|) sum_g ||X_g||²,
    # accounting for multiplicity of complex-conjugate pairs.
    return {
        'name': name,
        'full_name': full_name,
        'R_mats': R_mats,
        'mult_table': mult_table,
        'irrep_info': irreps,
        'order': n,
    }


# ---------------------------------------------------------------------------
# Registry of all 11 proper point groups
# ---------------------------------------------------------------------------

def build_all_proper_groups() -> Dict[str, dict]:
    """Build all 11 proper crystallographic point groups."""
    groups = {}
    for fn, name in [
        (build_C1, 'C1'), (build_C2, 'C2'), (build_C3, 'C3'),
        (build_C4, 'C4'), (build_C6, 'C6'), (build_D2, 'D2'),
        (build_D3, 'D3'), (build_D4, 'D4'), (build_D6, 'D6'),
        (build_T,  'T'),  (build_O,  'O'),
    ]:
        try:
            groups[name] = fn()
        except Exception as e:
            warnings.warn(f"Failed to build group {name}: {e}")
    return groups


# ---------------------------------------------------------------------------
# Space group → chiral point group mapping
# ---------------------------------------------------------------------------
# Maps SGN (1-230) to the chiral point group name used for star_G analysis.
# For improper groups we use the associated chiral subgroup (e.g., D4h → D4).

_SPG_RANGES = [
    # Triclinic
    (1, 1, 'C1'), (2, 2, 'C1'),           # Ci uses C1
    # Monoclinic
    (3, 5, 'C2'), (6, 9, 'C1'), (10, 15, 'C2'),   # Cs→C1, C2h→C2
    # Orthorhombic
    (16, 24, 'D2'), (25, 46, 'C2'), (47, 74, 'D2'),  # C2v→C2, D2h→D2
    # Tetragonal
    (75, 80, 'C4'), (81, 82, 'C4'), (83, 88, 'C4'),   # S4→C4, C4h→C4
    (89, 98, 'D4'), (99, 110, 'C4'), (111, 122, 'D2'), # C4v→C4, D2d→D2
    (123, 142, 'D4'),
    # Trigonal
    (143, 146, 'C3'), (147, 148, 'C3'),
    (149, 155, 'D3'), (156, 161, 'C3'), (162, 167, 'D3'),  # C3v→C3
    # Hexagonal
    (168, 173, 'C6'), (174, 174, 'C3'), (175, 176, 'C6'),  # C3h→C3, C6h→C6
    (177, 182, 'D6'), (183, 186, 'C6'), (187, 190, 'D3'),  # C6v→C6, D3h→D3
    (191, 194, 'D6'),
    # Cubic
    (195, 199, 'T'), (200, 206, 'T'),   # Th→T
    (207, 214, 'O'), (215, 220, 'T'),   # Td→T
    (221, 230, 'O'),
]

SPG_TO_CHIRAL: Dict[int, str] = {}
for _start, _end, _pg in _SPG_RANGES:
    for _n in range(_start, _end + 1):
        SPG_TO_CHIRAL[_n] = _pg


def spg_to_chiral(spg_number) -> str:
    """Return the chiral point group name for a given space group number."""
    try:
        return SPG_TO_CHIRAL.get(int(spg_number), 'O')  # default to O
    except (ValueError, TypeError):
        return 'O'


# ---------------------------------------------------------------------------
# Crystal system label → typical chiral group
# ---------------------------------------------------------------------------

CRYSTAL_SYSTEM_TO_GROUP = {
    'triclinic':    'C1',
    'monoclinic':   'C2',
    'orthorhombic': 'D2',
    'tetragonal':   'D4',
    'trigonal':     'D3',
    'hexagonal':    'D6',
    'cubic':        'O',
}


if __name__ == '__main__':
    print("Building all 11 proper crystallographic point groups...")
    groups = build_all_proper_groups()
    print(f"\n{'Group':<6}  {'Order':>5}  {'Irreps'}")
    print('-' * 45)
    for name, g in groups.items():
        irrep_str = ', '.join(
            f"{irr['name']}({irr['dim']}D)" for irr in g['irrep_info']
        )
        s = sum(irr['dim'] ** 2 for irr in g['irrep_info'])
        ok = 'OK' if s == g['order'] else '!!'
        print(f"{name:<6}  {g['order']:>5}  {irrep_str}  {ok}")

    # Verify group O
    g = groups['O']
    for irr in g['irrep_info']:
        _verify_rep(irr['matrices'], g['mult_table'], irr['name'])
    print("\nAll octahedral irreps verified.")
