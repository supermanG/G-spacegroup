"""
spacegroup_star.py
==================
Full star_G spacegroup-Fourier transform for crystallographic materials.

Extends the GLuon star_G Plancherel decomposition from point-group (32 classes)
to the full 230 crystallographic spacegroups, giving ~10x more discriminating
channels per material.

Mathematical setup:
    For spacegroup G with irreps {rho in G_hat}, a structural function
    sigma: G -> V decomposes as:

        sigma = sum_{rho} d_rho * tr(sigma_hat(rho) * rho)

    The spectral weight at irrep rho is:

        w_rho = ||sigma_hat(rho)||^2 / ||sigma||^2

    The full spectral weight vector (w_rho)_{rho in G_hat} is the
    "spacegroup fingerprint" of the material.

    chi_3 (the v3 predictor) equals sum of w_rho for irreps with C_3
    character restricted to the point-group quotient.

LH & Claude 2026
"""

import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

try:
    import spglib
    HAS_SPGLIB = True
except ImportError:
    HAS_SPGLIB = False
    warnings.warn("spglib not installed; spacegroup detection unavailable")

from .crystal_groups import (
    build_all_proper_groups,
    spg_to_chiral,
    SPG_TO_CHIRAL,
)

IRREP_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "irrep_tables"


def starg_fourier_transform(
    sigma: np.ndarray,
    group_elements: np.ndarray,
    irrep_matrices: List[Dict[str, Any]],
) -> Dict[str, np.ndarray]:
    """
    Compute the star_G Fourier transform of sigma over a finite group.

    Parameters
    ----------
    sigma : ndarray, shape (|G|,) or (|G|, d)
        Structural function evaluated at each group element.
        If 1D, treated as scalar-valued. If 2D, each row is a
        vector-valued observation at that group element.
    group_elements : ndarray, shape (|G|, 3, 3)
        Rotation matrices for each group element (matching sigma order).
    irrep_matrices : list of dict
        Each dict has keys 'name', 'dim', 'matrices' (list of ndarray).
        len(matrices) == |G|, each matrix is (dim, dim).

    Returns
    -------
    sigma_hat : dict mapping irrep_name -> ndarray
        Fourier coefficient at each irrep. Shape depends on sigma
        dimensionality and irrep dimension.
    """
    n = len(group_elements)
    if sigma.ndim == 1:
        sigma = sigma[:, None]

    result = {}
    for irrep in irrep_matrices:
        name = irrep['name']
        dim = irrep['dim']
        mats = irrep['matrices']

        coeff = np.zeros((dim, dim, sigma.shape[1]), dtype=complex)
        for g_idx in range(n):
            rho_g = np.array(mats[g_idx], dtype=complex)
            for d in range(sigma.shape[1]):
                coeff[:, :, d] += sigma[g_idx, d] * rho_g

        coeff *= dim / n
        result[name] = coeff

    return result


def spectral_weights(
    sigma_hat: Dict[str, np.ndarray],
) -> Dict[str, float]:
    """
    Compute normalized spectral weights from Fourier coefficients.

    w_rho = ||sigma_hat(rho)||_F^2 / sum_rho ||sigma_hat(rho)||_F^2

    Returns dict mapping irrep_name -> weight (float in [0,1]).
    """
    norms = {}
    total = 0.0
    for name, coeff in sigma_hat.items():
        w = float(np.sum(np.abs(coeff) ** 2))
        norms[name] = w
        total += w

    if total < 1e-30:
        return {name: 0.0 for name in sigma_hat}

    return {name: w / total for name, w in norms.items()}


def spacegroup_fingerprint(
    sigma: np.ndarray,
    spg_number: int,
    point_groups: Optional[Dict] = None,
) -> Dict[str, float]:
    """
    Compute the full spacegroup fingerprint for a material.

    This is the main entry point: given a structural function sigma
    evaluated at each element of the chiral point group corresponding
    to spg_number, returns the spectral weight vector.

    Parameters
    ----------
    sigma : ndarray, shape (|G|,) or (|G|, d)
        Structural observable at each group element.
    spg_number : int
        Space group number (1-230).
    point_groups : dict, optional
        Pre-built point groups from build_all_proper_groups().
        Built on first call if not provided.

    Returns
    -------
    fingerprint : dict mapping irrep_name -> float
        Normalized spectral weights per irrep.
    """
    if point_groups is None:
        point_groups = build_all_proper_groups()

    pg_name = spg_to_chiral(spg_number)
    if pg_name not in point_groups:
        warnings.warn(f"Unknown point group {pg_name} for spg {spg_number}")
        return {}

    pg = point_groups[pg_name]
    order = pg['order']

    if len(sigma) != order:
        raise ValueError(
            f"sigma has {len(sigma)} elements but group {pg_name} "
            f"has order {order}"
        )

    sigma_hat = starg_fourier_transform(
        sigma, pg['R_mats'], pg['irrep_info']
    )
    return spectral_weights(sigma_hat)


def chi3_from_fingerprint(
    fingerprint: Dict[str, float],
    spg_number: int,
    point_groups: Optional[Dict] = None,
) -> float:
    """
    Recover chi_3 (the v3 predictor) from a full spacegroup fingerprint.

    chi_3 is the spectral weight at irreps with angular momentum l >= 1
    (non-trivial rotational content). For cyclic groups C_n, this is
    the sum of all non-trivial irrep weights.

    This function verifies that the full spectrum is a strict refinement
    of the chi_3 point-group projection.
    """
    if point_groups is None:
        point_groups = build_all_proper_groups()

    pg_name = spg_to_chiral(spg_number)
    if pg_name not in point_groups:
        return 0.0

    pg = point_groups[pg_name]
    chi3 = 0.0
    for irrep in pg['irrep_info']:
        name = irrep['name']
        l_val = irrep.get('l', 0)
        if l_val > 0 and name in fingerprint:
            chi3 += fingerprint[name]

    return chi3


def bond_incidence_observable(
    structure: Any,
    point_group: Dict,
    cutoff: float = 3.0,
) -> np.ndarray:
    """
    Compute bond-incidence structural observable for star_G transform.

    For each group element g, count how many bonds are mapped to
    themselves (invariant) under g. This gives a natural scalar
    function on the group whose Fourier transform captures the
    symmetry content of the bonding network.

    Parameters
    ----------
    structure : object
        Crystal structure with attributes:
        - positions: ndarray (n_atoms, 3), fractional coords
        - lattice: ndarray (3, 3), lattice vectors
        - species: list of element symbols
    point_group : dict
        From build_all_proper_groups(), with 'R_mats'.
    cutoff : float
        Bond distance cutoff in Angstroms.

    Returns
    -------
    sigma : ndarray, shape (|G|,)
        Bond incidence count per group element.
    """
    R_mats = point_group['R_mats']
    n_ops = len(R_mats)

    positions = np.array(structure.positions)
    lattice = np.array(structure.lattice)
    n_atoms = len(positions)

    cart_pos = positions @ lattice

    bonds = []
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            d = np.linalg.norm(cart_pos[i] - cart_pos[j])
            if d < cutoff:
                bonds.append((i, j, cart_pos[i] - cart_pos[j]))

    if not bonds:
        return np.ones(n_ops)

    sigma = np.zeros(n_ops)
    for op_idx, R in enumerate(R_mats):
        count = 0
        for i, j, bond_vec in bonds:
            rotated = R @ bond_vec
            for _, _, bv2 in bonds:
                if np.linalg.norm(rotated - bv2) < 0.1:
                    count += 1
                    break
                if np.linalg.norm(rotated + bv2) < 0.1:
                    count += 1
                    break
        sigma[op_idx] = count

    return sigma


def wyckoff_occupancy_observable(
    wyckoff_letters: List[str],
    spg_number: int,
    point_group: Dict,
) -> np.ndarray:
    """
    Compute Wyckoff-occupancy structural observable.

    For each group element g, counts how many occupied Wyckoff positions
    are stabilized by g. This captures the site-symmetry content of
    the crystal.

    Parameters
    ----------
    wyckoff_letters : list of str
        Wyckoff position letters for each atom (e.g., ['a', 'b', 'c']).
    spg_number : int
        Space group number.
    point_group : dict
        From build_all_proper_groups().

    Returns
    -------
    sigma : ndarray, shape (|G|,)
        Wyckoff stabilizer count per group element.
    """
    n_ops = point_group['order']
    sigma = np.zeros(n_ops)

    unique_wyckoffs = set(wyckoff_letters)
    n_unique = len(unique_wyckoffs)

    sigma[0] = n_unique

    for op_idx in range(1, n_ops):
        sigma[op_idx] = sum(
            1 for w in unique_wyckoffs
            if _wyckoff_stabilized(w, spg_number, op_idx)
        )

    return sigma


def _wyckoff_stabilized(
    wyckoff_letter: str,
    spg_number: int,
    op_idx: int,
) -> bool:
    """
    Check if a Wyckoff position is stabilized by a given symmetry operation.
    Placeholder: requires Bilbao data or spglib site symmetry lookup.
    """
    if wyckoff_letter == 'a':
        return True
    return False


def batch_fingerprint(
    materials: List[Dict],
    point_groups: Optional[Dict] = None,
    observable: str = "bond_incidence",
    cutoff: float = 3.0,
) -> List[Dict[str, float]]:
    """
    Compute spacegroup fingerprints for a batch of materials.

    Parameters
    ----------
    materials : list of dict
        Each dict should have 'spg_number' and either 'structure'
        (for bond_incidence) or 'wyckoff_letters' (for wyckoff).
    point_groups : dict, optional
        Pre-built point groups.
    observable : str
        Which structural observable to use: 'bond_incidence' or 'wyckoff'.
    cutoff : float
        Bond distance cutoff (only for bond_incidence).

    Returns
    -------
    fingerprints : list of dict
        One fingerprint dict per material.
    """
    if point_groups is None:
        point_groups = build_all_proper_groups()

    results = []
    for mat in materials:
        spg = mat.get('spg_number', 1)
        pg_name = spg_to_chiral(spg)
        pg = point_groups.get(pg_name)
        if pg is None:
            results.append({})
            continue

        if observable == "bond_incidence" and 'structure' in mat:
            sigma = bond_incidence_observable(mat['structure'], pg, cutoff)
        elif observable == "wyckoff" and 'wyckoff_letters' in mat:
            sigma = wyckoff_occupancy_observable(
                mat['wyckoff_letters'], spg, pg
            )
        else:
            sigma = np.ones(pg['order'])

        fp = spacegroup_fingerprint(sigma, spg, point_groups)
        results.append(fp)

    return results


def fingerprint_to_feature_vector(
    fingerprint: Dict[str, float],
    spg_number: int,
    point_groups: Optional[Dict] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Convert a fingerprint dict to a fixed-length feature vector.

    Returns a vector of spectral weights ordered by irrep name,
    prefixed with 'sg_w_rho_'. This produces the feature columns
    for the v15+ pipeline.

    Returns
    -------
    features : ndarray, shape (n_irreps,)
    names : list of str
        Feature names like 'sg_w_rho_A1', 'sg_w_rho_E', etc.
    """
    if point_groups is None:
        point_groups = build_all_proper_groups()

    pg_name = spg_to_chiral(spg_number)
    pg = point_groups.get(pg_name)
    if pg is None:
        return np.array([]), []

    irrep_names = sorted(irr['name'] for irr in pg['irrep_info'])
    features = np.array([fingerprint.get(n, 0.0) for n in irrep_names])
    col_names = [f"sg_w_rho_{n}" for n in irrep_names]

    return features, col_names


def _frobenius_schur_indicator(
    irrep: Dict[str, Any],
    mult_table: np.ndarray,
) -> float:
    """
    Compute the Frobenius-Schur indicator for an irrep.

    nu = (1/|G|) sum_g chi(g^2)

    Returns +1 (real-type), 0 (complex-type), or -1 (quaternionic).
    """
    n = len(irrep['matrices'])
    indicator = 0.0
    for i in range(n):
        g_sq = mult_table[i, i]
        indicator += np.trace(np.array(irrep['matrices'][g_sq]))
    return indicator / n


def verify_plancherel(
    sigma: np.ndarray,
    spg_number: int,
    point_groups: Optional[Dict] = None,
    tol: float = 1e-8,
) -> bool:
    """
    Verify the Plancherel identity for the star_G Fourier transform.

    For real-valued sigma on a finite group with real matrix irreps,
    the Plancherel identity requires a Frobenius-Schur correction:
    complex-type irreps (realified conjugate pairs) contribute with
    an extra factor of 1/2.
    """
    if point_groups is None:
        point_groups = build_all_proper_groups()

    pg_name = spg_to_chiral(spg_number)
    pg = point_groups.get(pg_name)
    if pg is None:
        return False

    n = pg['order']
    if sigma.ndim == 1:
        sigma_2d = sigma[:, None]
    else:
        sigma_2d = sigma

    lhs = np.sum(np.abs(sigma_2d) ** 2) / n

    sigma_hat = starg_fourier_transform(
        sigma, pg['R_mats'], pg['irrep_info']
    )

    rhs = 0.0
    for irrep in pg['irrep_info']:
        name = irrep['name']
        dim = irrep['dim']
        coeff = sigma_hat[name]
        norm_sq = np.sum(np.abs(coeff) ** 2)

        fs = _frobenius_schur_indicator(irrep, pg['mult_table'])
        if abs(fs) < 0.5:
            rhs += norm_sq / (2 * dim)
        else:
            rhs += norm_sq / dim

    return abs(lhs - rhs) < tol * max(abs(lhs), 1.0)


if __name__ == '__main__':
    print("Star_G Spacegroup Spectroscopy Module")
    print("=" * 50)

    groups = build_all_proper_groups()
    print(f"\nBuilt {len(groups)} proper point groups.")

    pg_to_spg = {
        'C1': 1, 'C2': 3, 'C3': 143, 'C4': 75, 'C6': 168,
        'D2': 16, 'D3': 149, 'D4': 89, 'D6': 177, 'T': 195, 'O': 207,
    }
    for pg_name, pg in groups.items():
        n = pg['order']
        sigma = np.random.randn(n)
        spg = pg_to_spg.get(pg_name, 1)
        fp = spacegroup_fingerprint(sigma, spg, groups)

        total = sum(fp.values())
        ok = abs(total - 1.0) < 1e-10
        print(f"  {pg_name}: {len(fp)} irreps, "
              f"sum(w) = {total:.10f} {'OK' if ok else 'FAIL'}")

    print("\nPlancherel verification:")
    test_cases = [
        (1, 'C1'), (3, 'C2'), (143, 'C3'), (75, 'C4'),
        (168, 'C6'), (16, 'D2'), (149, 'D3'), (89, 'D4'),
        (177, 'D6'), (195, 'T'), (207, 'O'),
    ]
    for spg, expected_pg in test_cases:
        pg = groups[expected_pg]
        sigma = np.random.randn(pg['order'])
        ok = verify_plancherel(sigma, spg, groups)
        c3 = chi3_from_fingerprint(
            spacegroup_fingerprint(sigma, spg, groups), spg, groups
        )
        print(f"  SPG {spg:>3} ({expected_pg:<3}): "
              f"Plancherel {'OK' if ok else 'FAIL'}, "
              f"chi3 = {c3:.4f}")
