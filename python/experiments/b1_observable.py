#!/usr/bin/env python3
"""
b1_observable.py
================
Design and test a complementary observable that enhances B1 (d_{x^2-y^2})
spectral weight in D4 materials.

The standard angular spectrum observable is isotropic: it counts bond
preservation regardless of direction. For D4 materials, this gives large B2
(d_{xy}) weights but near-zero B1 (d_{x^2-y^2}) weights. Direct quadrupolar
weighting (cos 2*phi) gives zero by D4 symmetry.

The strain-response observable computes the derivative of the angular
spectrum with respect to a B1-symmetric strain (x stretch, y compress):
    sigma_B1(g) = [f(g; +eps) - f(g; -eps)] / (2*eps)

This derivative naturally transforms as B1, breaking the C2'/C2'' degeneracy
that suppresses B1 in the isotropic observable.

LH & Claude 2026
"""

import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent.parent))

from python.data.crystal_groups import build_all_proper_groups, spg_to_chiral
from python.data.spacegroup_star import (
    starg_fourier_transform,
    spectral_weights,
)
from python.data.irrep_cache import get_irrep_info


def _find_bonds_periodic(coords, Z, lattice, cutoff=3.5, metal_Z=None):
    """Find bonds with periodic boundary conditions."""
    if metal_Z is None:
        metal_Z = set(range(21, 31)) | set(range(39, 49)) | set(range(57, 80))

    metal_mask = np.array([z in metal_Z for z in Z])
    metal_idx = np.where(metal_mask)[0]
    if len(metal_idx) == 0:
        metal_idx = np.arange(len(Z))

    bonds = []
    for di in range(-1, 2):
        for dj in range(-1, 2):
            for dk in range(-1, 2):
                shift = di * lattice[0] + dj * lattice[1] + dk * lattice[2]
                for mi in metal_idx:
                    for j in range(len(Z)):
                        if j == mi and di == 0 and dj == 0 and dk == 0:
                            continue
                        d = (coords[j] + shift) - coords[mi]
                        dist = np.linalg.norm(d)
                        if 0.5 < dist < cutoff:
                            bonds.append(d)
    return bonds


def _angular_spectrum_soft(coords, Z, lattice, R_mats, cutoff=3.5,
                           metal_Z=None, kernel_sigma_sq=0.01):
    """
    Compute angular spectrum with periodic images and soft matching.

    Uses Gaussian kernel matching (no hard threshold) so that
    infinitesimal coordinate changes produce nonzero derivatives.
    This is essential for the strain-response observable.
    """
    n_ops = len(R_mats)
    bonds = _find_bonds_periodic(coords, Z, lattice, cutoff, metal_Z)

    if not bonds:
        return np.ones(n_ops)

    bond_vecs = np.array(bonds)
    sigma = np.zeros(n_ops)
    for op_idx, R in enumerate(R_mats):
        score = 0.0
        for bv in bond_vecs:
            rotated = R @ bv
            diffs = bond_vecs - rotated[None, :]
            min_dist_sq = np.min(np.sum(diffs ** 2, axis=1))
            diffs_neg = bond_vecs + rotated[None, :]
            min_dist_neg_sq = np.min(np.sum(diffs_neg ** 2, axis=1))
            best_sq = min(min_dist_sq, min_dist_neg_sq)
            score += np.exp(-best_sq / kernel_sigma_sq)
        sigma[op_idx] = score / len(bond_vecs)
    return sigma


def _cross_angular_spectrum(bonds_rotate, bonds_ref, R_mats, kernel_sigma_sq=0.01):
    """
    Cross angular spectrum: rotate one bond set, match against another.

    For each group element g, rotates bonds_rotate by R_g and computes
    the Gaussian overlap with bonds_ref. This breaks the self-consistency
    that makes the standard strain response vanish.
    """
    n_ops = len(R_mats)
    if len(bonds_rotate) == 0 or len(bonds_ref) == 0:
        return np.ones(n_ops)

    bv_rot = np.array(bonds_rotate)
    bv_ref = np.array(bonds_ref)

    sigma = np.zeros(n_ops)
    for op_idx, R in enumerate(R_mats):
        score = 0.0
        for bv in bv_rot:
            rotated = R @ bv
            diffs = bv_ref - rotated[None, :]
            min_dist_sq = np.min(np.sum(diffs ** 2, axis=1))
            diffs_neg = bv_ref + rotated[None, :]
            min_dist_neg_sq = np.min(np.sum(diffs_neg ** 2, axis=1))
            best_sq = min(min_dist_sq, min_dist_neg_sq)
            score += np.exp(-best_sq / kernel_sigma_sq)
        sigma[op_idx] = score / len(bv_rot)
    return sigma


def strain_response_observable(
    coords: np.ndarray,
    Z: np.ndarray,
    lattice: np.ndarray,
    point_group: Dict,
    cutoff: float = 3.5,
    metal_Z: Optional[set] = None,
    epsilon: float = 0.02,
    channel: str = "B1",
) -> np.ndarray:
    """
    Strain-response observable for enhanced B1 (or B2) spectral weight.

    Computes the cross-overlap between the rotated STRAINED crystal
    and the ORIGINAL crystal. By comparing strained bonds against
    unstrained reference bonds, the B1-symmetric strain breaks the
    C2'/C2'' degeneracy that suppresses B1 in the isotropic observable.

    sigma(g) = [overlap(R_g . strained+, original) -
                overlap(R_g . strained-, original)] / (2*eps)

    This derivative naturally transforms as B1 under the group.

    Parameters
    ----------
    coords : (n_atoms, 3) Cartesian coordinates
    Z : (n_atoms,) atomic numbers
    lattice : (3, 3) lattice matrix
    point_group : dict with 'R_mats'
    cutoff : bond distance cutoff in Angstroms
    metal_Z : metal center atomic numbers
    epsilon : strain magnitude (default 2%)
    channel : 'B1' for x^2-y^2 strain, 'B2' for xy shear strain

    Returns
    -------
    sigma : (|G|,) strain-response observable values
    """
    R_mats = point_group['R_mats']

    if channel == "B1":
        strain_plus = np.diag([1 + epsilon, 1 - epsilon, 1.0])
        strain_minus = np.diag([1 - epsilon, 1 + epsilon, 1.0])
    elif channel == "B2":
        strain_plus = np.array([[1, epsilon, 0],
                                [epsilon, 1, 0],
                                [0, 0, 1]], dtype=float)
        strain_minus = np.array([[1, -epsilon, 0],
                                 [-epsilon, 1, 0],
                                 [0, 0, 1]], dtype=float)
    else:
        raise ValueError(f"Unknown channel: {channel}")

    bonds_orig = _find_bonds_periodic(coords, Z, lattice, cutoff, metal_Z)

    coords_p = coords @ strain_plus.T
    lat_p = lattice @ strain_plus.T
    bonds_plus = _find_bonds_periodic(coords_p, Z, lat_p, cutoff, metal_Z)

    coords_m = coords @ strain_minus.T
    lat_m = lattice @ strain_minus.T
    bonds_minus = _find_bonds_periodic(coords_m, Z, lat_m, cutoff, metal_Z)

    f_plus = _cross_angular_spectrum(bonds_plus, bonds_orig, R_mats)
    f_minus = _cross_angular_spectrum(bonds_minus, bonds_orig, R_mats)

    sigma = (f_plus - f_minus) / (2 * epsilon)
    return sigma


def load_d4_materials():
    """Load D4 materials with spectroscopy results and v14 data."""
    spec_path = SCRIPT_DIR.parent.parent / "data" / "spectroscopy_results.json"
    v14_path = Path(r"C:\Users\superman\rtsc\roadmap\overnight\v14_candidate_pool.csv")

    with open(spec_path) as f:
        results = json.load(f)
    spec = {r["jid"]: r for r in results if r.get("status") == "ok"}

    v14 = {}
    with open(v14_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            jid = row["jid"]
            try:
                v14[jid] = {
                    "formula": row.get("formula", ""),
                    "Tc_central_K": float(row["Tc_central_K"]) if row["Tc_central_K"] else None,
                    "spg": int(row["spg"]) if row["spg"] else None,
                }
            except (ValueError, KeyError):
                continue

    matched = sorted(set(spec.keys()) & set(v14.keys()))
    d4_mats = [jid for jid in matched if spec[jid].get("pg") == "D4"]
    return d4_mats, spec, v14


def load_crystal_structures():
    """Load crystal structures from RTSC JARVIS cache."""
    cache_dir = Path(r"C:\Users\superman\rtsc\data\jarvis")

    ELEM_TO_Z = {
        'H':1,'He':2,'Li':3,'Be':4,'B':5,'C':6,'N':7,'O':8,'F':9,'Ne':10,
        'Na':11,'Mg':12,'Al':13,'Si':14,'P':15,'S':16,'Cl':17,'Ar':18,
        'K':19,'Ca':20,'Sc':21,'Ti':22,'V':23,'Cr':24,'Mn':25,'Fe':26,
        'Co':27,'Ni':28,'Cu':29,'Zn':30,'Ga':31,'Ge':32,'As':33,'Se':34,
        'Br':35,'Kr':36,'Rb':37,'Sr':38,'Y':39,'Zr':40,'Nb':41,'Mo':42,
        'Tc':43,'Ru':44,'Rh':45,'Pd':46,'Ag':47,'Cd':48,'In':49,'Sn':50,
        'Sb':51,'Te':52,'I':53,'Xe':54,'Cs':55,'Ba':56,'La':57,'Ce':58,
        'Pr':59,'Nd':60,'Pm':61,'Sm':62,'Eu':63,'Gd':64,'Tb':65,'Dy':66,
        'Ho':67,'Er':68,'Tm':69,'Yb':70,'Lu':71,'Hf':72,'Ta':73,'W':74,
        'Re':75,'Os':76,'Ir':77,'Pt':78,'Au':79,'Hg':80,'Tl':81,'Pb':82,
        'Bi':83,
    }

    structures = {}
    for fname in ['supercon_3d.json', 'alex_supercon.json']:
        path = cache_dir / fname
        if not path.exists():
            continue
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        for e in data:
            jid = e.get('jid', '')
            if not jid:
                continue
            atoms = e.get('atoms', {})
            lat = atoms.get('lattice_mat')
            frac = atoms.get('coords')
            elems = atoms.get('elements', [])
            if not lat or not frac or not elems:
                continue
            lat_arr = np.array(lat, dtype=float)
            raw = np.array(frac, dtype=float)
            z_arr = np.array([ELEM_TO_Z.get(el, 0) for el in elems])
            is_cart = atoms.get('cartesian', False)
            cart = raw if is_cart else raw @ lat_arr
            structures[jid] = {
                'coords': cart,
                'Z': z_arr,
                'lattice': lat_arr,
            }

    return structures


def run_b1_analysis():
    d4_mats, spec, v14 = load_d4_materials()
    structures = load_crystal_structures()
    all_groups = build_all_proper_groups()
    d4_group = all_groups['D4']

    print("=" * 80)
    print("B1 COMPLEMENTARY OBSERVABLE ANALYSIS")
    print("Strain-response observable: d/d(eps) f(g; eps_{x^2-y^2})")
    print("=" * 80)

    have_struct = [jid for jid in d4_mats if jid in structures]
    print(f"\nD4 materials with crystal structures: {len(have_struct)} / {len(d4_mats)}")

    irreps = d4_group['irrep_info']

    results = []
    for jid in have_struct:
        s = structures[jid]
        fp_std = spec[jid]["fingerprint"]

        sigma_sr = strain_response_observable(
            s['coords'], s['Z'], s['lattice'],
            d4_group, channel="B1",
        )

        if np.sum(np.abs(sigma_sr)) < 1e-12:
            continue

        sigma_hat = starg_fourier_transform(
            sigma_sr, d4_group['R_mats'], irreps,
        )
        w_sr = spectral_weights(sigma_hat)

        results.append({
            "jid": jid,
            "formula": v14[jid]["formula"],
            "Tc_central_K": v14[jid]["Tc_central_K"],
            "spg": v14[jid]["spg"],
            "std_B1": fp_std.get("B1", 0.0),
            "std_B2": fp_std.get("B2", 0.0),
            "sr_B1": w_sr.get("B1", 0.0),
            "sr_B2": w_sr.get("B2", 0.0),
            "sr_A1": w_sr.get("A1", 0.0),
            "sr_weights": w_sr,
        })

    print(f"Materials with non-zero strain-response signal: {len(results)}")

    print("\n--- COMPARISON: Standard vs Strain-Response B1 weights ---")
    print(f"{'JID':>15}  {'Formula':>12}  {'SPG':>4}  "
          f"{'std_B1':>7}  {'sr_B1':>7}  {'std_B2':>7}  {'sr_B2':>7}  {'Tc':>8}")

    by_sr_b1 = sorted(results, key=lambda x: -x["sr_B1"])
    for e in by_sr_b1[:30]:
        tc = f"{e['Tc_central_K']:8.2f}" if e['Tc_central_K'] is not None else "     N/A"
        print(f"{e['jid']:>15}  {e['formula']:>12}  {e['spg']:4d}  "
              f"{e['std_B1']:7.4f}  {e['sr_B1']:7.4f}  "
              f"{e['std_B2']:7.4f}  {e['sr_B2']:7.4f}  {tc}")

    if not results:
        print("\nNo results to analyze.")
        return

    print("\n--- STATISTICS ---")
    std_b1 = np.array([r["std_B1"] for r in results])
    sr_b1 = np.array([r["sr_B1"] for r in results])
    std_b2 = np.array([r["std_B2"] for r in results])
    sr_b2 = np.array([r["sr_B2"] for r in results])
    tc_vals = np.array([
        r["Tc_central_K"] if r["Tc_central_K"] is not None else np.nan
        for r in results
    ])
    valid = np.isfinite(tc_vals)

    print(f"  Standard B1:       mean={std_b1.mean():.4f}, max={std_b1.max():.4f}")
    print(f"  Strain-resp B1:    mean={sr_b1.mean():.4f}, max={sr_b1.max():.4f}")
    print(f"  Standard B2:       mean={std_b2.mean():.4f}, max={std_b2.max():.4f}")
    print(f"  Strain-resp B2:    mean={sr_b2.mean():.4f}, max={sr_b2.max():.4f}")

    if valid.sum() >= 3:
        corr_std_b1 = np.corrcoef(std_b1[valid], tc_vals[valid])[0, 1]
        corr_sr_b1 = np.corrcoef(sr_b1[valid], tc_vals[valid])[0, 1]
        corr_std_b2 = np.corrcoef(std_b2[valid], tc_vals[valid])[0, 1]
        corr_sr_b2 = np.corrcoef(sr_b2[valid], tc_vals[valid])[0, 1]
        print(f"\n  Tc correlations:")
        print(f"    Standard     B1: r = {corr_std_b1:.4f}")
        print(f"    Strain-resp  B1: r = {corr_sr_b1:.4f}")
        print(f"    Standard     B2: r = {corr_std_b2:.4f}")
        print(f"    Strain-resp  B2: r = {corr_sr_b2:.4f}")

    b1_boost = sr_b1.mean() / max(std_b1.mean(), 1e-10)
    print(f"\n  B1 amplification factor: {b1_boost:.1f}x")

    output = {
        "n_materials": len(results),
        "observable": "strain_response_B1",
        "method": "d/d(eps) f(g; eps_{x^2-y^2}), eps=0.02",
        "b1_amplification": float(b1_boost),
        "statistics": {
            "std_B1_mean": float(std_b1.mean()),
            "sr_B1_mean": float(sr_b1.mean()),
            "std_B2_mean": float(std_b2.mean()),
            "sr_B2_mean": float(sr_b2.mean()),
        },
        "ranking": [
            {
                "jid": r["jid"],
                "formula": r["formula"],
                "spg": r["spg"],
                "Tc_central_K": r["Tc_central_K"],
                "std_B1": r["std_B1"],
                "sr_B1": r["sr_B1"],
                "std_B2": r["std_B2"],
                "sr_B2": r["sr_B2"],
            }
            for r in by_sr_b1
        ],
    }

    out_path = SCRIPT_DIR.parent.parent / "data" / "b1_observable_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    run_b1_analysis()
