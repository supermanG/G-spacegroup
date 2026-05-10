#!/usr/bin/env python3
"""
run_spectroscopy.py
===================
Compute the full star_G spacegroup spectrum for RTSC candidate materials.

This is the main experiment for T1.2: replaces the chi_3 point-group
Z/3 character with the complete Plancherel decomposition over each
material's crystallographic spacegroup.

Usage:
    python -m python.experiments.run_spectroscopy [--rtsc-dir PATH]

Requires the RTSC data cache (JARVIS supercon_3d + Alexandria + dft_3d).

LH & Claude 2026
"""

import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from python.data.crystal_groups import build_all_proper_groups, spg_to_chiral
from python.data.spacegroup_star import (
    starg_fourier_transform,
    spectral_weights,
    spacegroup_fingerprint,
    chi3_from_fingerprint,
    fingerprint_to_feature_vector,
    verify_plancherel,
)
from python.data.irrep_cache import get_point_groups, get_irrep_info


def angular_spectrum_observable(
    coords: np.ndarray,
    Z: np.ndarray,
    lattice: np.ndarray,
    point_group: Dict,
    cutoff: float = 3.5,
    metal_Z: Optional[set] = None,
) -> np.ndarray:
    """
    Compute the angular spectrum structural observable for star_G transform.

    This matches the approach used for chi_3 in the RTSC predictor:
    for each group element g, measure how much the bonding network
    is preserved under g. This naturally captures the symmetry content
    of the metal-ligand coordination.

    Parameters
    ----------
    coords : (n_atoms, 3) Cartesian coordinates
    Z : (n_atoms,) atomic numbers
    lattice : (3, 3) lattice matrix
    point_group : dict from build_all_proper_groups()
    cutoff : bond distance cutoff in Angstroms
    metal_Z : set of atomic numbers to treat as "metal" centers.
        If None, uses transition metals + lanthanides.

    Returns
    -------
    sigma : (|G|,) structural observable values
    """
    if metal_Z is None:
        metal_Z = set(range(21, 31)) | set(range(39, 49)) | set(range(57, 80))

    R_mats = point_group['R_mats']
    n_ops = len(R_mats)

    metal_mask = np.array([z in metal_Z for z in Z])
    metal_idx = np.where(metal_mask)[0]

    if len(metal_idx) == 0:
        metal_idx = np.arange(len(Z))

    bonds = []
    for mi in metal_idx:
        for j in range(len(Z)):
            if j == mi:
                continue
            d = coords[j] - coords[mi]
            dist = np.linalg.norm(d)
            if 0.5 < dist < cutoff:
                bonds.append(d)

    if not bonds:
        return np.ones(n_ops)

    bond_vecs = np.array(bonds)

    sigma = np.zeros(n_ops)
    for op_idx, R in enumerate(R_mats):
        score = 0.0
        for bv in bond_vecs:
            rotated = R @ bv
            diffs = bond_vecs - rotated[None, :]
            min_dist = np.min(np.linalg.norm(diffs, axis=1))
            diffs_neg = bond_vecs + rotated[None, :]
            min_dist_neg = np.min(np.linalg.norm(diffs_neg, axis=1))
            best = min(min_dist, min_dist_neg)
            if best < 0.3:
                score += 1.0
            else:
                score += np.exp(-best ** 2 / 0.5)
        sigma[op_idx] = score / len(bond_vecs)

    return sigma


def load_rtsc_materials(rtsc_dir: Path) -> List[Dict]:
    """
    Load RTSC candidate materials from the JARVIS cache.

    Tries to import jarvis_sc.py from the RTSC project, falling back
    to direct JSON loading if the import fails.
    """
    jarvis_path = rtsc_dir / "python" / "data" / "jarvis_sc.py"
    cache_dir = rtsc_dir / "data" / "jarvis"

    sys.path.insert(0, str(rtsc_dir))

    try:
        import importlib
        jarvis_mod = importlib.import_module('python.data.jarvis_sc')
        load_jarvis_supercon_3d = jarvis_mod.load_jarvis_supercon_3d
        load_alex_supercon = jarvis_mod.load_alex_supercon
        _parse_atoms = jarvis_mod._parse_atoms
        spg_to_pg = jarvis_mod.spg_to_pg

        print("Loading JARVIS supercon_3d...")
        sc3d = load_jarvis_supercon_3d(cache_dir=str(cache_dir))
        print(f"  Loaded {len(sc3d)} entries.")

        print("Loading Alexandria SuperConDB...")
        alex = load_alex_supercon(cache_dir=str(cache_dir))
        print(f"  Loaded {len(alex)} entries.")

        materials = []
        for src_name, entries in [('jarvis', sc3d), ('alex', alex)]:
            for e in entries:
                parsed = _parse_atoms(e)
                if parsed is None:
                    continue
                coords, Z, lat = parsed
                if len(Z) > 100:
                    continue

                spg = e.get('spg_number', e.get('spg', 0))
                try:
                    spg = int(spg)
                except (ValueError, TypeError):
                    spg = 0

                tc_raw = e.get('Tc_supercon', e.get('Tc', 0))
                try:
                    tc = float(tc_raw)
                except (ValueError, TypeError):
                    tc = 0.0

                materials.append({
                    'jid': e.get('jid', ''),
                    'formula': e.get('formula', ''),
                    'Tc': tc,
                    'spg': spg,
                    'point_group': spg_to_pg(spg),
                    'coords': coords,
                    'Z': Z,
                    'lat': lat,
                    'n_atoms': len(Z),
                    'src': src_name,
                })

        print(f"Total materials with valid structure: {len(materials)}")
        return materials

    except Exception as exc:
        print(f"Could not load RTSC data: {exc}")
        print("Falling back to direct JSON loading...")
        return _load_from_json(cache_dir)


def _load_spg_lookup(cache_dir: Path) -> Dict[str, int]:
    """Load jid -> spg_number map from dft_3d cache."""
    dft_path = cache_dir / "dft_3d_2021_cache.json"
    if not dft_path.exists():
        print(f"  WARNING: {dft_path} not found, spg lookup will be empty")
        return {}

    with open(dft_path, encoding='utf-8') as f:
        data = json.load(f)

    spg_map = {}
    for e in data:
        jid = e.get('jid')
        spg_n = e.get('spg_number')
        if jid and spg_n:
            try:
                spg_map[jid] = int(spg_n)
            except (ValueError, TypeError):
                pass

    print(f"  Loaded {len(spg_map)} spg_number entries from dft_3d cache")
    return spg_map


def _load_from_json(cache_dir: Path) -> List[Dict]:
    """Direct JSON loading fallback."""
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
        'Bi':83,'Po':84,'At':85,'Rn':86,'Fr':87,'Ra':88,'Ac':89,'Th':90,
        'Pa':91,'U':92,'Np':93,'Pu':94,
    }

    spg_lookup = _load_spg_lookup(cache_dir)

    materials = []
    for fname in ['supercon_3d.json', 'alex_supercon.json']:
        path = cache_dir / fname
        if not path.exists():
            print(f"  {path} not found, skipping.")
            continue

        with open(path) as f:
            data = json.load(f)

        print(f"  Loaded {len(data)} entries from {fname}")
        for e in data:
            atoms = e.get('atoms', {})
            elements = atoms.get('elements', [])
            raw_coords = atoms.get('coords', [])
            lattice = atoms.get('lattice_mat', [])

            if not elements or not raw_coords or not lattice:
                continue

            try:
                lat = np.array(lattice, dtype=float)
                raw = np.array(raw_coords, dtype=float)
                if lat.shape != (3, 3) or raw.ndim != 2:
                    continue
                is_cart = atoms.get('cartesian', False)
                coords = raw if is_cart else raw @ lat
                Z = np.array([ELEM_TO_Z.get(el, 0) for el in elements])
                if np.any(Z == 0):
                    continue
            except (ValueError, TypeError):
                continue

            jid = e.get('jid', '')
            spg = e.get('spg_number', e.get('spg', 0))
            try:
                spg = int(spg)
            except (ValueError, TypeError):
                spg = 0
            if spg == 0 and jid:
                spg = spg_lookup.get(jid, 0)

            tc_raw = e.get('Tc_supercon', e.get('Tc', 0))
            try:
                tc = float(tc_raw)
            except (ValueError, TypeError):
                tc = 0.0

            materials.append({
                'jid': e.get('jid', ''),
                'formula': e.get('formula', ''),
                'Tc': tc,
                'spg': spg,
                'coords': coords,
                'Z': Z,
                'lat': lat,
                'n_atoms': len(Z),
            })

    print(f"Total: {len(materials)} materials")
    return materials


def compute_all_fingerprints(
    materials: List[Dict],
    point_groups: Dict,
    cutoff: float = 3.5,
) -> List[Dict]:
    """
    Compute star_G spacegroup fingerprints for all materials.

    Returns list of result dicts, one per material.
    """
    results = []
    n_total = len(materials)
    t0 = time.time()
    n_ok = 0
    n_fail = 0

    for idx, mat in enumerate(materials):
        spg = mat.get('spg', 0)
        if spg < 1 or spg > 230:
            n_fail += 1
            results.append({'jid': mat.get('jid', ''), 'status': 'bad_spg'})
            continue

        pg_name = spg_to_chiral(spg)
        pg = point_groups.get(pg_name)
        if pg is None:
            n_fail += 1
            results.append({'jid': mat.get('jid', ''), 'status': 'no_pg'})
            continue

        try:
            sigma = angular_spectrum_observable(
                mat['coords'], mat['Z'], mat['lat'],
                pg, cutoff=cutoff,
            )
            fp = spacegroup_fingerprint(sigma, spg, point_groups)
            chi3 = chi3_from_fingerprint(fp, spg, point_groups)
            feat_vec, feat_names = fingerprint_to_feature_vector(
                fp, spg, point_groups
            )
            plancherel_ok = verify_plancherel(sigma, spg, point_groups)

            results.append({
                'jid': mat.get('jid', ''),
                'formula': mat.get('formula', ''),
                'Tc': mat.get('Tc', 0.0),
                'spg': spg,
                'pg': pg_name,
                'fingerprint': fp,
                'chi3_recovered': chi3,
                'feature_vector': feat_vec.tolist(),
                'feature_names': feat_names,
                'plancherel_ok': plancherel_ok,
                'status': 'ok',
            })
            n_ok += 1
        except Exception as exc:
            n_fail += 1
            results.append({
                'jid': mat.get('jid', ''),
                'status': f'error: {exc}',
            })

        if (idx + 1) % 500 == 0:
            elapsed = time.time() - t0
            rate = (idx + 1) / elapsed
            eta = (n_total - idx - 1) / rate
            print(f"  [{idx+1}/{n_total}] "
                  f"{n_ok} ok, {n_fail} fail, "
                  f"{rate:.0f} mat/s, ETA {eta:.0f}s")

    elapsed = time.time() - t0
    print(f"\nDone: {n_ok} ok, {n_fail} fail in {elapsed:.1f}s")
    return results


def analyze_results(results: List[Dict], point_groups: Dict):
    """Print summary analysis of spectroscopy results."""
    ok_results = [r for r in results if r.get('status') == 'ok']
    if not ok_results:
        print("No successful results to analyze.")
        return

    print(f"\n{'='*60}")
    print(f"Star_G Spacegroup Spectroscopy Results")
    print(f"{'='*60}")
    print(f"Materials processed: {len(results)}")
    print(f"Successful: {len(ok_results)}")
    print(f"Failed: {len(results) - len(ok_results)}")

    pg_counts = {}
    for r in ok_results:
        pg = r.get('pg', 'unknown')
        pg_counts[pg] = pg_counts.get(pg, 0) + 1

    print(f"\nPoint group distribution:")
    for pg, count in sorted(pg_counts.items(), key=lambda x: -x[1]):
        info = get_irrep_info(next(
            s for s in range(1, 231) if spg_to_chiral(s) == pg
        ))
        print(f"  {pg:<4}: {count:>5} materials, "
              f"{info['n_irreps']} irreps")

    chi3_vals = [r['chi3_recovered'] for r in ok_results]
    print(f"\nchi_3 (recovered from full spectrum):")
    print(f"  mean = {np.mean(chi3_vals):.4f}")
    print(f"  std  = {np.std(chi3_vals):.4f}")
    print(f"  min  = {np.min(chi3_vals):.4f}")
    print(f"  max  = {np.max(chi3_vals):.4f}")

    plancherel_pass = sum(1 for r in ok_results if r.get('plancherel_ok'))
    print(f"\nPlancherel verification: {plancherel_pass}/{len(ok_results)} pass")

    print(f"\nTop 20 by chi_3:")
    sorted_by_chi3 = sorted(ok_results, key=lambda r: -r['chi3_recovered'])
    for r in sorted_by_chi3[:20]:
        fp = r['fingerprint']
        irrep_str = ', '.join(
            f"{k}={v:.3f}" for k, v in sorted(fp.items()) if v > 0.01
        )
        print(f"  {r['formula']:>12} (SPG {r['spg']:>3}, {r['pg']:<3}): "
              f"chi3={r['chi3_recovered']:.4f} | {irrep_str}")

    print(f"\nIrrep weight distribution (across all materials):")
    all_irrep_weights = {}
    for r in ok_results:
        for name, w in r['fingerprint'].items():
            if name not in all_irrep_weights:
                all_irrep_weights[name] = []
            all_irrep_weights[name].append(w)

    for name in sorted(all_irrep_weights.keys()):
        ws = all_irrep_weights[name]
        print(f"  {name:<4}: mean={np.mean(ws):.4f}, "
              f"std={np.std(ws):.4f}, "
              f"max={np.max(ws):.4f}, "
              f"n={len(ws)}")


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def save_results(results: List[Dict], output_path: Path):
    """Save results to JSON."""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, cls=_NumpyEncoder)
    print(f"Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Star_G spacegroup spectroscopy for RTSC candidates"
    )
    parser.add_argument(
        '--rtsc-dir', type=Path,
        default=Path(r'C:\Users\superman\rtsc'),
        help='Path to the RTSC project root'
    )
    parser.add_argument(
        '--cutoff', type=float, default=3.5,
        help='Bond distance cutoff in Angstroms'
    )
    parser.add_argument(
        '--output', type=Path,
        default=Path('data/spectroscopy_results.json'),
        help='Output path for results JSON'
    )
    parser.add_argument(
        '--max-materials', type=int, default=0,
        help='Max materials to process (0 = all)'
    )
    args = parser.parse_args()

    print("Star_G Spacegroup Spectroscopy")
    print("=" * 50)

    print("\nBuilding point groups...")
    point_groups = build_all_proper_groups()
    print(f"  {len(point_groups)} proper point groups ready.")

    print(f"\nLoading materials from {args.rtsc_dir}...")
    materials = load_rtsc_materials(args.rtsc_dir)

    if args.max_materials > 0:
        materials = materials[:args.max_materials]
        print(f"  Limited to {len(materials)} materials.")

    print(f"\nComputing star_G spectra (cutoff={args.cutoff} A)...")
    results = compute_all_fingerprints(materials, point_groups, args.cutoff)

    analyze_results(results, point_groups)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_results(results, args.output)


if __name__ == '__main__':
    main()
