#!/usr/bin/env python3
"""
run_expanded_spectroscopy.py
============================
Run star_G spectroscopy on the expanded material pool using spglib-detected
spacegroup numbers for Alexandria entries.

Integrates the extended coverage from extend_spglib_coverage.py with the
existing spectroscopy pipeline from run_spectroscopy.py.

LH & Claude 2026
"""

import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent.parent))

from python.data.crystal_groups import build_all_proper_groups, spg_to_chiral
from python.data.spacegroup_star import (
    starg_fourier_transform,
    spectral_weights,
    spacegroup_fingerprint,
    chi3_from_fingerprint,
    fingerprint_to_feature_vector,
    verify_plancherel,
)
from python.data.irrep_cache import get_irrep_info
from python.experiments.run_spectroscopy import (
    angular_spectrum_observable,
    _NumpyEncoder,
)

RTSC_DIR = Path(r"C:\Users\superman\rtsc")
CACHE_DIR = RTSC_DIR / "data" / "jarvis"
DATA_DIR = SCRIPT_DIR.parent.parent / "data"

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
    'Bi':83,'Po':84,'At':85,'Rn':86,
}


def load_extended_spg_map() -> Dict[str, int]:
    """Load spg numbers from extended coverage (includes spglib detections)."""
    ext_path = DATA_DIR / "extended_spg_coverage.json"
    if not ext_path.exists():
        print(f"  Extended coverage not found at {ext_path}")
        return {}
    with open(ext_path) as f:
        data = json.load(f)
    return {jid: entry['spg_number'] for jid, entry in data.items()}


def load_dft3d_spg_map() -> Dict[str, int]:
    """Load spg numbers from JARVIS dft_3d cache."""
    dft_path = CACHE_DIR / "dft_3d_2021_cache.json"
    if not dft_path.exists():
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
    return spg_map


def load_all_materials() -> List[Dict]:
    """Load all crystal structures from JARVIS and Alexandria."""
    materials = []

    ext_spg = load_extended_spg_map()
    dft_spg = load_dft3d_spg_map()
    print(f"  Extended spg map: {len(ext_spg)} entries")
    print(f"  DFT3D spg map: {len(dft_spg)} entries")

    for fname, src in [('supercon_3d.json', 'jarvis'), ('alex_supercon.json', 'alex')]:
        path = CACHE_DIR / fname
        if not path.exists():
            print(f"  {fname} not found, skipping")
            continue

        with open(path, encoding='utf-8') as f:
            data = json.load(f)

        count = 0
        for e in data:
            jid = e.get('jid', e.get('id', ''))
            if not jid:
                continue

            atoms = e.get('atoms', {})
            lat = atoms.get('lattice_mat')
            raw = atoms.get('coords')
            elems = atoms.get('elements', [])
            if not lat or not raw or not elems:
                continue

            lat_arr = np.array(lat, dtype=float)
            raw_arr = np.array(raw, dtype=float)
            z_arr = np.array([ELEM_TO_Z.get(el, 0) for el in elems])
            if 0 in z_arr:
                continue
            if len(z_arr) > 100:
                continue

            is_cart = atoms.get('cartesian', False)
            cart = raw_arr if is_cart else raw_arr @ lat_arr

            spg = e.get('spg_number', 0)
            try:
                spg = int(spg)
            except (ValueError, TypeError):
                spg = 0

            if spg < 1 or spg > 230:
                spg = ext_spg.get(jid, dft_spg.get(jid, 0))

            tc_raw = e.get('Tc_supercon', e.get('Tc', 0))
            try:
                tc = float(tc_raw)
            except (ValueError, TypeError):
                tc = 0.0

            formula = e.get('formula', '')
            if not formula and elems:
                from collections import Counter
                ec = Counter(elems)
                formula = ''.join(
                    f"{el}{n if n > 1 else ''}"
                    for el, n in sorted(ec.items())
                )

            materials.append({
                'jid': jid,
                'formula': formula,
                'Tc': tc,
                'spg': spg,
                'point_group': spg_to_chiral(spg) if 1 <= spg <= 230 else '',
                'coords': cart,
                'Z': z_arr,
                'lat': lat_arr,
                'n_atoms': len(z_arr),
                'src': src,
            })
            count += 1

        print(f"  {fname}: {count} valid structures loaded")

    return materials


def run_spectroscopy(materials, point_groups, cutoff=3.5):
    """Compute fingerprints for all materials."""
    results = []
    n_total = len(materials)
    t0 = time.time()
    n_ok = 0
    n_fail = 0
    n_bad_spg = 0

    for idx, mat in enumerate(materials):
        spg = mat.get('spg', 0)
        if spg < 1 or spg > 230:
            n_bad_spg += 1
            results.append({
                'jid': mat.get('jid', ''),
                'status': 'bad_spg',
            })
            continue

        pg_name = spg_to_chiral(spg)
        pg = point_groups.get(pg_name)
        if pg is None:
            n_fail += 1
            results.append({
                'jid': mat.get('jid', ''),
                'status': 'no_pg',
            })
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
                'n_atoms': mat.get('n_atoms', 0),
                'src': mat.get('src', ''),
                'status': 'ok',
            })
            n_ok += 1
        except Exception as exc:
            n_fail += 1
            results.append({
                'jid': mat.get('jid', ''),
                'status': f'error: {exc}',
            })

        if (idx + 1) % 1000 == 0:
            elapsed = time.time() - t0
            rate = (idx + 1) / elapsed
            eta = (n_total - idx - 1) / rate
            print(f"  [{idx+1}/{n_total}] "
                  f"{n_ok} ok, {n_fail} fail, {n_bad_spg} bad_spg, "
                  f"{rate:.0f} mat/s, ETA {eta:.0f}s")

    elapsed = time.time() - t0
    print(f"\nDone: {n_ok} ok, {n_fail} fail, {n_bad_spg} bad_spg "
          f"in {elapsed:.1f}s")
    return results


def analyze_expanded(results):
    """Print summary of expanded spectroscopy results."""
    ok = [r for r in results if r.get('status') == 'ok']
    if not ok:
        print("No successful results.")
        return

    print(f"\n{'='*70}")
    print(f"EXPANDED SPECTROSCOPY RESULTS")
    print(f"{'='*70}")
    print(f"Total processed: {len(results)}")
    print(f"Successful: {len(ok)}")
    bad_spg = sum(1 for r in results if r.get('status') == 'bad_spg')
    errors = sum(1 for r in results
                 if r.get('status', '').startswith('error'))
    print(f"Bad spg: {bad_spg}")
    print(f"Errors: {errors}")

    from collections import Counter
    pg_counts = Counter(r['pg'] for r in ok)
    src_counts = Counter(r.get('src', 'unknown') for r in ok)

    print(f"\nBy source:")
    for src, cnt in sorted(src_counts.items(), key=lambda x: -x[1]):
        print(f"  {src}: {cnt}")

    print(f"\nBy point group:")
    for pg, cnt in sorted(pg_counts.items(), key=lambda x: -x[1]):
        print(f"  {pg:>3}: {cnt:>5}")

    chi3_vals = np.array([r['chi3_recovered'] for r in ok])
    print(f"\nchi_3 statistics:")
    print(f"  mean={chi3_vals.mean():.4f}, std={chi3_vals.std():.4f}")
    print(f"  min={chi3_vals.min():.4f}, max={chi3_vals.max():.4f}")

    plancherel_pass = sum(1 for r in ok if r.get('plancherel_ok'))
    print(f"\nPlancherel: {plancherel_pass}/{len(ok)} pass")

    d4_ok = [r for r in ok if r['pg'] == 'D4']
    if d4_ok:
        print(f"\n--- D4 materials: {len(d4_ok)} ---")
        b1_vals = [r['fingerprint'].get('B1', 0) for r in d4_ok]
        b2_vals = [r['fingerprint'].get('B2', 0) for r in d4_ok]
        tc_vals = [r['Tc'] for r in d4_ok if r['Tc'] > 0]
        print(f"  B1: mean={np.mean(b1_vals):.4f}, max={np.max(b1_vals):.4f}")
        print(f"  B2: mean={np.mean(b2_vals):.4f}, max={np.max(b2_vals):.4f}")
        if tc_vals:
            print(f"  Tc (nonzero): mean={np.mean(tc_vals):.1f}K, "
                  f"max={np.max(tc_vals):.1f}K, n={len(tc_vals)}")

    print(f"\nTop 20 by chi_3:")
    by_chi3 = sorted(ok, key=lambda r: -r['chi3_recovered'])
    for r in by_chi3[:20]:
        fp = r['fingerprint']
        top_irreps = sorted(fp.items(), key=lambda x: -x[1])[:3]
        irrep_str = ', '.join(f"{k}={v:.3f}" for k, v in top_irreps)
        print(f"  {r['formula']:>12} ({r['pg']:<3} spg{r['spg']:>3}): "
              f"chi3={r['chi3_recovered']:.4f} Tc={r['Tc']:.1f}K | {irrep_str}")


def main():
    print("=" * 70)
    print("EXPANDED STAR_G SPECTROSCOPY")
    print("(with spglib-detected spacegroups for Alexandria entries)")
    print("=" * 70)

    print("\nBuilding point groups...")
    point_groups = build_all_proper_groups()

    print("\nLoading materials...")
    materials = load_all_materials()
    print(f"Total: {len(materials)} materials")

    has_spg = sum(1 for m in materials if 1 <= m.get('spg', 0) <= 230)
    print(f"With valid spg: {has_spg}")

    print(f"\nRunning spectroscopy (cutoff=3.5 A)...")
    results = run_spectroscopy(materials, point_groups, cutoff=3.5)

    analyze_expanded(results)

    out_path = DATA_DIR / "expanded_spectroscopy_results.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, cls=_NumpyEncoder)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
