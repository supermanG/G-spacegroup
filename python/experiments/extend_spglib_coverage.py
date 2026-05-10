#!/usr/bin/env python3
"""
extend_spglib_coverage.py
=========================
Extend spacegroup coverage to Alexandria entries via spglib detection.

Q1 (human-approved): many Alexandria materials lack spg_number in the
JARVIS dft_3d cache. Use spglib to detect spacegroups directly from
crystal structures, expanding our coverage from ~894 to potentially
5000+ materials.

LH & Claude 2026
"""

import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import spglib
    HAS_SPGLIB = True
except ImportError:
    HAS_SPGLIB = False

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent.parent))

from python.data.crystal_groups import spg_to_chiral

CACHE_DIR = Path(r"C:\Users\superman\rtsc\data\jarvis")

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


def detect_spacegroup(lattice, frac_coords, atomic_numbers, symprec=1e-3):
    """
    Detect spacegroup using spglib.

    Parameters
    ----------
    lattice : (3,3) lattice vectors (row format)
    frac_coords : (n_atoms, 3) fractional coordinates
    atomic_numbers : (n_atoms,) atomic numbers

    Returns
    -------
    spg_number : int or None
    spg_symbol : str or None
    """
    if not HAS_SPGLIB:
        return None, None

    cell = (lattice.tolist(), frac_coords.tolist(), atomic_numbers.tolist())
    try:
        dataset = spglib.get_symmetry_dataset(cell, symprec=symprec)
        if dataset is None:
            return None, None
        return dataset['number'], dataset['international']
    except Exception:
        return None, None


def load_existing_spg_map():
    """Load jid -> spg_number from dft_3d_2021 cache."""
    dft_path = CACHE_DIR / "dft_3d_2021_cache.json"
    if not dft_path.exists():
        print(f"  dft_3d cache not found at {dft_path}")
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


def load_all_structures():
    """Load all crystal structures from supercon_3d and alex_supercon."""
    all_entries = {}
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
            frac = atoms.get('coords')
            elems = atoms.get('elements', [])
            if not lat or not frac or not elems:
                continue

            lat_arr = np.array(lat, dtype=float)
            frac_arr = np.array(frac, dtype=float)
            z_arr = np.array([ELEM_TO_Z.get(el, 0) for el in elems])

            if 0 in z_arr:
                continue

            tc_raw = e.get('Tc_supercon', e.get('Tc', 0))
            try:
                tc = float(tc_raw)
            except (ValueError, TypeError):
                tc = 0.0

            formula = e.get('formula', '')
            if not formula and elems:
                from collections import Counter
                ec = Counter(elems)
                formula = ''.join(f"{el}{n if n>1 else ''}" for el, n in sorted(ec.items()))

            all_entries[jid] = {
                'lattice': lat_arr,
                'frac_coords': frac_arr,
                'Z': z_arr,
                'formula': formula,
                'Tc': tc,
                'n_atoms': len(z_arr),
                'src': src,
                'spg_number': e.get('spg_number', None),
            }
            count += 1

        print(f"  {fname}: {count} valid structures")

    return all_entries


def run_extension():
    if not HAS_SPGLIB:
        print("ERROR: spglib not installed. Run: pip install spglib")
        sys.exit(1)

    print("=" * 80)
    print("SPGLIB SPACEGROUP EXTENSION")
    print("=" * 80)

    print("\nStep 1: Load existing spg_number map from dft_3d cache...")
    existing_spg = load_existing_spg_map()
    print(f"  Existing coverage: {len(existing_spg)} materials")

    print("\nStep 2: Load all crystal structures...")
    all_structs = load_all_structures()
    print(f"  Total structures: {len(all_structs)}")

    have_spg = set()
    need_spg = set()
    for jid in all_structs:
        entry_spg = all_structs[jid].get('spg_number')
        if jid in existing_spg:
            have_spg.add(jid)
        elif entry_spg is not None:
            try:
                s = int(entry_spg)
                if 1 <= s <= 230:
                    have_spg.add(jid)
                    existing_spg[jid] = s
                else:
                    need_spg.add(jid)
            except (ValueError, TypeError):
                need_spg.add(jid)
        else:
            need_spg.add(jid)

    print(f"\n  Already have spg_number: {len(have_spg)}")
    print(f"  Need spglib detection: {len(need_spg)}")

    by_src = Counter(all_structs[jid]['src'] for jid in need_spg)
    for src, cnt in sorted(by_src.items()):
        print(f"    {src}: {cnt}")

    print(f"\nStep 3: Detecting spacegroups with spglib (symprec=1e-3)...")
    t0 = time.time()
    detected = {}
    failed = []
    for i, jid in enumerate(sorted(need_spg)):
        s = all_structs[jid]
        spg_n, spg_sym = detect_spacegroup(
            s['lattice'], s['frac_coords'], s['Z']
        )
        if spg_n is not None:
            detected[jid] = {
                'spg_number': spg_n,
                'spg_symbol': spg_sym,
                'point_group': spg_to_chiral(spg_n),
            }
            existing_spg[jid] = spg_n
        else:
            failed.append(jid)

        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            print(f"    {i+1}/{len(need_spg)} done ({elapsed:.1f}s)")

    elapsed = time.time() - t0
    print(f"\n  Detected: {len(detected)}")
    print(f"  Failed: {len(failed)}")
    print(f"  Time: {elapsed:.1f}s")

    pg_dist = Counter()
    for jid in existing_spg:
        if jid in all_structs:
            pg = spg_to_chiral(existing_spg[jid])
            pg_dist[pg] += 1

    print(f"\n  Total coverage now: {len(existing_spg)} materials with spg_number")
    print(f"\n  Point group distribution:")
    for pg, cnt in sorted(pg_dist.items(), key=lambda x: -x[1]):
        print(f"    {pg:>3}: {cnt:5d}")

    d4_jids = [jid for jid in existing_spg
                if spg_to_chiral(existing_spg[jid]) == 'D4'
                and jid in all_structs]
    print(f"\n  D4 materials (target for cuprate analysis): {len(d4_jids)}")
    d4_new = [jid for jid in d4_jids if jid in detected]
    print(f"    Newly detected via spglib: {len(d4_new)}")

    extended_map = {}
    for jid, spg_n in existing_spg.items():
        if jid in all_structs:
            entry = all_structs[jid]
            extended_map[jid] = {
                'spg_number': spg_n,
                'point_group': spg_to_chiral(spg_n),
                'formula': entry['formula'],
                'Tc': entry['Tc'],
                'n_atoms': entry['n_atoms'],
                'src': entry['src'],
                'spglib_detected': jid in detected,
            }

    out_path = SCRIPT_DIR.parent.parent / "data" / "extended_spg_coverage.json"
    with open(out_path, "w") as f:
        json.dump(extended_map, f, indent=2)
    print(f"\nExtended coverage map saved to {out_path}")

    summary = {
        "total_structures": len(all_structs),
        "existing_coverage": len(have_spg),
        "spglib_detected": len(detected),
        "spglib_failed": len(failed),
        "total_coverage": len(existing_spg),
        "point_group_distribution": dict(pg_dist),
        "d4_total": len(d4_jids),
        "d4_new": len(d4_new),
    }
    summary_path = SCRIPT_DIR.parent.parent / "data" / "spglib_extension_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    run_extension()
