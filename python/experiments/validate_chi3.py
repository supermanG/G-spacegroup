#!/usr/bin/env python3
"""
validate_chi3.py
================
Cross-validate chi_3 recovery from full spacegroup fingerprints
against the v14 predictor's z3 values.

The v14 z3 is computed from angular distribution of atoms projected onto
a plane perpendicular to the principal axis (see rtsc/python/z3_features.py).
Our chi3_recovered is computed from the full star_G spectral decomposition
using the angular spectrum observable.

These should be correlated but not identical, because:
1. v14 z3 uses a 2D angular projection; we use 3D bond geometry
2. v14 z3 uses a Z/3 cyclic restriction; we use the full point group
3. The observables are slightly different (angular distribution vs bond preservation)

This script quantifies the relationship.

LH & Claude 2026
"""

import csv
import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent


def load_spectroscopy_results(path: Path) -> dict:
    """Load our spectroscopy results keyed by jid."""
    with open(path) as f:
        results = json.load(f)
    return {r["jid"]: r for r in results if r.get("status") == "ok"}


def load_v14_z3(path: Path) -> dict:
    """Load v14 z3 values keyed by jid."""
    v14 = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            jid = row["jid"]
            z3_str = row.get("z3", "")
            if z3_str:
                try:
                    v14[jid] = float(z3_str)
                except ValueError:
                    continue
    return v14


def run_validation():
    spec_path = SCRIPT_DIR.parent.parent / "data" / "spectroscopy_results.json"
    v14_path = Path(r"C:\Users\superman\rtsc\roadmap\overnight\v14_candidate_pool.csv")

    if not spec_path.exists():
        print(f"ERROR: spectroscopy results not found at {spec_path}")
        sys.exit(1)
    if not v14_path.exists():
        print(f"ERROR: v14 CSV not found at {v14_path}")
        sys.exit(1)

    print("Loading spectroscopy results...")
    spec = load_spectroscopy_results(spec_path)
    print(f"  {len(spec)} materials with OK status")

    print("Loading v14 z3 values...")
    v14 = load_v14_z3(v14_path)
    print(f"  {len(v14)} materials with z3 values")

    matched_jids = sorted(set(spec.keys()) & set(v14.keys()))
    print(f"\nMatched materials: {len(matched_jids)}")

    if not matched_jids:
        print("No matching materials found!")
        sys.exit(1)

    chi3_ours = []
    z3_v14 = []
    pg_groups = defaultdict(list)

    for jid in matched_jids:
        c3 = spec[jid]["chi3_recovered"]
        z3 = v14[jid]
        chi3_ours.append(c3)
        z3_v14.append(z3)
        pg = spec[jid].get("pg", "?")
        pg_groups[pg].append((jid, c3, z3))

    chi3_ours = np.array(chi3_ours)
    z3_v14 = np.array(z3_v14)

    print("\n" + "=" * 60)
    print("GLOBAL STATISTICS")
    print("=" * 60)
    print(f"  Our chi3:   mean={chi3_ours.mean():.4f}, "
          f"std={chi3_ours.std():.4f}, "
          f"range=[{chi3_ours.min():.4f}, {chi3_ours.max():.4f}]")
    print(f"  v14 z3:     mean={z3_v14.mean():.4f}, "
          f"std={z3_v14.std():.4f}, "
          f"range=[{z3_v14.min():.4f}, {z3_v14.max():.4f}]")

    corr = np.corrcoef(chi3_ours, z3_v14)[0, 1]
    print(f"\n  Pearson correlation: {corr:.4f}")

    diff = chi3_ours - z3_v14
    print(f"  Mean difference (ours - v14): {diff.mean():.4f}")
    print(f"  Std difference: {diff.std():.4f}")
    print(f"  Max |difference|: {np.abs(diff).max():.4f}")

    for thresh in [0.01, 0.05, 0.1, 0.2]:
        n_close = np.sum(np.abs(diff) < thresh)
        print(f"  |diff| < {thresh}: {n_close}/{len(diff)} "
              f"({100*n_close/len(diff):.1f}%)")

    rank_ours = np.argsort(np.argsort(-chi3_ours))
    rank_v14 = np.argsort(np.argsort(-z3_v14))
    spearman_num = np.sum((rank_ours - rank_v14) ** 2)
    n = len(matched_jids)
    spearman = 1 - 6 * spearman_num / (n * (n**2 - 1))
    print(f"\n  Spearman rank correlation: {spearman:.4f}")

    print("\n" + "=" * 60)
    print("PER-POINT-GROUP BREAKDOWN")
    print("=" * 60)
    for pg in sorted(pg_groups.keys()):
        entries = pg_groups[pg]
        c3s = np.array([e[1] for e in entries])
        z3s = np.array([e[2] for e in entries])
        n_pg = len(entries)
        if n_pg >= 2:
            r = np.corrcoef(c3s, z3s)[0, 1]
            r_str = f"{r:.3f}"
        else:
            r_str = "N/A"
        d = c3s - z3s
        print(f"  {pg:>3}: n={n_pg:>4}, "
              f"mean_diff={d.mean():+.4f}, "
              f"std_diff={d.std():.4f}, "
              f"corr={r_str}")

    print("\n" + "=" * 60)
    print("LARGEST DISCREPANCIES (top 20)")
    print("=" * 60)
    disc = [(jid, spec[jid]["chi3_recovered"], v14[jid],
             abs(spec[jid]["chi3_recovered"] - v14[jid]),
             spec[jid].get("pg", "?"))
            for jid in matched_jids]
    disc.sort(key=lambda x: -x[3])
    print(f"  {'JID':>15}  {'PG':>3}  {'our_chi3':>8}  {'v14_z3':>8}  {'|diff|':>8}")
    for jid, c3, z3, d, pg in disc[:20]:
        print(f"  {jid:>15}  {pg:>3}  {c3:8.4f}  {z3:8.4f}  {d:8.4f}")

    print("\n" + "=" * 60)
    print("AGREEMENT ON TOP CANDIDATES")
    print("=" * 60)
    top20_ours = set(matched_jids[i] for i in np.argsort(-chi3_ours)[:20])
    top20_v14 = set(matched_jids[i] for i in np.argsort(-z3_v14)[:20])
    overlap = top20_ours & top20_v14
    print(f"  Top-20 overlap: {len(overlap)}/20")
    if overlap:
        print(f"  Shared top-20: {', '.join(sorted(overlap))}")

    top50_ours = set(matched_jids[i] for i in np.argsort(-chi3_ours)[:50])
    top50_v14 = set(matched_jids[i] for i in np.argsort(-z3_v14)[:50])
    overlap50 = top50_ours & top50_v14
    print(f"  Top-50 overlap: {len(overlap50)}/50")

    output = {
        "n_matched": len(matched_jids),
        "pearson_correlation": float(corr),
        "spearman_correlation": float(spearman),
        "mean_diff": float(diff.mean()),
        "std_diff": float(diff.std()),
        "max_abs_diff": float(np.abs(diff).max()),
        "top20_overlap": len(overlap),
        "top50_overlap": len(overlap50),
        "per_point_group": {},
    }
    for pg, entries in pg_groups.items():
        c3s = np.array([e[1] for e in entries])
        z3s = np.array([e[2] for e in entries])
        d = c3s - z3s
        output["per_point_group"][pg] = {
            "n": len(entries),
            "mean_diff": float(d.mean()),
            "std_diff": float(d.std()),
        }

    out_path = SCRIPT_DIR.parent.parent / "data" / "chi3_validation.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    run_validation()
