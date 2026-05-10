#!/usr/bin/env python3
"""
rerank_candidates.py
====================
Re-rank superconductor candidates using the full spacegroup fingerprint
versus the scalar chi_3.

For each point group with enough materials, compute how the ranking
changes when using individual irrep weights instead of just chi_3.
Highlight materials that move significantly up or down.

LH & Claude 2026
"""

import csv
import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent


def load_spectroscopy_results(path):
    with open(path) as f:
        results = json.load(f)
    return {r["jid"]: r for r in results if r.get("status") == "ok"}


def load_v14_features(path):
    rows = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            jid = row["jid"]
            try:
                tc = row.get("Tc_central_K", "")
                z3 = row.get("z3", "")
                rows[jid] = {
                    "Tc_central_K": float(tc) if tc else None,
                    "z3": float(z3) if z3 else None,
                    "formula": row.get("formula", ""),
                    "spg": int(row["spg"]) if row["spg"] else None,
                    "point_group": row.get("point_group", ""),
                }
            except (ValueError, KeyError):
                continue
    return rows


def composite_score(fingerprint, irrep_tc_corrs):
    """Compute a Tc-weighted composite score from irrep weights."""
    score = 0.0
    for irr, w in fingerprint.items():
        corr = irrep_tc_corrs.get(irr, 0.0)
        score += w * corr
    return score


def run_reranking():
    spec_path = SCRIPT_DIR.parent.parent / "data" / "spectroscopy_results.json"
    v14_path = Path(r"C:\Users\superman\rtsc\roadmap\overnight\v14_candidate_pool.csv")
    ortho_path = SCRIPT_DIR.parent.parent / "data" / "orthogonality_analysis.json"

    if not all(p.exists() for p in [spec_path, v14_path, ortho_path]):
        print("ERROR: required data files not found")
        sys.exit(1)

    print("Loading data...")
    spec = load_spectroscopy_results(spec_path)
    v14 = load_v14_features(v14_path)
    with open(ortho_path) as f:
        ortho = json.load(f)

    matched = sorted(set(spec.keys()) & set(v14.keys()))
    print(f"Matched materials: {len(matched)}")

    pg_materials = defaultdict(list)
    for jid in matched:
        pg = spec[jid].get("pg", "?")
        if v14[jid]["Tc_central_K"] is not None:
            pg_materials[pg].append(jid)

    print("\n" + "=" * 70)
    print("CANDIDATE RE-RANKING: FULL SPECTRUM vs CHI_3")
    print("=" * 70)

    all_movers = []

    for pg in sorted(pg_materials.keys()):
        jids = pg_materials[pg]
        if len(jids) < 10:
            continue

        tc_vals = np.array([v14[jid]["Tc_central_K"] for jid in jids])
        chi3_vals = np.array([spec[jid]["chi3_recovered"] for jid in jids])

        sample_fp = spec[jids[0]]["fingerprint"]
        irrep_names = sorted(sample_fp.keys())

        w_matrix = np.array([
            [spec[jid]["fingerprint"].get(irr, 0.0) for irr in irrep_names]
            for jid in jids
        ])

        irrep_tc_corrs = {}
        for col_idx, irr in enumerate(irrep_names):
            w_col = w_matrix[:, col_idx]
            valid = np.isfinite(tc_vals) & np.isfinite(w_col) & (np.std(w_col) > 1e-10)
            if valid.sum() >= 3:
                r = np.corrcoef(w_col[valid], tc_vals[valid])[0, 1]
                irrep_tc_corrs[irr] = r if np.isfinite(r) else 0.0
            else:
                irrep_tc_corrs[irr] = 0.0

        composite_scores = []
        for jid in jids:
            cs = composite_score(spec[jid]["fingerprint"], irrep_tc_corrs)
            composite_scores.append(cs)
        composite_scores = np.array(composite_scores)

        rank_chi3 = np.argsort(np.argsort(-chi3_vals))
        rank_composite = np.argsort(np.argsort(-composite_scores))
        rank_diff = rank_chi3.astype(int) - rank_composite.astype(int)

        print(f"\n--- Point group {pg} (n={len(jids)}) ---")
        print(f"  Irrep-Tc correlations: ", end="")
        for irr, r in sorted(irrep_tc_corrs.items(), key=lambda x: -abs(x[1])):
            if abs(r) > 0.05:
                print(f"{irr}:{r:+.2f} ", end="")
        print()

        big_movers = [(jids[i], rank_diff[i], chi3_vals[i],
                        composite_scores[i], tc_vals[i], pg)
                       for i in range(len(jids))
                       if abs(rank_diff[i]) >= max(3, len(jids) // 10)]
        big_movers.sort(key=lambda x: -abs(x[1]))

        if big_movers:
            print(f"  Materials with significant rank change (>= {max(3, len(jids) // 10)} positions):")
            print(f"    {'JID':>15}  {'rank_diff':>10}  {'chi3':>6}  {'composite':>10}  {'Tc':>8}")
            for jid, rd, c3, cs, tc, _ in big_movers[:10]:
                direction = "UP" if rd > 0 else "DOWN"
                print(f"    {jid:>15}  {rd:>+10} {direction:<4}  {c3:6.4f}  {cs:10.4f}  {tc:8.2f}")
                all_movers.append((jid, rd, c3, cs, tc, pg))
        else:
            print("  No significant rank changes.")

    print("\n" + "=" * 70)
    print("TOP 20 RANK MOVERS ACROSS ALL POINT GROUPS")
    print("=" * 70)
    all_movers.sort(key=lambda x: -abs(x[1]))
    print(f"  {'JID':>15}  {'PG':>3}  {'rank_diff':>10}  {'chi3':>6}  {'Tc':>8}  {'formula':>15}")
    for jid, rd, c3, cs, tc, pg in all_movers[:20]:
        formula = v14.get(jid, {}).get("formula", "?")
        direction = "UP" if rd > 0 else "DOWN"
        print(f"  {jid:>15}  {pg:>3}  {rd:>+10} {direction:<4}  {c3:6.4f}  {tc:8.2f}  {formula:>15}")

    output = {
        "n_matched": len(matched),
        "movers": [
            {"jid": jid, "pg": pg, "rank_diff": int(rd),
             "chi3": float(c3), "composite": float(cs),
             "Tc": float(tc), "formula": v14.get(jid, {}).get("formula", "")}
            for jid, rd, c3, cs, tc, pg in all_movers
        ],
    }
    out_path = SCRIPT_DIR.parent.parent / "data" / "reranking_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    run_reranking()
