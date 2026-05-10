#!/usr/bin/env python3
"""
orthogonality_analysis.py
=========================
Identify which spacegroup spectral channels carry information
orthogonal to the v14 feature set.

For each point group, the full spectrum gives d irrep weights
(e.g., O gives 5: A1, A2, E, T1, T2). The v14 predictor already
uses z3 (the sum of non-trivial weights). The question is: which
individual irrep weights carry information NOT captured by z3 or
other v14 features?

Method:
1. Load v14 feature matrix and our spectral weights
2. For each irrep weight column, compute its residual variance
   after projecting out z3 and key v14 features
3. Rank irreps by residual information content
4. Flag irreps that correlate with Tc independently of z3

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
    """Load selected v14 features for orthogonality analysis."""
    rows = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            jid = row["jid"]
            try:
                entry = {
                    "z3": float(row["z3"]) if row["z3"] else None,
                    "Tc_central_K": float(row["Tc_central_K"]) if row["Tc_central_K"] else None,
                    "K_GPa": float(row["K_GPa"]) if row["K_GPa"] else None,
                    "G_GPa": float(row["G_GPa"]) if row["G_GPa"] else None,
                    "pugh_ratio": float(row["pugh_ratio"]) if row["pugh_ratio"] else None,
                    "density_g_cc": float(row["density_g_cc"]) if row["density_g_cc"] else None,
                    "spg": int(row["spg"]) if row["spg"] else None,
                    "point_group": row.get("point_group", ""),
                }
                rows[jid] = entry
            except (ValueError, KeyError):
                continue
    return rows


def compute_residual_variance(target, predictors):
    """Variance of target not explained by linear combination of predictors."""
    if len(target) < 3:
        return np.nan
    X = np.column_stack(predictors) if len(predictors) > 0 else np.zeros((len(target), 0))
    X = np.column_stack([np.ones(len(target)), X])

    mask = np.all(np.isfinite(X), axis=1) & np.isfinite(target)
    if mask.sum() < X.shape[1] + 1:
        return np.nan

    X_clean = X[mask]
    y_clean = target[mask]

    try:
        beta, _, _, _ = np.linalg.lstsq(X_clean, y_clean, rcond=None)
        residual = y_clean - X_clean @ beta
        return np.var(residual) / np.var(y_clean) if np.var(y_clean) > 1e-30 else np.nan
    except np.linalg.LinAlgError:
        return np.nan


def run_analysis():
    spec_path = SCRIPT_DIR.parent.parent / "data" / "spectroscopy_results.json"
    v14_path = Path(r"C:\Users\superman\rtsc\roadmap\overnight\v14_candidate_pool.csv")

    if not spec_path.exists() or not v14_path.exists():
        print("ERROR: required data files not found")
        sys.exit(1)

    print("Loading data...")
    spec = load_spectroscopy_results(spec_path)
    v14 = load_v14_features(v14_path)

    matched = sorted(set(spec.keys()) & set(v14.keys()))
    print(f"Matched materials: {len(matched)}")

    pg_materials = defaultdict(list)
    for jid in matched:
        pg = spec[jid].get("pg", "?")
        pg_materials[pg].append(jid)

    print("\n" + "=" * 70)
    print("ORTHOGONALITY ANALYSIS: IRREP WEIGHTS vs V14 FEATURES")
    print("=" * 70)

    all_results = {}

    for pg in sorted(pg_materials.keys()):
        jids = pg_materials[pg]
        if len(jids) < 5:
            continue

        sample = spec[jids[0]]
        irrep_names = sorted(sample["fingerprint"].keys())
        n_irreps = len(irrep_names)

        if n_irreps < 2:
            continue

        print(f"\n--- Point group {pg} (n={len(jids)}, {n_irreps} irreps) ---")

        w_matrix = np.array([
            [spec[jid]["fingerprint"].get(irr, 0.0) for irr in irrep_names]
            for jid in jids
        ])
        z3_vals = np.array([v14[jid]["z3"] if v14[jid]["z3"] is not None else 0.0 for jid in jids])
        tc_vals = np.array([v14[jid]["Tc_central_K"] if v14[jid]["Tc_central_K"] is not None else np.nan for jid in jids])

        print(f"  {'Irrep':>6}  {'mean_w':>7}  {'std_w':>7}  "
              f"{'corr_z3':>8}  {'corr_Tc':>8}  "
              f"{'resid_var':>10}  {'info_gain':>10}")

        pg_result = {}
        for col_idx, irr in enumerate(irrep_names):
            w_col = w_matrix[:, col_idx]
            mean_w = np.mean(w_col)
            std_w = np.std(w_col)

            valid_z3 = np.isfinite(z3_vals) & np.isfinite(w_col)
            if valid_z3.sum() >= 3:
                corr_z3 = np.corrcoef(w_col[valid_z3], z3_vals[valid_z3])[0, 1]
            else:
                corr_z3 = np.nan

            valid_tc = np.isfinite(tc_vals) & np.isfinite(w_col)
            if valid_tc.sum() >= 3:
                corr_tc = np.corrcoef(w_col[valid_tc], tc_vals[valid_tc])[0, 1]
            else:
                corr_tc = np.nan

            resid_var = compute_residual_variance(w_col, [z3_vals])

            info_gain = 1.0 - resid_var if np.isfinite(resid_var) else np.nan

            print(f"  {irr:>6}  {mean_w:7.4f}  {std_w:7.4f}  "
                  f"{corr_z3:8.4f}  {corr_tc:8.4f}  "
                  f"{resid_var:10.4f}  {info_gain:10.4f}")

            pg_result[irr] = {
                "mean_w": float(mean_w),
                "std_w": float(std_w),
                "corr_z3": float(corr_z3) if np.isfinite(corr_z3) else None,
                "corr_Tc": float(corr_tc) if np.isfinite(corr_tc) else None,
                "residual_variance_after_z3": float(resid_var) if np.isfinite(resid_var) else None,
            }

        all_results[pg] = {
            "n_materials": len(jids),
            "n_irreps": n_irreps,
            "irreps": pg_result,
        }

    print("\n" + "=" * 70)
    print("SUMMARY: MOST INFORMATIVE NEW CHANNELS")
    print("=" * 70)
    print("(High residual variance = carries info NOT captured by z3)")
    print()

    candidates = []
    for pg, res in all_results.items():
        for irr, data in res["irreps"].items():
            rv = data.get("residual_variance_after_z3")
            ct = data.get("corr_Tc")
            if rv is not None and rv > 0.1 and ct is not None and abs(ct) > 0.05:
                candidates.append((pg, irr, rv, ct, res["n_materials"]))

    candidates.sort(key=lambda x: -abs(x[3]))
    print(f"  {'PG':>3}  {'Irrep':>6}  {'resid_var':>10}  {'corr_Tc':>8}  {'n':>5}")
    for pg, irr, rv, ct, n in candidates[:20]:
        print(f"  {pg:>3}  {irr:>6}  {rv:10.4f}  {ct:8.4f}  {n:5d}")

    out_path = SCRIPT_DIR.parent.parent / "data" / "orthogonality_analysis.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull results saved to {out_path}")


if __name__ == "__main__":
    run_analysis()
