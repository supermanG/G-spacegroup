#!/usr/bin/env python3
"""
cuprate_channel_ranking.py
==========================
Produce dedicated B1 (d_{x^2-y^2}) and B2 (d_{xy}) weight rankings
for D4 point group materials.

Per human decision Q2: highlight materials by cuprate-channel irrep
weights.

B1 = d_{x^2-y^2} (canonical cuprate pairing channel)
B2 = d_{xy} (shows stronger Tc correlation in our data, r=0.33)

LH & Claude 2026
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent


def load_data():
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
                    "Tc_lit_K": float(row["Tc_lit_K"]) if row.get("Tc_lit_K") else None,
                    "spg": int(row["spg"]) if row["spg"] else None,
                    "point_group": row.get("point_group", ""),
                    "z3": float(row["z3"]) if row["z3"] else None,
                    "K_GPa": float(row["K_GPa"]) if row.get("K_GPa") else None,
                    "G_GPa": float(row["G_GPa"]) if row.get("G_GPa") else None,
                    "is_ductile": row.get("is_ductile", ""),
                    "chem_has_Cu": row.get("chem_has_Cu", ""),
                    "chem_has_O": row.get("chem_has_O", ""),
                    "chem_has_CuO": row.get("chem_has_CuO", ""),
                    "src": row.get("src", ""),
                }
            except (ValueError, KeyError):
                continue

    return spec, v14


def run_ranking():
    spec, v14 = load_data()
    matched = sorted(set(spec.keys()) & set(v14.keys()))

    d4_materials = [
        jid for jid in matched
        if spec[jid].get("pg") == "D4"
    ]

    print("=" * 80)
    print("CUPRATE CHANNEL RANKINGS (D4 point group materials)")
    print("=" * 80)
    print(f"\nTotal D4 materials with both spectroscopy and v14 data: {len(d4_materials)}")

    entries = []
    for jid in d4_materials:
        fp = spec[jid]["fingerprint"]
        v = v14[jid]
        entries.append({
            "jid": jid,
            "formula": v["formula"],
            "spg": v["spg"],
            "Tc_central_K": v["Tc_central_K"],
            "Tc_lit_K": v["Tc_lit_K"],
            "z3": v["z3"],
            "w_A1": fp.get("A1", 0.0),
            "w_A2": fp.get("A2", 0.0),
            "w_B1": fp.get("B1", 0.0),
            "w_B2": fp.get("B2", 0.0),
            "w_E": fp.get("E", 0.0),
            "chi3": spec[jid]["chi3_recovered"],
            "K_GPa": v["K_GPa"],
            "is_ductile": v["is_ductile"],
            "has_Cu": v["chem_has_Cu"],
            "has_CuO": v["chem_has_CuO"],
        })

    print("\n--- B1 RANKING (d_{x^2-y^2}, canonical cuprate channel) ---")
    print(f"{'Rank':>4}  {'JID':>15}  {'Formula':>12}  {'SPG':>4}  "
          f"{'w_B1':>6}  {'w_B2':>6}  {'chi3':>6}  {'Tc':>8}  {'Cu?':>3}")
    b1_sorted = sorted(entries, key=lambda x: -x["w_B1"])
    for rank, e in enumerate(b1_sorted[:30], 1):
        tc_str = f"{e['Tc_central_K']:8.2f}" if e['Tc_central_K'] is not None else "     N/A"
        cu = "Y" if e["has_Cu"] == "1" else ""
        print(f"{rank:4d}  {e['jid']:>15}  {e['formula']:>12}  {e['spg']:4d}  "
              f"{e['w_B1']:6.4f}  {e['w_B2']:6.4f}  {e['chi3']:6.4f}  "
              f"{tc_str}  {cu:>3}")

    print("\n--- B2 RANKING (d_{xy}, strongest Tc correlation) ---")
    print(f"{'Rank':>4}  {'JID':>15}  {'Formula':>12}  {'SPG':>4}  "
          f"{'w_B2':>6}  {'w_B1':>6}  {'chi3':>6}  {'Tc':>8}  {'Cu?':>3}")
    b2_sorted = sorted(entries, key=lambda x: -x["w_B2"])
    for rank, e in enumerate(b2_sorted[:30], 1):
        tc_str = f"{e['Tc_central_K']:8.2f}" if e['Tc_central_K'] is not None else "     N/A"
        cu = "Y" if e["has_Cu"] == "1" else ""
        print(f"{rank:4d}  {e['jid']:>15}  {e['formula']:>12}  {e['spg']:4d}  "
              f"{e['w_B2']:6.4f}  {e['w_B1']:6.4f}  {e['chi3']:6.4f}  "
              f"{tc_str}  {cu:>3}")

    print("\n--- STATISTICS ---")
    b1_vals = np.array([e["w_B1"] for e in entries])
    b2_vals = np.array([e["w_B2"] for e in entries])
    tc_vals = np.array([e["Tc_central_K"] if e["Tc_central_K"] is not None else np.nan for e in entries])
    valid = np.isfinite(tc_vals)

    print(f"  B1: mean={b1_vals.mean():.4f}, std={b1_vals.std():.4f}, "
          f"max={b1_vals.max():.4f}")
    print(f"  B2: mean={b2_vals.mean():.4f}, std={b2_vals.std():.4f}, "
          f"max={b2_vals.max():.4f}")
    if valid.sum() >= 3:
        corr_b1 = np.corrcoef(b1_vals[valid], tc_vals[valid])[0, 1]
        corr_b2 = np.corrcoef(b2_vals[valid], tc_vals[valid])[0, 1]
        print(f"  corr(B1, Tc) = {corr_b1:.4f}")
        print(f"  corr(B2, Tc) = {corr_b2:.4f}")

    output = {
        "n_d4_materials": len(d4_materials),
        "b1_ranking": [
            {"rank": i+1, "jid": e["jid"], "formula": e["formula"],
             "spg": e["spg"], "w_B1": e["w_B1"], "w_B2": e["w_B2"],
             "chi3": e["chi3"],
             "Tc_central_K": e["Tc_central_K"]}
            for i, e in enumerate(b1_sorted)
        ],
        "b2_ranking": [
            {"rank": i+1, "jid": e["jid"], "formula": e["formula"],
             "spg": e["spg"], "w_B1": e["w_B1"], "w_B2": e["w_B2"],
             "chi3": e["chi3"],
             "Tc_central_K": e["Tc_central_K"]}
            for i, e in enumerate(b2_sorted)
        ],
    }
    out_path = SCRIPT_DIR.parent.parent / "data" / "cuprate_channel_ranking.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    run_ranking()
