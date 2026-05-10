"""
irrep_cache.py
==============
Manage cached irrep character tables for all 230 spacegroups.

Currently uses the 11 proper point groups from crystal_groups.py
(via the spg -> chiral point group mapping). Future versions will
incorporate full spacegroup irreps from Bilbao Crystallographic Server.

LH & Claude 2026
"""

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from .crystal_groups import build_all_proper_groups, spg_to_chiral

CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "irrep_tables"

_POINT_GROUPS_CACHE: Optional[Dict] = None


def get_point_groups() -> Dict:
    """Get cached point groups (built once per process)."""
    global _POINT_GROUPS_CACHE
    if _POINT_GROUPS_CACHE is None:
        _POINT_GROUPS_CACHE = build_all_proper_groups()
    return _POINT_GROUPS_CACHE


def get_irrep_info(spg_number: int) -> Dict:
    """
    Get irrep information for a spacegroup.

    Returns dict with keys:
      - pg_name: chiral point group name
      - order: group order
      - irreps: list of {name, dim, l} dicts
      - n_irreps: total number of irreps
      - n_channels: total number of spectral channels
    """
    groups = get_point_groups()
    pg_name = spg_to_chiral(spg_number)
    pg = groups.get(pg_name)
    if pg is None:
        return {'pg_name': pg_name, 'order': 0, 'irreps': [],
                'n_irreps': 0, 'n_channels': 0}

    irreps = [
        {'name': irr['name'], 'dim': irr['dim'], 'l': irr.get('l', 0)}
        for irr in pg['irrep_info']
    ]
    return {
        'pg_name': pg_name,
        'order': pg['order'],
        'irreps': irreps,
        'n_irreps': len(irreps),
        'n_channels': sum(irr['dim'] ** 2 for irr in irreps),
    }


def save_irrep_table(spg_number: int, table: Dict, cache_dir: Optional[Path] = None):
    """Save an irrep table to the cache directory."""
    d = cache_dir or CACHE_DIR
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"spg_{spg_number:03d}.json"

    serializable = {}
    for k, v in table.items():
        if isinstance(v, np.ndarray):
            serializable[k] = v.tolist()
        elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], np.ndarray):
            serializable[k] = [a.tolist() for a in v]
        else:
            serializable[k] = v

    with open(path, 'w') as f:
        json.dump(serializable, f, indent=2)


def load_irrep_table(spg_number: int, cache_dir: Optional[Path] = None) -> Optional[Dict]:
    """Load an irrep table from the cache directory."""
    d = cache_dir or CACHE_DIR
    path = d / f"spg_{spg_number:03d}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def spacegroup_summary() -> Dict[str, int]:
    """
    Summarize the point group distribution across all 230 spacegroups.

    Returns dict mapping point group name to count of spacegroups.
    """
    from .crystal_groups import SPG_TO_CHIRAL
    counts = {}
    for spg in range(1, 231):
        pg = SPG_TO_CHIRAL.get(spg, 'O')
        counts[pg] = counts.get(pg, 0) + 1
    return counts


if __name__ == '__main__':
    summary = spacegroup_summary()
    print("Spacegroup -> Point Group distribution:")
    for pg, count in sorted(summary.items(), key=lambda x: -x[1]):
        info = get_irrep_info(next(
            s for s in range(1, 231) if spg_to_chiral(s) == pg
        ))
        print(f"  {pg:<4}: {count:>3} spacegroups, "
              f"{info['n_irreps']} irreps, "
              f"order {info['order']}")
