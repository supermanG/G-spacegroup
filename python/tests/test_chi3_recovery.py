"""
test_chi3_recovery.py
=====================
Verify that the full spacegroup fingerprint is a strict refinement of
the chi_3 point-group predictor used in RTSC v3-v14.

Key property: chi_3 = sum of spectral weights at irreps with l > 0,
which equals the non-trivial rotational content of the structural
function.
"""

import numpy as np
import pytest

from python.data.crystal_groups import build_all_proper_groups
from python.data.spacegroup_star import (
    spacegroup_fingerprint,
    chi3_from_fingerprint,
)

GROUPS = build_all_proper_groups()

SPG_TO_PG = [
    (1, 'C1'), (3, 'C2'), (143, 'C3'), (75, 'C4'),
    (168, 'C6'), (16, 'D2'), (149, 'D3'), (89, 'D4'),
    (177, 'D6'), (195, 'T'), (207, 'O'),
]


class TestChi3IsSpecialCase:
    """
    Verify that chi_3 = 1 - w(trivial irreps), confirming
    the full spectrum is a refinement.
    """

    @pytest.mark.parametrize("spg,pg_name", SPG_TO_PG)
    def test_chi3_plus_trivial_equals_one(self, spg, pg_name):
        pg = GROUPS[pg_name]
        rng = np.random.default_rng(999 + spg)
        sigma = rng.standard_normal(pg['order'])
        fp = spacegroup_fingerprint(sigma, spg, GROUPS)
        c3 = chi3_from_fingerprint(fp, spg, GROUPS)

        trivial_weight = sum(
            fp.get(irr['name'], 0.0)
            for irr in pg['irrep_info']
            if irr.get('l', 0) == 0
        )

        assert abs(c3 + trivial_weight - 1.0) < 1e-10, (
            f"{pg_name}: chi3={c3}, trivial={trivial_weight}, "
            f"sum={c3 + trivial_weight}"
        )

    @pytest.mark.parametrize("spg,pg_name", SPG_TO_PG)
    def test_more_irreps_than_chi3(self, spg, pg_name):
        """Full fingerprint has >= as many components as chi_3 encodes."""
        pg = GROUPS[pg_name]
        n_irreps = len(pg['irrep_info'])
        assert n_irreps >= 2 or pg_name == 'C1', (
            f"{pg_name} has only {n_irreps} irreps"
        )

    def test_octahedral_has_five_irreps(self):
        """O group has 5 irreps: A1, A2, E, T1, T2."""
        pg = GROUPS['O']
        assert len(pg['irrep_info']) == 5

    def test_d4_captures_cuprate_channel(self):
        """D4 has B1 irrep (d_{x^2-y^2}), the cuprate pairing symmetry."""
        pg = GROUPS['D4']
        irrep_names = [irr['name'] for irr in pg['irrep_info']]
        assert 'B1' in irrep_names, "D4 missing B1 (cuprate channel)"
