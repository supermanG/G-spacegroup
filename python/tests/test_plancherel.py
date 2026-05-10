"""
test_plancherel.py
==================
Verify the Plancherel identity holds for all 11 proper point groups
and random structural functions.
"""

import numpy as np
import pytest

from python.data.crystal_groups import build_all_proper_groups
from python.data.spacegroup_star import (
    starg_fourier_transform,
    spectral_weights,
    spacegroup_fingerprint,
    verify_plancherel,
    chi3_from_fingerprint,
)

GROUPS = build_all_proper_groups()

SPG_TO_PG = [
    (1, 'C1'), (3, 'C2'), (143, 'C3'), (75, 'C4'),
    (168, 'C6'), (16, 'D2'), (149, 'D3'), (89, 'D4'),
    (177, 'D6'), (195, 'T'), (207, 'O'),
]


class TestPlancherel:
    @pytest.mark.parametrize("spg,pg_name", SPG_TO_PG)
    def test_plancherel_random(self, spg, pg_name):
        pg = GROUPS[pg_name]
        rng = np.random.default_rng(42 + spg)
        sigma = rng.standard_normal(pg['order'])
        assert verify_plancherel(sigma, spg, GROUPS)

    @pytest.mark.parametrize("spg,pg_name", SPG_TO_PG)
    def test_plancherel_constant(self, spg, pg_name):
        pg = GROUPS[pg_name]
        sigma = np.ones(pg['order'])
        assert verify_plancherel(sigma, spg, GROUPS)

    @pytest.mark.parametrize("spg,pg_name", SPG_TO_PG)
    def test_plancherel_delta(self, spg, pg_name):
        pg = GROUPS[pg_name]
        sigma = np.zeros(pg['order'])
        sigma[0] = 1.0
        assert verify_plancherel(sigma, spg, GROUPS)


class TestSpectralWeights:
    @pytest.mark.parametrize("spg,pg_name", SPG_TO_PG)
    def test_weights_sum_to_one(self, spg, pg_name):
        pg = GROUPS[pg_name]
        rng = np.random.default_rng(123 + spg)
        sigma = rng.standard_normal(pg['order'])
        fp = spacegroup_fingerprint(sigma, spg, GROUPS)
        total = sum(fp.values())
        assert abs(total - 1.0) < 1e-10, f"sum = {total}"

    @pytest.mark.parametrize("spg,pg_name", SPG_TO_PG)
    def test_weights_nonnegative(self, spg, pg_name):
        pg = GROUPS[pg_name]
        rng = np.random.default_rng(456 + spg)
        sigma = rng.standard_normal(pg['order'])
        fp = spacegroup_fingerprint(sigma, spg, GROUPS)
        for name, w in fp.items():
            assert w >= -1e-15, f"Negative weight {w} at {name}"

    def test_constant_function_all_trivial(self):
        """A constant function should have all weight in the trivial irrep."""
        for pg_name, pg in GROUPS.items():
            sigma = np.ones(pg['order']) * 3.14
            spg = next(s for s, p in SPG_TO_PG if p == pg_name)
            fp = spacegroup_fingerprint(sigma, spg, GROUPS)
            trivial_names = [
                irr['name'] for irr in pg['irrep_info']
                if irr['dim'] == 1 and irr.get('l', 0) == 0
            ]
            if trivial_names:
                assert fp[trivial_names[0]] > 0.99, (
                    f"{pg_name}: trivial irrep weight = "
                    f"{fp[trivial_names[0]]}"
                )


class TestChi3Recovery:
    @pytest.mark.parametrize("spg,pg_name", SPG_TO_PG)
    def test_chi3_in_unit_interval(self, spg, pg_name):
        pg = GROUPS[pg_name]
        rng = np.random.default_rng(789 + spg)
        sigma = rng.standard_normal(pg['order'])
        fp = spacegroup_fingerprint(sigma, spg, GROUPS)
        c3 = chi3_from_fingerprint(fp, spg, GROUPS)
        assert 0.0 <= c3 <= 1.0 + 1e-10, f"chi3 = {c3} out of range"

    def test_chi3_constant_is_zero(self):
        """chi_3 of a constant function should be 0 (all weight in trivial)."""
        for pg_name, pg in GROUPS.items():
            sigma = np.ones(pg['order'])
            spg = next(s for s, p in SPG_TO_PG if p == pg_name)
            fp = spacegroup_fingerprint(sigma, spg, GROUPS)
            c3 = chi3_from_fingerprint(fp, spg, GROUPS)
            assert c3 < 1e-10, f"{pg_name}: chi3 = {c3} for constant"
