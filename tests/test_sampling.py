import numpy as np
import pytest

from flux1km import sampling
from flux1km.table import FootprintTable


@pytest.fixture
def t():
    n = 12
    return FootprintTable({
        "granule": np.array(["A2014060.0000"] * n, dtype=object), "index": np.arange(n),
        "Fsw": np.array([50, 80, 120, 300, 450, 500, np.nan, 90, 410, 600, 200, 100], float),
        "SZA": np.array([10, 20, 30, 40, 50, 60, 70, 80, 20, 30, 40, 50], float),
        "VZA": np.array([0, 10, 20, 30, 40, 50, 60, 5, 15, 25, 35, 45], float),
        "glint": np.array([5, 30, 60, 90, 20, 40, 50, 60, 70, 80, 90, 100], float),
        "CF": np.array([0, 2, 50, 99, 100, 30, 0, 10, 97, 60, 3, 70], float),
        "meanCTH": np.array([1000] * 6 + [5000] * 6, float),
        "stdCTH": np.array([100] * 12, float),
        "cv": np.linspace(0.05, 0.6, 12),
    })


class TestRegime:
    def test_regime_labels(self):
        r = sampling.regime(np.array([0, 4.9, 5, 50, 95, 100, np.nan]))
        assert r.tolist() == ["clear", "clear", "broken", "broken", "overcast", "overcast", "unknown"]


class TestApply:
    def test_s0_drops_invalid_label(self, t):
        m, w = sampling.apply("S0", t)
        assert not m[6] and m.sum() == 11 and np.all(w[m] == 1)

    def test_s1_geometry(self, t):
        m, _ = sampling.apply("S1", t, sza_max=70, vza_max=55)
        assert not m[7] and not m[6]  # SZA 80; label NaN
        assert m[0]

    def test_s2_glint(self, t):
        m, _ = sampling.apply("S2", t, min_glint=25)
        assert not m[0] and not m[4] and m[1]

    def test_s3_homogeneity_cth_and_cv(self, t):
        m, _ = sampling.apply("S3", t, method="cth")
        assert m[:6].sum() == 6 and m[6:].sum() == 0
        m2, w2 = sampling.apply("S3", t, method="cv", cv_max=0.2)
        assert m2.sum() == np.sum(t["cv"] < 0.2)
        assert np.all(w2[m2] > 0)

    def test_s3_bad_method(self, t):
        with pytest.raises(ValueError):
            sampling.apply("S3", t, method="nope")

    def test_s4_bright_only(self, t):
        with pytest.raises(KeyError):  # S4 (Fsw >= 400) was dropped from the plan
            sampling.apply("S4", t, fsw_min=400)

    def test_s5_stratified_weights_balance_regimes(self, t):
        m, w = sampling.apply("S5", t)
        reg = sampling.regime(t["CF"])
        totals = {r: w[m & (reg == r)].sum() for r in ("clear", "broken", "overcast")}
        vals = list(totals.values())
        assert max(vals) == pytest.approx(min(vals))

    def test_s6_thinning(self, t):
        m, _ = sampling.apply("S6", t, period=3)
        assert np.all(t["index"][m] % 3 == 0)

    def test_s7_weights_inverse_variance(self, t):
        m, w = sampling.apply("S7", t, sigma={"clear": 2.0, "broken": 10.0, "overcast": 6.0})
        reg = sampling.regime(t["CF"])
        assert w[0] == pytest.approx(1 / 4) and w[2] == pytest.approx(1 / 100)
        assert m.sum() == 11 and np.all(w[~m] == 0)
        assert reg[0] == "clear"

    def test_unknown_strategy(self, t):
        with pytest.raises(KeyError):
            sampling.apply("S99", t)

    def test_restrict_to_rows(self, t):
        rows = np.array([0, 1, 2, 3])
        m, w = sampling.apply("S0", t, rows=rows)
        assert m.shape == (4,) and w.shape == (4,)


class TestSimilarity:
    def test_jaccard(self):
        a = np.array([1, 1, 0, 0], bool); b = np.array([1, 0, 1, 0], bool)
        assert sampling.jaccard(a, b) == pytest.approx(1 / 3)
        assert sampling.jaccard(np.zeros(3, bool), np.zeros(3, bool)) == 1.0

    def test_ks_distance(self):
        x = np.arange(100.0)
        assert sampling.ks_distance(x, x) == 0.0
        assert sampling.ks_distance(x, x + 1000) == pytest.approx(1.0)

    def test_rank_agreement(self):
        assert sampling.rank_agreement([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1.0)
        assert sampling.rank_agreement([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)


class TestCompareStrategies:
    def test_compare_strategies(self, t):
        res = sampling.compare_strategies(t, ["S0", "S3"], columns=("Fsw", "CF"))
        assert res["jaccard"][("S0", "S3")] == pytest.approx(6 / 11)
        assert ("S3", "Fsw") in res["ks_vs_reference"]
