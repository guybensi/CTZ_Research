import numpy as np
import pytest

from flux1km import metrics


class TestFootprintScores:
    def test_footprint_scores_perfect_and_bias(self):
        y = np.array([1.0, 2.0, 3.0])
        s = metrics.footprint_scores(y, y)
        assert s["rmse"] == 0 and s["r2"] == 1 and s["n"] == 3
        s = metrics.footprint_scores(y, y + 1)
        assert s["bias"] == pytest.approx(1) and s["rmse"] == pytest.approx(1) and s["mae"] == pytest.approx(1)

    def test_footprint_scores_ignores_nan_and_empty(self):
        s = metrics.footprint_scores(np.array([1.0, np.nan]), np.array([1.0, 5.0]))
        assert s["n"] == 1 and s["rmse"] == 0
        s = metrics.footprint_scores(np.array([]), np.array([]))
        assert s["n"] == 0 and np.isnan(s["rmse"])

    def test_footprint_scores_shape_mismatch(self):
        with pytest.raises(ValueError):
            metrics.footprint_scores(np.ones(2), np.ones(3))

    def test_weighted_scores(self):
        s = metrics.footprint_scores(np.array([0.0, 0.0]), np.array([1.0, 3.0]), weights=np.array([1.0, 0.0]))
        assert s["rmse"] == pytest.approx(1.0)


class TestRegimeScores:
    def test_scores_by_and_macro(self):
        y = np.zeros(6); p = np.array([1, 1, 2, 2, 4, 4.0])
        lab = np.array(["clear", "clear", "broken", "broken", "overcast", "overcast"], dtype=object)
        by = metrics.scores_by(y, p, lab)
        assert by["broken"]["rmse"] == 2
        assert metrics.macro_rmse(y, p, lab) == pytest.approx(7 / 3)

    def test_macro_rmse_missing_regime(self):
        y = np.zeros(2); p = np.array([1.0, 3.0]); lab = np.array(["clear", "overcast"], dtype=object)
        assert metrics.macro_rmse(y, p, lab) == pytest.approx(2.0)


class TestResidualDiagnostics:
    def test_binned_residuals_and_slope(self):
        x = np.linspace(0, 60, 200); r = 0.1 * x
        c, m, s, n = metrics.binned_residuals(r, x, bins=np.array([0, 20, 40, 60.0001]))
        assert len(c) == 3 and m[0] < m[1] < m[2] and n.sum() == 200
        slope, p = metrics.residual_trend(r, x)
        assert slope == pytest.approx(0.1) and p < 1e-6
        slope0, p0 = metrics.residual_trend(np.random.default_rng(0).normal(size=200), x)
        assert abs(slope0) < 0.05 and p0 > 0.01

    def test_swath_side_asymmetry(self):
        raa = np.array([30, 30, 150, 150.0]); vza = np.array([20, 20, 20, 20.0])
        r = np.array([1, 1, -1, -1.0])
        assert metrics.swath_side_asymmetry(r, raa, vza) == pytest.approx(2.0)
        assert metrics.swath_side_asymmetry(np.zeros(4), raa, vza) == 0

    def test_swath_side_asymmetry_no_overlap_nan(self):
        assert np.isnan(metrics.swath_side_asymmetry(np.zeros(2), np.array([30, 30.0]), np.array([10, 10.0])))


class TestSignificance:
    def test_unit_losses(self):
        y = np.zeros(4); p = np.array([1, 1, 2, 2.0]); u = np.array(["a", "a", "b", "b"], dtype=object)
        units, loss = metrics.unit_losses(y, p, u)
        assert units.tolist() == ["a", "b"] and loss.tolist() == [1.0, 4.0]

    def test_paired_bootstrap_detects_difference(self):
        rng = np.random.default_rng(0)
        la = rng.normal(10, 1, 40); lb = la + 2 + rng.normal(0, 0.1, 40)
        r = metrics.paired_bootstrap(la, lb, n_boot=2000, seed=0)
        assert r["mean_diff"] == pytest.approx(-2, abs=0.1) and r["ci_high"] < 0

    def test_paired_bootstrap_shape_mismatch(self):
        with pytest.raises(ValueError):
            metrics.paired_bootstrap(np.ones(3), np.ones(4))

    def test_wilcoxon_and_dm(self):
        rng = np.random.default_rng(1)
        la = rng.normal(5, 1, 60); lb = la + 1 + rng.normal(0, 0.3, 60)
        assert metrics.wilcoxon_p(la, lb) < 0.01
        stat, p = metrics.diebold_mariano(la, lb)
        assert stat < 0 and p < 0.01
        stat0, p0 = metrics.diebold_mariano(la, la + rng.normal(0, 1, 60))
        assert p0 > 0.01

    def test_dm_constant_difference(self):
        stat, p = metrics.diebold_mariano(np.ones(10), np.ones(10))
        assert np.isnan(stat) and np.isnan(p)

    def test_holm(self):
        adj = metrics.holm([0.01, 0.04, 0.03])
        np.testing.assert_allclose(adj, [0.03, 0.06, 0.06])
        assert metrics.holm([]).size == 0


class TestPixelScores:
    def test_pixel_scores_identity_and_ssim(self):
        rng = np.random.default_rng(0)
        t = rng.normal(100, 20, size=(3, 27, 55))
        s = metrics.pixel_scores(t, t)
        assert s["rmse"] == 0 and s["corr"] == pytest.approx(1) and s["ssim"] == pytest.approx(1)
        s2 = metrics.pixel_scores(t, np.broadcast_to(t.mean(axis=(1, 2), keepdims=True), t.shape))
        assert s2["ssim"] < 0.5 and s2["rmse"] > 10
