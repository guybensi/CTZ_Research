import numpy as np
import pytest

from flux1km import physics, synthetic


class TestDistanceToCloud:
    def test_distance_to_cloud_basic(self):
        m = np.zeros((1, 5, 5), bool); m[0, 2, 2] = True
        d = physics.distance_to_cloud(m)
        assert d[0, 2, 2] == 0 and d[0, 2, 4] == 2 and d[0, 0, 0] == pytest.approx(np.sqrt(8))

    def test_distance_to_cloud_no_cloud_is_inf(self):
        assert np.all(np.isinf(physics.distance_to_cloud(np.zeros((2, 3, 3), bool))))

    def test_distance_pixel_size(self):
        m = np.zeros((1, 1, 4), bool); m[0, 0, 0] = True
        assert physics.distance_to_cloud(m, pixel_km=2.0)[0, 0, 3] == 6


class TestFluxVsDistance:
    def test_flux_vs_distance_ignores_cloudy_and_far(self):
        maps = np.array([[[100.0, 90.0, 80.0, 70.0]]]); d = np.array([[[0.0, 1.0, 2.0, np.inf]]])
        c, mean, n = physics.flux_vs_distance(maps, d, bins=np.array([0.5, 1.5, 2.5]))
        assert mean.tolist() == [90.0, 80.0] and n.tolist() == [1, 1]

    def test_decay_score(self):
        assert physics.decay_score(np.array([5.0, 4, 3, 2])) == pytest.approx(-1.0)
        assert physics.decay_score(np.array([1.0, 2, 3, 4])) == pytest.approx(1.0)
        assert np.isnan(physics.decay_score(np.array([1.0])))


class TestMapChecks:
    def test_bound_violation_fraction(self):
        maps = np.array([[[-1.0, 100.0], [2000.0, 50.0]]]); sza = np.zeros((1, 2, 2))
        assert physics.bound_violation_fraction(maps, sza) == pytest.approx(0.5)

    def test_percentile_spread(self):
        lo, hi = physics.percentile_spread(np.arange(101.0).reshape(1, 1, 101))
        assert lo == pytest.approx(1) and hi == pytest.approx(99)

    def test_footprint_spacing_peak(self):
        x = np.arange(55)
        clean = np.ones((10, 27, 55))
        striped = clean + 5 * np.cos(2 * np.pi * x / 8)[None, None, :]
        rng = np.random.default_rng(0)
        noise = rng.normal(size=(10, 27, 55))
        assert physics.spacing_peak_ratio(striped + noise, period_px=8) > 5
        assert physics.spacing_peak_ratio(clean + noise, period_px=8) < 3

    def test_spacing_peak_ignores_nan_pixels(self):
        x = np.arange(55)
        striped = np.ones((4, 27, 55)) + 5 * np.cos(2 * np.pi * x / 8)[None, None, :]
        striped[0, 3, 7] = np.nan
        striped[1, 5, :] = np.nan                       # a whole invalid row is dropped
        assert physics.spacing_peak_ratio(striped, period_px=8) > 5
        assert np.isnan(physics.spacing_peak_ratio(np.full((1, 2, 55), np.nan)))

    def test_far_clear_flux(self):
        maps = np.full((1, 2, 2), 80.0); maps[0, 0, 0] = 500
        d = np.array([[[0.0, 40.0], [50.0, 60.0]]])
        assert physics.far_clear_flux(maps, d, min_km=30) == pytest.approx(80.0)
        assert np.isnan(physics.far_clear_flux(maps, d, min_km=100))


class TestRunChecks:
    def test_run_checks_on_synthetic_truth_passes_and_on_extreme_fails(self):
        g = synthetic.make_synthetic_granule("A2014060.0245", np.random.default_rng(0), n_rows=4, n_cols=8)
        sza = g.images["Imviews"][:, 0]
        good = physics.run_checks(g.truth["flux"], g.truth["dist_km"], sza, pcs_target=(40, 120))
        assert good["passed"], good
        bad_maps = g.truth["flux"].copy(); bad_maps[:, :, ::2] *= 3
        bad = physics.run_checks(bad_maps, g.truth["dist_km"], sza, pcs_target=(40, 120))
        assert not bad["passed"]


class TestHighPass:
    def test_constant_has_no_detail(self):
        assert np.allclose(physics.high_pass(np.full((2, 5, 7), 3.0)), 0.0)

    def test_nan_stays_nan_and_does_not_spread(self):
        x = np.random.default_rng(0).random((1, 9, 9)); x[0, 4, 4] = np.nan
        hp = physics.high_pass(x)
        assert np.isnan(hp[0, 4, 4]) and np.isfinite(hp).sum() == 80


class TestSharpness:
    def _scene(self):
        g = synthetic.make_synthetic_granule("A2014060.0245", np.random.default_rng(1), n_rows=3, n_cols=6)
        return g.truth["flux"], g.images["ImSW"][:, 2]           # flux and the 0.66 um channel

    def test_blur_lowers_the_score(self):
        from scipy import ndimage

        truth, img = self._scene()
        s = [physics.sharpness(ndimage.gaussian_filter(truth, (0, b, b)) if b else truth, img) for b in (0, 1, 2, 4)]
        assert s[0] > 0.9 and s[0] > s[1] > s[2] > s[3]

    def test_constant_per_footprint_scores_zero(self):
        truth, img = self._scene()
        const = np.broadcast_to(truth.mean(axis=(1, 2), keepdims=True), truth.shape)
        assert physics.sharpness(const, img) == 0.0

    def test_detail_in_the_wrong_place_is_negative(self):
        truth, img = self._scene()
        assert physics.sharpness(-truth, img) < -0.9

    def test_no_detail_in_the_image_is_nan(self):
        assert np.isnan(physics.sharpness(np.random.default_rng(0).random((2, 5, 5)), np.ones((2, 5, 5))))
