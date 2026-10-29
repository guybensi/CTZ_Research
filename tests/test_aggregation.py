import numpy as np
import pytest

from flux1km import aggregation as ag


class TestUniformWeights:
    def test_uniform_weights_sum_and_shape(self):
        w = ag.uniform_weights(27, 55)
        assert w.shape == (27, 55) and w.sum() == pytest.approx(1.0) and np.ptp(w) == 0

    def test_uniform_bad_shape(self):
        with pytest.raises(ValueError):
            ag.uniform_weights(0, 5)


class TestGaussianPsfWeights:
    def test_gaussian_psf_normalised_peak_center_and_symmetric(self):
        w = ag.gaussian_psf_weights(27, 55, fwhm_y=10, fwhm_x=20)
        assert w.sum() == pytest.approx(1.0)
        assert np.unravel_index(w.argmax(), w.shape) == (13, 27)
        np.testing.assert_allclose(w, w[::-1, ::-1])

    def test_gaussian_psf_bad_fwhm(self):
        with pytest.raises(ValueError):
            ag.gaussian_psf_weights(27, 55, 0, 10)


class TestAggregate:
    def test_aggregate_uniform_equals_mean(self):
        rng = np.random.default_rng(1); maps = rng.normal(size=(4, 27, 55))
        np.testing.assert_allclose(ag.aggregate(maps, ag.uniform_weights(27, 55)), maps.mean(axis=(1, 2)))

    def test_aggregate_per_footprint_weights(self):
        maps = np.stack([np.zeros((2, 2)), np.ones((2, 2))])
        w = np.stack([ag.uniform_weights(2, 2)] * 2)
        np.testing.assert_allclose(ag.aggregate(maps, w), [0.0, 1.0])

    def test_aggregate_unnormalised_weights_are_normalised(self):
        maps = np.ones((1, 3, 3)) * 5
        assert ag.aggregate(maps, np.full((3, 3), 7.0))[0] == pytest.approx(5.0)

    def test_aggregate_nan_renormalize_vs_propagate(self):
        maps = np.array([[[1.0, np.nan], [3.0, 5.0]]])
        w = ag.uniform_weights(2, 2)
        assert ag.aggregate(maps, w)[0] == pytest.approx(3.0)
        assert np.isnan(ag.aggregate(maps, w, nan_policy="propagate")[0])

    def test_aggregate_all_nan_gives_nan(self):
        assert np.isnan(ag.aggregate(np.full((1, 2, 2), np.nan), ag.uniform_weights(2, 2))[0])

    def test_aggregate_shape_mismatch_and_bad_policy(self):
        with pytest.raises(ValueError):
            ag.aggregate(np.ones((1, 3, 3)), ag.uniform_weights(2, 2))
        with pytest.raises(ValueError):
            ag.aggregate(np.ones((1, 2, 2)), ag.uniform_weights(2, 2), nan_policy="zzz")

    def test_aggregate_negative_weights_rejected(self):
        with pytest.raises(ValueError):
            ag.aggregate(np.ones((1, 2, 2)), np.array([[1.0, -1.0], [1.0, 1.0]]))


class TestShiftWeights:
    def test_shift_weights_moves_peak_and_renormalises(self):
        w = ag.gaussian_psf_weights(27, 55, 6, 6)
        s = ag.shift_weights(w, dy=1, dx=-2)
        assert s.sum() == pytest.approx(1.0)
        assert np.unravel_index(s.argmax(), s.shape) == (14, 25)

    def test_shift_out_of_range(self):
        with pytest.raises(ValueError):
            ag.shift_weights(ag.uniform_weights(3, 3), 3, 0)


class TestRandomSubsetWeights:
    def test_random_subset_weights(self):
        rng = np.random.default_rng(0)
        w = ag.random_subset_weights(ag.uniform_weights(27, 55), 0.2, rng)
        assert w.sum() == pytest.approx(1.0)
        frac = (w > 0).mean()
        assert 0.15 < frac < 0.25

    def test_random_subset_bad_fraction(self):
        with pytest.raises(ValueError):
            ag.random_subset_weights(ag.uniform_weights(3, 3), 0.0, np.random.default_rng(0))


class TestGetOperator:
    def test_get_operator(self):
        assert ag.get_operator("uniform", 27, 55).shape == (27, 55)
        assert ag.get_operator("gaussian_psf", 27, 55).sum() == pytest.approx(1.0)
        with pytest.raises(KeyError):
            ag.get_operator("nope", 27, 55)
