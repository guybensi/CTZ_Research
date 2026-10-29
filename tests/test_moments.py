import numpy as np
import pytest
from scipy import stats

from flux1km import moments


@pytest.fixture
def pix():
    return np.random.default_rng(0).gamma(2.0, 3.0, size=(5, 3, 27, 55))  # (N, C, H, W)


class TestMomentsFromPixels:
    def test_matches_scipy(self, pix):
        m, s, sk, ku = moments.moments_from_pixels(pix)
        flat = pix.reshape(5, 3, -1)
        np.testing.assert_allclose(m, flat.mean(-1))
        np.testing.assert_allclose(s, flat.std(-1))
        np.testing.assert_allclose(sk, stats.skew(flat, axis=-1), rtol=1e-8)
        np.testing.assert_allclose(ku, stats.kurtosis(flat, axis=-1, fisher=False), rtol=1e-8)

    def test_uniform_weights_equal_unweighted(self, pix):
        for a, b in zip(moments.moments_from_pixels(pix, np.ones((27, 55))), moments.moments_from_pixels(pix)):
            np.testing.assert_allclose(a, b)

    def test_constant_patch(self):
        m, s, sk, ku = moments.moments_from_pixels(np.full((1, 1, 4, 4), 2.0))
        assert m[0, 0] == 2.0 and s[0, 0] == 0.0
        assert np.isnan(sk[0, 0]) and np.isnan(ku[0, 0])


class TestRawMoments:
    def test_roundtrip(self, pix):
        raw = moments.raw_moments(*moments.moments_from_pixels(pix))
        flat = pix.reshape(5, 3, -1)
        for k in range(4):
            np.testing.assert_allclose(raw[..., k], (flat ** (k + 1)).mean(-1), rtol=1e-9)

    def test_excess_kurtosis_flag(self, pix):
        m, s, sk, ku = moments.moments_from_pixels(pix)
        np.testing.assert_allclose(moments.raw_moments(m, s, sk, ku),
                                   moments.raw_moments(m, s, sk, ku - 3.0, excess_kurtosis=True))

    def test_shape_mismatch(self):
        with pytest.raises(ValueError):
            moments.raw_moments(np.ones(3), np.ones(2), np.ones(3), np.ones(3))


class TestPolynomialFeatures:
    def test_pixel_mean_equals_footprint_features(self, pix):
        """Core M2 property: the footprint mean of per-pixel features equals features from moments."""
        fp = moments.footprint_polynomial_features(*moments.moments_from_pixels(pix), degree=3)
        px = moments.pixel_polynomial_features(pix, degree=3)
        np.testing.assert_allclose(px.mean(axis=(-2, -1)), fp, rtol=1e-9)
        assert fp.shape == (5, 9)

    @pytest.mark.parametrize("deg", [0, 5])
    def test_degree_bounds(self, pix, deg):
        with pytest.raises(ValueError):
            moments.pixel_polynomial_features(pix, degree=deg)


class TestHomogeneityCv:
    def test_values_and_zero_mean(self):
        cv = moments.homogeneity_cv(np.array([[10.0, 20.0], [10.0, 0.0]]), np.array([[1.0, 4.0], [0.0, 1.0]]))
        assert cv[0] == pytest.approx(0.15) and np.isnan(cv[1])
