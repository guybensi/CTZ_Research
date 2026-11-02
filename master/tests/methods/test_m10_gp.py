import numpy as np
import pytest

from flux1km import features
from flux1km.methods import m10_gp
from flux1km.models import get_model


def _rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


class TestRFF:
    def test_kernel_approximation(self):
        f = m10_gp.RFF(2, 4000, 1.0, np.random.default_rng(0))
        x = np.array([[0.0, 0.0], [1.0, 0.0]])
        phi = f(x)
        assert phi[0] @ phi[1] == pytest.approx(np.exp(-0.5), abs=0.05)
        with pytest.raises(ValueError):
            m10_gp.RFF(2, 10, 0.0, np.random.default_rng(0))


class TestVBAgg:
    def test_learns_and_gives_uncertainty(self, bigger):
        train, val, test = bigger
        W = features.footprint_weights(test)
        m = get_model("M10").fit(train)
        clim = get_model("B-clim").fit(train)
        assert _rmse(m.predict_footprint(test, W), test.label) < _rmse(clim.predict_footprint(test, W), test.label)
        sd = m.predict_map_std(test)
        assert sd.shape == (len(test), 27, 55) and np.all(sd >= 0) and sd.max() > 0


class TestMediated:
    def test_fit_mediated_on_moments_only_batch(self, easy):
        from dataclasses import replace
        train, _, test = easy
        moments_only = replace(train, images=None)
        m = get_model("M10-cmp").fit_mediated(train, moments_only)
        assert m.cme.shape[1] == m.params["n_features"] + 1
        assert np.all(np.isfinite(m.predict_map(test)))

    def test_mediating_features_constant_column(self, easy):
        z = m10_gp.mediating_features(easy[0])
        assert np.all(z[:, 0] == 1)
