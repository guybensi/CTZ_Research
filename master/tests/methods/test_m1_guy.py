import numpy as np
import pytest

from flux1km import aggregation, schema
from flux1km.methods import m1_guy
from flux1km.models import get_model

W = aggregation.uniform_weights(27, 55)


def _rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


class TestBroadbandReconstruction:
    def test_trapezoid_and_albedo(self):
        bb, alb = m1_guy.broadband_reconstruction(np.array([[2.0, 2.0, 2.0]]), ["0.47", "0.66", "0.85"], np.array([60.0]))
        assert bb[0] == pytest.approx(2.0 * (0.85 - 0.47))
        assert alb[0] == pytest.approx(np.pi * bb[0] / 680.5)

    def test_unsorted_channels_and_errors(self):
        bb1, _ = m1_guy.broadband_reconstruction(np.array([[1.0, 3.0]]), ["0.47", "1.375"], np.array([0.0]))
        bb2, _ = m1_guy.broadband_reconstruction(np.array([[3.0, 1.0]]), ["1.375", "0.47"], np.array([0.0]))
        assert bb1 == pytest.approx(bb2)
        with pytest.raises(ValueError):
            m1_guy.broadband_reconstruction(np.ones((1, 1)), ["0.47"], np.array([0.0]))


class TestGuyFeatures:
    def test_names_and_no_ceres(self, easy):
        X, names = m1_guy.guy_features(easy[0], "F5")
        assert X.shape == (len(easy[0]), len(names)) and np.all(np.isfinite(X))
        assert "ratio_0.66/0.47" in names and "ratio_12.0/11.0" in names and "ratio_3.75/4.05" in names
        assert {"L_BB", "albedo_BB", "glint"} <= set(names)
        schema.assert_no_ceres_inputs(names)

    def test_sw_only_input_set_has_no_lw_ratios(self, easy):
        _, names = m1_guy.guy_features(easy[0], "F1")
        assert not any("11.0" in n for n in names) and "glint" not in names


class TestModels:
    def test_gbm_learns(self, bigger):
        train, val, test = bigger
        m = get_model("M1-gbm").fit(train, val)
        clim = get_model("B-clim").fit(train)
        assert _rmse(m.predict_footprint(test, W), test.label) < _rmse(clim.predict_footprint(test, W), test.label)
        assert not m.produces_map

    def test_ann_runs(self, easy):
        pytest.importorskip("torch")
        train, val, test = easy
        p = get_model("M1-ann", epochs=3).fit(train, val).predict_footprint(test, W)
        assert np.all(np.isfinite(p))
