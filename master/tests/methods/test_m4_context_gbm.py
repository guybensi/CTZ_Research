import numpy as np

from flux1km import features
from flux1km.methods import m4_context_gbm
from flux1km.models import get_model


def _rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


class TestPixelTable:
    def test_shape_with_and_without_context(self, easy):
        T, names = m4_context_gbm.pixel_table(easy[2], "F5")
        T0, names0 = m4_context_gbm.pixel_table(easy[2], "F5", use_context=False)
        assert T.shape == (len(easy[2]) * 27 * 55, len(names)) and len(names) == len(names0) + 17
        assert T0.shape[1] == len(names0)


class TestContextGBM:
    def test_boosting_beats_climatology_and_is_bounded(self, bigger):
        train, val, test = bigger
        W = features.footprint_weights(test)
        m = get_model("M4", n_rounds=40).fit(train, val)
        clim = get_model("B-clim").fit(train)
        assert _rmse(m.predict_footprint(test, W), test.label) < _rmse(clim.predict_footprint(test, W), test.label)
        maps = m.predict_map(test)
        assert maps.shape == (len(test), 27, 55) and np.all(maps >= 0)
        assert 0 < len(m.trees) <= 40

    def test_early_stopping_truncates(self, easy):
        train, val, _ = easy
        m = get_model("M4", n_rounds=60, patience=2, learning_rate=1.0).fit(train, val)
        assert len(m.trees) <= 60
