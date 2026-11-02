import numpy as np
import pytest

from flux1km import features
from flux1km.methods import m6e_footprint_norm as m6e
from flux1km.models import get_model


class TestNormaliseToFootprints:
    @pytest.mark.parametrize("mode", m6e.MODES)
    def test_reproduces_labels_exactly(self, mode):
        rng = np.random.default_rng(0)
        maps = rng.uniform(50, 500, size=(4, 5, 6))
        w = rng.uniform(0, 1, size=(4, 5, 6))
        y = rng.uniform(100, 400, size=4)
        out = m6e.normalise_to_footprints(maps, w, y, mode)
        np.testing.assert_allclose((out * w).sum(axis=(1, 2)) / w.sum(axis=(1, 2)), y)

    def test_scene_mode_moves_bright_pixels_more(self):
        maps = np.array([[[10.0, 100.0]]])
        out = m6e.normalise_to_footprints(maps, np.full((1, 1, 2), 0.5), np.array([65.0]), "scene")
        d = out - maps
        assert d[0, 0, 1] > d[0, 0, 0] >= 0

    def test_nan_label_and_bad_mode(self):
        maps = np.ones((1, 2, 2))
        np.testing.assert_array_equal(m6e.normalise_to_footprints(maps, np.ones((2, 2)), np.array([np.nan])), maps)
        with pytest.raises(ValueError):
            m6e.normalise_to_footprints(maps, np.ones((2, 2)), np.array([1.0]), "box")


class TestModel:
    def test_reads_test_labels_and_matches_them(self, easy):
        train, _, test = easy
        m = get_model("M6e").fit(train)
        assert m.uses_test_labels
        W = features.footprint_weights(test)
        np.testing.assert_allclose(m.predict_footprint(test, W), test.label, rtol=1e-6)
