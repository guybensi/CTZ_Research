import numpy as np
import pytest

pytest.importorskip("torch")

from flux1km import features  # noqa: E402
from flux1km.methods import _common  # noqa: E402
from flux1km.models import get_model  # noqa: E402


class TestPixelMLP:
    def test_unknown_regulariser(self):
        with pytest.raises(ValueError):
            get_model("M3", reg="tv")

    @pytest.mark.parametrize("reg", ["none", "manifold", "subset"])
    def test_fit_map_bounded(self, easy, reg):
        train, val, test = easy
        m = get_model("M3", reg=reg, epochs=2).fit(train, val)
        maps = m.predict_map(test)
        assert maps.shape == (len(test), 27, 55)
        assert np.all(maps >= 0) and np.all(maps <= _common.pixel_insolation(test) + 1e-3)

    def test_learns_footprints(self, bigger):
        train, val, test = bigger
        m = get_model("M3", epochs=40, lr=3e-3, patience=40).fit(train, val)
        h = m.history["val_rmse"]
        assert min(h) < 0.5 * h[0]

    def test_needs_images(self, easy):
        train = easy[0]
        from dataclasses import replace
        with pytest.raises(ValueError):
            get_model("M3", epochs=1).fit(replace(train, images=None))

    def test_trains_with_scoring_operator(self, easy):
        m = get_model("M3", operator="uniform", epochs=1).fit(easy[0])
        np.testing.assert_allclose(m._weights(easy[0])[0], features.footprint_weights(easy[0], "uniform")[0])
