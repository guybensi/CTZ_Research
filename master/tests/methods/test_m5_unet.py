import numpy as np
import pytest

pytest.importorskip("torch")

from flux1km.methods import _common  # noqa: E402
from flux1km.models import get_model  # noqa: E402


class TestUNet:
    def test_fit_predict_bounded(self, easy):
        train, val, test = easy
        m = get_model("M5", base=4, epochs=2).fit(train, val)
        maps = m.predict_map(test)
        assert maps.shape == (len(test), 27, 55)
        assert np.all(maps >= 0) and np.all(maps <= _common.pixel_insolation(test) + 1e-3)

    def test_no_smoothness_term_when_weight_zero(self, easy):
        m = get_model("M5", smooth_weight=0.0)
        assert m.extra_loss(easy[0], None) is None
