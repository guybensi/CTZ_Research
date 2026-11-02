import numpy as np
import pytest

pytest.importorskip("torch")

from flux1km.methods import _common, m6d_distill_m1  # noqa: E402
from flux1km.models import get_model  # noqa: E402


class TestTeacherTileTargets:
    def test_shape_and_finite(self, easy):
        teacher = get_model("M1-gbm").fit(easy[0])
        t = m6d_distill_m1.teacher_tile_targets(teacher, easy[2].images, _common.tile_masks(27, 55, (9, 11)))
        assert t.shape == (len(easy[2]), 15) and np.all(np.isfinite(t))


class TestDistilled:
    def test_fit_predict(self, easy):
        train, val, test = easy
        m = get_model("M6d", epochs=2).fit(train, val)
        assert m.targets.shape == (len(train), 15)
        assert m.predict_map(test).shape == (len(test), 27, 55)
