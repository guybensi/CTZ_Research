import numpy as np
import pytest

pytest.importorskip("torch")

from flux1km.models import get_model  # noqa: E402


class TestUNetFromM2:
    def test_pretraining_reproduces_the_m2_map(self, easy):
        train, val, _ = easy
        m = get_model("M6c", base=4, epochs=0, pretrain_epochs=60, lr=3e-3).fit(train, val)  # pretraining only
        assert m.pretrain_loss[-1] < 0.1 * m.pretrain_loss[0]
        assert np.corrcoef(m.predict_map(train).ravel(), m.m2.predict_map(train).ravel())[0, 1] > 0.9

    def test_fine_tuning_runs(self, easy):
        train, val, test = easy
        m = get_model("M6c", base=4, epochs=2, pretrain_epochs=2).fit(train, val)
        assert m.predict_map(test).shape == (len(test), 27, 55)
