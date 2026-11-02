import numpy as np
import pytest

torch = pytest.importorskip("torch")

from flux1km.methods import m6a_cnn_tabular  # noqa: E402
from flux1km.models import get_model  # noqa: E402


class TestCNNTabular:
    def test_tabular_branch_changes_output(self):
        net = m6a_cnn_tabular.make_cnn_tabular(3, 2, base=4)
        x = torch.zeros(1, 5, 27, 55)
        y0 = net(x)
        x[:, 3:] = 2.0  # tabular channels only
        assert net(x).shape == (1, 1, 27, 55) and (net(x) - y0).abs().max() > 0

    def test_inputs_carry_constant_tabular_channels(self, easy):
        m = get_model("M6a", base=4, epochs=1)
        X = m.inputs(easy[2])
        assert X.shape[1] == m.n_pixel + m.n_tab
        np.testing.assert_allclose(X[:, m.n_pixel:].std(axis=(2, 3)), 0, atol=1e-9)

    def test_fit_predict(self, easy):
        train, val, test = easy
        maps = get_model("M6a", base=4, epochs=2).fit(train, val).predict_map(test)
        assert maps.shape == (len(test), 27, 55) and np.all(np.isfinite(maps))
