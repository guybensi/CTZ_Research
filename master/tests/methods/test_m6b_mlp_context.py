import pytest

torch = pytest.importorskip("torch")

from flux1km.methods import m6b_mlp_context  # noqa: E402
from flux1km.models import get_model  # noqa: E402


class TestMLPContext:
    def test_forward_is_pixel_plus_context(self):
        net = m6b_mlp_context.make_mlp_context(3, (8,), width=4)
        x = torch.randn(2, 3, 9, 11)
        torch.testing.assert_close(net(x), net.pixel(x) + net.context(x))

    def test_fit_predict_and_penalty(self, easy):
        train, val, test = easy
        m = get_model("M6b", epochs=2).fit(train, val)
        assert m.predict_map(test).shape == (len(test), 27, 55)
        assert get_model("M6b", context_penalty=0.0).extra_loss(train, None) is None
