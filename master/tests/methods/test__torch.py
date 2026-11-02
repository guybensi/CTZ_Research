import numpy as np
import pytest

torch = pytest.importorskip("torch")

from flux1km.methods import _torch  # noqa: E402


def _toy(n=12, h=5, w=7, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2, h, w)).astype(np.float32)
    ins = np.full((n, h, w), 1000.0)
    W = np.full((n, h, w), 1.0 / (h * w))
    true_alb = 1 / (1 + np.exp(-X[:, 0]))          # albedo = sigmoid(channel 0)
    y = (true_alb * ins * W).sum(axis=(1, 2))
    return X, ins, W, y


class TestDevice:
    def test_require_gpu_without_cuda_raises(self, monkeypatch):
        monkeypatch.setenv("CTZ_REQUIRE_GPU", "1")
        if torch.cuda.is_available():
            pytest.skip("a GPU is present")
        with pytest.raises(RuntimeError):
            _torch.device()


class TestArchitectures:
    def test_unet_keeps_odd_patch_shape(self):
        net = _torch.make_unet(4, base=4, depth=2)
        assert net(torch.zeros(2, 4, 27, 55)).shape == (2, 1, 27, 55)

    def test_pixel_mlp_has_no_spatial_context(self):
        net = _torch.make_pixel_mlp(3, (8,))
        x = torch.zeros(1, 3, 5, 5)
        y0 = net(x)
        x[0, :, 0, 0] = 5.0
        d = (net(x) - y0).abs()[0, 0]
        assert d[0, 0] > 0 and d[1:, :].max() == 0 and d[0, 1:].max() == 0


class TestTrainMapNet:
    def test_learns_and_is_bounded(self):
        X, ins, W, y = _toy()
        net = _torch.make_pixel_mlp(2, (16,))
        hist = _torch.train_map_net(net, X, ins, W, y, val=(X, ins, W, y), epochs=80, lr=1e-2, batch_size=4, patience=80)
        assert hist["val_rmse"][hist["best_epoch"]] < 0.3 * hist["val_rmse"][0]
        f = _torch.predict_map_net(net, X, ins)
        assert f.shape == ins.shape and np.all(f >= 0) and np.all(f <= ins)

    def test_shape_mismatch_and_no_labels(self):
        X, ins, W, y = _toy()
        with pytest.raises(ValueError):
            _torch.train_map_net(_torch.make_pixel_mlp(2), X[:-1], ins, W, y, epochs=1)
        with pytest.raises(ValueError):
            _torch.train_map_net(_torch.make_pixel_mlp(2), X, ins, W, np.full_like(y, np.nan), epochs=1)

    def test_extra_loss_and_subset_run(self):
        X, ins, W, y = _toy()
        calls = []

        def extra(xb, flux, idx, flipped):
            calls.append(flipped)
            return flux.mean() * 0.0

        _torch.train_map_net(_torch.make_pixel_mlp(2), X, ins, W, y, epochs=2, extra_loss=extra, augment=True,
                             subset_fraction=0.5)
        assert calls

    def test_pretrain_reduces_loss(self):
        X, ins, _, _ = _toy()
        target = 0.3 * ins
        losses = _torch.pretrain_map_net(_torch.make_pixel_mlp(2, (8,)), X, ins, target, epochs=30, lr=1e-2)
        assert losses[-1] < losses[0]


class TestRegularisers:
    def test_smoothness_zero_for_constant_map(self):
        xb = torch.randn(2, 3, 5, 6)
        assert float(_torch.edge_aware_smoothness(xb, torch.full((2, 5, 6), 100.0))) == 0.0

    def test_manifold_zero_when_outputs_equal(self):
        xb = torch.randn(2, 3, 5, 6)
        assert float(_torch.manifold_penalty(xb, torch.full((2, 5, 6), 7.0))) == 0.0


class TestTrainTabularNet:
    def test_fits_linear_target(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(300, 3))
        y = 500 + 50 * X[:, 0]
        pred = _torch.train_tabular_net(X, y, val=(X, y), hidden=(16,), epochs=200, lr=1e-2, batch_size=64, patience=50)
        assert np.sqrt(np.mean((pred(X) - y) ** 2)) < 10

    def test_advisor_loss_and_bad_loss(self):
        X = np.random.default_rng(0).normal(size=(50, 2))
        y = np.log1p(np.full(50, 300.0))
        assert np.isfinite(_torch.train_tabular_net(X, y, loss="wmae_log1p", l1=(0.01,), epochs=2)(X)).all()
        with pytest.raises(ValueError):
            _torch.train_tabular_net(X, y, loss="nope")
