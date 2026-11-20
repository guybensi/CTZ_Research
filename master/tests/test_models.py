import numpy as np
import pytest

from flux1km import aggregation, features, models, synthetic
from flux1km.table import build_table


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    gs = [synthetic.make_synthetic_granule(g, rng, n_rows=4, n_cols=10, label_noise=0.5) for g in synthetic.DEMO_GRANULES[:3]]
    t = build_table(gs)
    train = features.make_batch(gs, t, t.rows_of(gs[0].granule_id).tolist() + t.rows_of(gs[1].granule_id).tolist())
    test = features.make_batch(gs, t, t.rows_of(gs[2].granule_id))
    return train, test

W = aggregation.uniform_weights(27, 55)

def _rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


class TestRegistry:
    def test_registry_and_unknown(self):
        assert {"B-clim", "B-const", "M0-ridge", "B-bagpixel", "M2"} <= set(models.MODELS)
        with pytest.raises(KeyError):
            models.get_model("nope")

    def test_param_grid_nonempty(self):
        for cls in models.MODELS.values():
            assert len(cls.param_grid()) >= 1


class TestBaselines:
    def test_climatology_shapes(self, data):
        train, test = data
        m = models.get_model("B-clim").fit(train)
        assert m.predict_map(test).shape == (len(test), 27, 55)
        assert m.predict_footprint(test, W).shape == (len(test),)

    def test_unitr_reproduces_labels(self, data):
        _, test = data
        m = models.get_model("B-const").fit(test)
        np.testing.assert_allclose(m.predict_footprint(test, W), test.label)
        assert m.uses_test_labels

    def test_footprint_ridge_has_no_map(self, data):
        train, test = data
        m = models.get_model("M0-ridge").fit(train)
        assert m.predict_footprint(test, W).shape == (len(test),)
        with pytest.raises(NotImplementedError):
            m.predict_map(test)


class TestPixelPoly:
    def test_m2_map_aggregates_to_its_footprint_prediction(self, data):
        """Exactness of the M2 bridge: aggregated map == moments-based prediction (up to geometry)."""
        train, test = data
        m = models.get_model("M2", degree=2, target="flux").fit(train)
        w = aggregation.fov_weights(test.images["FOVorientation"])  # the operator the moments were built with
        from_map = m.predict_footprint(test, w)
        import copy
        no_img = copy.copy(test); no_img.images = None
        from_moments = m.predict_footprint(no_img, w)
        # exact for the radiance terms; geometry enters through cos/sin of averaged angles (~1%)
        np.testing.assert_allclose(from_map, from_moments, rtol=0.01)

    def test_learned_models_beat_climatology_on_synthetic(self, data):
        train, test = data
        clim = _rmse(models.get_model("B-clim").fit(train).predict_footprint(test, W), test.label)
        for name in ("M0-ridge", "M2", "B-bagpixel"):
            r = _rmse(models.get_model(name).fit(train).predict_footprint(test, W), test.label)
            assert r < clim, (name, r, clim)

    def test_m2_pixel_error_reasonable(self, data):
        train, test = data
        m = models.get_model("M2").fit(train)
        pm = m.predict_map(test)
        assert pm.shape == test.truth.shape
        assert _rmse(pm, test.truth) < _rmse(np.broadcast_to(test.truth.mean(), test.truth.shape), test.truth)

    def test_sample_weights_used(self, data):
        train, _ = data
        import copy
        t2 = copy.copy(train); t2.weight = np.zeros(len(train))
        with pytest.raises(ValueError):
            models.get_model("M2").fit(t2)


ALL_METHODS = {"M0-ridge", "M0-allsky37", "M0-advisor", "M1-gbm", "M1-ann", "M2", "M3", "M4", "M5", "M6a", "M6b", "M6c", "M6d",
               "M6e", "M6f", "M7", "M8", "M9", "M9-local", "M10", "M10-cmp", "B-clim", "B-const", "B-bagpixel"}


class TestRegister:
    def test_every_plan_method_is_registered(self):
        assert ALL_METHODS <= set(models.MODELS)

    def test_duplicate_name_rejected(self):
        class Dup(models.FluxModel):
            name = "M2"

            def fit(self, train, val=None):
                return self

        with pytest.raises(ValueError):
            models.register(Dup)

    def test_test_label_readers_are_flagged(self):
        assert {n for n, c in models.MODELS.items() if c.uses_test_labels} == {"B-const", "M6e", "M7", "M9-local"}


class TestNoCeresInputs:
    """PLAN.md rule 1: scrambling every CERES-derived scalar of the TEST footprints must not change any prediction."""

    @pytest.mark.parametrize("name", sorted(ALL_METHODS))
    def test_predictions_ignore_ceres_quantities(self, data, name):
        from dataclasses import replace
        cls = models.MODELS[name]
        if cls.uses_test_labels:
            pytest.skip("reads the test labels by design (never ranked)")
        if cls.needs_torch:
            pytest.importorskip("torch")
        train, test = data
        kw = {"epochs": 1} if cls.needs_torch else {}
        if name == "M4":
            kw = {"n_rounds": 3}
        m = models.get_model(name, **kw).fit(train)
        rng = np.random.default_rng(1)
        junk = {k: rng.normal(size=len(test)) * 1e3 for k in ("Fsw", "Flw", "Isw", "Ilw", "Clrf")}
        scrambled = replace(test, scalars={**test.scalars, **junk}, label=junk["Fsw"])
        np.testing.assert_allclose(m.predict_footprint(test, W), m.predict_footprint(scrambled, W), rtol=1e-6)


class TestChunkPositions:
    def test_slices_cover_every_footprint_once(self):
        from flux1km.models import chunk_positions
        b = type("B", (), {"__len__": lambda s: 7, "granule": None})()
        cov = np.concatenate([np.arange(7)[p] for p in chunk_positions(b, 3)])
        assert cov.tolist() == list(range(7))
        assert len(chunk_positions(b, 100)) == 1

    def test_whole_granules(self):
        from flux1km.models import chunk_positions
        b = type("B", (), {"__len__": lambda s: 5, "granule": np.array(["b", "a", "b", "a", "c"])})()
        assert [p.tolist() for p in chunk_positions(b, 1, whole_granules=True)] == [[0, 2], [1, 3], [4]]

    def test_whole_granules_without_ids_is_one_chunk(self):
        from flux1km.models import chunk_positions
        b = type("B", (), {"__len__": lambda s: 4, "granule": None})()
        assert chunk_positions(b, 1, whole_granules=True) == [slice(0, 4)]

    def test_empty_and_bad_size(self):
        from flux1km.models import chunk_positions
        b = type("B", (), {"__len__": lambda s: 0, "granule": None})()
        assert chunk_positions(b, 3) == []
        with pytest.raises(ValueError):
            chunk_positions(b, 0)


class TestPredictMapChunked:
    @pytest.mark.parametrize("name", ["M2", "B-bagpixel", "M9", "M7"])
    def test_same_as_one_call(self, name, monkeypatch):
        from flux1km import features, synthetic
        from flux1km import models as mdl
        from flux1km.table import build_table
        rng = np.random.default_rng(1)
        gs = [synthetic.make_synthetic_granule(g, rng, n_rows=2, n_cols=3) for g in ("A2014060.0245", "A2014061.0530")]
        t = build_table(gs)
        n = len(t["granule"])
        b = features.make_batch(gs, t, np.arange(n))
        m = mdl.get_model(name, seed=0).fit(b)
        full = m.predict_map(b)
        np.testing.assert_allclose(mdl.predict_map_chunked(m, b, size=4), full, atol=1e-9)
        w = np.full(full.shape[1:], 1.0 / np.prod(full.shape[1:]))
        monkeypatch.setattr(mdl, "PREDICT_CHUNK", 3)
        np.testing.assert_allclose(m.predict_footprint(b, w) if m.produces_map else 0, (full * w).sum(axis=(1, 2)),
                                   atol=1e-9)


class TestDiskBackedTraining:
    @pytest.mark.parametrize("name", ["M5", "M6c", "M6a"])
    def test_same_maps_as_in_memory(self, name, tmp_path):
        pytest.importorskip("torch")
        rng = np.random.default_rng(2)
        gs = [synthetic.make_synthetic_granule(g, rng, n_rows=3, n_cols=4) for g in ("A2014060.0245", "A2014061.0530")]
        t = build_table(gs)
        n = len(t["granule"])
        tr, te = np.arange(0, n - 6), np.arange(n - 6, n)
        params = dict(epochs=2, seed=0) if name != "M6c" else dict(epochs=2, seed=0, pretrain_epochs=1)
        maps = []
        for store in (None, tmp_path):
            train = features.make_batch(gs, t, tr, store=store)
            test = features.make_batch(gs, t, te, store=store)
            m = models.get_model(name, **params).fit(train)
            maps.append(m.predict_map(test))
        np.testing.assert_array_equal(maps[0], maps[1])
        assert list(tmp_path.glob("batch_*/X_*.npy"))      # the training inputs went to disk


class TestMapNetReproducible:
    @pytest.mark.parametrize("name", ["M3", "M5"])
    def test_same_seed_same_map(self, name):
        pytest.importorskip("torch")
        rng = np.random.default_rng(3)
        gs = [synthetic.make_synthetic_granule(g, rng, n_rows=3, n_cols=4) for g in ("A2014060.0245", "A2014061.0530")]
        t = build_table(gs)
        b = features.make_batch(gs, t, np.arange(len(t["granule"])))
        a = models.get_model(name, epochs=2, seed=0).fit(b).predict_map(b)
        import torch
        torch.rand(5)                                   # disturb the global generator in between
        c = models.get_model(name, epochs=2, seed=0).fit(b).predict_map(b)
        np.testing.assert_array_equal(a, c)
