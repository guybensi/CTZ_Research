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
