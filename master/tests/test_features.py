import numpy as np
import pytest

from flux1km import aggregation, features, synthetic
from flux1km.table import build_table


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    gs = [synthetic.make_synthetic_granule(g, rng, n_rows=2, n_cols=3) for g in ("A2014060.0245", "A2014060.0530")]
    return gs, build_table(gs)


class TestMakeBatch:
    def test_make_batch_order_and_alignment(self, data):
        gs, t = data
        rows = np.array([7, 0, 6, 1])
        b = features.make_batch(gs, t, rows)
        assert len(b) == 4 and b.images["ImSW"].shape == (4, 11, 27, 55)
        for k, r in enumerate(rows):
            g = gs[0] if t["granule"][r] == "A2014060.0245" else gs[1]
            i = t["index"][r]
            assert b.label[k] == g.scalars["Fsw"][i]
            np.testing.assert_array_equal(b.images["ImSW"][k], g.images["ImSW"][i])
            np.testing.assert_array_equal(b.truth[k], g.truth["flux"][i])

    def test_make_batch_missing_granule(self, data):
        gs, t = data
        with pytest.raises(KeyError):
            features.make_batch(gs[:1], t, np.array([7]))

    def test_make_batch_moments_only(self, data):
        gs, t = data
        import copy
        g2 = [copy.copy(g) for g in gs]
        g2[1].images = None
        b = features.make_batch(g2, t, np.array([0, 7]))
        assert not b.has_images and b.moments is not None
        with pytest.raises(ValueError):
            features.pixel_features(b)


class TestPixelFeatures:
    def test_pixel_features_shapes_and_names(self, data):
        gs, t = data
        b = features.make_batch(gs, t, np.arange(3))
        X, names = features.pixel_features(b, "F1")
        assert X.shape == (3, 4 + 4, 27, 55) and names[-1] == "cosRAA"
        X5, n5 = features.pixel_features(b, "F5")
        assert n5[-1] == "glint" and X5.shape[1] == len(n5)


class TestFootprintFeatures:
    def test_footprint_features(self, data):
        gs, t = data
        b = features.make_batch(gs, t, np.arange(5))
        X, names = features.footprint_features(b, "F2", stats=("m", "std"))
        assert X.shape == (5, 2 * 7 + 4) and np.all(np.isfinite(X))

    def test_no_ceres_inputs_possible(self, data):
        gs, t = data
        b = features.make_batch(gs, t, np.arange(2))
        _, names = features.footprint_features(b, "F5", stats=("m", "std", "skew", "kurt"))
        assert not any(n in ("Isw", "Fsw", "Ilw", "Flw") for n in names)


class TestPolyFeatures:
    def test_poly_features_aggregate_to_footprint_features(self, data):
        """The M2 bridge holds end-to-end through the feature builders (geometry up to pixel vs scalar)."""
        gs, t = data
        b = features.make_batch(gs, t, np.arange(6))
        F, fn = features.footprint_poly_features(b, "F2", degree=2)
        P, pn = features.pixel_poly_features(b, "F2", degree=2)
        assert fn == pn
        n_poly = 7 * 2
        w = aggregation.fov_weights(b.images["FOVorientation"])[:, None]  # moments are FOV-weighted
        np.testing.assert_allclose((P[:, :n_poly] * w).sum(axis=(-2, -1)), F[:, :n_poly], rtol=1e-4)


class TestPixelGeometry:
    def test_converts_imviews_raa_to_ceres_convention_and_folds(self):
        v = np.zeros((1, 3, 1, 3))
        v[0, 2, 0] = [110.0, 0.0, 270.0]  # 270 can come out of the advisor's fold; it means 90
        _, _, raa = features.pixel_geometry({"Imviews": v})
        np.testing.assert_allclose(raa[0, 0], [70.0, 180.0, 90.0])


class TestFootprintWeights:
    def test_normalised_and_oriented(self):
        g = synthetic.make_synthetic_granule("A2014060.0245", np.random.default_rng(0), n_rows=1, n_cols=3)
        t = build_table([g])
        b = features.make_batch([g], t, np.arange(3))
        w = features.footprint_weights(b)
        np.testing.assert_allclose(w.sum(axis=(1, 2)), 1.0)
        np.testing.assert_allclose(w, aggregation.fov_weights(g.images["FOVorientation"]))
        assert list(b.granule) == [g.granule_id] * 3
