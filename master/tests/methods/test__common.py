import numpy as np
import pytest

from flux1km import moments, schema
from flux1km.methods import _common


class TestStandardizer:
    def test_zero_mean_unit_std_and_constant_feature(self):
        X = np.stack([np.arange(10.0), np.full(10, 3.0)], axis=1)
        z = _common.Standardizer(axis=1).fit(X).transform(X)
        np.testing.assert_allclose(z[:, 0].mean(), 0, atol=1e-12)
        np.testing.assert_allclose(z[:, 0].std(), 1, atol=1e-12)
        assert np.all(z[:, 1] == 0)  # constant feature: sd 1, centred

    def test_nan_becomes_zero(self):
        s = _common.Standardizer(axis=1).fit(np.array([[1.0], [3.0]]))
        assert s.transform(np.array([[np.nan]]))[0, 0] == 0.0

    def test_pixel_axis(self):
        X = np.random.default_rng(0).normal(size=(4, 3, 5, 6))
        z = _common.Standardizer(axis=1).fit(X).transform(X)
        np.testing.assert_allclose(z.mean(axis=(0, 2, 3)), 0, atol=1e-10)


    def test_fit_chunks_equals_fit(self):
        X = np.random.default_rng(1).normal(size=(7, 3, 4, 5))
        X[2, 1, 0, 0] = np.nan
        for cap in (10_000, 37):                                   # all entries, and a random sample of them
            a = _common.Standardizer(axis=1).fit(X, max_samples=cap, seed=4)
            b = _common.Standardizer(axis=1).fit_chunks([X[:3], X[3:4], X[4:]], total=7 * 4 * 5, max_samples=cap, seed=4)
            np.testing.assert_array_equal(a.mu, b.mu)
            np.testing.assert_array_equal(a.sd, b.sd)

    def test_fit_chunks_wrong_total(self):
        with pytest.raises(ValueError):
            _common.Standardizer(axis=1).fit_chunks([np.zeros((2, 3, 4, 5))], total=41)


class TestChunkedRows:
    def test_equals_whole_batch_in_memory_and_on_disk(self, easy, tmp_path, monkeypatch):
        from flux1km import features
        from flux1km import models as mdl
        from flux1km.methods.m4_context_gbm import pixel_table
        train = easy[0]
        monkeypatch.setattr(mdl, "PREDICT_CHUNK", 3)
        ref, names = pixel_table(train, "F5")
        got, n2 = _common.chunked_rows(train, lambda b: pixel_table(b, "F5"))
        np.testing.assert_array_equal(got, ref)
        assert n2 == names
        # a disk-backed batch gives a memory-mapped result with the same values
        disk = features.take_batch(train, slice(0, len(train)))
        disk.images = {k: _to_memmap(v, tmp_path / f"{k}.npy") for k, v in train.images.items()}
        got_d, _ = _common.chunked_rows(disk, lambda b: pixel_table(b, "F5"))
        assert isinstance(got_d, np.memmap)
        np.testing.assert_array_equal(got_d, ref)


class TestPredictRows:
    def test_blocks_give_the_same_prediction(self):
        from sklearn.tree import DecisionTreeRegressor
        rng = np.random.default_rng(0)
        X, y = rng.normal(size=(500, 4)), rng.normal(size=500)
        tree = DecisionTreeRegressor(max_depth=4, random_state=0).fit(X, y)
        np.testing.assert_array_equal(_common.predict_rows(tree, X, chunk_rows=37), tree.predict(X))


def _to_memmap(a, path):
    m = np.lib.format.open_memmap(path, mode="w+", dtype=a.dtype, shape=a.shape)
    m[:] = a
    m.flush()
    return np.load(path, mmap_mode="r")


class TestInsolationAndTarget:
    def test_shapes_and_albedo(self, easy):
        train, _, _ = easy
        assert _common.pixel_insolation(train).shape == (len(train), 27, 55)
        a = _common.albedo_target(train)
        assert np.all((a > 0) & (a < 1))

    def test_moments_only_batch_raises(self, easy):
        from flux1km.methods._common import moment_batch
        z, zl = np.zeros((1, 11)), np.zeros((1, 16))
        b = moment_batch((z, z, z, z), (zl, zl, zl, zl), [0.0], [0.0], [0.0])
        with pytest.raises(ValueError):
            _common.pixel_insolation(b)


class TestReflectanceAndBright:
    def test_clouds_brighter_than_clear(self, easy):
        _, _, test = easy
        r = _common.reflectance(test, "0.66")
        cloud = test.truth > np.percentile(test.truth, 90)
        assert r[cloud].mean() > r[~cloud].mean()

    def test_lw_channel_rejected(self, easy):
        with pytest.raises(KeyError):
            _common.reflectance(easy[0], "11.0")

    def test_distance_to_bright(self):
        m = np.zeros((2, 3, 3), bool)
        m[0, 1, 1] = True
        d = _common.distance_to_bright(m, cap_km=5.0)
        assert d[0, 1, 1] == 0 and d[0, 0, 1] == 1.0
        assert np.all(d[1] == 5.0)  # no bright pixel -> cap


class TestContextFeatures:
    def test_shape_names_and_no_ceres(self, easy):
        C, names = _common.context_features(easy[2])
        assert C.shape == (len(easy[2]), len(names), 27, 55)
        assert len(names) == 3 * 5 + 2
        schema.assert_no_ceres_inputs(names)
        assert np.all(np.isfinite(C))

    def test_needs_images(self):
        z, zl = np.zeros((1, 11)), np.zeros((1, 16))
        b = _common.moment_batch((z, z, z, z), (zl, zl, zl, zl), [0.0], [0.0], [0.0])
        with pytest.raises(ValueError):
            _common.context_features(b)


class TestTilesAndWindows:
    def test_tiles_cover_patch_once(self):
        t = _common.tile_masks(27, 55, (9, 11))
        assert t.shape == (15, 27, 55) and np.all(t.sum(0) == 1)

    def test_ragged_edges_and_bad_tile(self):
        assert np.all(_common.tile_masks(5, 5, (2, 2)).sum(0) == 1)
        with pytest.raises(ValueError):
            _common.tile_masks(5, 5, (0, 2))

    def test_window_moments_match_uniform_stats(self):
        x = np.random.default_rng(1).normal(size=(2, 3, 4, 5))
        mask = np.zeros((4, 5), bool); mask[1:3, 1:4] = True
        m, s, sk, ku = _common.window_moments(x, mask)
        m2, s2, sk2, ku2 = moments.moments_from_pixels(x[..., 1:3, 1:4])
        np.testing.assert_allclose(m, m2); np.testing.assert_allclose(s, s2)
        np.testing.assert_allclose(sk, sk2); np.testing.assert_allclose(ku, ku2 - 3.0)

    def test_window_batch_geometry_and_labels(self, easy):
        b = _common.window_batch(easy[2].images, np.ones((27, 55), bool))
        assert len(b) == len(easy[2]) and np.isnan(b.label).all()
        np.testing.assert_allclose(b.moments["mSWspec"], easy[2].images["ImSW"].mean(axis=(2, 3)), rtol=1e-5)
