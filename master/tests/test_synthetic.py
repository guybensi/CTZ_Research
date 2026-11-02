import numpy as np
import pytest

from flux1km import aggregation, geometry, io, moments, synthetic


@pytest.fixture
def gran():
    return synthetic.make_synthetic_granule("A2014060.0245", np.random.default_rng(0), n_rows=3, n_cols=4)


class TestMakeSyntheticGranule:
    def test_shapes_and_keys(self, gran):
        assert gran.images["Imviews"].shape == (12, 3, 27, 55)
        assert gran.images["ImLW"].shape == (12, 16, 27, 55)
        assert gran.moments["mSWspec"].shape == (12, 11)
        for k in ("Fsw", "Isw", "SZA", "VZA", "RAA", "Lat", "Lon", "CF", "glintf"):
            assert gran.scalars[k].shape == (12,)

    def test_labels_are_aggregated_truth(self):
        g = synthetic.make_synthetic_granule("A2014060.0245", np.random.default_rng(1), n_rows=2, n_cols=2, label_noise=0.0)
        expected = aggregation.aggregate(g.truth["flux"], aggregation.fov_weights(g.images["FOVorientation"]))
        np.testing.assert_allclose(g.scalars["Fsw"], expected, rtol=1e-5)

    def test_moments_equal_fov_weighted_image_stats(self, gran):
        m, _, _, _ = moments.moments_from_pixels(gran.images["ImSW"], aggregation.fov_weights(gran.images["FOVorientation"])[:, None])
        np.testing.assert_allclose(gran.moments["mSWspec"], m, rtol=1e-6)

    def test_neighbouring_footprints_overlap(self, gran):
        r0, c0 = gran.truth["row0"], gran.truth["col0"]
        dx = c0[1] - c0[0]
        assert 0 < dx < 55 and r0[0] == r0[1]
        np.testing.assert_allclose(gran.truth["flux"][0][:, dx:], gran.truth["flux"][1][:, :55 - dx])


class TestMakeScene:
    def test_ctz_switch_changes_clear_sky_near_clouds_only(self):
        s1 = synthetic.make_scene((60, 80), np.random.default_rng(3), ctz=True)
        s0 = synthetic.make_scene((60, 80), np.random.default_rng(3), ctz=False)
        diff = s1["flux"] - s0["flux"]
        assert np.allclose(diff[s1["cloud"]], 0)
        near = ~s1["cloud"] & (s1["dist_km"] <= 2)
        far = ~s1["cloud"] & (s1["dist_km"] > 30)
        assert diff[near].mean() > 0
        if far.any():
            assert diff[far].mean() < diff[near].mean()

    def test_flux_within_physical_bounds(self):
        s = synthetic.make_scene((50, 50), np.random.default_rng(4))
        assert np.all(s["flux"] > 0) and np.all(s["flux"] <= geometry.toa_insolation(s["sza"]))


class TestWriteDemoDataset:
    def test_writes_all_parts(self, tmp_path):
        ids = synthetic.write_demo_dataset(tmp_path, n_rows=2, n_cols=2)
        assert io.find_granules(tmp_path, require=("moments", "scalars", "images", "truth")) == sorted(ids)


class TestHardScene:
    def test_bounds_and_isw_anisotropy(self):
        g = synthetic.make_synthetic_granule("A2014060.0245", np.random.default_rng(2), n_rows=3, n_cols=8, hard=True, label_noise=0)
        assert np.all(g.truth["flux"] >= 0)
        r = np.pi * g.scalars["Isw"] / g.scalars["Fsw"]
        assert r.std() > 0.01  # the broadband radiance is not flux / pi: angle-dependent anisotropy

    def test_part_of_the_ctz_is_hidden_from_the_channels(self):
        kw = dict(shape=(60, 80), hard=True)
        full = synthetic.make_scene(rng=np.random.default_rng(3), ctz=True, ctz_visible=1.0, **kw)
        hidden = synthetic.make_scene(rng=np.random.default_rng(3), ctz=True, ctz_visible=0.0, **kw)
        np.testing.assert_allclose(full["flux"], hidden["flux"], rtol=1e-6)    # same truth...
        near = ~full["cloud"] & (full["dist_km"] <= 2)
        assert full["sw"][2][near].mean() > hidden["sw"][2][near].mean()     # ...but the channels see less of the CTZ

    def test_flux_is_nonlinear_in_the_channels(self):
        assert synthetic.broadband_albedo(0.8, 0.8, 0.0) < 2 * synthetic.broadband_albedo(0.4, 0.4, 0.0) * 1.1
        assert synthetic.broadband_albedo(0.4, 0.4, 60.0) > synthetic.broadband_albedo(0.4, 0.4, 0.0)

    def test_invalid_ctz_visible(self):
        with pytest.raises(ValueError):
            synthetic.make_scene((10, 10), np.random.default_rng(0), ctz_visible=1.5)

    def test_anisotropy_glint_lobe(self):
        assert synthetic.anisotropy(30.0, 30.0, 0.0, 0.0, 0.0) > 2 * synthetic.anisotropy(30.0, 30.0, 180.0, 60.0, 0.0)
