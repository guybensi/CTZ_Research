"""Tests for flux1km.mapviz (1 km map figures)."""
import json

import numpy as np
import pytest

from flux1km import mapviz


class TestCloudMasks:
    def test_myd35_masks_and_proxy(self):
        conf = np.array([[[0, 1, 2, 3]]], np.uint8)
        m = mapviz.cloud_masks({"ImCloud": conf}, np.array([[[0.5, 0.3, 0.1, 0.0]]]))
        assert list(m) == ["MYD35 relaxed", "MYD35 strict", "0.66 um proxy"]
        assert m["MYD35 relaxed"][0, 0].tolist() == [True, False, False, False]
        assert m["MYD35 strict"][0, 0].tolist() == [True, True, False, False]
        assert m["0.66 um proxy"][0, 0].tolist() == [True, True, False, False]

    def test_without_imcloud_only_the_proxy(self):
        assert list(mapviz.cloud_masks({}, np.zeros((1, 2, 2)))) == ["0.66 um proxy"]

    def test_threshold(self):
        r = np.full((1, 1, 1), 0.25)
        assert not mapviz.cloud_masks({}, r, proxy_threshold=0.3)["0.66 um proxy"].any()


class TestCloudFraction:
    def test_plain_and_weighted(self):
        m = np.array([[[True, False], [False, False]]])
        assert mapviz.cloud_fraction(m).tolist() == [25.0]
        assert mapviz.cloud_fraction(m, np.array([[0.7, 0.1], [0.1, 0.1]]))[0] == pytest.approx(70.0)

    def test_per_footprint_weights_and_negative_lobes(self):
        m = np.ones((2, 1, 2), bool)
        w = np.array([[[1.0, -0.5]], [[0.0, 2.0]]])
        assert mapviz.cloud_fraction(m, w).tolist() == [100.0, 100.0]

    def test_zero_weights_do_not_divide_by_zero(self):
        assert mapviz.cloud_fraction(np.ones((1, 1, 1), bool), np.zeros((1, 1))).tolist() == [0.0]


class TestChooseWindow:
    def test_prefers_all_regimes(self):
        r0 = np.array([0, 0, 100, 0]); c0 = np.array([0, 10, 0, 300]); cf = np.array([0.0, 50.0, 99.0, 99.0])
        top, left = mapviz.choose_window(r0, c0, cf, size=(40, 80))
        inside = mapviz.footprints_in_window(r0, c0, (top, left), size=(40, 80))
        assert inside[:2].all()

    def test_window_inside_the_footprints_area(self):
        r0 = np.arange(0, 200, 9); c0 = np.zeros_like(r0)
        top, left = mapviz.choose_window(r0, c0, np.full(len(r0), 50.0), size=(60, 55))
        assert top >= 0 and left >= 0

    def test_no_footprints(self):
        with pytest.raises(ValueError):
            mapviz.choose_window([], [], [])


class TestFootprintsInWindow:
    def test_whole_patch_only(self):
        assert mapviz.footprints_in_window([0, 124, 123], [0, 0, 0], (0, 0), size=(150, 100)).tolist() == [True, False, True]

    def test_left_of_window(self):
        assert mapviz.footprints_in_window([10], [5], (0, 10), size=(150, 100)).tolist() == [False]


class TestMosaic:
    def test_overlap_mean_and_gap(self):
        p = np.stack([np.ones((1, 2)), 3 * np.ones((1, 2))])
        assert mapviz.mosaic(p, [0, 0], [0, 1]).tolist() == [[1.0, 2.0, 3.0]]
        g = mapviz.mosaic(p, [0, 0], [0, 3])
        assert np.isnan(g[0, 2]) and g[0, 3] == 3.0

    def test_corner_and_shape(self):
        g = mapviz.mosaic(np.ones((1, 1, 1)), [5], [7], corner=(5, 6), shape=(2, 3))
        assert g[0, 1] == 1.0 and np.isnan(g).sum() == 5

    def test_nan_pixels_do_not_count(self):
        p = np.array([[[np.nan]], [[4.0]]])
        assert mapviz.mosaic(p, [0, 0], [0, 0]).tolist() == [[4.0]]

    def test_patch_outside_grid(self):
        with pytest.raises(ValueError):
            mapviz.mosaic(np.ones((1, 2, 2)), [0], [0], shape=(1, 1))

    def test_bool_mask_gives_fraction(self):
        p = np.array([[[True]], [[False]]])
        assert mapviz.mosaic(p, [0, 0], [0, 0]).tolist() == [[0.5]]


class TestPaintFootprints:
    def test_nearest_centre(self):
        out = mapviz.paint_footprints([1.0, 5.0], [0, 0], [0, 2], (0, 0), (1, 4), patch=(1, 2))
        assert out.tolist() == [[1.0, 1.0, 5.0, 5.0]]

    def test_uncovered_is_nan(self):
        out = mapviz.paint_footprints([1.0], [0], [0], (0, 0), (1, 3), patch=(1, 2))
        assert out[0, :2].tolist() == [1.0, 1.0] and np.isnan(out[0, 2])


class TestClearFluxVsDistance:
    def test_decay(self):
        f = np.array([[500.0, 90.0, 80.0, 70.0, 60.0, 50.0]]); c = np.array([[1.0, 0, 0, 0, 0, 0]])
        x, m = mapviz.clear_flux_vs_distance(f, c, bins=(0.5, 1.5, 2.5))
        assert x.tolist() == [1.0, 2.0] and m.tolist() == [90.0, 80.0]

    def test_no_cloud_all_nan(self):
        _, m = mapviz.clear_flux_vs_distance(np.ones((3, 3)), np.zeros((3, 3)), bins=(0, 1, 2))
        assert np.isnan(m).all()

    def test_pixels_nearer_the_edge_than_the_cloud_are_dropped(self):
        f = np.full((1, 7), 10.0); c = np.zeros((1, 7)); c[0, 0] = 1
        f[0, 6] = np.nan                          # uncovered: the pixel at distance 5 is 1 from the edge
        _, m = mapviz.clear_flux_vs_distance(f, c, bins=(4.5, 5.5))
        assert np.isnan(m[0])


class TestPickFootprints:
    def test_closest_to_targets(self):
        assert mapviz.pick_footprints([0.0, 40.0, 60.0, 99.0, 100.0]) == {"clear": 0, "broken": 1, "overcast": 3}

    def test_missing_regimes_and_nan(self):
        assert mapviz.pick_footprints([np.nan, 50.0]) == {"broken": 1}
        assert mapviz.pick_footprints([]) == {}


class TestLoadParams:
    def test_drops_operator_and_path(self, tmp_path):
        p = tmp_path / "s.json"
        p.write_text(json.dumps({"folds": [{"name": "f0", "params": {"M5": {"lr": 1e-3, "operator": "x", "psf_path": "y"}}}]}))
        assert mapviz.load_params(p, "f0") == {"M5": {"lr": 1e-3}}

    def test_unknown_fold(self, tmp_path):
        p = tmp_path / "s.json"
        p.write_text(json.dumps({"folds": []}))
        with pytest.raises(KeyError):
            mapviz.load_params(p, "f9")

    def test_real_img6_summary(self):
        from pathlib import Path

        s = Path(__file__).resolve().parents[1] / "docs/results/2008_Pacific_img6/summary.json"
        if not s.exists():
            pytest.skip("img6 summary not in the repo")
        params = mapviz.load_params(s, "day-kfold:0")
        assert params["M5"]["base"] in (16, 32) and "psf_path" not in params["M5"]


class TestSharpnessTable:
    def test_per_method_and_truth(self):
        img = np.random.default_rng(0).random((2, 27, 55))
        t = mapviz.sharpness_table({"vis": img, "truth": 2 * img}, {"A": img, "B": np.ones((2, 27, 55))})
        assert t["A"] == pytest.approx(1.0) and t["B"] == 0.0 and t["truth"] == pytest.approx(1.0)


class TestWriteSharpness:
    def test_sorted_by_mean_and_nan_last(self, tmp_path):
        p = mapviz.write_sharpness(tmp_path / "s.txt", {"g1": {"A": 0.2, "B": 0.8, "C": float("nan")}, "g2": {"A": 0.4, "B": 0.6}})
        rows = [line.split()[0] for line in p.read_text().splitlines()[4:]]
        assert rows == ["B", "A", "C"]
        assert "0.70" in p.read_text().splitlines()[4]


class TestSaveLoadMaps:
    def test_round_trip(self, tmp_path):
        data = {"label": np.array([1.0, 2.0]), "mask__MYD35 relaxed": np.zeros((2, 3, 4), bool)}
        maps = {"M2": np.ones((2, 3, 4), np.float32), "B-const": np.zeros((2, 3, 4), np.float32)}
        mapviz.save_maps(tmp_path / "maps_g.npz", data, maps)
        d, m = mapviz.load_maps(tmp_path / "maps_g.npz")
        assert set(m) == {"M2", "B-const"} and d["label"].tolist() == [1.0, 2.0]
        assert d["mask__MYD35 relaxed"].dtype == bool


@pytest.fixture(scope="module")
def demo_run(tmp_path_factory):
    pytest.importorskip("matplotlib")
    pytest.importorskip("sklearn")
    out = tmp_path_factory.mktemp("viz")
    paths = mapviz.main(["--demo", "--models", "B-const", "M2", "M9", "--out", str(out)])
    return out, paths


class TestMain:
    def test_demo_writes_figures_and_maps(self, demo_run):
        out, paths = demo_run
        names = {p.name.split("_A")[0] for p in paths}
        assert {"fig1_scene", "fig3_scatter", "fig4_ctz", "fig5_masks"} <= names
        assert any(n.startswith("fig2_footprint_") for n in names)
        assert (paths[0].parent / "sharpness.txt").exists()
        run = paths[0].parent
        data, maps = mapviz.load_maps(next(run.glob("maps_*.npz")))
        assert set(maps) == {"B-const", "M2", "M9"}
        assert {"mask__MYD35 relaxed", "mask__MYD35 strict", "mask__0.66 um proxy", "truth"} <= set(data)
        n = len(data["label"])
        assert maps["M2"].shape == (n, 27, 55) and data["weights"].shape == (n, 27, 55)
        # B-const aggregates back to every label exactly
        w = data["weights"]
        agg = (maps["B-const"] * w).sum(axis=(1, 2)) / w.sum(axis=(1, 2))
        np.testing.assert_allclose(agg, data["label"], rtol=1e-4)

    def test_replot(self, demo_run):
        run = demo_run[1][0].parent
        before = {p.name for p in run.glob("*.png")}
        paths = mapviz.main(["--replot", str(run)])
        assert {p.name for p in paths} == before

    def test_needs_a_source(self):
        with pytest.raises(SystemExit):
            mapviz.parse_args([])

    def test_unknown_fold(self, tmp_path):
        pytest.importorskip("sklearn")
        with pytest.raises(SystemExit):
            mapviz.main(["--demo", "--models", "M2", "--fold", "nope", "--out", str(tmp_path)])
