import zipfile

import numpy as np
import pytest

from flux1km import e0, io, splits, synthetic
from flux1km.table import build_table


@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    root = tmp_path_factory.mktemp("e0data")
    synthetic.write_demo_dataset(root, n_rows=3, n_cols=8)
    return root, io.load_dataset(root, parts=("moments", "scalars", "images"))


class TestUnpackZip:
    def test_extracts_and_rejects_traversal(self, tmp_path):
        good = tmp_path / "good.zip"
        with zipfile.ZipFile(good, "w") as z:
            z.writestr("scalars/a.txt", "x")
        assert e0.unpack_zip(good, tmp_path / "out") == ["scalars/a.txt"]
        bad = tmp_path / "bad.zip"
        with zipfile.ZipFile(bad, "w") as z:
            z.writestr("../evil.txt", "x")
        with pytest.raises(ValueError):
            e0.unpack_zip(bad, tmp_path / "out2")


class TestInventory:
    def test_counts_keys_and_roots(self, demo):
        root, _ = demo
        inv = e0.inventory(root)
        assert inv["n_granules"] == 6 and inv["n_footprints"] == 6 * 24
        assert inv["keys"]["images"]["ImSW"] == (11, 27, 55) and inv["keys"]["moments"]["SWwavlngs"] == (11,)
        assert not inv["inconsistent"] and inv["months"] == ["2014-03"]
        assert e0.find_data_roots(root) == [root]


class TestValueRanges:
    def test_flags(self, demo):
        _, gs = demo
        r = e0.value_ranges(gs)
        assert r["fov_values"] == [0.0, 1.0] and "Fsw" in r["scalars"]
        g = gs[0]
        bad = io.Granule(g.granule_id, {**g.scalars, "CF": g.scalars["CF"] * 100}, g.moments,
                         {**g.images, "ImSW": g.images["ImSW"] * 1000}, None)
        flags = e0.value_ranges([bad])["flags"]
        assert any("CF exceeds 1" in f for f in flags) and any("ImSW" in f for f in flags)


class TestOperatorChecks:
    def test_fov_ellipse_reproduces_synthetic_moments(self, demo):
        _, gs = demo
        err = e0.moments_vs_patches(gs)
        assert err["fov_ellipse"] < 1e-6 < err["uniform"]

    def test_orientation_as_stored_fits_best(self, demo):
        r = e0.check_fov_orientation(demo[1])
        assert r["as_stored"] < r["inverted"]

    def test_fit_and_save_operator(self, demo, tmp_path):
        out = e0.fit_aggregation_operator(demo[1], tmp_path / "op.npz", max_footprints=60, maxiter=40)
        assert out["ellipse_error"] < 0.01 and out["psf_error"] is not None
        from flux1km import aggregation
        assert aggregation.load_psf_file(out["path"]).shape == (27, 55)

    def test_needs_images_and_moments(self, demo):
        g = demo[1][0]
        with pytest.raises(ValueError):
            e0.moments_vs_patches([io.Granule(g.granule_id, g.scalars, g.moments, None)])


class TestOverlapAndStripes:
    def test_overlap_of_synthetic_lattice(self, demo):
        o = e0.footprint_overlap(demo[1][0])
        assert 0 < o["spacing_km_min"] < 15 and 0.3 < o["shared_max"] <= 1.0

    def test_single_footprint(self, demo):
        g = demo[1][0]
        one = io.Granule(g.granule_id, {k: v[:1] for k, v in g.scalars.items()})
        assert np.isnan(e0.footprint_overlap(one)["spacing_km_median"])

    def test_detects_artificial_stripes(self, demo):
        g = demo[1][0]
        sw = g.images["ImSW"].copy()
        sw[:, 2, ::2, :] += 500.0  # strong offset on every other row of the 0.66 um channel
        st = e0.stripe_scores([io.Granule(g.granule_id, g.scalars, g.moments, {**g.images, "ImSW": sw})])
        assert "0.66" in st["bad"] and "0.47" not in st["bad"]


class TestManifest:
    def test_frozen_manifest_has_zero_reuse(self, demo, tmp_path):
        _, gs = demo
        t = build_table(gs, splits.group_granules([g.granule_id for g in gs]))
        folds, rep = e0.build_frozen_manifest(t, tmp_path / "m.json", image_granules=[g.granule_id for g in gs], k=3)
        assert any(n.startswith("A:logo") for n in rep["folds"]) and any(n.startswith("A:interleave") for n in rep["folds"])
        assert all(v["max_reuse"] == 0 for v in rep["folds"].values() if not v["leaky"])
        loaded, meta = splits.load_manifest(tmp_path / "m.json", t)
        assert len(loaded) == len(folds) and meta["built_by"] == "flux1km.e0"
        assert len(e0.arena_folds(folds, "A")) >= 4

    def test_exclude_days(self, demo):
        _, gs = demo
        t = build_table(gs)
        f = splits.Fold("x", np.arange(len(t)), np.array([], int), np.array([0]))
        day = t["day"][0]
        out = e0.exclude_days_from_training(t, [f], {day})[0]
        assert day not in set(t["day"][out.train]) and np.array_equal(out.test, f.test)


class TestRunE0:
    def test_writes_report_and_artifacts(self, demo, tmp_path):
        root, _ = demo
        res = e0.run_e0(root, tmp_path, min_group_size=1, max_footprints=60, maxiter=20)
        text = (tmp_path / "E0_report.txt").read_text()
        for key in ("1. Inventory", "Moments vs patch", "Fitted aggregation operator", "Frozen split manifest", "LiqPhsf"):
            assert key in text
        assert (tmp_path / "split_manifest.json").exists() and (tmp_path / "fitted_operator.npz").exists()
        assert (tmp_path / "advisor_bug_report.txt").exists() and res["manifest"]["folds"]
