import json

import numpy as np
import pytest

from flux1km import splits, synthetic
from flux1km.table import FootprintTable, build_table


def _table(lat, lon, granule, minutes=None, vza=None):
    n = len(lat)
    return FootprintTable({
        "granule": np.array(granule, dtype=object), "index": np.arange(n),
        "group": np.array(granule, dtype=object), "day": np.array([g[:8] for g in granule], dtype=object),
        "minutes": np.zeros(n) if minutes is None else np.asarray(minutes, float),
        "Lat": np.asarray(lat, float), "Lon": np.asarray(lon, float),
        "VZA": np.zeros(n) if vza is None else np.asarray(vza, float),
        "year": np.full(n, 2014),
    })


@pytest.fixture(scope="module")
def demo():
    rng = np.random.default_rng(0)
    gs = [synthetic.make_synthetic_granule(g, rng, n_rows=3, n_cols=5, lat0=-10 + 3 * i)
          for i, g in enumerate(synthetic.DEMO_GRANULES)]
    groups = splits.group_granules([g.granule_id for g in gs], sizes={g.granule_id: g.n_footprints for g in gs})
    return gs, groups, build_table(gs, groups)


class TestGroupGranules:
    def test_group_consecutive_granules_same_orbit(self):
        gr = splits.group_granules(["A2014090.2350", "A2014090.2355", "A2014090.2240"])
        assert gr["A2014090.2350"] == gr["A2014090.2355"] != gr["A2014090.2240"]

    def test_group_chain_and_satellite_separation(self):
        gr = splits.group_granules(["A2014060.0000", "A2014060.0005", "A2014060.0010", "T2014060.0005"])
        assert len({gr["A2014060.0000"], gr["A2014060.0005"], gr["A2014060.0010"]}) == 1
        assert gr["T2014060.0005"] != gr["A2014060.0005"]

    def test_group_across_midnight(self):
        gr = splits.group_granules(["A2014060.2355", "A2014061.0000"])
        assert gr["A2014060.2355"] == gr["A2014061.0000"]

    def test_merge_small_groups_into_nearest_same_day(self):
        ids = ["A2014060.0245", "A2014060.0415", "A2014060.0530"]
        gr = splits.group_granules(ids, sizes={"A2014060.0245": 9, "A2014060.0415": 3000, "A2014060.0530": 670}, min_size=100)
        assert gr["A2014060.0245"] == gr["A2014060.0415"] != gr["A2014060.0530"]

    def test_demo_matches_plan_grouping(self, demo):
        _, groups, _ = demo
        assert groups["A2014090.2350"] == groups["A2014090.2355"]
        assert len(set(groups.values())) == 5  # 0245 not merged: synthetic granules are all the same size


class TestLogoFolds:
    def test_logo_folds_partition_and_nested(self, demo):
        _, _, t = demo
        folds = splits.logo_folds(t)
        assert len(folds) == len(set(t["group"]))
        seen = []
        for f in folds:
            assert not set(f.train) & set(f.test) and not set(f.val) & set(f.test) and not set(f.train) & set(f.val)
            assert len(set(t["group"][f.test])) == 1 and len(f.val) > 0
            assert not set(t["group"][f.train]) & set(t["group"][f.test])
            seen.extend(f.test)
        assert sorted(seen) == list(range(len(t)))

    def test_logo_needs_three_groups(self):
        t = _table([0, 0], [0, 1], ["A2014060.0000", "A2014061.0000"])
        with pytest.raises(ValueError):
            splits.logo_folds(t)


class TestGroupKfold:
    def test_group_kfold_by_day(self, demo):
        _, _, t = demo
        folds = splits.group_kfold(t, key="day", k=2, seed=0)
        for f in folds:
            assert not set(t["day"][f.train]) & set(t["day"][f.test])
            assert not set(t["day"][f.val]) & set(t["day"][f.test])

    def test_group_kfold_too_many_folds(self, demo):
        _, _, t = demo
        with pytest.raises(ValueError):
            splits.group_kfold(t, key="day", k=5)


class TestHoldout:
    def test_holdout(self, demo):
        _, _, t = demo
        f = splits.holdout(t, test_mask=t["day"] == "A2014-03-31", name="time", val_fraction=0.5, seed=0)
        assert set(t["day"][f.test]) == {"A2014-03-31"} and "A2014-03-31" not in set(t["day"][f.train])
        assert not set(t["group"][f.val]) & set(t["group"][f.train])

    def test_holdout_empty_side_raises(self, demo):
        _, _, t = demo
        with pytest.raises(ValueError):
            splits.holdout(t, test_mask=np.zeros(len(t), bool))


class TestSpatialBlockFolds:
    def test_spatial_block_folds_buffer(self):
        lat = np.zeros(400); lon = np.linspace(0, 10, 400)  # ~1100 km line, ~2.8 km spacing
        t = _table(lat, lon, ["A2014060.0000"] * 400)
        folds = splits.spatial_block_folds(t, block_km=200, buffer_km=40, k=3, seed=0)
        for f in folds:
            gap = splits.min_distance_km(t, f.train, f.test)
            assert gap >= 40 - 1e-6
            assert len(f.test) > 0 and len(f.train) > 0

    def test_spatial_blocks_ignore_other_overpasses(self):
        # identical location, different overpass -> no pixel sharing, buffer must not drop it
        t = _table([0, 0, 0, 0], [0, 0.01, 5, 5.01], ["A2014060.0000", "A2014061.0000", "A2014060.0000", "A2014061.0000"])
        t.columns["group"] = t["granule"]
        folds = splits.spatial_block_folds(t, block_km=200, buffer_km=40, k=2, seed=0)
        for f in folds:
            assert len(f.train) + len(f.test) + len(f.val) == 4


class TestInterleaveSplit:
    def test_interleave_is_marked_leaky(self, demo):
        _, _, t = demo
        f = splits.interleave_split(t, period=2)
        assert f.leaky and len(f.test) > 0
        assert np.all(t["index"][f.test] % 2 == 1)


class TestPixelReuseFraction:
    def test_pixel_reuse_zero_when_far(self):
        t = _table([0, 0], [0, 5], ["A2014060.0000"] * 2)
        r = splits.pixel_reuse_fraction(t, np.array([0]), np.array([1]), radius_km=15)
        assert r[0] == 0.0

    def test_pixel_reuse_full_when_identical(self):
        t = _table([0, 0], [0, 0], ["A2014060.0000"] * 2)
        r = splits.pixel_reuse_fraction(t, np.array([0]), np.array([1]), radius_km=15)
        assert r[0] == pytest.approx(1.0)

    def test_pixel_reuse_partial_and_monotone(self):
        d = np.array([0.05, 0.15, 0.25])  # deg ~ 5.5, 16.7, 27.8 km
        fr = []
        for x in d:
            t = _table([0, 0], [0, x], ["A2014060.0000"] * 2)
            fr.append(splits.pixel_reuse_fraction(t, np.array([0]), np.array([1]), radius_km=15)[0])
        assert 1 > fr[0] > fr[1] > fr[2] >= 0

    def test_pixel_reuse_other_group_ignored(self):
        t = _table([0, 0], [0, 0], ["A2014060.0000", "A2014061.0000"])
        assert splits.pixel_reuse_fraction(t, np.array([0]), np.array([1]), radius_km=15)[0] == 0.0

    def test_pixel_reuse_radius_grows_with_vza(self):
        t = _table([0, 0], [0, 0.3], ["A2014060.0000"] * 2, vza=[60, 60])
        assert splits.pixel_reuse_fraction(t, np.array([0]), np.array([1]), radius_km=15)[0] > 0


class TestCheckFold:
    def test_random_footprint_split_leaks_logo_does_not(self, demo):
        _, _, t = demo
        rng = np.random.default_rng(0)
        perm = rng.permutation(len(t))
        leaky = splits.Fold("random", perm[: len(t) // 2], np.array([], int), perm[len(t) // 2:])
        with pytest.raises(splits.LeakageError):
            splits.check_fold(t, leaky, radius_km=20)
        for f in splits.logo_folds(t):
            splits.check_fold(t, f, radius_km=20)

    def test_check_fold_skips_leaky_by_design(self, demo):
        _, _, t = demo
        splits.check_fold(t, splits.interleave_split(t), radius_km=20)


class TestManifest:
    def test_manifest_roundtrip(self, tmp_path, demo):
        _, _, t = demo
        folds = splits.logo_folds(t)
        p = tmp_path / "m.json"
        splits.save_manifest(p, t, folds, meta={"note": "x"})
        loaded, meta = splits.load_manifest(p, t)
        assert meta["note"] == "x" and len(loaded) == len(folds)
        for a, b in zip(folds, loaded):
            np.testing.assert_array_equal(a.test, b.test)
            assert a.name == b.name and a.leaky == b.leaky
        json.loads(p.read_text())

    def test_manifest_detects_changed_table(self, tmp_path, demo):
        gs, groups, t = demo
        p = tmp_path / "m.json"
        splits.save_manifest(p, t, splits.logo_folds(t))
        t2 = build_table(gs[:-1], groups)
        with pytest.raises(ValueError):
            splits.load_manifest(p, t2)
