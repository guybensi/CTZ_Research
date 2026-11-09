"""Tests for scripts/download_modis.py (the pure helpers; no network)."""
import importlib.util
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "download_modis.py"
_spec = importlib.util.spec_from_file_location("download_modis", _PATH)
dm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dm)


class TestParseGranuleName:
    def test_l1b(self):
        assert dm.parse_granule_name("MYD021KM.A2017060.0245.061.2017061.hdf") == ("MYD021KM", "A2017060.0245", "2017", "060")

    def test_l2_with_underscore_and_path(self):
        assert dm.parse_granule_name("/x/y/MYD06_L2.A2017001.2355.061.hdf")[:2] == ("MYD06_L2", "A2017001.2355")

    @pytest.mark.parametrize("bad", ["", "readme.txt", "MYD03.2017060.0245.hdf", "XYZ.A2017060.0245.061.hdf"])
    def test_rejects_non_granules(self, bad):
        with pytest.raises(ValueError):
            dm.parse_granule_name(bad)


class TestDateChunks:
    def test_single_day(self):
        assert dm.date_chunks("2017-03-01", "2017-03-01") == [("2017-03-01", "2017-03-01")]

    def test_year_boundary_and_leap_february(self):
        assert dm.date_chunks("2019-12-30", "2020-02-29") == [
            ("2019-12-30", "2019-12-31"), ("2020-01-01", "2020-01-31"), ("2020-02-01", "2020-02-29")]

    def test_full_year_is_12_chunks(self):
        assert len(dm.date_chunks("2020-01-01", "2020-12-31")) == 12

    @pytest.mark.parametrize("start,end", [("2017-03-03", "2017-03-01"), ("2017-13-01", "2017-13-02"), ("x", "2017-01-01")])
    def test_invalid(self, start, end):
        with pytest.raises(ValueError):
            dm.date_chunks(start, end)


class TestNormalizePolygon:
    def test_clockwise_becomes_ccw_and_closed(self):
        out = dm.normalize_polygon([(0, 0), (0, 1), (1, 1), (1, 0)])
        assert out[0] == out[-1] == (0.0, 0.0)
        area2 = sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(out[:-1], out[1:]))
        assert area2 > 0

    def test_already_closed_input_is_not_doubled(self):
        assert len(dm.normalize_polygon([(0, 0), (1, 0), (1, 1), (0, 0)])) == 4

    @pytest.mark.parametrize("bad", [[(0, 0), (1, 1)], [(0, 0), (1, 1), (2, 2)], [(181, 0), (1, 1), (0, 1)], [(0, 91), (1, 1), (0, 1)]])
    def test_invalid(self, bad):
        with pytest.raises(ValueError):
            dm.normalize_polygon(bad)

    def test_default_polygons_are_valid(self):
        for poly in dm.POLYGONS:
            dm.normalize_polygon(poly)


class TestCompleteSets:
    def test_intersection_sorted(self):
        assert dm.complete_sets({"A": {"k2": 0, "k1": 0}, "B": {"k1": 0, "k2": 0, "k3": 0}}) == ["k1", "k2"]

    def test_empty(self):
        assert dm.complete_sets({}) == []
        assert dm.complete_sets({"A": {}, "B": {"k1": 0}}) == []


class TestSizeMb:
    @pytest.mark.parametrize("size,expected", [(lambda: 12.5, 12.5), (7.0, 7.0), (None, 0.0), (lambda: None, 0.0), ("x", 0.0)])
    def test_method_or_number(self, size, expected):
        class G:
            pass
        g = G(); g.size = size
        assert dm._size_mb(g) == expected


class TestTargetFolder:
    @pytest.mark.parametrize("product,year,doy,expected", [
        ("MYD021KM", "2008", "001", "AQUA/2008/Pacific/jan/MOD02"),
        ("MYD03", "2008", "060", "AQUA/2008/Pacific/feb/MOD03"),       # 2008 is a leap year: day 60 = Feb 29
        ("MYD06_L2", "2017", "060", "AQUA/2017/Pacific/mrch/MOD06"),
        ("MYD35_L2", "2008", "366", "AQUA/2008/Pacific/dec/MOD35"),
        ("MYD021KM", "2014", "152", "AQUA/2014/Pacific/june/MOD02"),
    ])
    def test_advisor_layout(self, tmp_path, product, year, doy, expected):
        assert dm.target_folder(tmp_path, product, year, doy) == tmp_path / expected

    @pytest.mark.parametrize("product,year,doy", [("MYD04_L2", "2008", "001"), ("MYD03", "2017", "366"), ("MYD03", "2008", "000")])
    def test_invalid(self, tmp_path, product, year, doy):
        with pytest.raises(ValueError):
            dm.target_folder(tmp_path, product, year, doy)


class TestReorganize:
    def test_moves_old_layout_and_removes_empty_folders(self, tmp_path):
        old = tmp_path / "MYD021KM" / "2008" / "060"
        old.mkdir(parents=True)
        (old / "MYD021KM.A2008060.0245.061.2017.hdf").write_bytes(b"x")
        (tmp_path / "manifest.csv").write_text("keep")
        assert dm.reorganize(tmp_path) == 1
        assert (tmp_path / "AQUA/2008/Pacific/feb/MOD02/MYD021KM.A2008060.0245.061.2017.hdf").read_bytes() == b"x"
        assert not (tmp_path / "MYD021KM").exists()
        assert (tmp_path / "manifest.csv").exists()

    def test_nothing_to_move(self, tmp_path):
        assert dm.reorganize(tmp_path) == 0

    def test_new_layout_is_left_alone(self, tmp_path):
        new = tmp_path / "AQUA/2008/Pacific/feb/MOD02"
        new.mkdir(parents=True)
        (new / "MYD021KM.A2008060.0245.061.2017.hdf").write_bytes(b"x")
        assert dm.reorganize(tmp_path) == 0
        assert (new / "MYD021KM.A2008060.0245.061.2017.hdf").exists()
