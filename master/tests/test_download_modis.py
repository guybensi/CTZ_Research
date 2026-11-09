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
