"""Tests for scripts/organize_ceres.py (synthetic netCDF files; no network)."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

nc = pytest.importorskip("netCDF4")

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "organize_ceres.py"
_spec = importlib.util.spec_from_file_location("organize_ceres", _PATH)
oc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(oc)


def make_ssf(path, t0_days, n, drop=(), fsw=300.0):
    """Tiny SSF-like file: n footprints, 'time' in days since 2008-01-01, all REQUIRED_VARS but `drop`."""
    with nc.Dataset(path, "w") as ds:
        ds.createDimension("record", n)
        ds.createDimension("two", 2)
        for v in oc.REQUIRED_VARS:
            if v in drop:
                continue
            var = ds.createVariable(v, "f8", ("record",), fill_value=-999.0)
            var[:] = np.arange(n, dtype="f8") + t0_days if v == "time" else np.full(n, fsw)
            if v == "time":
                var.units = "days since 2008-01-01 00:00:00"
        ds.createVariable("static", "f4", ("two",))[:] = [1.0, 2.0]
        ds.title = "x"


class TestParseUrls:
    def test_filters_dedups_and_keeps_order(self):
        txt = "# c\nhttps://a/x.nc\n\nnotes\nhttps://a/x.nc\n ftp://b/y.nc \n"
        assert oc.parse_urls(txt) == ["https://a/x.nc", "ftp://b/y.nc"]

    def test_empty(self):
        assert oc.parse_urls("") == []


class TestDateFromName:
    @pytest.mark.parametrize("name,ym", [
        ("CER_SSF_Aqua-FM3-MODIS_Edition4A_400405.2008010100", (2008, 1)),
        ("x_20081231.nc", (2008, 12)),
        ("/a/b/CER_SSF_Aqua-FM3-MODIS_Edition4A_400405.2008061523.nc", (2008, 6)),
    ])
    def test_found(self, name, ym):
        assert oc.date_from_name(name) == ym

    @pytest.mark.parametrize("name", ["order.nc", "x_200813.nc", "run_20081345.nc"])
    def test_not_found(self, name):
        assert oc.date_from_name(name) is None


class TestMonthFolder:
    def test_layout(self):
        assert str(oc.month_folder("/r", 2008, "Pacific", 3)) == "/r/AQUA/2008/Pacific/mrch"

    @pytest.mark.parametrize("m", [0, 13])
    def test_bad_month(self, m):
        with pytest.raises(ValueError):
            oc.month_folder("/r", 2008, "Pacific", m)


class TestMergeNetcdf:
    def test_concatenates_in_time_order_and_keeps_attrs(self, tmp_path):
        a, b = tmp_path / "a.nc", tmp_path / "b.nc"
        make_ssf(a, 10, 3)
        make_ssf(b, 0, 2)   # earlier in time but listed second
        n = oc.merge_netcdf([a, b], tmp_path / "m.nc")
        assert n == 5
        with nc.Dataset(tmp_path / "m.nc") as m:
            assert m.variables["time"][:].tolist() == [0, 1, 10, 11, 12]
            assert m.variables["static"][:].tolist() == [1.0, 2.0]
            assert m.title == "x"
            assert m.variables["time"].units.startswith("days since")

    def test_fill_values_survive(self, tmp_path):
        a = tmp_path / "a.nc"
        make_ssf(a, 0, 2)
        with nc.Dataset(a, "a") as d:
            d.variables["CERES_SW_TOA_flux___upwards"][0] = -999.0
        oc.merge_netcdf([a], tmp_path / "m.nc")
        with nc.Dataset(tmp_path / "m.nc") as m:
            assert np.ma.is_masked(m.variables["CERES_SW_TOA_flux___upwards"][0])

    def test_no_parts(self, tmp_path):
        with pytest.raises(ValueError):
            oc.merge_netcdf([], tmp_path / "m.nc")

    def test_different_variables(self, tmp_path):
        a, b = tmp_path / "a.nc", tmp_path / "b.nc"
        make_ssf(a, 0, 2)
        make_ssf(b, 5, 2, drop=("Cloud_mask_clear_strong_percent_coverage",))
        with pytest.raises(ValueError):
            oc.merge_netcdf([a, b], tmp_path / "m.nc")


class TestVerifyFile:
    def test_ok(self, tmp_path):
        f = tmp_path / "a.nc"
        make_ssf(f, 3, 4)   # days 3..6 of January 2008
        assert oc.verify_file(f, 2008, 1) == []

    def test_missing_variable(self, tmp_path):
        f = tmp_path / "a.nc"
        make_ssf(f, 3, 4, drop=("CERES_LW_radiance___upwards",))
        assert "CERES_LW_radiance___upwards" in oc.verify_file(f, 2008, 1)[0]

    def test_wrong_month(self, tmp_path):
        f = tmp_path / "a.nc"
        make_ssf(f, 3, 4)
        assert "expected 2008-02" in oc.verify_file(f, 2008, 2)[0]

    def test_night_only(self, tmp_path):
        f = tmp_path / "a.nc"
        make_ssf(f, 3, 4, fsw=0.0)
        assert any("Fsw" in p for p in oc.verify_file(f, 2008, 1))

    def test_empty_file(self, tmp_path):
        f = tmp_path / "a.nc"
        make_ssf(f, 0, 0)
        assert oc.verify_file(f, 2008, 1) == ["no records"]


class TestOrganize:
    def test_sorts_merges_and_leaves_one_ceres_file(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        make_ssf(src / "CER_SSF_x.2008010100.nc", 0, 2)
        make_ssf(src / "CER_SSF_x.2008010112.nc", 0.5, 2)
        make_ssf(src / "CER_SSF_x.2008020100.nc", 31, 2)
        make_ssf(src / "CER_SSF_x.2009010100.nc", 0, 2)   # other year: skipped
        res = oc.organize(sorted(src.glob("*.nc")), tmp_path / "out", "2008", "Pacific")
        assert sorted(res) == [1, 2]
        jan = tmp_path / "out/AQUA/2008/Pacific/jan"
        assert [p.name for p in jan.glob("CERES*")] == ["CERES_SSF_Aqua_2008_jan.nc"]   # parts dir has no "CERES"
        assert len(list((jan / "ssf_parts").glob("*.nc"))) == 2
        assert oc.verify_file(res[1], 2008, 1) == []

    def test_rerun_is_idempotent(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        make_ssf(src / "CER_x.2008030100.nc", 60, 3)
        oc.organize(list(src.glob("*.nc")), tmp_path / "out", "2008", "Pacific")
        res = oc.organize(list(src.glob("*.nc")), tmp_path / "out", "2008", "Pacific")
        with nc.Dataset(res[3]) as m:
            assert m.dimensions["record"].size == 3
