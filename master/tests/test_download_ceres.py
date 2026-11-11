"""Tests for scripts/download_ceres.py (pure helpers and the subsetting of a synthetic SSF-like file; no network)."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

nc = pytest.importorskip("netCDF4")

_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_DIR))
_spec = importlib.util.spec_from_file_location("download_ceres", _DIR / "download_ceres.py")
dc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dc)

# HDF4-style names (spaces and punctuation), as in the raw SSF file
RAW = {
    "Time of observation": None,
    "Colatitude of CERES FOV at surface": None,
    "Longitude of CERES FOV at surface": None,
    "CERES solar zenith at surface": 30.0,
    "CERES viewing zenith at surface": 20.0,
    "CERES relative azimuth at surface": 90.0,
    "CERES SW TOA flux - upwards": 250.0,
    "CERES LW TOA flux - upwards": 240.0,
    "CERES SW radiance - upwards": 80.0,
    "CERES LW radiance - upwards": 75.0,
    "Cloud mask clear strong percent coverage": 10.0,
    "PSF-wtd MOD04 effective optical depth average ocean (0.550)": 0.1,
}


def make_raw(path, lat, lon, drop=()):
    n = len(lat)
    with nc.Dataset(path, "w") as ds:
        ds.createDimension("n", n)
        for name, val in RAW.items():
            if name in drop:
                continue
            v = ds.createVariable(name, "f8", ("n",))
            if name == "Time of observation":
                v[:] = 2454466.5 + np.arange(n) / 86400.0     # Julian date: 2008-01-01 00:00 UTC + seconds
            elif name.startswith("Colatitude"):
                v[:] = 90.0 - np.asarray(lat)
            elif name.startswith("Longitude"):
                v[:] = lon
            else:
                v[:] = val
        ds.createVariable("something else", "f8", ("n",))[:] = 0.0


class TestSanitize:
    def test_matches_order_tool_names(self):
        assert dc.sanitize("CERES SW TOA flux - upwards") == "CERES_SW_TOA_flux___upwards"
        assert dc.sanitize("PSF-wtd MOD04 effective optical depth average ocean (0.550)").endswith("ocean__0_550_")

    def test_every_required_name_is_reachable(self):
        assert {dc.sanitize(k) for k in RAW} >= {v for v in dc.oc.REQUIRED_VARS if v != "time"} | {"Time_of_observation"}


class TestSsfHour:
    def test_catalogue_name(self):
        assert dc.ssf_hour("CER_SSF_Aqua-FM3-MODIS_Edition4A_400403.2008010100") == (2008, 1, 0)
        assert dc.ssf_hour("CER_SSF_Aqua-FM3-MODIS_Edition4A_400403.2008123123") == (2008, 366, 23)

    @pytest.mark.parametrize("bad", ["x.nc", "CER_2008011.nc", "x_2008010125"])
    def test_none(self, bad):
        assert dc.ssf_hour(bad) is None


class TestNeededHours:
    def test_reads_modis_granule_names(self, tmp_path):
        d = tmp_path / "AQUA/2008/Pacific/feb/MOD02"
        d.mkdir(parents=True)
        for n in ["MYD021KM.A2008032.0305.061.2018.hdf", "MYD021KM.A2008032.0310.061.2018.hdf", "MYD021KM.A2008033.2355.061.x.hdf", "junk.txt"]:
            (d / n).touch()
        assert dc.needed_hours(tmp_path, "2008", "Pacific") == {(2008, 32, 3), (2008, 33, 23)}

    def test_missing_dir(self, tmp_path):
        assert dc.needed_hours(tmp_path, "2008", "Pacific") == set()


class TestKeepMask:
    def test_inside_with_one_neighbour_each_side(self):
        m = dc.keep_mask(np.zeros(6), np.array([10, 10, 160, 10, 10, 10.0]))
        assert m.tolist() == [False, True, True, True, False, False]

    def test_latitude_limit_and_nan(self):
        m = dc.keep_mask(np.array([70.0, np.nan, 0.0]), np.array([160.0, 160.0, 160.0]))
        assert m.tolist() == [False, True, True]    # only the last is inside; its predecessor is kept as neighbour

    def test_nothing_inside_and_empty(self):
        assert not dc.keep_mask(np.zeros(3), np.full(3, 10.0)).any()
        assert dc.keep_mask(np.array([]), np.array([])).size == 0


class TestSubsetSsfFile:
    def test_output_matches_the_advisors_reader(self, tmp_path):
        lat = [0, 0, 0, 0, 0, 0]
        lon = [10, 10, 160, 170, 10, 10]
        make_raw(tmp_path / "raw", lat, lon)
        n = dc.subset_ssf_file(tmp_path / "raw", tmp_path / "part.nc")
        assert n == 4                                     # 2 inside + 1 neighbour each side
        with nc.Dataset(tmp_path / "part.nc") as ds:
            for v in dc.oc.REQUIRED_VARS:
                assert v in ds.variables
                assert "_FillValue" in ds.variables[v].ncattrs()      # the advisor's code reads this attribute
            d = nc.num2date(ds.variables["time"][0], units=ds.variables["time"].units)
            assert (d.year, d.month, d.day, d.hour) == (2008, 1, 1, 0)
            assert ds.variables["CERES_SW_TOA_flux___upwards"][:].tolist() == [250.0] * 4
        assert dc.oc.verify_file(tmp_path / "part.nc", 2008, 1) == []

    def test_fill_values_become_masked(self, tmp_path):
        make_raw(tmp_path / "raw", [0, 0], [160, 160])
        with nc.Dataset(tmp_path / "raw", "a") as ds:
            ds.variables["CERES SW TOA flux - upwards"][0] = 3.4028235e38
        dc.subset_ssf_file(tmp_path / "raw", tmp_path / "part.nc")
        with nc.Dataset(tmp_path / "part.nc") as ds:
            assert np.ma.is_masked(ds.variables["CERES_SW_TOA_flux___upwards"][0])

    def test_no_footprint_in_box_writes_nothing(self, tmp_path):
        make_raw(tmp_path / "raw", [0, 0], [10, 20])
        assert dc.subset_ssf_file(tmp_path / "raw", tmp_path / "part.nc") == 0
        assert not (tmp_path / "part.nc").exists()

    def test_missing_variable_is_reported(self, tmp_path):
        make_raw(tmp_path / "raw", [0], [160], drop=("CERES LW radiance - upwards",))
        with pytest.raises(ValueError, match="CERES_LW_radiance___upwards"):
            dc.subset_ssf_file(tmp_path / "raw", tmp_path / "part.nc")


class TestPartPath:
    def test_month_folder_and_name(self):
        assert dc.part_path(Path("/r"), "2008", "Pacific", 2008, 32, 3).as_posix() == "/r/AQUA/2008/Pacific/feb/ssf_parts/ssf_2008020103.nc"
        assert dc.part_path(Path("/r"), "2008", "Pacific", 2008, 366, 0).name == "ssf_2008123100.nc"


class TestBoxRanges:
    def test_pacific_default(self):
        assert dc.box_ranges(150, -120, -60, 60) == ((-60.0, 60.0), (150.0, 240.0))
        assert (dc.LAT_RANGE, dc.LON360_RANGE) == ((-60.0, 60.0), (150.0, 240.0))

    def test_western_hemisphere_box(self):
        assert dc.box_ranges(-60, -10, -50, 50) == ((-50.0, 50.0), (300.0, 350.0))

    def test_box_across_greenwich_wraps(self):
        assert dc.box_ranges(-10, 10, -5, 5)[1] == (350.0, 10.0)

    def test_invalid_box(self):
        with pytest.raises(ValueError):
            dc.box_ranges(0, 0, -5, 5)


class TestKeepMaskWrap:
    def test_wrapped_range(self):
        m = dc.keep_mask(np.zeros(5), np.array([100, 355, 5, 100, 100.0]), lon_range=(350, 10))
        assert m.tolist() == [True, True, True, True, False]   # 100 next to 355 / 5 is kept as neighbour

    def test_nan_coordinates_not_inside(self):
        assert dc.keep_mask(np.array([np.nan]), np.array([160.0])).tolist() == [False]


class TestYearsAndMonths:
    def test_years(self):
        assert dc.years_in("2008-12-30", "2009-01-02") == [2008, 2009]
        assert dc.years_in("2008-05-01", "2008-05-01") == [2008]

    def test_years_bad_order(self):
        with pytest.raises(ValueError):
            dc.years_in("2009-01-01", "2008-01-01")

    def test_months(self):
        assert dc.months_in(2008, "2008-03-15", "2008-03-15") == [3]
        assert dc.months_in(2008, "2008-11-15", "2009-02-01") == [11, 12]
        assert dc.months_in(2009, "2008-11-15", "2009-02-01") == [1, 2]
        assert dc.months_in(2010, "2008-11-15", "2009-02-01") == []


class TestSubsetWithBox:
    def test_custom_box(self, tmp_path):
        make_raw(tmp_path / "raw", [0, 0, 0], [10, 20, 100])      # all outside the Pacific default
        n = dc.subset_ssf_file(tmp_path / "raw", tmp_path / "p.nc", (-10.0, 10.0), (5.0, 25.0))
        assert n == 3                                              # 2 inside + 1 neighbour


class TestMainArguments:
    @pytest.mark.parametrize("args", [
        ["--start", "2009-01-01", "--end", "2008-01-01"],
        ["--start", "nonsense"],
        ["--box", "5", "5", "0", "10"],
    ])
    def test_bad_arguments_exit_before_login(self, args):
        with pytest.raises(SystemExit) as e:
            dc.main(args + ["--dry-run"])
        assert e.value.code == 2

    def test_merge_only_multi_year_range_verifies_only_requested_months(self, tmp_path, capsys):
        rc = dc.main(["--start", "2008-03-01", "--end", "2008-03-31", "--out", str(tmp_path), "--merge-only"])
        out = capsys.readouterr().out
        assert rc == 1 and out.count("NO CERES FILE") == 1 and "mrch" in out
