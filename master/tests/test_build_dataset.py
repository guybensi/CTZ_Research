"""Tests for scripts/build_dataset.py (synthetic MODIS/CERES files; no network)."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

nc = pytest.importorskip("netCDF4")
pytest.importorskip("scipy")

_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_DIR))
_spec = importlib.util.spec_from_file_location("build_dataset", _DIR / "build_dataset.py")
bd = importlib.util.module_from_spec(_spec)
sys.modules["build_dataset"] = bd      # worker processes pickle functions by module name
_spec.loader.exec_module(bd)

H, W = 90, 100
CY, CX = 45, 50


def grid(lon0=150.0):
    rows, cols = np.mgrid[0:H, 0:W]
    lat = 20.0 + 0.009 * rows
    lon = lon0 + 0.0095 * cols
    lon = np.where(lon > 180, lon - 360, lon)
    return lat.astype("float32"), lon.astype("float32")


def psf27():
    p = np.exp(-0.5 * (((np.arange(27) - 13) / 5.0)[:, None] ** 2 + ((np.arange(55) - 27) / 9.0)[None, :] ** 2))
    return p / p.sum()


def ceres_arrays(lat, lon, spots, fsw=300.0, hour_key=200800100):
    """CERES dict with one footprint per (row, col) in spots, in scan order."""
    n = len(spots)
    la = np.array([lat[r, c] for r, c in spots], dtype="float64")
    lo = np.array([lon[r, c] for r, c in spots], dtype="float64")
    full = lambda v: np.full(n, v, dtype="float64")  # noqa: E731
    return {"key": np.full(n, hour_key, dtype="int64"), "lat": la, "lon": lo, "sza": full(30.0), "vza": full(20.0),
            "raa": full(90.0), "fsw": full(fsw), "isw": full(90.0), "flw": full(240.0), "ilw": full(75.0),
            "aod": full(0.1), "clrf": full(10.0)}


def modis_arrays(lon0=150.0, nan_at=None):
    lat, lon = grid(lon0)
    solar = np.stack([np.full((H, W), 10.0 + i, dtype="float32") for i in range(11)])
    thermal = np.stack([np.full((H, W), 1.0 + i, dtype="float32") for i in range(16)])
    solar[0] += np.arange(W, dtype="float32")[None, :]      # a gradient so patch position matters
    if nan_at:
        solar[0][nan_at] = np.nan
    one = lambda v: np.full((H, W), v, dtype="float32")  # noqa: E731
    return dict(solar=solar, thermal=thermal, lat=lat, lon=lon, szen=one(30), vzen=one(20), raa=one(90),
                cloud=np.zeros((H, W), bool), glint=np.zeros((H, W), bool), land=np.zeros((H, W), "int8"),
                cth=one(2000), phase=one(1))


def extract(m, ceres, hour_key=200800100, **kw):
    return bd.extract_footprints(m["solar"], m["thermal"], m["lat"], m["lon"], m["szen"], m["vzen"], m["raa"],
                                 m["cloud"], m["glint"], m["land"], m["cth"], m["phase"], ceres, hour_key,
                                 kw.pop("psf", psf27()), **kw)


class TestHourKeys:
    def test_values(self):
        assert bd.hour_keys([0.0, 1.5, 31.0], "days since 2008-01-01 00:00:00").tolist() == [200800100, 200800212, 200803200]

    def test_other_units_and_epoch_with_time(self):
        assert bd.hour_keys([36.0], "hours since 2008-12-31 12:00:00").tolist() == [200900200]   # 2009-01-02 00:00
        assert bd.hour_keys([0.5], "days since 2000-01-01 12:00:00").tolist() == [200000200]    # 2000-01-02 00:00

    def test_nan_is_minus_one(self):
        assert bd.hour_keys([np.nan, 0.0], "days since 2008-01-01").tolist() == [-1, 200800100]

    def test_leap_day(self):
        assert bd.hour_keys([59.0], "days since 2008-01-01").tolist() == [200806000]

    @pytest.mark.parametrize("units", ["", "julian", "furlongs since 2000-01-01", "days since yesterday"])
    def test_bad_units(self, units):
        with pytest.raises(ValueError):
            bd.hour_keys([0.0], units)

    def test_matches_granule_key(self):
        k = bd.hour_keys([5.0 + 3 / 24 + 0.2 / 24], "days since 2008-01-01")[0]     # 2008-01-06 03:12 UTC
        assert k == bd.granule_hour_key("MYD021KM.A2008006.0305.061.2008006123456.hdf")


class TestGranuleHourKey:
    def test_example(self):
        assert bd.granule_hour_key("MYD021KM.A2008032.0305.061.x.hdf") == 200803203

    def test_single_digit_day(self):
        assert bd.granule_hour_key("MYD03.A2008005.2355.061.x.hdf") == 200800523

    def test_not_a_granule(self):
        with pytest.raises(ValueError):
            bd.granule_hour_key("readme.txt")


class TestDecodeCloudMask:
    def test_bits(self):
        d = bd.decode_cloud_mask(np.array([0b00000000, 0b00000001, 0b00000111, 0b00010111, 0b11000111]))
        assert d["cloud"].tolist() == [True, True, False, False, False]      # undetermined counts as cloudy
        assert d["glint"].tolist() == [False, True, True, False, True]
        assert d["land"].tolist() == [0, 0, 0, 0, 3]

    def test_land_bits_are_not_swapped(self):
        # bit 6 only = coastal (1); bit 7 only = desert (2): the advisor's code swapped these
        assert bd.decode_cloud_mask(np.array([0b01000001, 0b10000001]))["land"].tolist() == [1, 2]

    def test_confidence_order(self):
        b = np.array([0b00000011, 0b00000101])                              # confidence 1 (uncertain), 2 (prob. clear)
        assert bd.decode_cloud_mask(b, "relaxed")["cloud"].tolist() == [False, False]
        assert bd.decode_cloud_mask(b, "strict")["cloud"].tolist() == [True, False]

    def test_negative_int8_bytes(self):
        assert bd.decode_cloud_mask(np.array([-1], dtype="int8"))["land"].tolist() == [3]

    def test_bad_kind(self):
        with pytest.raises(ValueError):
            bd.decode_cloud_mask(np.array([1]), "loose")


class TestRelativeAzimuth:
    def test_fold_stays_in_range(self):
        sol, sen = np.meshgrid(np.arange(-180, 181, 15.0), np.arange(-180, 181, 15.0))
        r = bd.relative_azimuth(sol, sen)
        assert r.min() >= 0 and r.max() <= 180

    def test_known_values(self):
        assert bd.relative_azimuth(np.array([-170.0]), np.array([170.0])).tolist() == [20.0]
        assert bd.relative_azimuth(np.array([0.0]), np.array([180.0])).tolist() == [180.0]


class TestDistanceToCloud:
    def test_no_cloud_is_nan(self):
        assert np.isnan(bd.distance_to_cloud(np.zeros((5, 5), bool))).all()

    def test_cloud_pixels_are_zero(self):
        c = np.zeros((9, 9), bool)
        c[4, 4] = True
        assert bd.distance_to_cloud(c)[4, 4] == 0


class TestWeightedMoments:
    def test_matches_direct_formula(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(3, 5, 6))
        w = rng.random((5, 6))
        w /= w.sum()
        mean, std, skew, kurt = bd.weighted_moments(x, w)
        for c in range(3):
            m = (x[c] * w).sum()
            s = np.sqrt((w * (x[c] - m) ** 2).sum())
            assert mean[c] == pytest.approx(m)
            assert std[c] == pytest.approx(s)
            assert skew[c] == pytest.approx((w * ((x[c] - m) / s) ** 3).sum())
            assert kurt[c] == pytest.approx((w * ((x[c] - m) / s) ** 4).sum() - 3)

    def test_constant_channel_has_zero_std_and_finite_output(self):
        mean, std, skew, kurt = bd.weighted_moments(np.full((1, 3, 3), 7.0), np.full((3, 3), 1 / 9))
        assert mean[0] == pytest.approx(7.0) and std[0] == pytest.approx(0, abs=1e-12) and np.isfinite([skew[0], kurt[0]]).all()

    def test_nan_pixel_ignored(self):
        x = np.array([[[1.0, np.nan, 3.0]]])
        assert np.isfinite(bd.weighted_moments(x, np.array([[0.5, 0.0, 0.5]]))[0]).all()


class TestExtractFootprints:
    def test_one_footprint_keys_shapes_dtypes(self):
        m = modis_arrays()
        ce = ceres_arrays(m["lat"], m["lon"], [(CY, CX), (CY, CX + 3)])
        res = extract(m, ce)
        assert list(res["scalars"]) == bd.SCALAR_KEYS
        assert list(res["moments"]) == bd.MOMENT_KEYS
        assert list(res["images"]) == bd.IMAGE_KEYS
        assert res["scalars"]["Fsw"].shape == (1,) and res["scalars"]["Fsw"].dtype == np.float32
        assert res["moments"]["mSWspec"].shape == (1, 11) and res["moments"]["mLWspec"].shape == (1, 16)
        assert res["images"]["ImSW"].shape == (1, 11, 27, 55) and res["images"]["Imviews"].shape == (1, 3, 27, 55)
        assert res["moments"]["SWwavlngs"].tolist() == bd.SW_WAVELENGTHS.tolist()

    def test_patch_is_centred_on_the_footprint_and_psf_peak_is_on_it(self):
        m = modis_arrays()
        ce = ceres_arrays(m["lat"], m["lon"], [(CY, CX), (CY, CX + 3)])
        res = extract(m, ce)
        patch = res["images"]["ImSW"][0, 0]                       # channel 0 = 10 + column index
        assert patch[13, 27] == pytest.approx(10 + CX)            # centre row, PSF-peak column = footprint pixel
        assert res["scalars"]["mSumIsw"][0] == pytest.approx(float(np.sum(m["solar"].sum(0)[CY - 13:CY + 14, CX - 27:CX + 28] * psf27())), rel=1e-5)

    def test_advisor_column_offset_reproduces_his_patch(self):
        m = modis_arrays()
        ce = ceres_arrays(m["lat"], m["lon"], [(CY, CX), (CY, CX + 3)])
        patch = extract(m, ce, psf_column_offset=1)["images"]["ImSW"][0, 0]
        assert patch[13, 26] == pytest.approx(10 + CX)

    def test_orientation_and_mirrored_psf(self):
        m = modis_arrays()
        p = psf27() * np.linspace(0.5, 1.5, 55)[None, :]          # asymmetric along columns
        p /= p.sum()
        fwd = extract(m, ceres_arrays(m["lat"], m["lon"], [(CY, CX), (CY, CX + 3)]), psf=p)
        bwd = extract(m, ceres_arrays(m["lat"], m["lon"], [(CY, CX), (CY, CX - 3)]), psf=p)
        assert fwd["images"]["FOVorientation"].tolist() == [1.0] and bwd["images"]["FOVorientation"].tolist() == [0.0]
        # constant channel 5: weighted mean is 10+5 either way, and weights sum to 1 in both orientations
        assert fwd["moments"]["mSWspec"][0, 5] == pytest.approx(15.0, rel=1e-5)
        assert bwd["moments"]["mSWspec"][0, 5] == pytest.approx(15.0, rel=1e-5)
        assert fwd["moments"]["mSWspec"][0, 0] != pytest.approx(bwd["moments"]["mSWspec"][0, 0])

    def test_night_footprint_skipped(self):
        m = modis_arrays()
        assert extract(m, ceres_arrays(m["lat"], m["lon"], [(CY, CX), (CY, CX + 3)], fsw=0.5)) is None

    def test_other_hour_is_ignored(self):
        m = modis_arrays()
        assert extract(m, ceres_arrays(m["lat"], m["lon"], [(CY, CX), (CY, CX + 3)]), hour_key=200800101) is None

    def test_last_footprint_without_neighbour_does_not_crash(self):
        m = modis_arrays()
        assert extract(m, ceres_arrays(m["lat"], m["lon"], [(CY, CX)])) is None

    def test_footprint_near_the_edge_is_skipped(self):
        m = modis_arrays()
        assert extract(m, ceres_arrays(m["lat"], m["lon"], [(5, CX), (5, CX + 3)])) is None
        assert extract(m, ceres_arrays(m["lat"], m["lon"], [(CY, 10), (CY, 13)])) is None

    def test_footprint_outside_the_granule_is_skipped(self):
        m = modis_arrays()
        ce = ceres_arrays(m["lat"], m["lon"], [(CY, CX), (CY, CX + 3)])
        ce["lat"][:] += 5.0
        ce["lon"][:] += 5.0
        assert extract(m, ce) is None

    def test_invalid_pixels_under_the_psf_skip_the_footprint(self):
        m = modis_arrays(nan_at=(slice(CY - 2, CY + 3), slice(CX - 2, CX + 3)))     # under the PSF peak
        assert extract(m, ceres_arrays(m["lat"], m["lon"], [(CY, CX), (CY, CX + 3)])) is None

    def test_invalid_pixels_in_the_far_corner_are_tolerated(self):
        m = modis_arrays(nan_at=(CY - 13, CX - 27))                                 # corner, PSF weight < 0.001
        assert extract(m, ceres_arrays(m["lat"], m["lon"], [(CY, CX), (CY, CX + 3)])) is not None

    def test_dateline_granule(self):
        m = modis_arrays(lon0=179.525)                                              # lon wraps from +180 to -180 at col 50
        assert (m["lon"] < 0).any() and (m["lon"] > 0).any()
        res = extract(m, ceres_arrays(m["lat"], m["lon"], [(CY, 45), (CY, 48), (CY, 55), (CY, 58)]))
        assert len(res["scalars"]["Fsw"]) == 3                                      # the last has no next footprint
        assert (res["scalars"]["Lon"] > 0).tolist() == [True, True, False]          # one is across the dateline

    def test_liquid_fraction_is_a_fraction(self):
        m = modis_arrays()
        m["cloud"][:] = True
        res = extract(m, ceres_arrays(m["lat"], m["lon"], [(CY, CX), (CY, CX + 3)]))
        assert res["scalars"]["LiqPhsf"][0] == pytest.approx(1.0) and res["scalars"]["CF"][0] == pytest.approx(1.0)
        m["phase"][:] = 2
        assert extract(m, ceres_arrays(m["lat"], m["lon"], [(CY, CX), (CY, CX + 3)]))["scalars"]["LiqPhsf"][0] == 0.0

    def test_liquid_fraction_counts_only_cloudy_pixels(self):
        m = modis_arrays()
        m["cloud"][:, ::2] = True                                  # half the columns are cloudy, phase 1 everywhere
        res = extract(m, ceres_arrays(m["lat"], m["lon"], [(CY, CX), (CY, CX + 3)]))
        assert res["scalars"]["LiqPhsf"][0] == pytest.approx(1.0)
        m["phase"][:, :50:2] = 2                                   # ice in the cloudy columns of the left half
        assert 0 < extract(m, ceres_arrays(m["lat"], m["lon"], [(CY, CX), (CY, CX + 3)]))["scalars"]["LiqPhsf"][0] < 1

    def test_no_cloud_means_no_liquid_fraction(self):
        m = modis_arrays()
        res = extract(m, ceres_arrays(m["lat"], m["lon"], [(CY, CX), (CY, CX + 3)]))
        assert np.isnan(res["scalars"]["LiqPhsf"][0]) and res["scalars"]["meanCTH"][0] == pytest.approx(2000.0)

    def test_without_mod06_cth_is_nan(self):
        m = modis_arrays()
        m["cth"] = m["phase"] = None
        res = extract(m, ceres_arrays(m["lat"], m["lon"], [(CY, CX), (CY, CX + 3)]))
        assert np.isnan(res["scalars"]["meanCTH"][0]) and np.isnan(res["scalars"]["LiqPhsf"][0])

    def test_no_images(self):
        m = modis_arrays()
        assert extract(m, ceres_arrays(m["lat"], m["lon"], [(CY, CX), (CY, CX + 3)]), gen_images=False)["images"] is None

    def test_more_than_10000_footprints_would_not_overflow(self):
        m = modis_arrays()
        spots = [(CY, CX), (CY, CX + 1)] * 3
        assert len(extract(m, ceres_arrays(m["lat"], m["lon"], spots))["scalars"]["Fsw"]) == 5

    def test_lw_only_footprint_is_kept(self):
        m = modis_arrays()
        ce = ceres_arrays(m["lat"], m["lon"], [(CY, CX), (CY, CX + 3)])
        ce["fsw"][0] = np.nan
        res = extract(m, ce)
        assert np.isnan(res["scalars"]["Fsw"][0]) and res["scalars"]["Flw"][0] == pytest.approx(240.0)


# ---- file level -------------------------------------------------------------------------------------
def write_modis_files(root, key="A2008001.0300", lon0=150.0, with06=True, with35=True):
    m = modis_arrays(lon0)
    stem = f"{key}.061.2008001999999.hdf"
    d = {}

    def band_var(ds, name, nb, data):
        ds.createDimension(f"b_{name}", nb)
        v = ds.createVariable(name, "u2", (f"b_{name}", "y", "x"), fill_value=65535)
        v[:] = data
        v.setncattr("radiance_scales", np.full(nb, 0.01, dtype="float32"))
        v.setncattr("radiance_offsets", np.zeros(nb, dtype="float32"))

    (root / "MOD02").mkdir(parents=True, exist_ok=True)
    f02 = root / "MOD02" / f"MYD021KM.{stem}"
    with nc.Dataset(f02, "w") as ds:
        ds.createDimension("y", H)
        ds.createDimension("x", W)
        raw = lambda a: np.round(a / 0.01).astype("uint16")  # noqa: E731
        s = m["solar"]
        r500 = np.stack([s[0], s[1], s[7], s[8], s[9]])        # bands 3,4,5,6,7
        r250 = np.stack([s[2], s[3]])                           # bands 1,2
        r1 = np.stack([np.full((H, W), 5.0)] * 11 + [s[4], s[5], s[6], s[10]])     # ..., 17, 18, 19, 26
        band_var(ds, "EV_500_Aggr1km_RefSB", 5, raw(r500))
        band_var(ds, "EV_250_Aggr1km_RefSB", 2, raw(r250))
        band_var(ds, "EV_1KM_RefSB", 15, raw(r1))
        band_var(ds, "EV_1KM_Emissive", 16, raw(m["thermal"]))
    d["MOD02"] = str(f02)
    (root / "MOD03").mkdir(exist_ok=True)
    f03 = root / "MOD03" / f"MYD03.{stem}"
    with nc.Dataset(f03, "w") as ds:
        ds.createDimension("y", H)
        ds.createDimension("x", W)
        for name, arr in (("Latitude", m["lat"]), ("Longitude", m["lon"])):
            ds.createVariable(name, "f4", ("y", "x"), fill_value=-999.0)[:] = arr
        for name in ("SolarZenith", "SensorZenith", "SolarAzimuth", "SensorAzimuth"):
            v = ds.createVariable(name, "i2", ("y", "x"), fill_value=-32768)
            v.scale_factor = 0.01
            v[:] = {"SolarZenith": 30.0, "SensorZenith": 20.0, "SolarAzimuth": 100.0, "SensorAzimuth": 10.0}[name]
    d["MOD03"] = str(f03)
    if with35:
        (root / "MOD35").mkdir(exist_ok=True)
        f35 = root / "MOD35" / f"MYD35_L2.{stem}"
        with nc.Dataset(f35, "w") as ds:
            ds.createDimension("seg", 6)
            ds.createDimension("y", H)
            ds.createDimension("x", W)
            v = ds.createVariable("Cloud_Mask", "i1", ("seg", "y", "x"), fill_value=0)
            data = np.zeros((6, H, W), dtype="int8")
            data[0] = np.int8(0b11010111 - 256)                 # determined, clear, no glint, land code 3
            v[:] = data
        d["MOD35"] = str(f35)
    if with06:
        (root / "MOD06").mkdir(exist_ok=True)
        f06 = root / "MOD06" / f"MYD06_L2.{stem}"
        with nc.Dataset(f06, "w") as ds:
            ds.createDimension("y", H)
            ds.createDimension("x", W)
            ds.createVariable("Cloud_Phase_Infrared_1km", "i1", ("y", "x"), fill_value=-127)[:] = 1
            ds.createVariable("cloud_top_height_1km", "i2", ("y", "x"), fill_value=-32768)[:] = 1500
        d["MOD06"] = str(f06)
    return d, m


class TestReaders:
    def test_radiance_scaled_and_saturated_values_removed(self, tmp_path):
        d, m = write_modis_files(tmp_path)
        with nc.Dataset(d["MOD02"], "a") as ds:
            ds.variables["EV_250_Aggr1km_RefSB"][0, 3, 3] = 65534       # saturated
            ds.variables["EV_250_Aggr1km_RefSB"][0, 3, 4] = 65535       # fill
        r = bd.read_radiance(d["MOD02"], "EV_250_Aggr1km_RefSB")
        assert r.shape == (2, H, W) and np.isnan(r[0, 3, 3]) and np.isnan(r[0, 3, 4])
        assert r[0, 10, 10] == pytest.approx(m["solar"][2][10, 10], rel=1e-3)

    def test_scaled_integer_geometry_in_degrees_and_fill_is_nan(self, tmp_path):
        d, _ = write_modis_files(tmp_path)
        with nc.Dataset(d["MOD03"], "a") as ds:
            ds.set_auto_maskandscale(False)
            ds.variables["SolarZenith"][0, 0] = -32768          # raw fill value
        z = bd.read_plain(d["MOD03"], "SolarZenith")
        assert z[5, 5] == pytest.approx(30.0) and np.isnan(z[0, 0])

    def test_cloud_mask_zero_bytes_are_not_masked(self, tmp_path):
        d, _ = write_modis_files(tmp_path)
        b = bd.read_cloud_mask_byte0(d["MOD35"])
        assert b.dtype.kind == "i" and b.min() == 0b11010111 and not np.ma.isMaskedArray(b)


class TestProcessGranule:
    def task(self, tmp_path, files, m, **kw):
        ce = ceres_arrays(m["lat"], m["lon"], [(CY, CX), (CY, CX + 3)], hour_key=200800103)
        return {"key": "A2008001.0300", "hour_key": 200800103, "dir": files, "ceres": ce, "psf": psf27(),
                "gen_images": True, "month_out": str(tmp_path / "out"), **kw}

    def test_writes_three_files_readable_by_the_project_loader(self, tmp_path):
        files, m = write_modis_files(tmp_path / "in")
        res = bd.process_granule(self.task(tmp_path, files, m))
        assert res["status"] == "ok" and res["n"] == 1 and res["msg"] == ""
        from flux1km import io

        g = io.load_granule(tmp_path / "out", "A2008001.0300")
        assert g.n_footprints == 1 and g.moments["mSWspec"].shape == (1, 11) and g.images["ImSW"].shape == (1, 11, 27, 55)
        assert g.scalars["landf"][0] == pytest.approx(3.0) and g.scalars["glintf"][0] == 0
        assert g.scalars["SZA"][0] == pytest.approx(30.0)
        assert g.images["Imviews"][0, 2, 0, 0] == pytest.approx(90.0, abs=0.1)     # |100-10| = 90

    def test_missing_cloud_products_give_nan_and_a_message(self, tmp_path):
        files, m = write_modis_files(tmp_path / "in", with06=False, with35=False)
        res = bd.process_granule(self.task(tmp_path, files, m))
        assert res["status"] == "ok" and "MOD35" in res["msg"] and "MOD06" in res["msg"]
        with np.load(tmp_path / "out/scalars/A2008001.0300.npz") as s:
            assert np.isnan(s["CF"]).all() and np.isnan(s["landf"]).all() and np.isnan(s["meanCTH"]).all()
            assert np.isfinite(s["Fsw"]).all()

    def test_error_is_returned_not_raised(self, tmp_path):
        files, m = write_modis_files(tmp_path / "in")
        files["MOD02"] = str(tmp_path / "nope.hdf")
        res = bd.process_granule(self.task(tmp_path, files, m))
        assert res["status"] == "error" and res["msg"]
        assert not (tmp_path / "out/scalars/A2008001.0300.npz").exists()

    def test_no_footprint_is_empty_and_writes_nothing(self, tmp_path):
        files, m = write_modis_files(tmp_path / "in")
        t = self.task(tmp_path, files, m)
        t["ceres"]["fsw"][:] = 0.0
        assert bd.process_granule(t)["status"] == "empty"
        assert not (tmp_path / "out").exists()

    def test_no_images_option(self, tmp_path):
        files, m = write_modis_files(tmp_path / "in")
        bd.process_granule(self.task(tmp_path, files, m, gen_images=False))
        assert not (tmp_path / "out/images").exists() and (tmp_path / "out/scalars/A2008001.0300.npz").exists()


def write_ceres_month(path, ceres_rows, units="days since 2008-01-01 00:00:00"):
    n = len(ceres_rows["time"])
    with nc.Dataset(path, "w") as ds:
        ds.createDimension("record", n)
        for name in [*bd.CERES_FIELDS.values(), "time"]:
            v = ds.createVariable(name, "f8", ("record",), fill_value=3.4028235e38)
            if name == "time":
                v.units = units
        ds.variables["time"][:] = ceres_rows["time"]
        for k, name in bd.CERES_FIELDS.items():
            ds.variables[name][:] = ceres_rows[k]


class TestLoadCeres:
    def rows(self, lat, lon, times):
        n = len(times)
        full = lambda v: np.full(n, v, dtype="float64")  # noqa: E731
        return {"time": times, "colat": 90 - np.asarray(lat), "lon": np.asarray(lon) % 360, "sza": full(30.0),
                "vza": full(20.0), "raa": full(90.0), "fsw": full(300.0), "flw": full(240.0), "isw": full(90.0),
                "ilw": full(75.0), "clrf": full(10.0), "aod": full(0.1)}

    def test_conversions(self, tmp_path):
        write_ceres_month(tmp_path / "CERES_x.nc", self.rows([10.0, 20.0], [-170.0, 10.0], np.array([0.1, 0.2])))
        c = bd.load_ceres(tmp_path / "CERES_x.nc")
        assert c["lat"].tolist() == pytest.approx([10.0, 20.0]) and c["lon"].tolist() == pytest.approx([-170.0, 10.0])
        assert c["key"].tolist() == [200800102, 200800104]

    def test_sorted_by_time_and_fill_becomes_nan(self, tmp_path):
        r = self.rows([10.0, 20.0, 30.0], [10.0, 20.0, 30.0], np.array([2.0, 1.0, 3.0]))
        r["fsw"][1] = 3.4028235e38
        write_ceres_month(tmp_path / "CERES_x.nc", r)
        c = bd.load_ceres(tmp_path / "CERES_x.nc")
        assert c["lat"].tolist() == pytest.approx([20.0, 10.0, 30.0]) and np.isnan(c["fsw"][0])

    def test_missing_variable(self, tmp_path):
        write_ceres_month(tmp_path / "CERES_x.nc", self.rows([1.0], [1.0], np.array([0.0])))
        with nc.Dataset(tmp_path / "CERES_x.nc", "a") as ds:
            ds.renameVariable("CERES_SW_TOA_flux___upwards", "other")
        with pytest.raises(ValueError, match="CERES_SW_TOA_flux___upwards"):
            bd.load_ceres(tmp_path / "CERES_x.nc")


class TestCeresSliceAndFiles:
    def test_slice_includes_next_footprint(self):
        c = {"key": np.array([1, 1, 2, 2, 3]), "lat": np.arange(5.0)}
        assert bd.ceres_slice(c, 2)["lat"].tolist() == [2.0, 3.0, 4.0]
        assert bd.ceres_slice(c, 3)["lat"].tolist() == [4.0]
        assert bd.ceres_slice(c, 9) is None

    def test_find_ceres_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            bd.find_ceres_file(tmp_path)
        (tmp_path / "CERES_a.nc").write_bytes(b"")
        assert bd.find_ceres_file(tmp_path).name == "CERES_a.nc"
        (tmp_path / "CERES_b.nc").write_bytes(b"")
        with pytest.raises(ValueError):
            bd.find_ceres_file(tmp_path)


# ---- whole month ------------------------------------------------------------------------------------
def make_month(tmp_path, n_granules=2):
    """Two granules in jan 2008 (days 1 and 2) + a CERES month file with footprints in their hours."""
    mdir = tmp_path / "data/AQUA/2008/Pacific/jan"
    m = None
    times, spots = [], []
    for i in range(n_granules):
        key = f"A2008{i + 1:03d}.0300"
        _, m = write_modis_files(mdir, key=key)
        times += [i + 3 / 24 + 0.001, i + 3 / 24 + 0.0011]
        spots += [(CY, CX), (CY, CX + 3)]
    lat, lon = m["lat"], m["lon"]
    la = [lat[r, c] for r, c in spots]
    lo = [lon[r, c] for r, c in spots]
    n = len(spots)
    full = lambda v: np.full(n, v, dtype="float64")  # noqa: E731
    write_ceres_month(mdir / "CERES_SSF_Aqua_2008_jan.nc",
                      {"time": np.array(times), "colat": 90 - np.array(la), "lon": np.array(lo) % 360,
                       "sza": full(30.0), "vza": full(20.0), "raa": full(90.0), "fsw": full(300.0),
                       "flw": full(240.0), "isw": full(90.0), "ilw": full(75.0), "clrf": full(10.0), "aod": full(0.1)})
    return tmp_path / "data", tmp_path / "out"


class TestBuildMonth:
    def run(self, data, out, **kw):
        logs = []
        c = bd.build_month(data, out, 2008, "Pacific", 1, psf27(), start="2008-01-01", end="2008-12-31", workers=1,
                           gen_images=True, limit=None, dry_run=False, log=logs.append, **kw)
        return c, logs

    def test_builds_resumes_gathers_and_verifies(self, tmp_path):
        data, out = make_month(tmp_path)
        c, _ = self.run(data, out)
        assert c["ok"] == 2 and c["error"] == 0
        month = out / "Pacific/2008/jan"
        assert sorted(p.name for p in (month / "scalars").glob("*.npz")) == ["A2008001.0300.npz", "A2008002.0300.npz"]
        assert (month / "granules.csv").read_text().count("ok") == 2
        c2, logs2 = self.run(data, out)                                   # resume: nothing to do
        assert c2["ok"] == 0 and c2["skipped"] == 2 and "0 to build" in logs2[0]
        assert bd.gather(out) == {"Pacific": 6}
        assert sorted(p.name for p in (out / "Pacific/scalars").iterdir()) == ["A2008001.0300.npz", "A2008002.0300.npz"]
        assert (out / "Pacific/images/A2008001.0300.npz").is_symlink()
        assert bd.gather(out) == {"Pacific": 6}                           # idempotent
        assert bd.verify_triplets(out / "Pacific") == []
        from flux1km import io

        assert io.find_granules(out / "Pacific") == ["A2008001.0300", "A2008002.0300"]
        assert io.load_granule(out / "Pacific", "A2008002.0300").n_footprints == 1

    def test_failed_granule_is_retried_and_others_go_on(self, tmp_path):
        data, out = make_month(tmp_path)
        bad = next((data / "AQUA/2008/Pacific/jan/MOD03").glob("*A2008001*"))
        bad.write_bytes(b"corrupt")
        c, _ = self.run(data, out)
        assert c["ok"] == 1 and c["error"] == 1
        bad.unlink()
        make_month(tmp_path)                                              # restores the file
        c, _ = self.run(data, out)
        assert c["ok"] == 1 and c["skipped"] == 1

    def test_hour_without_ceres_is_recorded_empty_and_not_retried(self, tmp_path):
        data, out = make_month(tmp_path)
        write_modis_files(data / "AQUA/2008/Pacific/jan", key="A2008005.0700")
        c, _ = self.run(data, out)
        assert c["ok"] == 2 and c["empty"] == 1
        c2, _ = self.run(data, out)
        assert c2["ok"] == 0 and c2["empty"] == 0 and c2["skipped"] == 3

    def test_date_range_selects_granules(self, tmp_path):
        data, out = make_month(tmp_path)
        c = bd.build_month(data, out, 2008, "Pacific", 1, psf27(), start="2008-01-02", end="2008-01-02", workers=1,
                           gen_images=False, limit=None, dry_run=False, log=lambda s: None)
        assert c["ok"] == 1 and not (out / "Pacific/2008/jan/images").exists()

    def test_dry_run_and_limit(self, tmp_path):
        data, out = make_month(tmp_path)
        c = bd.build_month(data, out, 2008, "Pacific", 1, psf27(), start="2008-01-01", end="2008-12-31", workers=1,
                           gen_images=True, limit=None, dry_run=True, log=lambda s: None)
        assert c["ok"] == 0 and not out.exists()
        c = bd.build_month(data, out, 2008, "Pacific", 1, psf27(), start="2008-01-01", end="2008-12-31", workers=1,
                           gen_images=True, limit=1, dry_run=False, log=lambda s: None)
        assert c["ok"] == 1

    def test_missing_month_folder(self, tmp_path):
        assert bd.build_month(tmp_path, tmp_path / "o", 2008, "Pacific", 3, psf27(), start="2008-01-01",
                              end="2008-12-31", workers=1, gen_images=True, limit=None, dry_run=False,
                              log=lambda s: None)["ok"] == 0

    def test_two_workers_give_the_same_files(self, tmp_path):
        data, out = make_month(tmp_path)
        c = bd.build_month(data, out, 2008, "Pacific", 1, psf27(), start="2008-01-01", end="2008-12-31", workers=2,
                           gen_images=True, limit=None, dry_run=False, log=lambda s: None)
        assert c["ok"] == 2


class TestVerify:
    def test_detects_row_count_mismatch_and_missing_keys(self, tmp_path):
        data, out = make_month(tmp_path)
        TestBuildMonth().run(data, out)
        bd.gather(out)
        month = out / "Pacific/2008/jan"
        with np.load(month / "moments/A2008001.0300.npz") as f:
            arrs = {k: f[k] for k in f.files}
        arrs["mSWspec"] = np.vstack([arrs["mSWspec"]] * 2)
        np.savez(month / "moments/A2008001.0300.npz", **arrs)
        assert any("row count" in p for p in bd.verify_triplets(out / "Pacific"))

    def test_empty_folder(self, tmp_path):
        (tmp_path / "scalars").mkdir()
        assert "no scalars" in bd.verify_triplets(tmp_path)[0]


class TestCommandLine:
    def test_missing_psf_file(self, tmp_path):
        with pytest.raises(SystemExit) as e:
            bd.main(["--out", str(tmp_path), "--psf", str(tmp_path / "nope.csv")])
        assert e.value.code == 2

    def test_default_psf_is_the_advisors_file(self):
        from flux1km import aggregation

        assert bd.DEFAULT_PSF.is_file()
        p = bd.load_psf(str(bd.DEFAULT_PSF))
        assert p.shape == (27, 55) and p.sum() == pytest.approx(1.0)
        assert int(p.sum(axis=0).argmax()) == 18                        # the PSF peak column, not the patch centre
        assert np.allclose(p, aggregation.load_ceres_psf(bd.DEFAULT_PSF))

    def test_real_psf_puts_its_peak_on_the_footprint_pixel(self):
        p = bd.load_psf(str(bd.DEFAULT_PSF))
        m = modis_arrays()
        fwd = extract(m, ceres_arrays(m["lat"], m["lon"], [(CY, CX), (CY, CX + 3)]), psf=p)
        bwd = extract(m, ceres_arrays(m["lat"], m["lon"], [(CY, CX), (CY, CX - 3)]), psf=p)
        assert fwd["images"]["ImSW"][0, 0, 13, 18] == pytest.approx(10 + CX)
        assert bwd["images"]["ImSW"][0, 0, 13, 36] == pytest.approx(10 + CX)     # mirrored: 54 - 18

    def test_bad_dates(self, tmp_path):
        with pytest.raises(SystemExit):
            bd.main(["--out", str(tmp_path), "--psf", "ellipse", "--start", "2009-01-01", "--end", "2008-01-01"])

    def test_end_to_end_with_ellipse_psf(self, tmp_path):
        data, out = make_month(tmp_path)
        rc = bd.main(["--data", str(data), "--out", str(out), "--workers", "1"])               # default PSF
        assert rc == 0 and (out / "Pacific/scalars/A2008001.0300.npz").exists()
        assert bd.main(["--out", str(out), "--verify"]) == 0
        assert "granules" in (out / "build_log.txt").read_text() or "done" in (out / "build_log.txt").read_text()


class TestInvalidChannels:
    """Dead detectors (Aqua band 6 = SW channel 8, 1.64 um) make whole row groups invalid in one channel."""

    def dead_band6(self):
        m = modis_arrays()
        rows = np.arange(H)
        m["solar"][8][(rows % 10) < 7] = np.nan                  # 70% of the rows: dead detectors
        return m

    def spots(self, m):
        return ceres_arrays(m["lat"], m["lon"], [(CY, CX), (CY, CX + 3)])

    def test_footprint_is_kept_with_nan_moments_for_the_bad_channel(self):
        m = self.dead_band6()
        res = extract(m, self.spots(m))
        assert res is not None
        mom = res["moments"]
        assert np.isnan(mom["mSWspec"][0, 8]) and np.isnan(mom["stdSWspec"][0, 8])
        good = [i for i in range(11) if i != 8]
        assert np.isfinite(mom["mSWspec"][0, good]).all() and np.isfinite(mom["mLWspec"][0]).all()

    def test_bad_channel_stays_nan_in_the_images_and_out_of_the_sum(self):
        m = self.dead_band6()
        res = extract(m, self.spots(m))
        assert np.isnan(res["images"]["ImSW"][0, 8]).any() and np.isfinite(res["images"]["ImSW"][0, 0]).all()
        # channels hold constants 10..20 (+ the column gradient on channel 0): the sum skips channel 8
        full = extract(modis_arrays(), self.spots(m))["scalars"]["mSumIsw"][0]
        assert full - res["scalars"]["mSumIsw"][0] == pytest.approx(10 + 8, rel=1e-4)

    def test_requiring_all_channels_rejects_it_as_the_old_code_did(self):
        m = self.dead_band6()
        stats = {}
        assert extract(m, self.spots(m), required_sw=range(11), required_lw=range(16), stats=stats) is None
        assert stats["invalid"] == 1 and stats["accepted"] == 0

    def test_invalid_required_channel_rejects(self):
        m = modis_arrays()
        m["solar"][3][CY - 3:CY + 4, CX - 3:CX + 4] = np.nan     # 0.85 um, under the PSF core
        stats = {}
        assert extract(m, self.spots(m), stats=stats) is None
        assert stats["invalid"] == 1

    def test_stats_count_every_reason(self):
        m = modis_arrays()
        ce = ceres_arrays(m["lat"], m["lon"], [(CY, CX), (CY, CX + 3), (5, CX), (5, CX + 3)])
        stats = {}
        extract(m, ce, stats=stats)
        assert stats == {"candidates": 3, "outside": 0, "edge": 1, "invalid": 0, "accepted": 2}   # the 4th has no next footprint

    def test_a_fully_valid_granule_is_unchanged_by_the_channel_logic(self):
        m = modis_arrays()
        res = extract(m, self.spots(m))
        assert np.isfinite(res["moments"]["mSWspec"]).all() and np.isfinite(res["moments"]["kurtLWspec"]).all()


class TestSpread:
    def test_picks_are_spread_and_include_the_ends(self):
        assert bd.spread(list(range(100)), 3) == [0, 50, 99]

    def test_more_than_available(self):
        assert bd.spread([1, 2], 5) == [1, 2]
        assert bd.spread([], 3) == []


class TestLimitDiagnoseRetry:
    def build(self, data, out, **kw):
        base = dict(start="2008-01-01", end="2008-12-31", workers=1, gen_images=False, limit=None, dry_run=False,
                    log=lambda s: None)
        return bd.build_month(data, out, 2008, "Pacific", 1, psf27(), **{**base, **kw})

    def test_limit_takes_granules_spread_over_the_month(self, tmp_path):
        data, out = make_month(tmp_path, n_granules=3)
        c = self.build(data, out, limit=2)
        assert c["ok"] == 2
        assert sorted(p.stem for p in (out / "Pacific/2008/jan/scalars").glob("*.npz")) == ["A2008001.0300", "A2008003.0300"]

    def test_empty_granule_records_why(self, tmp_path):
        data, out = make_month(tmp_path)
        write_modis_files(data / "AQUA/2008/Pacific/jan", key="A2008001.0900")
        self.build(data, out)
        text = (out / "Pacific/2008/jan/granules.csv").read_text()
        assert "no CERES footprint in this hour" in text

    def test_retry_empty_rebuilds_them(self, tmp_path):
        data, out = make_month(tmp_path)
        jan = out / "Pacific/2008/jan"
        jan.mkdir(parents=True)
        bd._append_log(jan, {"key": "A2008001.0300", "status": "empty", "n": 0, "sec": 0.0, "msg": "old rule"})
        assert self.build(data, out)["ok"] == 1                    # only the other granule: this one is "done"
        c = self.build(data, out, retry_empty=True)
        assert c["ok"] == 1 and c["skipped"] == 1
        assert self.build(data, out)["skipped"] == 2               # the new "ok" row wins over the old "empty" one

    def test_diagnose_builds_nothing_and_reports(self, tmp_path):
        data, out = make_month(tmp_path)
        logs = []
        c = self.build(data, out, diagnose=2, log=logs.append)
        assert c["ok"] == 0 and not (out / "Pacific/2008/jan/scalars").exists()
        text = "\n".join(logs)
        assert "A2008001.0300" in text and "accepted=1" in text and "invalid pixels per SW channel: none" in text

    def test_diagnose_names_the_bad_channel(self, tmp_path):
        data, out = make_month(tmp_path, n_granules=1)
        f02 = next((data / "AQUA/2008/Pacific/jan/MOD02").glob("*.hdf"))
        with nc.Dataset(f02, "a") as ds:
            ds.variables["EV_500_Aggr1km_RefSB"][3, ::2, :] = 65533          # band 6 = 1.64 um: saturated rows
        logs = []
        self.build(data, out, diagnose=1, log=logs.append)
        assert "1.64um=50%" in "\n".join(logs)

    def test_cli_options(self, tmp_path):
        data, out = make_month(tmp_path)
        assert bd.main(["--data", str(data), "--out", str(out), "--workers", "1", "--limit", "1", "--no-images",
                        "--required-channels", "all"]) == 0
        assert len(list((out / "Pacific/2008/jan/scalars").glob("*.npz"))) == 1
        assert bd.main(["--data", str(data), "--out", str(out), "--diagnose", "1"]) == 0
