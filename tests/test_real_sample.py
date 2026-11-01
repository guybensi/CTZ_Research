"""Smoke tests on the real sample granule in data/sample/ (skipped when it is absent).

They pin the storage conventions PLAN.md section 2.1 documents, so a change in the data format
shows up as a failing test rather than as silently wrong features.
"""
from pathlib import Path

import numpy as np
import pytest

from flux1km import aggregation, features, io, models, schema
from flux1km.table import build_table

ROOT = Path(__file__).resolve().parents[1] / "data" / "sample"
GID = "A2014060.0245"
pytestmark = pytest.mark.skipif(not (ROOT / "images" / f"{GID}.npz").exists(), reason="no real sample data")


@pytest.fixture(scope="module")
def gran():
    return io.load_granule(ROOT, GID)


class TestFormat:
    def test_shapes(self, gran):
        n = gran.n_footprints
        assert gran.images["Imviews"].shape == (n, 3, 27, 55)
        assert gran.images["ImSW"].shape == (n, 11, 27, 55) and gran.images["ImLW"].shape == (n, 16, 27, 55)
        assert gran.moments["mSWspec"].shape == (n, 11) and gran.moments["kurtLWspec"].shape == (n, 16)

    def test_wavelengths_match_schema(self, gran):
        np.testing.assert_allclose(gran.moments["SWwavlngs"], schema.SW_WAVELENGTHS)
        np.testing.assert_allclose(gran.moments["LWwavlngs"], schema.LW_WAVELENGTHS)

    def test_kurtosis_is_excess(self, gran):
        assert schema.MOMENTS_EXCESS_KURTOSIS
        assert np.nanmin(gran.moments["kurtSWspec"]) < 1.0  # impossible for Pearson kurtosis

    def test_cf_is_fraction(self, gran):
        assert np.nanmax(gran.scalars["CF"]) <= 1.0

    def test_imviews_raa_is_supplement_of_scalars(self, gran):
        img = gran.images["Imviews"][:, 2].mean(axis=(1, 2))
        np.testing.assert_allclose(img, 180.0 - gran.scalars["RAA"], atol=1.5)
        _, _, raa = features.pixel_geometry(gran.images)
        np.testing.assert_allclose(raa.mean(axis=(1, 2)), gran.scalars["RAA"], atol=1.5)

    def test_imviews_sza_vza_match_scalars(self, gran):
        v = gran.images["Imviews"].mean(axis=(2, 3))
        np.testing.assert_allclose(v[:, 0], gran.scalars["SZA"], atol=0.5)
        np.testing.assert_allclose(v[:, 1], gran.scalars["VZA"], atol=3.0)


class TestPipelineOnRealSample:
    def test_table_features_and_m2(self, gran):
        t = build_table([gran])
        assert t["CF"].max() <= 100 and t["month"][0] == "2014-03"
        b = features.make_batch([gran], t, np.arange(len(t)))
        X, names = features.pixel_features(b, "F5")
        assert np.all(np.isfinite(X)) and names[-1] == "glint"
        F, _ = features.footprint_poly_features(b, "F5", degree=2)
        assert np.all(np.isfinite(F))
        m = models.get_model("M2", degree=1, alpha=10.0).fit(b)
        assert m.predict_map(b).shape == (len(b), 27, 55)
        assert np.all(np.isfinite(m.predict_footprint(b, aggregation.uniform_weights(27, 55))))
