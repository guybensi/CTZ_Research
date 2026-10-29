import numpy as np
import pytest

from flux1km import geometry


class TestGlintAngle:
    def test_zero_at_specular(self):
        assert geometry.glint_angle(30.0, 30.0, 0.0) == pytest.approx(0.0, abs=1e-6)

    def test_backscatter(self):
        assert geometry.glint_angle(30.0, 30.0, 180.0) == pytest.approx(60.0)

    def test_nadir_view_equals_sza(self):
        assert geometry.glint_angle(40.0, 0.0, 77.0) == pytest.approx(40.0)

    def test_convention_specular_at_180(self):
        assert geometry.glint_angle(30.0, 30.0, 180.0, raa_convention="specular_at_180") == pytest.approx(0.0, abs=1e-6)

    def test_bad_convention(self):
        with pytest.raises(ValueError):
            geometry.glint_angle(1.0, 1.0, 1.0, raa_convention="x")

    def test_broadcast_and_nan(self):
        out = geometry.glint_angle(np.array([[30.0, np.nan]]), 30.0, np.array([0.0, 0.0]))
        assert out.shape == (1, 2) and np.isnan(out[0, 1])

    def test_rounding_never_gives_nan(self):
        sza = np.linspace(0, 89, 50)
        out = geometry.glint_angle(sza, sza, 0.0)
        assert np.all(np.isfinite(out)) and np.all(out >= 0)


class TestEncodeAngles:
    def test_values(self):
        e = geometry.encode_angles(np.array([0.0, 60.0]), np.array([0.0, 0.0]), np.array([90.0, 0.0]))
        assert e.shape == (2, 4)
        np.testing.assert_allclose(e, [[1, 1, 1, 0], [0.5, 1, 0, 1]], atol=1e-12)


class TestInsolationAlbedo:
    def test_insolation(self):
        np.testing.assert_allclose(geometry.toa_insolation(np.array([0.0, 60.0, 95.0])), [1361.0, 680.5, 0.0])
        assert geometry.toa_insolation(0.0, earth_sun_au=2.0) == pytest.approx(1361.0 / 4)

    def test_albedo_roundtrip_and_night(self):
        a = geometry.flux_to_albedo(np.array([136.1, 50.0]), np.array([0.0, 95.0]))
        assert a[0] == pytest.approx(0.1) and np.isnan(a[1])
        assert geometry.albedo_to_flux(np.array([0.1]), np.array([0.0]))[0] == pytest.approx(136.1)
