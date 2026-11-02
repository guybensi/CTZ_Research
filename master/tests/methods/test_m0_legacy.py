import numpy as np
import pytest

from flux1km import aggregation, schema
from flux1km.methods import m0_legacy
from flux1km.models import get_model

W = aggregation.uniform_weights(27, 55)


class TestNormReference:
    def test_bands(self):
        assert [m0_legacy.norm_reference(c) for c in ("0.47", "2.13", "3.75", "4.5", "6.7", "14.23")] == \
            ["0.47", "0.47", "4.05", "4.05", "11.0", "11.0"]


class TestLegacyFeatures:
    def test_37_and_55_features(self, easy):
        X37, n37 = m0_legacy.legacy_features(easy[0], m0_legacy.ALLSKY37_CHANNELS, ("m", "skew"))
        X55, n55 = m0_legacy.legacy_features(easy[0], m0_legacy.ADVISOR_CHANNELS, ("m", "std", "skew", "kurt"))
        assert X37.shape[1] == 37 and X55.shape[1] == 55
        schema.assert_no_ceres_inputs(n37 + n55)
        assert n37[:3] == ["SZA", "VZA", "RAA"]

    def test_cross_normalisation(self, easy):
        b = easy[0]
        X, names = m0_legacy.legacy_features(b, ("0.47", "0.66", "11.0", "12.0"), ("m",))
        m047, m066 = b.moments["mSWspec"][:, 0], b.moments["mSWspec"][:, 2]
        np.testing.assert_allclose(X[:, names.index("m0.66")], m066 / m047)
        np.testing.assert_allclose(X[:, names.index("m0.47")], m047)  # the reference itself is kept
        m11, m12 = b.moments["mLWspec"][:, 10], b.moments["mLWspec"][:, 11]
        np.testing.assert_allclose(X[:, names.index("m12.0")], m12 / m11)

    def test_errors(self, easy):
        with pytest.raises(KeyError):
            m0_legacy.legacy_features(easy[0], ("0.99",), ("m",))


class TestWidths:
    def test_widths(self):
        assert m0_legacy.legacy_widths(37) == (18, 9, 5, 5) and m0_legacy.legacy_widths(2) == (1, 1, 1, 5)


class TestModels:
    @pytest.mark.parametrize("name", ["M0-allsky37", "M0-advisor"])
    def test_fit_predict_footprint_only(self, easy, name):
        pytest.importorskip("torch")
        train, val, test = easy
        m = get_model(name, epochs=3).fit(train, val)
        p = m.predict_footprint(test, W)
        assert p.shape == (len(test),) and np.all(np.isfinite(p))
        assert not m.produces_map
        with pytest.raises(NotImplementedError):
            m.predict_map(test)
