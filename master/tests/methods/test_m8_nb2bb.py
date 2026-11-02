import numpy as np
import pytest

from flux1km import aggregation, features, geometry
from flux1km.methods import m8_nb2bb
from flux1km.models import get_model


def _rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


class TestStage1Features:
    def test_footprint_features_aggregate_pixel_features_exactly(self, easy):
        b = easy[2]
        fp = m8_nb2bb.stage1_footprint_features(b)
        px = m8_nb2bb.stage1_pixel_features(b)
        w = aggregation.fov_weights(b.images["FOVorientation"])
        agg = (px[:, :4] * w[:, None]).sum(axis=(2, 3))  # the moments use the same operator
        np.testing.assert_allclose(fp[:, :4], agg, rtol=1e-4)

    def test_needs_moments(self, easy):
        from dataclasses import replace
        with pytest.raises(ValueError):
            m8_nb2bb.stage1_footprint_features(replace(easy[0], moments=None))


class TestADMFeatures:
    def test_glint_lobe(self):
        f = m8_nb2bb.adm_features(np.array([30.0, 30.0]), np.array([30.0, 30.0]), np.array([0.0, 180.0]), np.array([0.1, 0.1]))
        assert f[0, 5] > 0.99 and f[1, 5] < 0.01  # specular vs backscatter


class TestModel:
    def test_requires_isw(self, easy):
        from dataclasses import replace
        train = easy[0]
        with pytest.raises(ValueError):
            get_model("M8").fit(replace(train, scalars={k: v for k, v in train.scalars.items() if k != "Isw"}))

    def test_hard_scene_beats_climatology_and_bounded(self, hard):
        train, val, test = hard
        W = features.footprint_weights(test)
        m = get_model("M8", cv_max=1.0).fit(train)
        clim = get_model("B-clim").fit(train)
        assert _rmse(m.predict_footprint(test, W), test.label) < _rmse(clim.predict_footprint(test, W), test.label)
        maps = m.predict_map(test)
        sza = features.pixel_geometry(test.images)[0]
        assert np.all(maps >= 0) and np.all(maps <= geometry.toa_insolation(sza) + 1e-6)

    def test_isw_is_never_an_input(self, easy):
        from dataclasses import replace
        train, _, test = easy
        m = get_model("M8").fit(train)
        W = features.footprint_weights(test)
        scrambled = replace(test, scalars={**test.scalars, "Isw": np.zeros(len(test))})
        np.testing.assert_allclose(m.predict_footprint(test, W), m.predict_footprint(scrambled, W))
