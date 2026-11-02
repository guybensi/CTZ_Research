import numpy as np
import pytest

from flux1km import features
from flux1km.methods import m7_atprk
from flux1km.models import get_model


class TestExponentialKernel:
    def test_values(self):
        k = m7_atprk.exponential_kernel(3, 4, 2.0)
        assert k.shape == (5, 7) and k[2, 3] == 1.0 and k[2, 4] == pytest.approx(np.exp(-0.5))
        with pytest.raises(ValueError):
            m7_atprk.exponential_kernel(2, 2, 0.0)


class TestATPK:
    def test_coherent_alone(self):
        w = np.random.default_rng(0).uniform(size=(3, 6, 8))
        R = np.array([1.0, -2.0, 5.0])
        r = m7_atprk.atpk_residual_maps(w, R, 3.0)
        wn = w / w.sum(axis=(1, 2), keepdims=True)
        np.testing.assert_allclose((r * wn).sum(axis=(1, 2)), R, rtol=1e-4)

    def test_coherent_joint_with_overlap(self):
        w = np.full((3, 6, 8), 1 / 48)
        off = np.array([[0, 0], [0, 4], [3, 2]])
        R = np.array([2.0, -1.0, 4.0])
        r = m7_atprk.atpk_residual_maps(w, R, 5.0, offsets=off, nugget=1e-9)
        np.testing.assert_allclose((r * w).sum(axis=(1, 2)), R, atol=1e-3)
        # overlapping pixels get one value: patch 0 cols 4..7 == patch 1 cols 0..3
        np.testing.assert_allclose(r[0][:, 4:], r[1][:, :4], atol=1e-8)


class TestModel:
    def test_reproduces_labels_and_flags_test_labels(self, easy):
        train, _, test = easy
        m = get_model("M7").fit(train)
        assert m.uses_test_labels
        W = features.footprint_weights(test)
        np.testing.assert_allclose(m.predict_footprint(test, W), test.label, rtol=1e-3)
