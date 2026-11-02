import numpy as np

from flux1km import features
from flux1km.methods import m9_homogeneous as m9
from flux1km.models import get_model


class TestHomogeneousWeights:
    def test_mask_and_inverse_cv(self, easy):
        mask, w = m9.homogeneous_weights(easy[0], 10.0)
        assert mask.all() and np.all(w > 0)
        mask0, w0 = m9.homogeneous_weights(easy[0], 0.0)
        assert not mask0.any() and np.all(w0 == 0)


class TestSubset:
    def test_slices_every_part(self, easy):
        s = m9._subset(easy[0], np.array([0, 2]))
        assert len(s) == 2 and s.images["ImSW"].shape[0] == 2 and s.granule.shape == (2,)


class TestModels:
    def test_global_fit_predict(self, easy):
        train, _, test = easy
        for learner in ("gbm", "tree"):
            m = get_model("M9", learner=learner, min_rows=5).fit(train)
            maps = m.predict_map(test)
            assert maps.shape == (len(test), 27, 55) and np.all(maps >= 0)
        assert not m.uses_test_labels

    def test_local_variant_reads_labels_and_improves_fit(self, bigger):
        train, _, test = bigger
        W = features.footprint_weights(test)
        g = get_model("M9", cv_max=1.0).fit(train)
        loc = get_model("M9-local", cv_max=1.0, min_local=5).fit(train)
        assert loc.uses_test_labels
        e_g = np.abs(g.predict_footprint(test, W) - test.label).mean()
        e_l = np.abs(loc.predict_footprint(test, W) - test.label).mean()
        assert e_l <= e_g + 1e-6
