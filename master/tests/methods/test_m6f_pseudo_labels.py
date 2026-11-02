import numpy as np
import pytest

from flux1km import io, synthetic
from flux1km.methods import m6f_pseudo_labels as m6f
from flux1km.models import get_model


class TestRandomWindows:
    def test_sizes_and_errors(self):
        m = m6f.random_windows(27, 55, 20, np.random.default_rng(0))
        assert m.shape == (20, 27, 55) and np.all(m.sum(axis=(1, 2)) >= 9 * 11)
        with pytest.raises(ValueError):
            m6f.random_windows(5, 5, 1, np.random.default_rng(0), (6, 2))


class TestLoadUnlabeled:
    def test_images_only_folder(self, tmp_path):
        g = synthetic.make_synthetic_granule("A2014060.0245", np.random.default_rng(0), n_rows=1, n_cols=3)
        io.save_granule(tmp_path, io.Granule(g.granule_id, g.scalars, None, g.images, None))
        imgs = m6f.load_unlabeled_images(tmp_path, max_patches=2)
        assert imgs["ImSW"].shape == (2, 11, 27, 55) and set(imgs) == {"ImSW", "ImLW", "Imviews"}
        assert m6f.load_unlabeled_images(tmp_path / "none") is None


class TestPseudoLabelModel:
    def test_fit_with_unlabeled_granules(self, easy, tmp_path):
        pytest.importorskip("torch")
        g = synthetic.make_synthetic_granule("A2015001.0100", np.random.default_rng(5), n_rows=1, n_cols=3)
        io.save_granule(tmp_path, io.Granule(g.granule_id, g.scalars, None, g.images, None))
        train, val, test = easy
        m = get_model("M6f", epochs=2, pseudo_per_patch=2, unlabeled_root=str(tmp_path)).fit(train, val)
        assert m.n_pseudo == 2 * len(train) + 2 * 3
        assert m.predict_map(test).shape == (len(test), 27, 55)
