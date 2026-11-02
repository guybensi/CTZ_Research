"""Small synthetic batches shared by the method tests (train / val / test from different granules)."""
import numpy as np
import pytest

from flux1km import features, synthetic
from flux1km.table import build_table


def _batches(hard: bool, n_rows: int = 2, n_cols: int = 5, n_granules: int = 4):
    rng = np.random.default_rng(0)
    gs = [synthetic.make_synthetic_granule(g, rng, n_rows=n_rows, n_cols=n_cols, label_noise=0.5, hard=hard, lat0=3.0 * i)
          for i, g in enumerate(synthetic.DEMO_GRANULES[:n_granules])]
    t = build_table(gs)

    def rows(ids):
        return np.concatenate([t.rows_of(gs[i].granule_id) for i in ids])

    train = features.make_batch(gs, t, rows(range(n_granules - 2)))
    val = features.make_batch(gs, t, rows([n_granules - 2]))
    test = features.make_batch(gs, t, rows([n_granules - 1]))
    return train, val, test


@pytest.fixture(scope="session")
def easy():
    return _batches(hard=False)


@pytest.fixture(scope="session")
def hard():
    return _batches(hard=True)


@pytest.fixture(scope="session")
def bigger():
    """More footprints, for tests that check that a method actually learns."""
    return _batches(hard=False, n_rows=4, n_cols=10, n_granules=5)

