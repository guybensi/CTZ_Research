import numpy as np
import pytest

from flux1km import benchmark, sampling_study, splits, synthetic
from flux1km.table import build_table


@pytest.fixture(scope="module")
def setup():
    rng = np.random.default_rng(0)
    gs = [synthetic.make_synthetic_granule(g, rng, n_rows=3, n_cols=8, lat0=-10 + 3 * i) for i, g in enumerate(synthetic.DEMO_GRANULES)]
    t = build_table(gs, splits.group_granules([g.granule_id for g in gs]))
    return gs, t, splits.logo_folds(t)


class TestSubpopulations:
    def test_masks_partition_regimes(self, setup):
        _, t, _ = setup
        m = sampling_study.subpopulations(t)
        reg = m["regime:clear"].astype(int) + m["regime:broken"] + m["regime:overcast"]
        assert np.all(reg <= 1) and np.all(m["glint<25"] ^ m["glint>=25"])


class TestSubpopulationRmse:
    def test_values(self):
        recs = [{"model": "A", "leaky": False, "row": 0, "pred": 1.0, "label": 0.0},
                {"model": "A", "leaky": False, "row": 1, "pred": 3.0, "label": 0.0},
                {"model": "A", "leaky": True, "row": 1, "pred": 99.0, "label": 0.0}]
        out = sampling_study.subpopulation_rmse(recs, {"first": np.array([True, False]), "none": np.array([False, False])}, "A")
        assert out["first"] == 1.0 and np.isnan(out["none"])


class TestRunSamplingStudy:
    def test_study_and_report(self, setup, tmp_path):
        gs, t, folds = setup
        cfg = benchmark.BenchmarkConfig(models=["M2", "M0-ridge"], budget=1)
        res = sampling_study.run_sampling_study(gs, t, folds, cfg, ("S0", "S6", "S7"), {"S6": {"period": 2}}, out_dir=tmp_path)
        assert set(res["per_strategy"]) == {"S0", "S6", "S7"} and res["tau_vs_S0"]["S0"] == pytest.approx(1.0)
        assert "mmd_vs_reference" in res["similarity"] and "Kendall tau" in res["recommendation"]
        text = open(res["results_txt"]).read()
        assert "Train-on-Si x test-on-subpopulation" in text and "RECOMMENDATION" in text
