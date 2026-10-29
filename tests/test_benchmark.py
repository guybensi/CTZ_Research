import json

import numpy as np
import pytest

from flux1km import benchmark, splits, synthetic
from flux1km.table import build_table

MODELS = ["B-clim", "B-const", "M0-ridge", "B-bagpixel", "M2"]


@pytest.fixture(scope="module")
def setup():
    rng = np.random.default_rng(0)
    gs = [synthetic.make_synthetic_granule(g, rng, n_rows=3, n_cols=8, lat0=-10 + 3 * i)
          for i, g in enumerate(synthetic.DEMO_GRANULES)]
    t = build_table(gs, splits.group_granules([g.granule_id for g in gs]))
    return gs, t, splits.logo_folds(t)


@pytest.fixture(scope="module")
def full_run(setup):
    gs, t, folds = setup
    return benchmark.run_benchmark(gs, t, folds + [splits.interleave_split(t)],
                                   benchmark.BenchmarkConfig(models=MODELS, budget=3))


class TestRunBenchmark:
    def test_records_for_every_model_and_test_row(self, setup, full_run):
        _, _, folds = setup
        n_test = sum(len(f.test) for f in folds)
        for m in MODELS:
            assert sum(r["model"] == m and not r["leaky"] for r in full_run["records"]) == n_test

    def test_summary_contents(self, full_run):
        s = full_run["summary"]
        m2 = s["models"]["M2"]
        assert m2["overall"]["n"] > 0 and np.isfinite(m2["macro_rmse"])
        assert {"pixel", "physics", "macro_rmse_gaussian_psf", "interleave"} <= set(m2)
        assert "pixel" not in s["models"]["M0-ridge"]  # footprint-only model has no map
        assert any("M2" in k for k in s["comparisons"])
        assert all("B-const" not in k for k in s["comparisons"])  # reads test labels: never ranked

    def test_learned_model_beats_climatology(self, full_run):
        m = full_run["summary"]["models"]
        assert m["M2"]["macro_rmse"] < m["B-clim"]["macro_rmse"]

    def test_decision(self, full_run):
        d = full_run["summary"]["decision"]
        assert d["winner"] in ("B-clim", "M0-ridge", "B-bagpixel", "M2") and d["log"]

    def test_leakage_check_blocks_random_split(self, setup):
        gs, t, _ = setup
        perm = np.random.default_rng(0).permutation(len(t))
        bad = splits.Fold("random", perm[:100], perm[100:120], perm[120:])
        with pytest.raises(splits.LeakageError):
            benchmark.run_benchmark(gs, t, [bad], benchmark.BenchmarkConfig(models=["B-clim"]))

    def test_unknown_model(self, setup):
        gs, t, folds = setup
        with pytest.raises(KeyError):
            benchmark.run_benchmark(gs, t, folds, benchmark.BenchmarkConfig(models=["nope"]))

    def test_writes_outputs(self, tmp_path, setup):
        gs, t, folds = setup
        res = benchmark.run_benchmark(gs, t, folds[:1], benchmark.BenchmarkConfig(models=["M2"], budget=2), out_dir=tmp_path)
        run = next(tmp_path.glob("run_*"))
        assert (run / "predictions.csv").exists()
        json.loads((run / "summary.json").read_text())
        assert res["summary"]["run_dir"] == str(run)

    def test_sampling_strategy_shrinks_training(self, setup):
        gs, t, folds = setup
        cfg = {"models": ["M2"], "budget": 1}
        s6 = benchmark.run_benchmark(gs, t, folds[:1], benchmark.BenchmarkConfig(**cfg, sampling="S6", sampling_params={"period": 2}))
        s0 = benchmark.run_benchmark(gs, t, folds[:1], benchmark.BenchmarkConfig(**cfg))
        assert s6["summary"]["folds"][0]["n_train"] < s0["summary"]["folds"][0]["n_train"]


class TestDecide:
    def test_gates_and_noise_floor(self):
        summ = {"models": {
            "A": {"overall": {"rmse": 10}, "macro_rmse": 10.0, "complexity": 0, "trend_effect": {}, "uses_test_labels": False},
            "B": {"overall": {"rmse": 5}, "macro_rmse": 5.0, "complexity": 1,
                  "trend_effect": {"VZA": {"effect_wm2": 9.0, "p": 1e-5}}, "uses_test_labels": False},
            "C": {"overall": {"rmse": 9}, "macro_rmse": 9.5, "complexity": 2, "trend_effect": {}, "uses_test_labels": False}},
            "comparisons": {"C vs A": {"p_holm": 0.001}}}
        d = benchmark.decide(summ, benchmark.BenchmarkConfig(models=[]))
        assert d["winner"] == "A"  # B fails the trend gate; C's gain (0.5) is below the noise floor
        assert any("B fails" in line for line in d["log"])

    def test_significant_large_gain_replaces(self):
        summ = {"models": {
            "A": {"overall": {"rmse": 10}, "macro_rmse": 10.0, "complexity": 0, "trend_effect": {}, "uses_test_labels": False},
            "C": {"overall": {"rmse": 5}, "macro_rmse": 5.0, "complexity": 2, "trend_effect": {}, "uses_test_labels": False}},
            "comparisons": {"A vs C": {"p_holm": 0.01}}}
        assert benchmark.decide(summ, benchmark.BenchmarkConfig(models=[]))["winner"] == "C"
