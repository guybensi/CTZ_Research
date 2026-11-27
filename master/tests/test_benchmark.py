import json
from pathlib import Path

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

    def test_clear_sky_check_and_clrf_in_records(self, full_run):
        m = full_run["summary"]["models"]
        assert {"bias", "n", "mean_label"} <= set(m["M2"]["clear_check"])
        assert m["B-const"]["clear_check"].get("bias", 0.0) in (0.0,) or np.isnan(m["B-const"]["clear_check"]["bias"])
        assert "Clrf" in full_run["records"][0]

    def test_physics_distance_from_truth_on_synthetic(self, full_run):
        phys = full_run["summary"]["models"]["M2"]["physics"]
        assert phys["distance_source"] == "synthetic truth"
        assert -1 <= phys["sharpness"] <= 1 and phys["sharpness_truth"] > 0.9

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


class TestFixedParams:
    def test_operator_only_for_aggregation_loss_models(self):
        cfg = benchmark.BenchmarkConfig(models=[], operator="uniform", model_params={"M3": {"input_set": "F2"}})
        assert benchmark.fixed_params("M3", cfg, 2) == {"seed": 2, "operator": "uniform", "psf_path": None, "input_set": "F2"}
        assert benchmark.fixed_params("M2", cfg) == {"seed": 0}


class TestRobustness:
    def test_failing_model_is_reported_not_fatal(self, setup, monkeypatch):
        gs, t, folds = setup
        from flux1km import models

        def boom(self, train, val=None):
            raise ValueError("synthetic failure")

        monkeypatch.setattr(models.MODELS["M9"], "fit", boom)
        res = benchmark.run_benchmark(gs, t, folds[:1], benchmark.BenchmarkConfig(models=["M2", "M9"], budget=1))
        assert res["summary"]["errors"] and res["summary"]["errors"][0]["model"] == "M9"
        assert "M2" in res["summary"]["models"]

    def test_results_txt_written(self, tmp_path, setup):
        gs, t, folds = setup
        res = benchmark.run_benchmark(gs, t, folds[:1], benchmark.BenchmarkConfig(models=["M2"], budget=1), out_dir=tmp_path)
        text = open(res["summary"]["results_txt"]).read()
        assert "WINNER" in text and "Glossary" in text

    def test_physics_cloud_source_without_truth(self, setup):
        from dataclasses import replace
        gs, t, folds = setup
        no_truth = [replace(g, truth=None) for g in gs]
        res = benchmark.run_benchmark(no_truth, t, folds[:1], benchmark.BenchmarkConfig(models=["M2"], budget=1))
        m2 = res["summary"]["models"]["M2"]
        assert "pixel" not in m2 and m2["physics"]["distance_source"] == "MYD35 strict"
        no_mask = [replace(g, truth=None, images={k: v for k, v in g.images.items() if k != "ImCloud"}) for g in gs]
        res = benchmark.run_benchmark(no_mask, t, folds[:1], benchmark.BenchmarkConfig(models=["M2"], budget=1))
        assert res["summary"]["models"]["M2"]["physics"]["distance_source"] == "0.66 um brightness proxy"


class TestDiskCache:
    def test_same_results_with_cache_and_lazy_images(self, setup, tmp_path):
        from flux1km import io
        gs, t, folds = setup
        for g in gs:
            io.save_granule(tmp_path / "data", g)
        lazy = io.load_dataset(tmp_path / "data", parts=("moments", "scalars", "images", "truth"), lazy_images=True)
        models = ["M2", "M7"] + (["M5"] if _has_torch() else [])
        params = {"M5": {"epochs": 2}}
        cfg = dict(models=models, budget=1, model_params=params)
        ref = benchmark.run_benchmark(gs, t, folds, benchmark.BenchmarkConfig(**cfg))
        got = benchmark.run_benchmark(lazy, t, folds, benchmark.BenchmarkConfig(**cfg, cache_dir=str(tmp_path / "cache")))
        assert [r["pred"] for r in got["records"]] == [r["pred"] for r in ref["records"]]
        same = json.dumps(got["summary"]["models"]["M2"]["physics"], sort_keys=True, default=str)   # NaN-safe compare
        assert same == json.dumps(ref["summary"]["models"]["M2"]["physics"], sort_keys=True, default=str)
        assert not list((tmp_path / "cache").iterdir())          # every fold's folder was deleted

    def test_no_cache_files_left_after_each_model(self, setup, tmp_path, monkeypatch):
        import flux1km.methods._base as base
        from flux1km import io
        from flux1km import models as mdl
        gs, t, folds = setup
        for g in gs:
            io.save_granule(tmp_path / "data", g)
        lazy = io.load_dataset(tmp_path / "data", parts=("moments", "scalars", "images"), lazy_images=True)
        monkeypatch.setattr(mdl, "PREDICT_CHUNK", 4)            # so the folds are "large" and use disk inputs
        monkeypatch.setattr(base, "PREDICT_CHUNK", 4)
        seen = []
        orig = benchmark._sweep_cache
        def spy(store):
            seen.append(sorted(p.name.split("_")[0] for p in Path(store).glob("batch_*/*_*.npy")))
            return orig(store)
        monkeypatch.setattr(benchmark, "_sweep_cache", spy)
        models = ["M4"] + (["M5"] if _has_torch() else [])
        cfg = benchmark.BenchmarkConfig(models=models, budget=2, cache_dir=str(tmp_path / "cache"),
                                        model_params={"M5": {"epochs": 1}, "M4": {"n_rounds": 2}})
        benchmark.run_benchmark(lazy, t, folds[:1], cfg)
        assert seen and all(s == [] for s in seen)              # every fit deleted its own X_ / rows_ files

    def test_physics_subset(self):
        assert benchmark._physics_subset(3, 5, 0).tolist() == [0, 1, 2]
        k = benchmark._physics_subset(1000, 50, 1)
        assert len(k) == 50 and np.all(np.diff(k) > 0) and k.max() < 1000


def _has_torch():
    import importlib.util
    return importlib.util.find_spec("torch") is not None


class TestTuneProgress:
    def test_one_line_per_candidate(self, capsys):
        from flux1km.features import make_batch
        rng = np.random.default_rng(0)
        gs = [synthetic.make_synthetic_granule(g, rng, n_rows=3, n_cols=4) for g in ("A2014060.0245", "A2014061.0245")]
        t = build_table(gs)
        tr, va = make_batch(gs, t, np.arange(12)), make_batch(gs, t, np.arange(12, 24))
        cfg = benchmark.BenchmarkConfig(models=("M0-ridge",), progress=True)
        n = len(benchmark.get_model("M0-ridge").param_grid()[: cfg.budget])
        benchmark.tune("M0-ridge", tr, va, cfg)
        lines = [ln for ln in capsys.readouterr().out.splitlines() if "candidate" in ln]
        assert len(lines) == n and lines[-1].split("candidate ")[1].startswith(f"{n}/{n}")
        cfg.progress = False
        benchmark.tune("M0-ridge", tr, va, cfg)
        assert "candidate" not in capsys.readouterr().out


class TestCloudMask:
    def _batch(self, images):
        from flux1km.features import Batch

        n = len(images["ImSW"])
        return Batch(rows=np.arange(n), label=np.zeros(n), scalars={}, moments=None, images=images, truth=None,
                     dist_km=None, group=np.zeros(n), weight=np.ones(n))

    def test_myd35_strict_when_present(self):
        g = synthetic.make_synthetic_granule("A2014060.0245", np.random.default_rng(0), n_rows=1, n_cols=2)
        m = benchmark.cloud_mask(self._batch(g.images))
        np.testing.assert_array_equal(m, g.images["ImCloud"] <= 1)

    def test_proxy_without_imcloud(self):
        g = synthetic.make_synthetic_granule("A2014060.0245", np.random.default_rng(0), n_rows=1, n_cols=2)
        images = {k: v for k, v in g.images.items() if k != "ImCloud"}
        m = benchmark.cloud_mask(self._batch(images))
        assert m.shape == (2, 27, 55) and m.dtype == bool
