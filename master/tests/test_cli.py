import pytest

from flux1km import cli

DEMO = ["--demo", "--demo-size", "2x5", "--min-group-size", "1"]
FAST = ["B-clim", "B-const", "M0-ridge", "B-bagpixel", "M2", "M8", "M9"]


def _args(tmp_path, *extra):
    return [*DEMO, "--data-root", str(tmp_path / "data"), "--out", str(tmp_path / "out"),
            "--manifest", str(tmp_path / "m.json"), *extra]


class TestCli:
    def test_demo_run_end_to_end(self, tmp_path, capsys):
        res = cli.main(_args(tmp_path, "--budget", "2", "--models", *FAST))
        assert "winner:" in capsys.readouterr().out and (tmp_path / "m.json").exists()
        assert res["summary"]["decision"]["winner"] is not None

    def test_manifest_is_reused(self, tmp_path):
        args = _args(tmp_path, "--budget", "1", "--models", "B-clim")
        a, b = cli.main(args), cli.main(args)
        assert [f["name"] for f in a["summary"]["folds"]] == [f["name"] for f in b["summary"]["folds"]]

    def test_requires_data_or_demo(self):
        with pytest.raises(SystemExit):
            cli.parse_args([])

    def test_unknown_split(self):
        with pytest.raises(ValueError):
            cli.make_folds(None, "nope", 3, 0, 100, 40)

    def test_month_and_year_splits(self):
        import numpy as np

        from flux1km import splits, synthetic
        from flux1km.table import build_table
        rng = np.random.default_rng(0)
        ids = ["A2014060.0245", "A2014152.0245", "A2015060.0245", "A2015152.0245"]  # Mar/Jun x 2014/2015
        t = build_table([synthetic.make_synthetic_granule(g, rng, n_rows=1, n_cols=2) for g in ids])
        assert sorted(set(t["month"])) == ["2014-03", "2014-06", "2015-03", "2015-06"]
        for f in cli.make_folds(t, "month", 5, 0, 100, 40)[:-1]:
            assert not set(t["month"][f.train]) & set(t["month"][f.test])
        year = cli.make_folds(t, "year", 5, 0, 100, 40)[0]
        assert set(t["year"][year.test]) == {2015} and set(t["year"][year.train]) == {2014}
        assert isinstance(year, splits.Fold)


class TestCliOptions:
    def test_hard_demo_seeds_and_results_txt(self, tmp_path, capsys):
        res = cli.main(_args(tmp_path, "--budget", "1", "--models", "M2", "--demo-hard", "--seeds", "0", "1"))
        assert "full results (text):" in capsys.readouterr().out
        assert {r["seed"] for r in res["records"]} == {0, 1}

    def test_unknown_model_exits(self, tmp_path):
        with pytest.raises(SystemExit):
            cli.main(_args(tmp_path, "--models", "M99"))

    def test_torch_models_dropped_without_torch(self, monkeypatch):
        import importlib.util
        monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
        assert cli.available_models(["M2", "M3", "M5"]) == ["M2"]

    def test_default_model_list_is_every_registered_method(self):
        from flux1km.models import MODELS
        assert cli.parse_args(["--demo"]).models == sorted(MODELS)
