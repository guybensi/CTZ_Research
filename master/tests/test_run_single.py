from flux1km import run_single


class TestSingleArgs:
    def test_adds_baselines_once(self):
        assert run_single.single_args(["M3", "B-clim"], ["--demo"]) == ["--demo", "--models", "M3", "B-clim", "B-const", "M0-ridge"]

    def test_user_models_and_no_baselines(self):
        assert run_single.single_args(["M3"], ["--models", "M2"]) == ["--models", "M2"]
        assert run_single.single_args(["M3"], ["--no-baselines", "--demo"]) == ["--demo", "--models", "M3"]


class TestMainFor:
    def test_runs_a_method_module_end_to_end(self, tmp_path):
        res = run_single.main_for(["M9"], ["--demo", "--demo-size", "2x4", "--min-group-size", "1", "--budget", "1",
                                           "--data-root", str(tmp_path / "d"), "--out", str(tmp_path / "o")])
        assert set(res["summary"]["models"]) == {"M9", "B-clim", "B-const", "M0-ridge"}
        assert any((tmp_path / "o").glob("run_*/results.txt"))
