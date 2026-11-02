import numpy as np

from flux1km import synthetic, synthetic_study


class TestCtzSignal:
    def test_values(self):
        assert synthetic_study.ctz_signal([10.0, 8.0, None, 2.0, 0.0]) == 8.0
        assert np.isnan(synthetic_study.ctz_signal([1.0, np.nan]))


class TestTruthCurve:
    def test_ctz_and_no_ctz(self):
        rng = np.random.default_rng(0)
        with_ctz = [synthetic.make_synthetic_granule("A2014060.0245", rng, n_rows=3, n_cols=8, hard=True)]
        rng = np.random.default_rng(0)
        no_ctz = [synthetic.make_synthetic_granule("A2014060.0245", rng, n_rows=3, n_cols=8, hard=True, ctz=False)]
        assert synthetic_study.ctz_signal(synthetic_study.truth_curve(with_ctz)) > synthetic_study.ctz_signal(
            synthetic_study.truth_curve(no_ctz))


class TestStudy:
    def test_cases_and_report(self, tmp_path):
        res = synthetic_study.run_synthetic_truth_study(["M2"], lattices=((2, 5),), scales=(0.5,), hard=True, budget=1,
                                                        out_dir=tmp_path)
        assert [(c["scale"], c["ctz"]) for c in res["cases"]] == [(0.5, True), (1.0, False)]
        case = res["cases"][0]
        assert {"M2", "B-const"} <= set(case["models"]) and np.isfinite(case["models"]["M2"]["pixel"]["rmse"])
        assert res["cases"][1]["models"]["M2"]["flat_ok"] in (True, False)
        text = open(res["results_txt"]).read()
        assert "NO CTZ (control)" in text and "beat B-const" in text
