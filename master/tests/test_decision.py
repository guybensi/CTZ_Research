import numpy as np

from flux1km import decision
from flux1km.benchmark import BenchmarkConfig

CFG = BenchmarkConfig(models=[], sensitivity_operators=("uniform",))


def _m(macro, complexity=0, **kw):
    return {"overall": {"rmse": macro}, "macro_rmse": macro, "complexity": complexity, "trend_effect": {},
            "uses_test_labels": False, "produces_map": True, **kw}


def _boot(lo, hi, p):
    return {"ci_low": lo, "ci_high": hi, "p_holm_boot": p, "p_holm": p}


class TestRegimeNoiseFloor:
    def test_mean_over_scored_regimes(self):
        res = {"by_regime": {"clear": {"n": 3}, "overcast": {"n": 2}, "broken": {"n": 0}}}
        assert decision.regime_noise_floor(res, CFG) == np.mean([1.9, 6.0])


class TestRankingFlips:
    def test_flip_beyond_tolerance_only_for_map_models(self):
        s = {"models": {"A": _m(5.0, macro_rmse_uniform=9.0), "B": _m(7.0, macro_rmse_uniform=6.0),
                        "C": _m(6.0, macro_rmse_uniform=20.0, produces_map=False)}}
        flips = decision.ranking_flips(s, ["A", "B", "C"], ("uniform",), tol=0.5)
        assert set(flips) == {"A", "B"}

    def test_small_differences_ignored(self):
        s = {"models": {"A": _m(5.0, macro_rmse_uniform=5.3), "B": _m(5.2, macro_rmse_uniform=5.1)}}
        assert decision.ranking_flips(s, ["A", "B"], ("uniform",), tol=0.5) == {}


class TestGateFailures:
    def test_each_gate(self):
        res = _m(5.0, physics={"passed": False, "ok_pcs": False, "ok_bounds": True, "ok_spacing": True},
                 trend_effect={"SZA": {"effect_wm2": 9.0, "p": 1e-4}}, swath_asymmetry=7.0)
        fails = decision.gate_failures("A", res, CFG, {"A": ["B@uniform"]})
        assert fails == ["ok_pcs", "trend:SZA", "swath_asymmetry", "ranking_flip:B@uniform"]
        assert decision.gate_failures("A", _m(5.0), CFG, {}) == []


class TestCompare:
    def test_bootstrap_ci_direction(self):
        s = {"models": {"A": _m(5.0), "B": _m(9.0)}, "comparisons": {"A vs B": _boot(-3.0, -1.0, 0.01)}}
        assert decision.compare(s, "A", "B")[:2] == (True, False)
        assert decision.compare(s, "B", "A")[:2] == (False, True)

    def test_ci_straddling_zero_is_not_significant(self):
        s = {"models": {"A": _m(5.0), "B": _m(9.0)}, "comparisons": {"A vs B": _boot(-3.0, 1.0, 0.01)}}
        assert decision.compare(s, "A", "B")[:2] == (False, False)

    def test_missing_comparison(self):
        assert decision.compare({"models": {}, "comparisons": {}}, "A", "B") == (False, False, None)


class TestDecide:
    def test_replacement_needs_significance_and_floor(self):
        s = {"models": {"A": _m(30.0, 0), "B": _m(10.0, 2), "C": _m(9.0, 3)},
             "comparisons": {"B vs A": _boot(-50, -10, 0.001), "C vs B": _boot(-5, -0.1, 0.01), "C vs A": _boot(-60, -10, 0.001)}}
        d = decision.decide(s, CFG)
        assert d["winner"] in ("B", "C") and any("B replaces A" in x for x in d["log"])
        assert any("C does not replace B" in x for x in d["log"])  # gain 1.0 < floor 2.0

    def test_test_label_methods_never_ranked(self):
        s = {"models": {"A": _m(10.0), "X": {**_m(0.0), "uses_test_labels": True}}, "comparisons": {}}
        d = decision.decide(s, CFG)
        assert d["winner"] == "A" and "X" not in d["ranking_by_macro_rmse"]

    def test_tie_default_m3_or_m2_when_scarce(self):
        s = {"models": {"M2": _m(10.0, 2), "M3": _m(9.8, 4)}, "comparisons": {}, "n_image_granules": 50}
        assert decision.decide(s, CFG)["winner"] == "M3"
        s["n_image_granules"] = 6
        assert decision.decide(s, CFG)["winner"] == "M2"

    def test_tie_broken_by_pixel_truth_first(self):
        s = {"models": {"M2": _m(10.0, 2, pixel={"rmse": 20.0}), "M9": _m(10.1, 3, pixel={"rmse": 5.0})}, "comparisons": {}}
        d = decision.decide(s, CFG)
        assert d["winner"] == "M9" and "tie-breaker 1" in d["log"][-1]

    def test_unet_must_beat_m6b(self):
        # M5 would win the tie on synthetic pixel truth, but it is not significantly better than M6b
        s = {"models": {"M6b": _m(10.0, 5, pixel={"rmse": 30.0}), "M5": _m(9.0, 6, pixel={"rmse": 10.0})},
             "comparisons": {"M5 vs M6b": _boot(-3, 1, 0.3)}}
        d = decision.decide(s, CFG)
        assert d["winner"] == "M6b" and "spatial context adds little" in d["log"][-1]

    def test_unet_wins_when_significantly_better_than_m6b(self):
        s = {"models": {"M6b": _m(30.0, 5), "M5": _m(5.0, 6)}, "comparisons": {"M5 vs M6b": _boot(-30, -10, 0.001)}}
        assert decision.decide(s, CFG)["winner"] == "M5"

    def test_no_model_passes(self):
        s = {"models": {"A": _m(5.0, swath_asymmetry=50.0)}, "comparisons": {}}
        assert decision.decide(s, CFG)["winner"] is None
