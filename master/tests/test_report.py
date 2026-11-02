from flux1km import report


class TestFormatTable:
    def test_nan_none_and_strings(self):
        out = report.format_table(["m", "x", "y", "z"], [["a", float("nan"), None, "s"]], [4, 6, 6, 6])
        line = out.splitlines()[1]
        assert "nan" in line and "-" in line and line.endswith("s")

    def test_long_headers_are_truncated(self):
        assert len(report.format_table(["model", "a_very_long_header"], [], [6, 8]).splitlines()[0]) == 14


class TestFormatReport:
    def test_all_sections(self):
        summ = {"models": {"M2": {"overall": {"rmse": 3.0, "n": 5}, "macro_rmse": 3.0, "by_regime": {"clear": {"rmse": 1.0, "n": 5}},
                                  "macro_rmse_uniform": 4.0, "trend_effect": {"SZA": {"effect_wm2": 1.0, "p": 0.5}},
                                  "physics": {"pcs_flux": 80.0, "passed": True, "flux_vs_distance": [90.0, 80.0]},
                                  "pixel": {"rmse": 7.0}}},
                "comparisons": {"M2 vs B": {"mean_diff": -1.0, "ci_low": -2.0, "ci_high": -0.5, "p_boot": 0.01}},
                "decision": {"winner": "M2", "log": ["x"], "gates": {"M2": []}},
                "folds": [{"name": "logo:a", "leaky": False, "n_train": 1, "n_val": 1, "n_test": 1, "params": {"M2": {"alpha": 1}},
                           "val_rmse": {"M2": 2.0}}],
                "errors": [{"model": "M5", "fold": "logo:a", "seed": 0, "error": "ImportError: torch"}],
                "config": {"models": ["M2"]}}
        text = report.format_report(summ)
        for key in ("WINNER", "winner: M2", "Configuration", "Folds", "Errors", "macro@uniform", "Scores by cloud regime",
                    "Residual flatness", "Physics checks", "Pairwise comparisons", "Gate results", "Glossary", "ImportError"):
            assert key in text
