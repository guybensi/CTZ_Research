import pytest

from flux1km import cli

DEMO = ["--demo", "--demo-size", "2x5", "--min-group-size", "1"]


def _args(tmp_path, *extra):
    return [*DEMO, "--data-root", str(tmp_path / "data"), "--out", str(tmp_path / "out"),
            "--manifest", str(tmp_path / "m.json"), *extra]


class TestCli:
    def test_demo_run_end_to_end(self, tmp_path, capsys):
        res = cli.main(_args(tmp_path, "--budget", "2"))
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
