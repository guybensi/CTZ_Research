"""Tests for the Windows versions in scripts/windows/ (they run on Linux too; the Windows-only paths are forced)."""
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

nc = pytest.importorskip("netCDF4")
pytest.importorskip("scipy")

WIN = Path(__file__).resolve().parents[1] / "scripts" / "windows"
MASTER = WIN.parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_build_dataset as tb  # noqa: E402  (synthetic MODIS/CERES month: make_month, psf27)


def load_windows(name):
    """Load scripts/windows/<name>.py with its sibling imports resolved to the Windows versions too."""
    saved = {k: sys.modules.get(k) for k in ("download_modis", "organize_ceres")}
    try:
        for sib in ("download_modis", "organize_ceres"):
            spec = importlib.util.spec_from_file_location(sib, WIN / f"{sib}.py")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            sys.modules[sib] = mod
        spec = importlib.util.spec_from_file_location(f"win_{name}", WIN / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


wdm = load_windows("download_modis")
wbd = load_windows("build_dataset")
wdc = load_windows("download_ceres")


def run_cli(script, *args, env_extra=None):
    env = {**os.environ, **(env_extra or {})}
    return subprocess.run([sys.executable, str(WIN / script), *args], cwd=MASTER, env=env,
                          capture_output=True, text=True, timeout=300)


class TestLockFolder:
    def test_second_lock_fails_until_released(self, tmp_path):
        first = wdm.lock_folder(tmp_path, ".x.lock")
        assert first is not None and wdm.lock_folder(tmp_path, ".x.lock") is None
        first.close()
        again = wdm.lock_folder(tmp_path, ".x.lock")
        assert again is not None
        again.close()


class TestSpread:
    def test_examples(self):
        assert wdm.spread(list(range(10)), 3) == [0, 4, 9]
        assert wdm.spread([1, 2], 5) == [1, 2] and wdm.spread([1, 2, 3], 0) == []


class FakeGranule:
    def __init__(self, name):
        self.name, self.size = name, 1.0

    def data_links(self):
        return [f"https://example.invalid/{self.name}"]


class FakeEarthaccess:
    """6 daytime granules per product on 2008-01-15."""

    def __init__(self):
        self.downloaded = []

    def login(self, **kw):
        return True

    def search_data(self, short_name, **kw):
        return [FakeGranule(f"{short_name}.A2008015.{h:02d}00.061.2008016000000.hdf") for h in range(0, 24, 4)]

    def download(self, granules, local_path, threads=4):
        for g in granules:
            (Path(local_path) / g.name).write_bytes(b"x" * (1024 * 1024))
            self.downloaded.append(g.name)


class TestMaxGranules:
    def test_only_n_complete_sets_spread_over_the_day(self, tmp_path, monkeypatch):
        fake = FakeEarthaccess()
        monkeypatch.setitem(sys.modules, "earthaccess", fake)
        assert wdm.main(["--start", "2008-01-15", "--end", "2008-01-15", "--out", str(tmp_path), "--max-granules", "2"]) == 0
        names = sorted(p.name for p in (tmp_path / "AQUA/2008/Pacific/jan/MOD02").iterdir())
        assert [n.split(".")[2] for n in names] == ["0000", "2000"] and len(fake.downloaded) == 4

    def test_lock_refuses_a_second_copy(self, tmp_path, monkeypatch):
        fake = FakeEarthaccess()
        monkeypatch.setitem(sys.modules, "earthaccess", fake)
        held = wdm.lock_folder(tmp_path, ".download.lock")
        rc = wdm.main(["--start", "2008-01-15", "--end", "2008-01-15", "--out", str(tmp_path), "--max-granules", "1"])
        held.close()
        assert rc == 1 and fake.downloaded == []


class TestWiring:
    def test_siblings_are_the_windows_versions(self):
        assert hasattr(wbd.dm, "lock_folder") and hasattr(wdc.dm, "lock_folder")
        assert Path(wdc.oc.__file__).parent == WIN

    def test_defaults(self):
        assert wbd.DEFAULT_PSF.is_file()
        assert wdm.OUT_DIR == str(Path.home() / "GUY_DATA") and wdc.OUT_DIR == wdm.OUT_DIR

    def test_linux_versions_are_untouched(self):
        linux = (WIN.parent / "download_modis.py").read_text()
        assert "import fcntl" in linux and "lock_folder" not in linux and "--max-granules" not in linux
        assert 'mp.get_context("fork")' in (WIN.parent / "build_dataset.py").read_text()


class TestGather:
    def built(self, tmp_path):
        data, out = tb.make_month(tmp_path)
        c = wbd.build_month(data, out, 2008, "Pacific", 1, tb.psf27(), start="2008-01-01", end="2008-12-31", workers=1,
                            gen_images=False, limit=None, dry_run=False, log=lambda s: None)
        assert c["ok"] == 2
        return out

    def test_hard_links_without_symlink_rights(self, tmp_path, monkeypatch):
        out = self.built(tmp_path)
        monkeypatch.setattr(Path, "symlink_to", lambda *a, **k: (_ for _ in ()).throw(OSError("no rights")))
        assert wbd.gather(out) == {"Pacific": 4} and wbd.gather(out) == {"Pacific": 4}
        link = out / "Pacific/scalars/A2008001.0300.npz"
        assert not link.is_symlink() and link.samefile(out / "Pacific/2008/jan/scalars/A2008001.0300.npz")
        assert wbd.verify_triplets(out / "Pacific") == []

    def test_copies_without_any_links(self, tmp_path, monkeypatch):
        out = self.built(tmp_path)

        def refuse(*a, **k):
            raise OSError("not supported")

        monkeypatch.setattr(Path, "symlink_to", refuse)
        monkeypatch.setattr(wbd.os, "link", refuse)
        assert wbd.gather(out) == {"Pacific": 4} and wbd.verify_triplets(out / "Pacific") == []


class TestSpawnEndToEnd:
    def test_command_line_with_spawn_workers(self, tmp_path):
        """The real Windows path: the script run as a program, 2 workers started with spawn."""
        data, out = tb.make_month(tmp_path)
        r = run_cli("build_dataset.py", "--data", str(data), "--out", str(out), "--workers", "2",
                    env_extra={"CTZ_START_METHOD": "spawn"})
        assert r.returncode == 0, r.stdout + r.stderr
        assert sorted(p.name for p in (out / "Pacific/scalars").iterdir()) == ["A2008001.0300.npz", "A2008002.0300.npz"]
        with np.load(out / "Pacific/images/A2008002.0300.npz") as im:
            assert im["ImSW"].shape[1:] == (11, 27, 55)
        r = run_cli("build_dataset.py", "--out", str(out), "--verify")
        assert r.returncode == 0 and "all triplets OK" in r.stdout

    def test_diagnose_and_limit_from_the_command_line(self, tmp_path):
        data, out = tb.make_month(tmp_path)
        r = run_cli("build_dataset.py", "--data", str(data), "--out", str(out), "--diagnose", "1")
        assert r.returncode == 0 and "accepted=" in r.stdout


class TestEarthdataLogin:
    class Fake:
        def __init__(self, netrc_result):
            self.netrc_result, self.calls = netrc_result, []

        def login(self, strategy, persist=False):
            self.calls.append((strategy, persist))
            if strategy == "netrc" and isinstance(self.netrc_result, Exception):
                raise self.netrc_result
            return self.netrc_result if strategy == "netrc" else True

    def test_missing_netrc_raises_then_asks_and_saves(self):
        f = self.Fake(FileNotFoundError("No .netrc found at C:\\Users\\x\\_netrc"))
        wdm.earthdata_login(f)
        assert f.calls == [("netrc", False), ("interactive", True)]

    def test_netrc_returning_false_asks(self):
        f = self.Fake(False)
        wdm.earthdata_login(f)
        assert f.calls[-1] == ("interactive", True)

    def test_working_netrc_does_not_ask(self):
        f = self.Fake(True)
        wdm.earthdata_login(f)
        assert f.calls == [("netrc", False)]
