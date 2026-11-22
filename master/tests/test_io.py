import numpy as np
import pytest

from flux1km import io, synthetic


@pytest.fixture
def gran():
    return synthetic.make_synthetic_granule("A2014060.0245", np.random.default_rng(0), n_rows=3, n_cols=4)


class TestParseGranuleId:
    def test_valid(self):
        g = io.parse_granule_id("A2014090.2355")
        assert (g.satellite, g.time.hour, g.time.minute, g.day) == ("Aqua", 23, 55, "A2014-03-31")
        assert io.parse_granule_id("T2015001.0000.npz").satellite == "Terra"

    @pytest.mark.parametrize("bad", ["X2014060.0245", "A2014400.0245", "A2014060.2460", "A201406.0245", ""])
    def test_invalid(self, bad):
        with pytest.raises(ValueError):
            io.parse_granule_id(bad)


class TestGranule:
    def test_validate_row_mismatch(self, gran):
        gran.images["ImSW"] = gran.images["ImSW"][:-1]
        with pytest.raises(ValueError):
            gran.validate()


class TestSaveLoad:
    def test_roundtrip(self, tmp_path, gran):
        io.save_granule(tmp_path, gran)
        g2 = io.load_granule(tmp_path, gran.granule_id, parts=("moments", "scalars", "images", "truth"))
        np.testing.assert_array_equal(g2.images["ImSW"], gran.images["ImSW"])
        np.testing.assert_allclose(g2.scalars["Fsw"], gran.scalars["Fsw"])
        assert g2.truth is not None and g2.n_footprints == 12

    def test_missing_optional_and_required(self, tmp_path, gran):
        gran.images = None
        io.save_granule(tmp_path, gran)
        assert io.load_granule(tmp_path, gran.granule_id).images is None
        with pytest.raises(FileNotFoundError):
            io.load_granule(tmp_path, "A2014060.0000")

    def test_unknown_part(self, tmp_path, gran):
        io.save_granule(tmp_path, gran)
        with pytest.raises(ValueError):
            io.load_granule(tmp_path, gran.granule_id, parts=("bogus",))


class TestLazyNpz:
    @pytest.mark.parametrize("save", [np.savez, np.savez_compressed])
    def test_shapes_without_reading_and_values(self, tmp_path, save):
        a, b = np.arange(12.0).reshape(3, 4), np.ones((3,), np.float32)
        save(tmp_path / "x.npz", a=a, b=b)
        z = io.LazyNpz(tmp_path / "x.npz")
        assert z.shapes == {"a": (3, 4), "b": (3,)} and set(z) == {"a", "b"} and len(z) == 2
        np.testing.assert_array_equal(z["a"], a)
        assert z["b"].dtype == np.float32

    def test_nothing_is_kept(self, tmp_path):
        np.savez(tmp_path / "x.npz", a=np.zeros(3))
        z = io.LazyNpz(tmp_path / "x.npz")
        assert z["a"] is not z["a"]

    def test_missing_key_and_file(self, tmp_path):
        np.savez(tmp_path / "x.npz", a=np.zeros(3))
        with pytest.raises(KeyError):
            io.LazyNpz(tmp_path / "x.npz")["b"]
        with pytest.raises(FileNotFoundError):
            io.LazyNpz(tmp_path / "missing.npz")

    def test_lazy_granule_equals_eager(self, tmp_path, gran):
        io.save_granule(tmp_path, gran)
        lazy = io.load_granule(tmp_path, gran.granule_id, lazy_images=True)
        eager = io.load_granule(tmp_path, gran.granule_id)
        assert isinstance(lazy.images, io.LazyNpz) and set(lazy.images) == set(eager.images)
        for k in eager.images:
            np.testing.assert_array_equal(lazy.images[k], eager.images[k])

    def test_lazy_validate_catches_row_mismatch(self, tmp_path, gran):
        io.save_granule(tmp_path, gran)
        imgs = dict(np.load(tmp_path / "images" / f"{gran.granule_id}.npz"))
        imgs["ImSW"] = imgs["ImSW"][:-1]
        np.savez(tmp_path / "images" / f"{gran.granule_id}.npz", **imgs)
        with pytest.raises(ValueError):
            io.load_granule(tmp_path, gran.granule_id, lazy_images=True)


class TestDiskArray:
    def test_creates_memmap_and_discard_deletes_it(self, tmp_path):
        a = io.disk_array(tmp_path / "a.npy", np.float32, (3, 2), min_free_gb=0)
        a[:] = 1.0
        assert isinstance(a, np.memmap) and (tmp_path / "a.npy").exists()
        io.discard(a, np.zeros(2), None)
        assert not (tmp_path / "a.npy").exists()

    def test_refuses_when_the_disk_would_get_too_full(self, tmp_path):
        import shutil
        free_gb = shutil.disk_usage(tmp_path).free / 2**30
        with pytest.raises(io.DiskFullError):
            io.disk_array(tmp_path / "b.npy", np.float32, (4,), min_free_gb=free_gb + 1)
        assert not (tmp_path / "b.npy").exists()


class TestFindGranules:
    def test_requires_all_parts(self, tmp_path, gran):
        io.save_granule(tmp_path, gran)
        assert io.find_granules(tmp_path, require=("moments", "scalars", "images")) == ["A2014060.0245"]
        (tmp_path / "images" / "A2014060.0245.npz").unlink()
        assert io.find_granules(tmp_path, require=("images",)) == []
