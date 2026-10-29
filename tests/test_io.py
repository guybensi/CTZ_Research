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


class TestFindGranules:
    def test_requires_all_parts(self, tmp_path, gran):
        io.save_granule(tmp_path, gran)
        assert io.find_granules(tmp_path, require=("moments", "scalars", "images")) == ["A2014060.0245"]
        (tmp_path / "images" / "A2014060.0245.npz").unlink()
        assert io.find_granules(tmp_path, require=("images",)) == []
