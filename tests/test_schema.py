import pytest

from flux1km import schema


class TestBandTables:
    def test_lengths(self):
        assert len(schema.SW_LABELS) == len(schema.SW_WAVELENGTHS) == 11
        assert len(schema.LW_LABELS) == len(schema.LW_WAVELENGTHS) == 16


class TestChannelIndex:
    def test_known(self):
        assert schema.channel_index("0.66") == ("SW", 2)
        assert schema.channel_index("11.0") == ("LW", 10)
        assert schema.channel_index("1.375") == ("SW", 10)

    def test_unknown(self):
        with pytest.raises(KeyError):
            schema.channel_index("0.123")


class TestAssertNoCeresInputs:
    def test_modis_only_passes(self):
        schema.assert_no_ceres_inputs(["SZA", "mSWspec_0.66", "glint"])

    @pytest.mark.parametrize("bad", [["Isw"], ["SZA", "Fsw"], ["mSumIsw"], ["stdSumIlw"], ["scalars/Isw"]])
    def test_ceres_keys_rejected(self, bad):
        with pytest.raises(schema.LeakageError):
            schema.assert_no_ceres_inputs(bad)


class TestInputSet:
    def test_nested_and_without_bad_channels(self):
        prev = set()
        for name in ("F1", "F2", "F3", "F4", "F5"):
            s = schema.input_set(name)
            cur = set(s.sw) | set(s.lw)
            assert prev <= cur
            assert not cur & set(schema.BAD_CHANNELS)
            prev = cur

    def test_glint_only_in_f5_and_f1_has_no_lw(self):
        assert not schema.input_set("F4").use_glint and schema.input_set("F5").use_glint
        assert schema.input_set("F1").lw == ()

    def test_unknown(self):
        with pytest.raises(KeyError):
            schema.input_set("F9")
