"""Shared fixtures for all tests."""
import pytest


@pytest.fixture(autouse=True)
def _no_cache_free_space_floor(monkeypatch):
    """Tests write tiny cache files; the 20 GB free-space floor of io.disk_array is for real runs."""
    from flux1km import io
    monkeypatch.setattr(io, "MIN_FREE_GB", 0.0)
