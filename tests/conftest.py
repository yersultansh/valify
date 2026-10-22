"""Shared pytest configuration: the tests never touch your real data/cache, data/raw or .env settings."""
import pytest

from backend import config, storage


@pytest.fixture(scope="session", autouse=True)
def isolated_cache(tmp_path_factory):
    """Point every cache path at a temporary folder for the whole run.

    Files created by real use of the app (live_listings.csv, refresh_state.json, the saved model, the
    processed dataset) and settings such as USE_LIVE_PARSER=true in .env must not change test results. Individual
    tests may still override these paths with their own temporary folders.
    """
    mp = pytest.MonkeyPatch()
    cache = tmp_path_factory.mktemp("cache")
    mp.setattr(config, "CACHE_DIR", cache)
    mp.setattr(config, "RAW_DIR", tmp_path_factory.mktemp("raw"))
    mp.setattr(config, "FALLBACK_CSV", cache / "no_fallback.csv")
    mp.setattr(config, "USE_LIVE_PARSER", False)
    mp.setattr(storage, "PROCESSED_CSV", cache / "dataset.csv")
    mp.setattr(storage, "PROCESSED_META", cache / "dataset.meta.json")
    mp.setattr(storage, "MODEL_PATH", cache / "model.joblib")
    yield
    mp.undo()
