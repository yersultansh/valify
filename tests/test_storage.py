"""Storage loading rules: size-aware precedence, live merge, model persistence."""
import pandas as pd
import pytest

from backend import config, parser, storage
from tests.test_metrics import synthetic_frame
from tests.test_service import frame

ROOT = config.BASE_DIR


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(storage, "PROCESSED_CSV", tmp_path / "cache" / "dataset.csv")
    monkeypatch.setattr(storage, "PROCESSED_META", tmp_path / "cache" / "dataset.meta.json")
    monkeypatch.setattr(config, "FALLBACK_CSV", tmp_path / "fallback.csv")
    return tmp_path


def _write_fallback(env, n=150):
    storage.save_listings(frame(n).to_dict("records"), env / "fallback.csv")


def _listing_html():
    for base in (ROOT / "tests" / "fixtures", ROOT / "data" / "raw"):
        if (base / "listing.html").exists():
            return (base / "listing.html").read_text(encoding="utf-8")
    pytest.skip("listing.html fixture missing")


def _cache_pages(n_pages):
    base = _listing_html()
    ids = [c["listing_id"] for c in parser.parse_listing_cards(base)]
    for page in range(1, n_pages + 1):
        html = base
        for i in ids:
            html = html.replace(i, f"{i}{page}")
        url = config.SOURCE_URL if page == 1 else f"{config.SOURCE_URL}?page={page}"
        storage.write_raw_cache(url, html, 200)


def test_small_raw_cache_does_not_replace_the_collected_dataset(env):
    _write_fallback(env)
    _cache_pages(2)                                              # ~38 rows only
    res = storage.load_dataset(use_live=False)
    assert res.data_source == "offline_fallback" and res.rows == 150


def test_large_raw_cache_does_not_replace_stored_data(env):
    _write_fallback(env)
    _cache_pages(7)                                              # > 100 resale rows
    res = storage.load_dataset(use_live=False)
    assert res.data_source == "offline_fallback" and res.rows == 150


def test_small_raw_cache_is_last_resort_without_fallback(env):
    _cache_pages(2)
    res = storage.load_dataset(use_live=False)
    assert res.data_source == "raw_cache" and 0 < res.rows < storage.MIN_USABLE_ROWS
    assert any("Мало данных" in w for w in res.warnings)


def test_processed_cache_rules(env):
    _write_fallback(env)
    storage._save_processed(storage.rows_to_frame(frame(20).to_dict("records")), "live")      # too small
    assert storage.load_dataset(use_live=False).data_source == "offline_fallback"
    storage._save_processed(storage.rows_to_frame(frame(130).to_dict("records")), "live")
    res = storage.load_dataset(use_live=False)
    assert res.data_source == "live" and res.rows == 130


def test_live_rows_are_merged_into_collected_data(env, monkeypatch):
    _write_fallback(env, 150)
    fresh = frame(30, seed=5, start=100)                          # ids 100..129: 50 overlap the stored 0..149
    fresh["listing_id"] = [str(500_000 + 140 + i) for i in range(30)]   # 10 overlap (140-149), 20 new
    monkeypatch.setattr(parser, "collect", lambda stats=None, **kw: (stats.update(blocked=False),
                                                                  fresh.to_dict("records"))[1])
    res = storage.load_dataset(use_live=True)
    assert res.data_source == "live" and res.rows == 170
    assert res.df["listing_id"].is_unique


def test_blocked_live_run_falls_back_with_warning(env, monkeypatch):
    _write_fallback(env, 150)
    monkeypatch.setattr(parser, "collect", lambda stats=None, **kw: (stats.update(blocked=True), [])[1])
    res = storage.load_dataset(use_live=True)
    assert res.data_source == "offline_fallback" and res.rows == 150
    assert any("заблокирован" in w for w in res.warnings)


def test_no_data_at_all_raises(env):
    with pytest.raises(FileNotFoundError):
        storage.load_dataset(use_live=False)


def test_model_roundtrip_and_corrupt_file(tmp_path):
    path = tmp_path / "m.joblib"
    assert storage.load_model(path) is None
    storage.save_model({"fingerprint": "x", "model": [1, 2, 3]}, path)
    assert storage.load_model(path)["fingerprint"] == "x"
    path.write_bytes(b"garbage")
    assert storage.load_model(path) is None


def test_fallback_change_invalidates_fresh_processed_cache(env):
    _write_fallback(env, 150)
    storage._save_processed(storage._read_stored(), "offline_fallback")
    _write_fallback(env, 180)
    result = storage.load_dataset(use_live=False)
    assert result.rows == 180


def test_live_csv_change_invalidates_fresh_processed_cache(env):
    storage.save_listings(frame(150).to_dict("records"), storage._live_path())
    storage._save_processed(storage._read_stored(), "live")
    storage.save_listings(frame(190).to_dict("records"), storage._live_path())
    assert storage.load_dataset(use_live=False).rows == 190


def test_legacy_raw_processed_cache_is_ignored(env):
    import json
    _write_fallback(env, 150)
    storage._save_processed(storage.rows_to_frame(frame(120).to_dict("records")), "raw_cache")
    meta = json.loads(storage.PROCESSED_META.read_text())
    meta.pop("stored_signature")
    storage.PROCESSED_META.write_text(json.dumps(meta))
    assert storage.load_dataset(use_live=False).rows == 150


def test_empty_fallback_allows_raw_recovery(env):
    storage.save_listings([], config.FALLBACK_CSV)
    _cache_pages(7)
    result = storage.load_dataset(use_live=False)
    assert result.data_source == "raw_cache"
    assert result.rows >= storage.MIN_USABLE_ROWS


def test_large_raw_cache_does_not_replace_live_store(env):
    storage.save_listings(frame(200).to_dict("records"), storage._live_path())
    _cache_pages(7)
    assert storage.load_dataset(use_live=False).rows == 200
