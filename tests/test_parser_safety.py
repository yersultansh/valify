"""Parser block handling and cache isolation; no spatial imputation remains."""
import pytest
import requests
from pathlib import Path
from backend import config, parser, storage

class Resp:
    def __init__(self, status, text="", url="https://krisha.kz/a/show/x"):
        self.status_code, self.text, self.url, self.encoding = status, text, url, "utf-8"

    def raise_for_status(self):
        if self.status_code >= 400:
            err = requests.HTTPError(str(self.status_code))
            err.response = self
            raise err


@pytest.fixture
def http(monkeypatch):
    monkeypatch.setattr(config, "MIN_DELAY", 0.0)
    monkeypatch.setattr(config, "MAX_DELAY", 0.0)
    monkeypatch.setattr(parser, "_last_request_ts", None)
    monkeypatch.setattr(parser.time, "sleep", lambda s: None)
    calls = []

    def reply(status):
        monkeypatch.setattr(parser.requests.Session, "get", lambda self, url, **kw: calls.append(url) or Resp(status, "x", url))
        return calls

    return reply


@pytest.mark.parametrize("status", [468, 403, 429, 401, 418])
def test_blocks_stop_at_once_without_retries(http, status):
    calls = http(status)
    with pytest.raises(parser.BlockedError):
        parser.fetch("https://krisha.kz/a/show/1", use_cache=False)
    assert len(calls) == 1


@pytest.mark.parametrize("status", [404, 410])
def test_removed_listing_is_not_a_block(http, status):
    calls = http(status)
    with pytest.raises(requests.HTTPError):
        parser.fetch("https://krisha.kz/a/show/1", use_cache=False)
    assert len(calls) == 1


def test_server_errors_are_retried_but_not_treated_as_blocks(http):
    calls = http(503)
    with pytest.raises(RuntimeError, match="giving up"):
        parser.fetch("https://krisha.kz/a/show/1", use_cache=False)
    assert len(calls) == 3


def test_collect_stops_on_468(http, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RAW_DIR", tmp_path / "raw")
    calls = http(468)
    stats = {}
    assert parser.collect(max_pages=3, stats=stats) == [] and stats["blocked"] is True and len(calls) == 1



def test_use_cache_false_neither_reads_nor_writes_the_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(config, "MIN_DELAY", 0.0)
    monkeypatch.setattr(config, "MAX_DELAY", 0.0)
    monkeypatch.setattr(parser, "_last_request_ts", None)
    calls = []

    class R:
        status_code, text, url, encoding = 200, "<html>x</html>", "https://krisha.kz/a/show/9", "utf-8"

        def raise_for_status(self):
            pass

    monkeypatch.setattr(parser.requests.Session, "get", lambda self, url, **kw: calls.append(url) or R())
    parser.fetch("https://krisha.kz/a/show/9", use_cache=False)
    parser.fetch("https://krisha.kz/a/show/9", use_cache=False)
    assert len(calls) == 2 and not (tmp_path / "raw").exists()
    parser.fetch("https://krisha.kz/a/show/9")                                  # default: written to the cache
    parser.fetch("https://krisha.kz/a/show/9")
    assert len(calls) == 3 and any((tmp_path / "raw").glob("*.html"))



def test_tests_are_isolated_from_the_real_cache_and_env():
    assert config.CACHE_DIR != Path(__file__).resolve().parent.parent / "data" / "cache" and config.RAW_DIR != Path(__file__).resolve().parent.parent / "data" / "raw"
    assert config.USE_LIVE_PARSER is False and not config.FALLBACK_CSV.exists()
    assert storage.MODEL_PATH.parent == config.CACHE_DIR
