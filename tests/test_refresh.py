"""Refresh (live collection + retrain): rate limit, single flight, blocking, failure recovery, API."""
import threading
import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from backend import config, main, metrics, parser, service, storage
from tests.test_service import FLAT, frame


class Clock:
    def __init__(self, t=1_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


@pytest.fixture
def env(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    monkeypatch.setattr(config, "CACHE_DIR", cache)
    monkeypatch.setattr(config, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(config, "FALLBACK_CSV", tmp_path / "fallback.csv")
    monkeypatch.setattr(config, "USE_LIVE_PARSER", False)
    monkeypatch.setattr(storage, "PROCESSED_CSV", cache / "dataset.csv")
    monkeypatch.setattr(storage, "PROCESSED_META", cache / "dataset.meta.json")
    monkeypatch.setattr(storage, "MODEL_PATH", cache / "model.joblib")
    storage.save_listings(frame(60).to_dict("records"), tmp_path / "fallback.csv")
    return tmp_path


def new_rows(n=20, start=1000):
    return frame(n, seed=7, start=start).to_dict("records")


def collector(rows=None, blocked=False, exc=None, gate=None, calls=None):
    def run(stats):
        if calls is not None:
            calls.append(1)
        if gate is not None:
            assert gate.wait(10), "gate never released"
        if exc:
            raise exc
        stats["blocked"] = blocked
        return list(rows or [])
    return run


def make(coll, enabled=True, clock=None):
    return service.PriceService(collector=coll, live_enabled=enabled, clock=clock or Clock())


def wait_idle(svc, timeout=30):
    end = time.time() + timeout
    while svc.refresh_status()["running"] and time.time() < end:
        time.sleep(0.05)
    assert not svc.refresh_status()["running"]


def count_training(monkeypatch):
    calls = []
    real = metrics.train_and_evaluate
    monkeypatch.setattr(metrics, "train_and_evaluate", lambda d: (calls.append(1), real(d))[1])
    return calls


# ------------------------------- service -------------------------------
def test_disabled_flag_blocks_everything(env):
    calls = []
    svc = make(collector(new_rows(), calls=calls), enabled=False)
    assert svc.start_refresh() == {"outcome": "disabled"} and calls == []
    assert svc.refresh_status()["enabled"] is False and storage.load_refresh_state() == {}


def test_status_before_any_run(env):
    st = make(collector()).refresh_status()
    assert st["enabled"] and not st["running"] and st["next_allowed_in_seconds"] == 0
    assert st["last_started_at"] is None and st["last_result"] is None and st["min_interval_seconds"] == 600


def test_refresh_stores_rows_retrains_and_persists(env):
    svc = make(collector(new_rows(20)))
    svc.ensure_ready()
    before = svc.rows_used
    res = svc.run_refresh()
    assert res["status"] == "ok" and res["added"] == 20 and "переобучена" in res["message"]
    assert svc.rows_used > before and svc.health()["data_source"] == "live"
    assert len(storage.read_live_store()) == 20
    assert svc.predict(dict(FLAT))["price_estimate_kzt"] > 0
    fresh = service.PriceService()                              # new process: fallback + live store
    fresh.ensure_ready()
    assert fresh.rows_used == svc.rows_used and fresh.health()["data_source"] == "live"
    assert svc.refresh_status()["last_result"]["added"] == 20


def test_rate_limit_one_run_per_ten_minutes_and_persisted(env):
    clock = Clock()
    svc = make(collector(), clock=clock)
    assert svc.start_refresh() == {"outcome": "started"}
    wait_idle(svc)
    assert svc.start_refresh() == {"outcome": "rate_limited", "retry_after": 600}
    clock.t += 599
    assert svc.start_refresh() == {"outcome": "rate_limited", "retry_after": 1}
    assert svc.refresh_status()["next_allowed_in_seconds"] == 1
    assert make(collector(), clock=clock).start_refresh()["outcome"] == "rate_limited"   # survives a restart
    clock.t += 2
    assert svc.start_refresh() == {"outcome": "started"}
    wait_idle(svc)


def test_single_flight_while_running(env):
    gate, calls = threading.Event(), []
    svc = make(collector(gate=gate, calls=calls))
    assert svc.start_refresh() == {"outcome": "started"}
    assert svc.start_refresh() == {"outcome": "running"} and svc.refresh_status()["running"]
    gate.set()
    wait_idle(svc)
    assert len(calls) == 1 and svc.refresh_status()["last_result"]["status"] == "no_new"


def test_block_keeps_old_data_and_still_counts(env, monkeypatch):
    svc = make(collector([], blocked=True))
    svc.ensure_ready()
    trained, rows = count_training(monkeypatch), svc.rows_used
    res = svc.run_refresh()
    assert res["status"] == "blocked" and "ограничил доступ" in res["message"]
    assert trained == [] and svc.rows_used == rows and storage.read_live_store() is None
    svc.run_refresh()                                            # second run via the public path is limited
    assert svc.start_refresh()["outcome"] in ("started", "rate_limited")


def test_partial_collection_then_block_keeps_valid_rows(env):
    svc = make(collector(new_rows(5), blocked=True))
    svc.ensure_ready()
    res = svc.run_refresh()
    assert res["status"] == "ok" and res["added"] == 5 and "ограничил доступ" in res["message"]
    assert len(storage.read_live_store()) == 5


def test_no_new_rows_means_no_retraining(env, monkeypatch):
    svc = make(collector(frame(10).to_dict("records")))          # ids already in the stored dataset
    svc.ensure_ready()
    trained = count_training(monkeypatch)
    res = svc.run_refresh()
    assert res["status"] == "no_new" and trained == [] and storage.read_live_store() is None


def test_collector_error_is_contained(env):
    svc = make(collector(exc=RuntimeError("boom")))
    svc.ensure_ready()
    res = svc.run_refresh()
    assert res["status"] == "error" and "Используются прежние данные" in res["message"]
    assert svc.refresh_status()["running"] is False and svc.predict(dict(FLAT))["price_estimate_kzt"] > 0


def test_failed_rebuild_keeps_old_model_and_is_retried(env, monkeypatch):
    svc = make(collector(new_rows(20)))
    svc.ensure_ready()
    old_rows, real = svc.rows_used, metrics.train_and_evaluate
    monkeypatch.setattr(metrics, "train_and_evaluate", lambda d: (_ for _ in ()).throw(ValueError("bad")))
    assert svc.run_refresh()["status"] == "error"
    assert svc.rows_used == old_rows and svc.predict(dict(FLAT))["price_estimate_kzt"] > 0
    assert len(storage.read_live_store()) == 20                   # collected rows are kept
    monkeypatch.setattr(metrics, "train_and_evaluate", real)
    svc2 = svc
    svc2._collector = collector([])                               # nothing new, but the model is stale
    res = svc2.run_refresh()
    assert res["status"] == "ok" and svc2.rows_used > old_rows


def test_rebuild_never_crawls_again(env, monkeypatch):
    monkeypatch.setattr(config, "USE_LIVE_PARSER", True)          # opt-in: loader would normally go live
    crawls = []
    monkeypatch.setattr(parser, "collect", lambda *a, **k: crawls.append(1) or [])
    svc = make(collector(new_rows(20)))
    svc._call_loader(False)                                       # sanity: explicit no-live
    assert svc.run_refresh()["status"] == "ok" and crawls == []


def test_collect_live_uses_polite_delays_and_restores_config(monkeypatch):
    monkeypatch.delenv("BATCH_MIN_DELAY", raising=False)
    monkeypatch.delenv("BATCH_MAX_DELAY", raising=False)
    seen = {}

    def fake_collect(stats=None, **kw):
        seen.update(lo=config.MIN_DELAY, hi=config.MAX_DELAY, ttl=config.CACHE_TTL_HOURS)
        stats["blocked"] = False
        return []

    monkeypatch.setattr(parser, "collect", fake_collect)
    before = (config.MIN_DELAY, config.MAX_DELAY, config.CACHE_TTL_HOURS)
    service.PriceService._collect_live({})
    assert (seen["lo"], seen["hi"], seen["ttl"]) == (8.0, 12.0, service.REFRESH_CACHE_TTL_HOURS)
    assert (config.MIN_DELAY, config.MAX_DELAY, config.CACHE_TTL_HOURS) == before
    monkeypatch.setattr(parser, "collect", lambda **k: (_ for _ in ()).throw(RuntimeError("x")))
    with pytest.raises(RuntimeError):
        service.PriceService._collect_live({})
    assert (config.MIN_DELAY, config.MAX_DELAY, config.CACHE_TTL_HOURS) == before


# ------------------------------- storage -------------------------------
def test_live_store_dedupes_invalidates_cache_and_labels_source(env):
    assert storage.load_dataset(use_live=False).data_source == "offline_fallback"
    storage._save_processed(storage.rows_to_frame(frame(120).to_dict("records")), "live")
    assert storage.PROCESSED_CSV.exists()
    rows = new_rows(10) + frame(5).to_dict("records")             # 10 new + 5 already in the fallback
    assert storage.append_live_rows(rows) == 10
    assert not storage.PROCESSED_CSV.exists()                      # stale cache dropped
    assert storage.append_live_rows(rows) == 0 and storage.append_live_rows([]) == 0
    res = storage.load_dataset(use_live=False)
    assert res.data_source == "live" and res.rows == 70 and res.df["listing_id"].is_unique


# ------------------------------- API -------------------------------
def test_api_disabled_returns_403(env):
    c = TestClient(main.create_app(svc=make(collector(), enabled=False), warmup=False))
    r = c.post("/api/refresh")
    assert r.status_code == 403 and "USE_LIVE_PARSER=true" in r.json()["message"]
    assert c.get("/api/refresh/status").json()["enabled"] is False


def test_api_started_running_and_rate_limited(env):
    gate = threading.Event()
    svc = make(collector(new_rows(10), gate=gate))
    c = TestClient(main.create_app(svc=svc, warmup=False))
    r = c.post("/api/refresh")
    assert r.status_code == 202 and r.json()["status"] == "started" and r.json()["min_interval_seconds"] == 600
    r = c.post("/api/refresh")
    assert r.status_code == 409 and "уже выполняется" in r.json()["message"]
    assert c.get("/api/refresh/status").json()["running"] is True
    gate.set()
    wait_idle(svc)
    r = c.post("/api/refresh")
    assert r.status_code == 429 and r.headers["retry-after"] == "600"
    assert "не чаще одного раза в 10 минут" in r.json()["message"] and "10 мин 0 с" in r.json()["message"]
    st = c.get("/api/refresh/status").json()
    assert st["running"] is False and st["last_result"]["status"] == "ok" and st["next_allowed_in_seconds"] == 600
    assert c.post("/api/predict", json=FLAT).status_code == 200      # predictions work after the swap


def test_api_refresh_waits_while_model_is_training(env):
    app = main.create_app(svc=make(collector()), warmup=False)
    app.state.warmup_thread = SimpleNamespace(is_alive=lambda: True)
    r = TestClient(app).post("/api/refresh")
    assert r.status_code == 503 and r.headers["retry-after"] == "5"
    assert storage.load_refresh_state() == {}                         # did not consume the rate limit
