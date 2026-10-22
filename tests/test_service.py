"""Service-layer tests (no network, no FastAPI). Synthetic frames are test-only."""
import json
import threading

import pandas as pd
import pytest

from backend import config, metrics, service, storage
from tests.test_metrics import synthetic_frame

FLAT = dict(district="Бостандыкский", area_total=54.0, floor=5, floor_total=9, house_year=2012, building_type="монолит")


def frame(n=400, seed=0, start=0):
    df = synthetic_frame(n, seed)
    df["listing_id"] = [str(500_000 + start + i) for i in range(n)]
    df["url"] = "https://krisha.kz/a/show/" + df["listing_id"]
    df["scraped_at"] = "2026-10-05T10:00:00+00:00"
    return df


def loader_for(df, source="offline_fallback", warnings=None):
    return lambda: storage.DatasetResult(df, source, len(df), list(warnings or []))


@pytest.fixture(autouse=True)
def tmp_model_path(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "MODEL_PATH", tmp_path / "model.joblib")


@pytest.fixture(scope="module")
def svc(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    mp.setattr(storage, "MODEL_PATH", tmp_path_factory.mktemp("model") / "model.joblib")
    s = service.PriceService(loader=loader_for(frame()))
    s.ensure_ready()
    yield s
    mp.undo()


# ------------------------------- predict -------------------------------
def test_predict_has_spec_fields_and_is_json_safe(svc):
    r = svc.predict(dict(FLAT))
    for key in ("price_estimate_kzt", "price_low_kzt", "price_high_kzt", "price_per_m2_kzt",
                "district_median_price_per_m2_kzt", "comparables", "model_metrics", "data_source",
                "rows_used", "warnings"):
        assert key in r
    assert r["price_low_kzt"] <= r["price_estimate_kzt"] <= r["price_high_kzt"]
    assert r["price_estimate_kzt"] % 100_000 == 0 and r["price_per_m2_kzt"] % 1_000 == 0
    assert abs(r["price_per_m2_kzt"] * 54.0 / r["price_estimate_kzt"] - 1) < 0.05
    assert r["data_source"] == "offline_fallback" and r["rows_used"] > 300
    assert len(r["comparables"]) == 5
    json.dumps(r, allow_nan=False)                                   # no numpy types, no NaN


def test_metrics_reported_are_cross_validation(svc):
    mm = svc.predict(dict(FLAT))["model_metrics"]
    assert mm["evaluation"] == "5-fold CV"
    assert mm["mape"] == pytest.approx(svc.metrics["cv"][svc.metrics["chosen_model"]]["mape"])
    assert set(("mae", "mape", "r2")) <= set(mm)


def test_rooms_are_ignored_with_fixed_renovation(svc):
    a = svc.predict(dict(FLAT, renovation=1))
    b = svc.predict(dict(FLAT, renovation="свежий ремонт", rooms=3))
    assert {k: a[k] for k in ("price_estimate_kzt", "price_low_kzt", "price_high_kzt")} == \
           {k: b[k] for k in ("price_estimate_kzt", "price_low_kzt", "price_high_kzt")}


def test_price_does_not_fall_when_area_grows(svc):
    prices = [svc.predict(dict(FLAT, area_total=a))["price_estimate_kzt"] for a in range(30, 200, 10)]
    assert prices == sorted(prices)


def test_district_comparison(svc):
    r = svc.predict(dict(FLAT))
    assert r["district_median_price_per_m2_kzt"] > 0
    assert r["vs_district_median_pct"] == pytest.approx(
        (r["price_per_m2_kzt"] / r["district_median_price_per_m2_kzt"] - 1) * 100, abs=1.5)


def test_input_errors(svc):
    with pytest.raises(ValueError):
        svc.predict(dict(FLAT, district="Нигде"))
    bad = dict(FLAT)
    bad.pop("floor")
    with pytest.raises(ValueError):
        svc.predict(bad)


def test_warnings(svc):
    w = " ".join(svc.predict(dict(FLAT, area_total=380.0))["warnings"])
    assert "сохранённые данные" in w and "собраны 2026-10-05" in w
    assert "Площадь вне основного диапазона" in w
    assert "Площадь вне" not in " ".join(svc.predict(dict(FLAT))["warnings"])


def test_live_data_has_no_saved_data_warning():
    s = service.PriceService(loader=loader_for(frame(150), source="live"), use_saved_model=False)
    assert not any("сохранённые данные" in w for w in s.predict(dict(FLAT))["warnings"])


def test_small_dataset_uses_baseline_and_warns():
    s = service.PriceService(loader=loader_for(frame(60)))
    r = s.predict(dict(FLAT))
    assert s.metrics["chosen_model"] == "naive"
    text = " ".join(r["warnings"])
    assert "Недостаточно данных" in text and "мало объявлений" in text
    assert r["model_metrics"]["evaluation"].startswith("in-sample")
    assert 0 < r["price_low_kzt"] <= r["price_estimate_kzt"] <= r["price_high_kzt"]


# ------------------------------- stats / health / options -------------------------------
def test_stats_and_options(svc):
    s = svc.stats()
    assert set(s["all_districts"]) == set(config.DISTRICTS)
    d = svc.stats("Медеуский")
    assert d["stats"]["count"] > 0 and d["model_details"]["chosen_model"] == svc.metrics["chosen_model"]
    with pytest.raises(KeyError):
        svc.stats("Нигде")
    json.dumps(d, allow_nan=False, default=str)
    o = svc.options()
    assert len(o["districts"]) == 8 and "renovation" not in json.dumps(o) and o["year_min"] == 1950


def test_health_does_not_train_and_reports_errors():
    calls = []
    s = service.PriceService(loader=lambda: calls.append(1))
    assert s.health()["status"] == "not_ready" and calls == []

    def broken():
        raise FileNotFoundError("no data")

    b = service.PriceService(loader=broken)
    with pytest.raises(service.ServiceError):
        b.ensure_ready()
    h = b.health()
    assert h["status"] == "error" and "no data" in h["error"]


# ------------------------------- persistence / concurrency -------------------------------
def test_saved_model_reused_until_data_changes(monkeypatch):
    calls = []
    real = metrics.train_and_evaluate
    monkeypatch.setattr(metrics, "train_and_evaluate", lambda d: (calls.append(1), real(d))[1])
    df = frame(60)
    service.PriceService(loader=loader_for(df)).ensure_ready()
    assert len(calls) == 1 and storage.MODEL_PATH.exists()
    service.PriceService(loader=loader_for(df)).ensure_ready()
    assert len(calls) == 1                                           # loaded from disk
    service.PriceService(loader=loader_for(pd.concat([df, frame(5, seed=9, start=1000)]))).ensure_ready()
    assert len(calls) == 2                                           # new data -> retrain


def test_corrupt_saved_model_triggers_retrain(monkeypatch):
    storage.MODEL_PATH.write_bytes(b"not a model")
    s = service.PriceService(loader=loader_for(frame(60)))
    s.ensure_ready()
    assert s.health()["status"] == "ok"


def test_concurrent_start_trains_once(monkeypatch):
    calls = []
    real = metrics.train_and_evaluate
    monkeypatch.setattr(metrics, "train_and_evaluate", lambda d: (calls.append(1), real(d))[1])
    s = service.PriceService(loader=loader_for(frame(60)), use_saved_model=False)
    threads = [threading.Thread(target=s.ensure_ready) for _ in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert len(calls) == 1 and s.health()["status"] == "ok"


# ------------------------------- comparables -------------------------------
@pytest.fixture(scope="module")
def pool():
    return metrics.add_features(metrics.clean_dataset(frame()))


def test_comparables_same_district_within_area_window(pool):
    comps = service.find_comparables(pool, dict(FLAT))
    assert len(comps) == 5
    assert all(c["district"] == "Бостандыкский" for c in comps)
    assert all(abs(c["area_total"] - 54.0) / 54.0 <= 0.2 for c in comps)
    for key in ("url", "price_kzt", "area_total", "floor", "floor_total", "house_year"):
        assert key in comps[0]


def test_exact_match_ranks_first(pool):
    row = dict(FLAT)
    twin = pd.DataFrame([dict(listing_id="twin", price_kzt=50_000_000, district=row["district"], area_total=54.0,
                              floor=5, floor_total=9, house_year=2012, building_type="монолит",
                              url="https://krisha.kz/a/show/twin")])
    df = metrics.add_features(pd.concat([pool, twin], ignore_index=True))
    assert service.find_comparables(df, row)[0]["url"].endswith("/twin")


def test_comparables_relax_to_whole_city(pool):
    others = pool[pool["district"] != "Жетысуский"]
    twins = pd.DataFrame([dict(listing_id=f"z{i}", price_kzt=40_000_000 + i, district="Жетысуский", area_total=54.0 + i,
                               floor=5, floor_total=9, house_year=2012, building_type="монолит",
                               url=f"https://krisha.kz/a/show/z{i}") for i in range(2)])
    df = metrics.add_features(pd.concat([others, twins], ignore_index=True))      # only 2 listings in the district
    comps = service.find_comparables(df, dict(FLAT, district="Жетысуский"))
    assert len(comps) == 5                                         # relaxed to the whole city
    assert [c["district"] for c in comps[:2]] == ["Жетысуский", "Жетысуский"]     # same district ranks first
    assert any(c["district"] != "Жетысуский" for c in comps)


def test_comparables_handle_unknowns_and_tiny_pools(pool):
    comps = service.find_comparables(pool, dict(FLAT, house_year=None, building_type=None, floor=None))
    assert len(comps) == 5
    json.dumps(comps, allow_nan=False)
    assert len(service.find_comparables(pool.head(3), dict(FLAT))) <= 3
    assert service.find_comparables(pool.iloc[0:0], dict(FLAT)) == []
