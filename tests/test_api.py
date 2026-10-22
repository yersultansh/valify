"""API tests with FastAPI's TestClient. A synthetic service stands in for the real one (test-only data)."""
import json
import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from backend import config, main, service, storage
from tests.test_service import frame, loader_for

FLAT = dict(district="Бостандыкский", area_total=54.0, floor=5, floor_total=9, house_year=2012, building_type="монолит")


@pytest.fixture(scope="module")
def ready_service(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    mp.setattr(storage, "MODEL_PATH", tmp_path_factory.mktemp("m") / "model.joblib")
    s = service.PriceService(loader=loader_for(frame(130)))
    s.ensure_ready()
    yield s
    mp.undo()


@pytest.fixture(scope="module")
def client(ready_service):
    return TestClient(main.create_app(svc=ready_service, warmup=False))


def post(client, body):
    return client.post("/api/predict", json=body)


# ------------------------------- happy paths -------------------------------
def test_health_options_and_docs(client):
    h = client.get("/api/health").json()
    assert h["status"] == "ok" and h["data_source"] == "offline_fallback" and h["rows"] > 100 and h["model_trained_at"]
    o = client.get("/api/options").json()
    assert len(o["districts"]) == 8 and o["building_types"] == config.BUILDING_TYPES
    assert "renovation" not in json.dumps(o, ensure_ascii=False) and o["year_min"] == 1950
    assert client.get("/docs").status_code == 200
    paths = client.get("/openapi.json").json()["paths"]
    assert "/api/predict" in paths and "/api/stats" in paths and "/api/health" in paths


def test_predict_ok_and_ignores_unknown_fields(client):
    r = post(client, dict(FLAT, renovation=1))
    assert r.status_code == 200
    d = r.json()
    assert d["price_low_kzt"] <= d["price_estimate_kzt"] <= d["price_high_kzt"]
    assert len(d["comparables"]) == 5 and all(c["url"].startswith("https://krisha.kz/") for c in d["comparables"])
    assert d["model_metrics"]["evaluation"] and d["data_source"] == "offline_fallback" and d["warnings"]
    r2 = post(client, dict(FLAT, renovation="свежий ремонт", rooms=4, unknown="x"))
    assert r2.status_code == 200 and r2.json()["price_estimate_kzt"] == d["price_estimate_kzt"]


def test_predict_accepts_numeric_strings_and_whole_floats(client):
    assert post(client, dict(FLAT, area_total="54.5", floor="5", house_year=2012.0)).status_code == 200


def test_stats(client):
    s = client.get("/api/stats").json()
    assert len(s["all_districts"]) == 8 and s["model_metrics"]["mape"] > 0
    d = client.get("/api/stats", params={"district": "Медеуский"}).json()
    assert d["stats"]["count"] > 0
    assert client.get("/api/stats", params={"district": ""}).status_code == 200     # empty = all districts


def test_stats_errors(client, ready_service):
    r = client.get("/api/stats", params={"district": "Нигде"})
    assert r.status_code == 422 and "Район должен быть одним из" in r.json()["message"]
    df = frame(130)
    df = df[df["district"] != "Жетысуский"]
    app = main.create_app(svc=service.PriceService(loader=loader_for(df), use_saved_model=False), warmup=False)
    r = TestClient(app).get("/api/stats", params={"district": "Жетысуский"})
    assert r.status_code == 404 and "Нет данных по району" in r.json()["message"]


# ------------------------------- Russian validation -------------------------------
@pytest.mark.parametrize("patch,field,expected", [
    ({"area_total": 5}, "area_total", "не меньше 10."),
    ({"area_total": 401}, "area_total", "не больше 400."),
    ({"area_total": "много"}, "area_total", "должно быть числом"),
    ({"area_total": None}, "area_total", "должно быть числом"),
    ({"floor": 0}, "floor", "не меньше 1."),
    ({"floor": 5.5}, "floor", "целым числом"),
    ({"floor": 12}, "floor_total", "Этаж (12) не может быть больше этажности дома (9)"),
    ({"house_year": 1900}, "house_year", "Год постройки должен быть от 1950 до"),
    ({"house_year": 2100}, "house_year", "Год постройки должен быть от 1950 до"),
    ({"district": "Нигде"}, "district", "Выберите район из списка"),
    ({"building_type": "дом"}, "building_type", "Выберите тип дома из списка"),
    ({"district": None}, "district", "должно быть текстом"),
])
def test_validation_messages_are_russian(client, patch, field, expected):
    r = post(client, dict(FLAT, **patch))
    assert r.status_code == 422
    body = r.json()
    assert expected in body["message"]
    assert field in [e["field"] for e in body["detail"]]
    for banned in ("Input should", "Field required", "Value error", "valid", "ensure this"):
        assert banned not in body["message"]


@pytest.mark.parametrize("missing", ["district", "area_total", "floor", "floor_total", "house_year", "building_type"])
def test_missing_fields(client, missing):
    body = dict(FLAT)
    body.pop(missing)
    r = post(client, body)
    assert r.status_code == 422 and "обязательно для заполнения" in r.json()["message"]
    assert r.json()["detail"][0]["field"] == missing


def test_bad_bodies(client):
    r = client.post("/api/predict", content=b"{broken", headers={"Content-Type": "application/json"})
    assert r.status_code == 422 and r.json()["message"] == "Некорректный JSON в теле запроса."
    r = client.post("/api/predict", content=b"[1,2]", headers={"Content-Type": "application/json"})
    assert r.status_code == 422 and "JSON-объектом" in r.json()["message"]
    r = client.post("/api/predict")
    assert r.status_code == 422 and "Тело запроса отсутствует" in r.json()["message"]


def test_multiple_errors_are_reported_together(client):
    r = post(client, dict(FLAT, area_total=1, house_year=1800, district="Нигде"))
    fields = {e["field"] for e in r.json()["detail"]}
    assert {"area_total", "house_year", "district"} <= fields


def test_not_found_and_method_not_allowed_are_russian(client):
    assert client.get("/nope").json()["message"] == "Страница не найдена."
    r = client.get("/api/predict")
    assert r.status_code == 405 and r.json()["message"] == "Метод не поддерживается."


# ------------------------------- readiness / failures -------------------------------
def test_predict_503_while_training_then_lazy_start(ready_service):
    s = service.PriceService(loader=loader_for(frame(60)), use_saved_model=False)
    app = main.create_app(svc=s, warmup=False)
    app.state.warmup_thread = SimpleNamespace(is_alive=lambda: True)          # pretend warm-up is running
    c = TestClient(app)
    assert c.get("/api/health").json()["status"] == "training"
    r = post(c, FLAT)
    assert r.status_code == 503 and r.headers["retry-after"] == "5" and "обучается" in r.json()["message"]
    app.state.warmup_thread = SimpleNamespace(is_alive=lambda: False)         # warm-up finished/never ran
    assert post(c, FLAT).status_code == 200                                    # lazy start trains on demand
    assert c.get("/api/health").json()["status"] == "ok"


def test_data_failure_gives_503_and_server_stays_up():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] == 1:
            raise FileNotFoundError("no data at /secret/path.csv")
        return storage.DatasetResult(frame(60), "offline_fallback", 60, [])

    app = main.create_app(svc=service.PriceService(loader=flaky, use_saved_model=False), warmup=False)
    c = TestClient(app)
    r = post(c, FLAT)
    assert r.status_code == 503 and "недоступны" in r.json()["message"]
    assert "/secret/path" not in r.text                                       # internals not leaked in predict
    assert c.get("/api/health").json()["status"] == "error"
    assert c.get("/api/options").status_code == 200                           # server alive
    assert post(c, FLAT).status_code == 200                                   # recovers on retry


def test_startup_warmup_runs_in_background():
    app = main.create_app(svc=service.PriceService(loader=loader_for(frame(60)), use_saved_model=False), warmup=True)
    with TestClient(app) as c:
        for _ in range(100):
            if c.get("/api/health").json()["status"] == "ok":
                break
            time.sleep(0.2)
        assert c.get("/api/health").json()["status"] == "ok"
        assert post(c, FLAT).status_code == 200


def test_unexpected_error_is_generic_500(ready_service, monkeypatch):
    monkeypatch.setattr(ready_service, "predict", lambda inputs: 1 / 0)
    c = TestClient(main.create_app(svc=ready_service, warmup=False), raise_server_exceptions=False)
    r = post(c, FLAT)
    assert r.status_code == 500 and r.json()["message"] == "Внутренняя ошибка сервера. Попробуйте позже."
    assert "ZeroDivision" not in r.text


# ------------------------------- CORS / static -------------------------------
def test_cors_only_for_local_origins(client):
    ok = client.options("/api/predict", headers={"Origin": "http://localhost:8000",
                                                 "Access-Control-Request-Method": "POST"})
    assert ok.headers.get("access-control-allow-origin") == "http://localhost:8000"
    ok2 = client.options("/api/predict", headers={"Origin": "http://127.0.0.1:8000",
                                                  "Access-Control-Request-Method": "POST"})
    assert ok2.headers.get("access-control-allow-origin") == "http://127.0.0.1:8000"
    bad = client.options("/api/predict", headers={"Origin": "http://evil.example",
                                                  "Access-Control-Request-Method": "POST"})
    assert "access-control-allow-origin" not in bad.headers


def test_index_static_and_traversal(ready_service, tmp_path, monkeypatch):
    # no frontend yet -> placeholder page, API links, 200
    monkeypatch.setattr(config, "FRONTEND_DIR", tmp_path / "frontend")
    c = TestClient(main.create_app(svc=ready_service, warmup=False))
    r = c.get("/")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"] and "/docs" in r.text
    assert c.get("/favicon.ico").status_code == 204
    # with a frontend folder: index and static files are served, traversal is not
    (tmp_path / "frontend").mkdir()
    (tmp_path / "frontend" / "index.html").write_text("<html>HELLO-INDEX</html>", encoding="utf-8")
    (tmp_path / "frontend" / "style.css").write_text("body{}", encoding="utf-8")
    (tmp_path / "secret.txt").write_text("secret", encoding="utf-8")
    c = TestClient(main.create_app(svc=ready_service, warmup=False))
    assert "HELLO-INDEX" in c.get("/").text
    assert c.get("/static/style.css").text == "body{}"
    assert c.get("/static/missing.js").status_code == 404
    assert c.get("/static/%2e%2e/secret.txt").status_code == 404
    assert c.get("/static/../secret.txt").status_code == 404


def test_default_app_object_exists():
    assert main.app.title.startswith("Almaty") and main.app.state.warmup_thread is None
