"""Coordinates: parsing from detail pages and the resumable `batch --enrich` mode. No network."""
from pathlib import Path

import pytest
import requests

from backend import batch, config, parser
from backend.drive_store import LocalStore

ROOT = Path(__file__).resolve().parent.parent


def detail_html():
    for base in (ROOT / "tests" / "fixtures", ROOT / "data" / "raw"):
        if (base / "detail.html").exists():
            return (base / "detail.html").read_text(encoding="utf-8")
    pytest.skip("detail.html fixture missing")


# ------------------------------- parsing -------------------------------
def test_detail_page_has_coordinates_and_no_personal_data():
    d = parser.parse_listing_detail(detail_html())
    assert (d["json_lat"], d["json_lon"]) == (43.188962, 76.821399)
    assert not any("owner" in k.lower() or "phone" in k.lower() for k in d)


def test_merge_and_normalize_coordinates():
    raw = {"listing_id": "1", "price_text": "35 000 000 ₸", "area_text": "48", "subtitle": "", "floor": None,
           "year_raw": None, "building_raw": None, "renovation_raw": None}
    assert parser.normalize_listing(raw)["latitude"] is None                   # cards carry no coordinates
    merged = parser.merge_detail(raw, parser.parse_listing_detail(detail_html()))
    row = parser.normalize_listing(merged)
    assert (row["latitude"], row["longitude"]) == (43.188962, 76.821399)
    again = parser.merge_detail(merged, {"json_lat": 1.0, "json_lon": 2.0})
    assert again["latitude"] == 43.188962                                      # known value is never overwritten


@pytest.mark.parametrize("lat,lon", [(0, 0), (51.1, 71.4), ("x", 76.9), (None, None), (float("nan"), 76.9), ("", "")])
def test_out_of_range_or_bad_coordinates_are_dropped(lat, lon):
    row = parser.normalize_listing({"listing_id": "1", "price_text": "1 ₸", "latitude": lat, "longitude": lon})
    assert row["latitude"] is None and row["longitude"] is None


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


# ------------------------------- enrich -------------------------------
class Resp:
    def __init__(self, status, text="", url="https://krisha.kz/a/show/x"):
        self.status_code, self.text, self.url, self.encoding = status, text, url, "utf-8"

    def raise_for_status(self):
        if self.status_code >= 400:
            err = requests.HTTPError(str(self.status_code))
            err.response = self
            raise err


def listing_rows(n):
    return [dict(listing_id=str(1000 + i), price_kzt=50_000_000, district="Бостандыкский", area_total=50,
                 url=f"https://krisha.kz/a/show/{1000 + i}", source="krisha", is_new_build="False") for i in range(n)]


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(batch, "WORK_DIR", tmp_path / "work")
    monkeypatch.setattr(batch.Tracker.__init__, "__defaults__", (tmp_path / "work",))
    monkeypatch.setattr(parser.time, "sleep", lambda s: None)
    monkeypatch.setattr(parser, "_last_request_ts", None)
    store = LocalStore(tmp_path / "store")
    store.prepare()
    (tmp_path / "work").mkdir()
    batch._write_csv(tmp_path / "work" / batch.DATASET_NAME, [])
    tmp = tmp_path / "seed.csv"
    batch._write_csv(tmp, listing_rows(6))
    store.write(batch.DATASET_NAME, tmp.read_bytes())
    calls = []
    html = detail_html()
    pages = {"1002": Resp(404), "1003": Resp(200, html.replace('"map"', '"mapx"'))}      # removed / no map block

    def fake_get(self, url, **kw):
        calls.append(url)
        return pages.get(url.rsplit("/", 1)[1], Resp(200, html, url))

    monkeypatch.setattr(parser.requests.Session, "get", fake_get)
    return dict(store=store, calls=calls, pages=pages, s=batch.Settings(batch_size=10), tmp=tmp_path)


def rows_of(store):
    return batch._parse_csv(store.read(batch.DATASET_NAME))


def test_enrich_fills_coordinates_and_remembers_unavailable(env):
    assert batch.run_enrich(env["store"], env["s"]) == 0
    rows = {r["listing_id"]: r for r in rows_of(env["store"])}
    assert [bool(rows[i]["latitude"]) for i in sorted(rows)] == [True, True, False, False, True, True]
    assert float(rows["1000"]["latitude"]) == 43.188962 and float(rows["1000"]["longitude"]) == 76.821399
    state = batch.json.loads(env["store"].read(batch.STATE_NAME))
    assert state["no_coords"] == ["1002", "1003"] and state["runs"][-1]["kind"] == "enrich"
    assert not (env["tmp"] / "raw").exists()                                    # detail pages are not cached
    first = len(env["calls"])
    assert batch.run_enrich(env["store"], env["s"]) == 0                         # nothing left: no requests
    assert len(env["calls"]) == first == 6


def test_enrich_respects_limit_and_resumes(env):
    assert batch.run_enrich(env["store"], env["s"], limit=2) == 0
    assert len(env["calls"]) == 2 and sum(bool(r["latitude"]) for r in rows_of(env["store"])) == 2
    batch.run_enrich(env["store"], env["s"], limit=2)
    assert len(env["calls"]) == 4 and len(set(env["calls"])) == 4                # no listing requested twice


def test_enrich_stops_on_block_and_keeps_progress(env, capsys):
    env["pages"]["1001"] = Resp(403)
    assert batch.run_enrich(env["store"], env["s"]) == 2
    assert "STOPPED" in capsys.readouterr().out
    got = [r["listing_id"] for r in rows_of(env["store"]) if r["latitude"]]
    assert got == ["1000"] and len(env["calls"]) == 2                           # nothing after the 403
    state = batch.json.loads(env["store"].read(batch.STATE_NAME))
    assert state["runs"][-1]["reason"] == "blocked"


def test_enrich_delays_and_no_network_when_everything_is_done(env, monkeypatch):
    sleeps = []
    monkeypatch.setattr(parser.time, "sleep", lambda s: sleeps.append(s))
    batch.run_enrich(env["store"], env["s"])
    assert sleeps and all(8.0 <= d <= 12.0 for d in sleeps)


def test_pull_keeps_the_more_complete_copy(env):
    batch.run_enrich(env["store"], env["s"], limit=1)                            # local + store both enriched
    older = env["tmp"] / "older.csv"
    batch._write_csv(older, listing_rows(6))                                     # store copy without coordinates
    env["store"].write(batch.DATASET_NAME, older.read_bytes())
    t = batch.Tracker()
    t.pull(env["store"])
    merged = {r["listing_id"]: r for r in batch._parse_csv(t.csv_path.read_bytes())}
    assert merged["1000"]["latitude"] and len(merged) == 6                       # coordinates survive the merge


def test_status_shows_coordinate_coverage(env, capsys):
    batch.run_enrich(env["store"], env["s"], limit=2)
    batch.show_status(env["store"], env["s"])
    assert "Listings with coordinates: 2/6" in capsys.readouterr().out
