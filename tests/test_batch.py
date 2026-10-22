"""Batch collector tests: fake Drive API, fake HTTP, saved fixtures. No network."""
import itertools
import re
from pathlib import Path

import pytest

from backend import batch, config, drive_store, parser
from backend.drive_store import FOLDER_MIME, DriveStore, LocalStore

ROOT = Path(__file__).resolve().parent.parent


def _fixture(name: str) -> str:
    for base in (ROOT / "tests" / "fixtures", ROOT / "data" / "raw"):
        if (base / name).exists():
            return (base / name).read_text(encoding="utf-8")
    pytest.skip(f"fixture {name} missing")


# ----------------------------- fake Google Drive -----------------------------
class _Req:
    def __init__(self, fn):
        self.fn = fn

    def execute(self, num_retries=0):
        return self.fn()


class FakeDrive:
    """Just enough of the Drive v3 API: files().list/create/update/get_media."""

    def __init__(self):
        self.files_db = {}
        self._n = itertools.count(1)
        self.add("root", "My Drive", None, FOLDER_MIME, fid="root")

    def add(self, fid_hint, name, parent, mime, data=b"", modified="2026-01-01T00:00:00Z", fid=None):
        fid = fid or f"id{next(self._n)}"
        self.files_db[fid] = dict(id=fid, name=name, parents=[parent] if parent else [], mimeType=mime,
                                  data=data, modifiedTime=modified, trashed=False)
        return fid

    def files(self):
        return self

    def list(self, q, **kw):
        def run():
            parent = re.search(r"'([^']+)' in parents", q).group(1)
            res = []
            for f in self.files_db.values():
                if parent not in f["parents"] or f["trashed"]:
                    continue
                if f"mimeType='{FOLDER_MIME}'" in q and f["mimeType"] != FOLDER_MIME:
                    continue
                if f"mimeType!='{FOLDER_MIME}'" in q and f["mimeType"] == FOLDER_MIME:
                    continue
                m = re.search(r"name='([^']+)'", q)
                if m and f["name"] != m.group(1):
                    continue
                res.append({k: f[k] for k in ("id", "name", "modifiedTime")})
            return {"files": res}
        return _Req(run)

    def create(self, body, media_body=None, fields=None):
        def run():
            data = media_body.getbytes(0, media_body.size()) if media_body else b""
            mime = media_body.mimetype() if media_body else body.get("mimeType", "text/plain")
            fid = self.add("", body["name"], body["parents"][0], mime, data)
            return {"id": fid}
        return _Req(run)

    def update(self, fileId, media_body, fields=None):
        def run():
            self.files_db[fileId]["data"] = media_body.getbytes(0, media_body.size())
            return {"id": fileId}
        return _Req(run)

    def get_media(self, fileId):
        return _Req(lambda: self.files_db[fileId]["data"])


def _drive(fake, parent="Forecasting", path="almaty-flat-calculator/data/collected"):
    return DriveStore(parent, path, "x.json", "t.json", service=fake)


def test_drive_finds_existing_parent_case_insensitive_and_creates_subfolders():
    fake = FakeDrive()
    existing = fake.add("", "forecasting", "root", FOLDER_MIME)
    fake.add("", "API key.rtf", existing, "text/rtf", b"secret")        # unrelated file must stay untouched
    store = _drive(fake)
    url = store.prepare()
    folders = [f for f in fake.files_db.values() if f["mimeType"] == FOLDER_MIME and f["id"] != "root"]
    assert sorted(f["name"] for f in folders) == ["almaty-flat-calculator", "collected", "data", "forecasting"]
    assert len([f for f in folders if f["name"].lower() == "forecasting"]) == 1    # no duplicate parent
    assert url.endswith(store.folder_id)
    store.prepare()                                                              # idempotent
    assert len([f for f in fake.files_db.values() if f["mimeType"] == FOLDER_MIME]) == 5
    assert fake.files_db[[k for k, v in fake.files_db.items() if v["name"] == "API key.rtf"][0]]["data"] == b"secret"


def test_drive_creates_parent_when_missing_and_picks_newest_duplicate():
    fake = FakeDrive()
    store = _drive(fake)
    store.prepare()
    assert any(f["name"] == "Forecasting" for f in fake.files_db.values())
    fake2 = FakeDrive()
    old = fake2.add("", "Forecasting", "root", FOLDER_MIME, modified="2026-01-01T00:00:00Z")
    new = fake2.add("", "FORECASTING", "root", FOLDER_MIME, modified="2026-09-01T00:00:00Z")
    s2 = _drive(fake2, path="")
    s2.prepare()
    assert s2.folder_id == new


class FakeMedia:
    """Stands in for googleapiclient's MediaIoBaseUpload, so this test needs no Google packages."""

    def __init__(self, data, mime):
        self._data, self._mime = data, mime

    def getbytes(self, begin, length):
        return self._data[begin:begin + length]

    def size(self):
        return len(self._data)

    def mimetype(self):
        return self._mime


def test_drive_write_updates_in_place_without_duplicates(monkeypatch):
    monkeypatch.setattr(drive_store, "_media", FakeMedia)
    fake = FakeDrive()
    store = _drive(fake)
    store.prepare()
    assert store.read("a.csv") is None
    store.write("a.csv", b"one")
    store.write("a.csv", b"two")
    assert store.read("a.csv") == b"two"
    assert len([f for f in fake.files_db.values() if f["name"] == "a.csv"]) == 1
    assert next(f for f in fake.files_db.values() if f["name"] == "a.csv")["mimeType"] == "text/csv"


# ----------------------------- batch logic -----------------------------
class _Resp:
    def __init__(self, status, text="", url="https://krisha.kz/x"):
        self.status_code, self.text, self.url, self.encoding = status, text, url, "utf-8"

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


def _page(base_html, n):
    """Make page n distinct by suffixing every listing id with n."""
    html = base_html
    for c in parser.parse_listing_cards(base_html):
        html = html.replace(c["listing_id"], f"{c['listing_id']}{n}")
    return html


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(batch, "WORK_DIR", tmp_path / "work")
    monkeypatch.setattr(batch.Tracker.__init__, "__defaults__", (tmp_path / "work",))
    sleeps = []
    monkeypatch.setattr(parser.time, "sleep", lambda s: sleeps.append(s))
    monkeypatch.setattr(parser, "_last_request_ts", None)
    base = _fixture("listing.html")
    calls = []

    def fake_get(self, url, **kw):
        calls.append(url)
        m = re.search(r"page=(\d+)", url)
        n = int(m.group(1)) if m else 1
        return _Resp(200, _page(base, n) if n > 1 else base, url)

    monkeypatch.setattr(parser.requests.Session, "get", fake_get)
    store = LocalStore(tmp_path / "store")
    s = batch.Settings(batch_size=10, max_pages=5, target_total=500)
    return dict(store=store, s=s, calls=calls, sleeps=sleeps, tmp=tmp_path, base=base)


def _rows(store):
    return batch._parse_csv(store.read(batch.DATASET_NAME))


def test_batch_stops_at_batch_size_and_resumes_without_duplicates(env):
    assert batch.run_batch(env["store"], env["s"]) == 0
    first = _rows(env["store"])
    assert len(first) == 10
    visited = env["store"].read(batch.VISITED_NAME).decode().splitlines()
    assert len(visited) == 10 and all(v.startswith("https://krisha.kz/a/show/") for v in visited)
    assert batch.run_batch(env["store"], env["s"]) == 0
    second = _rows(env["store"])
    ids = [r["listing_id"] for r in second]
    assert len(ids) == 20 and len(set(ids)) == 20 and ids[:10] == [r["listing_id"] for r in first]


def test_batch_never_returns_new_builds_and_respects_max_pages(env):
    env["s"].batch_size, env["s"].max_pages = 200, 2
    batch.run_batch(env["store"], env["s"])
    rows = _rows(env["store"])
    assert len(rows) == 38                         # 19 resale rows per page x 2 pages
    assert all(r["is_new_build"] == "False" and r["source"] == "krisha" for r in rows)
    assert len([c for c in env["calls"] if "krisha.kz" in c]) == 2


def test_delays_between_requests_are_8_to_12_seconds(env):
    env["s"].batch_size, env["s"].max_pages = 200, 3
    batch.run_batch(env["store"], env["s"])
    assert len(env["calls"]) == 3 and len(env["sleeps"]) == 2
    assert all(8.0 <= d <= 12.0 for d in env["sleeps"])


def test_visited_ids_are_skipped_before_parsing(env):
    first_id = parser.parse_listing_cards(env["base"])[0]["listing_id"]
    env["store"].prepare()
    env["store"].write(batch.VISITED_NAME, f"https://krisha.kz/a/show/{first_id}\n".encode())
    env["s"].batch_size, env["s"].max_pages = 200, 1
    batch.run_batch(env["store"], env["s"])
    ids = [r["listing_id"] for r in _rows(env["store"])]
    assert first_id not in ids and len(ids) == 18


def test_block_saves_progress_and_exits_with_code_2(env, monkeypatch, capsys):
    base = env["base"]

    def fake_get(self, url, **kw):
        return _Resp(403) if "page=2" in url else _Resp(200, base, url)

    monkeypatch.setattr(parser.requests.Session, "get", fake_get)
    env["s"].batch_size = 200
    assert batch.run_batch(env["store"], env["s"]) == 2
    assert len(_rows(env["store"])) == 19          # page 1 kept
    assert "STOPPED" in capsys.readouterr().out
    state = batch.json.loads(env["store"].read(batch.STATE_NAME))
    assert state["next_page"] == 2 and state["runs"][-1]["reason"] == "blocked"


def test_target_reached_stops_and_later_runs_do_nothing(env):
    env["s"].target_total, env["s"].batch_size = 7, 50
    batch.run_batch(env["store"], env["s"])
    assert len(_rows(env["store"])) == 7
    calls_before = len(env["calls"])
    assert batch.run_batch(env["store"], env["s"]) == 0
    assert len(env["calls"]) == calls_before


def test_local_leftovers_are_merged_when_store_upload_failed(env):
    batch.run_batch(env["store"], env["s"])                     # 10 rows in store and local
    # simulate a failed upload: local has 10 extra rows the store never received
    t = batch.Tracker()
    extra = parser.parse_listing_cards(_page(env["base"], 9))
    for raw in extra[:3]:
        if not raw["is_new_build"]:
            t.add_row(parser.normalize_listing(raw))
    env["store"].write(batch.DATASET_NAME, env["store"].read(batch.DATASET_NAME))   # store unchanged
    t2 = batch.Tracker()
    t2.pull(env["store"])
    assert len(t2.ids) >= 11 and len(t2.ids) == len({r["listing_id"] for r in batch._parse_csv(t2.csv_path.read_bytes())})


def test_stalls_when_pages_are_already_collected(env, monkeypatch, capsys):
    monkeypatch.setattr(parser.requests.Session, "get", lambda self, url, **kw: _Resp(200, env["base"], url))
    env["s"].batch_size, env["s"].max_pages = 200, 5
    batch.run_batch(env["store"], env["s"])
    batch.run_batch(env["store"], env["s"], start_page=1)
    out = capsys.readouterr().out
    assert "no_new_listings" in out and "--start-page" in out


def test_status_makes_no_http_calls(env, capsys):
    assert batch.show_status(env["store"], env["s"]) == 0
    assert env["calls"] == []


def test_settings_respect_hard_limits(monkeypatch):
    monkeypatch.setattr(config, "MAX_PAGES", 5)
    monkeypatch.setattr(config, "MAX_LISTINGS", 200)
    s = batch.Settings(batch_size=999, max_pages=50, min_delay=12, max_delay=8).clamped()
    assert (s.batch_size, s.max_pages, s.min_delay, s.max_delay) == (200, 5, 8, 12)
