#!/usr/bin/env python3
"""Update 2: anti-bot (HTTP 468) handling, enrichment cooldown, district imputation of spatial features,
a race fix in the refresh status, and tests that no longer depend on your real cache or .env.

Run from the project root (the folder that contains backend/ and tests/), AFTER apply_spatial_update.py:

    python3 apply_update_2.py

Safe by design: all edits are checked first (every anchor must match exactly once, otherwise NOTHING is
written), re-running is harmless, a backup <file>.before_update2 is saved, and changed files are compiled.
"""
import pathlib
import py_compile
import shutil
import sys

EDITS = {}


EDITS["backend/parser.py"] = [
(r'''BLOCK_STATUS_CODES = (403, 429)''',
 r'''BLOCK_STATUS_CODES = (403, 429, 468)       # 468 is the Qrator anti-bot block page
GONE_STATUS_CODES = (404, 410)              # a removed listing is not a block'''),
(r'''        if resp.status_code in BLOCK_STATUS_CODES:
            raise BlockedError(f"HTTP {resp.status_code} from {url}")''',
 r'''        if resp.status_code in BLOCK_STATUS_CODES or (400 <= resp.status_code < 500
                                                      and resp.status_code not in GONE_STATUS_CODES):
            raise BlockedError(f"HTTP {resp.status_code} from {url}")   # any other 4xx is treated as a block too'''),
]

EDITS["backend/batch.py"] = [
(r'''import sys
from dataclasses import dataclass''', r'''import sys
import time
from dataclasses import dataclass'''),
(r'''    include_new_builds: bool = False

    @classmethod''', r'''    include_new_builds: bool = False
    enrich_cooldown_hours: float = 6.0     # after a block, --enrich refuses to run again for this long (0 = off)

    @classmethod'''),
(r'''            include_new_builds=g("INCLUDE_NEW_BUILDS", "false").lower() == "true",
        )''', r'''            include_new_builds=g("INCLUDE_NEW_BUILDS", "false").lower() == "true",
            enrich_cooldown_hours=float(g("ENRICH_COOLDOWN_HOURS", "6")),
        )'''),
(r'''    skip = set(t.state.get("no_coords", []))
    todo = [r for r in rows if not r.get("latitude") and r["listing_id"] not in skip]
''', r'''    until = t.state.get("blocked_until") or 0
    if time.time() < until:
        left = (until - time.time()) / 3600
        print(f"krisha.kz blocked the last enrichment run; pausing for another {left:.1f} h. Nothing was requested.")
        print("Meanwhile the model uses district-typical spatial values for listings without coordinates.")
        return 2
    skip = set(t.state.get("no_coords", []))
    todo = [r for r in rows if not r.get("latitude") and r["listing_id"] not in skip]
'''),
(r'''        reason = "limit_reached" if len(todo) > limit else "all_done"
''', r'''        reason = "limit_reached" if len(todo) > limit else "all_done"
        t.state.pop("blocked_until", None)                 # a clean pass ends any cooldown
'''),
(r'''    except parser.BlockedError as exc:
        reason, code = "blocked", 2
        print(f"\nSTOPPED: krisha.kz blocked the request ({exc}). Progress is saved; wait several hours before retrying.")
''', r'''    except parser.BlockedError as exc:
        reason, code = "blocked", 2
        t.state["blocked_until"] = time.time() + s.enrich_cooldown_hours * 3600
        print(f"\nSTOPPED: krisha.kz blocked the request ({exc}) - its anti-bot protection (HTTP 468 / 403 / 429).")
        print("Progress is saved and no further requests were made. No attempt is made to get around the block.")
        print(f"Enrichment pauses for {s.enrich_cooldown_hours:g} h. Meanwhile the model uses district-typical spatial "
              "values for listings without coordinates; nothing is invented in the dataset.")
'''),
]

EDITS["backend/geo.py"] = [
(r'''    d.loc[ok, "dist_to_metro"] = nearest_km(la, lo, poi.get("metro"))
    return d
''', r'''    d.loc[ok, "dist_to_metro"] = nearest_km(la, lo, poi.get("metro"))
    return d


def impute_by_district(df: pd.DataFrame, min_rows: int = 5) -> pd.DataFrame:
    """Fill missing spatial features with typical values and flag those rows (``geo_imputed`` = 1).

    A listing without coordinates (e.g. its detail page was blocked) gets the median of the REAL values of its
    district, or the city median if the district has fewer than ``min_rows`` real rows. This is preprocessing
    for the model only: nothing is written back to the dataset, and the model counts imputed rows as missing
    when it decides whether enough listings have real spatial data.
    """
    d = df.copy()
    real = d[GEO_FEATURES].notna().all(axis=1)
    d["geo_imputed"] = (~real).astype(int)
    if not real.any() or real.all():
        return d
    city = d.loc[real, GEO_FEATURES].median()
    by_district = d[real].groupby("district")[GEO_FEATURES].median()
    by_district = by_district[d[real].groupby("district").size() >= min_rows]
    for col in GEO_FEATURES:
        local = d["district"].map(by_district[col]) if len(by_district) else np.nan
        d[col] = d[col].fillna(local).fillna(city[col])
    return d
'''),
]

EDITS["backend/metrics.py"] = [
(r'''    coverage = float(d[SPATIAL_FEATURES].notna().all(axis=1).mean()) if n else 0.0
''', r'''    real = 1 - d["geo_imputed"] if "geo_imputed" in d.columns else d[SPATIAL_FEATURES].notna().all(axis=1)
    coverage = float(real.mean()) if n else 0.0                   # share of rows with REAL spatial data
'''),
]

EDITS["backend/service.py"] = [
(r'''        self.spatial_used = False
''', r'''        self.spatial_used = False
        self.spatial_typical: dict = {}
'''),
(r'''            clean = geo.add_geo_features(clean, poi)          # schools / hospitals / kindergartens / metro
''', r'''            clean = geo.add_geo_features(clean, poi)          # schools / hospitals / kindergartens / metro
            clean = geo.impute_by_district(clean)             # no coordinates: district-typical values, flagged
'''),
(r'''            "spatial_used": bool(m.get("spatial_features_used", False)),
''', r'''            "spatial_used": bool(m.get("spatial_features_used", False)),
            "spatial_typical": self._typical_spatial(feats),
'''),
(r'''    def _apply(self, state: dict) -> None:
''', r'''    @staticmethod
    def _typical_spatial(feats: pd.DataFrame) -> dict:
        """Median spatial values of REAL rows per district (and city-wide under '_city'), used for blank inputs."""
        cols = metrics.SPATIAL_FEATURES
        if "geo_imputed" not in feats.columns:
            return {}
        real = feats[feats["geo_imputed"] == 0]
        if real.empty:
            return {}
        out = {"_city": real[cols].median().to_dict()}
        for dist, g in real.groupby("district"):
            if len(g) >= 5:
                out[str(dist)] = g[cols].median().to_dict()
        return out

    def _apply(self, state: dict) -> None:
'''),
(r'''        est, low, high = metrics.predict_with_range(self.model, inputs)
''', r'''        filled = dict(inputs)                          # blank spatial inputs -> typical values of the district
        typical = self.spatial_typical.get(inputs["district"]) or self.spatial_typical.get("_city") or {}
        for key in metrics.SPATIAL_FEATURES:
            if filled.get(key) is None and key in typical:
                filled[key] = typical[key]
        est, low, high = metrics.predict_with_range(self.model, filled)
'''),
(r'''". Использованы типичные значения.")''', r'''". Использованы типичные значения района.")'''),
]

EDITS["backend/drive_store.py"] = [
(r'''class DriveStore:
''', r'''def _media(data: bytes, mime: str):
    """Upload payload for the Drive API (google-api-python-client is imported lazily; only STORE_BACKEND=drive needs it)."""
    from googleapiclient.http import MediaIoBaseUpload
    return MediaIoBaseUpload(io.BytesIO(data), mimetype=mime, resumable=False)


class DriveStore:
'''),
(r'''        from googleapiclient.http import MediaIoBaseUpload
        if not self.folder_id:
            raise StoreError("prepare() must be called first")
        mime = _MIME_BY_SUFFIX.get(Path(name).suffix, "application/octet-stream")
        media = MediaIoBaseUpload(io.BytesIO(data), mimetype=mime, resumable=False)''',
 r'''        if not self.folder_id:
            raise StoreError("prepare() must be called first")
        mime = _MIME_BY_SUFFIX.get(Path(name).suffix, "application/octet-stream")
        media = _media(data, mime)'''),
]

EDITS["tests/test_batch.py"] = [
(r'''from backend import batch, config, parser
''', r'''from backend import batch, config, drive_store, parser
'''),
(r'''def test_drive_write_updates_in_place_without_duplicates():
''', r'''class FakeMedia:
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
'''),
]

EDITS["tests/test_parser.py"] = [
(r'''    monkeypatch.setattr(storage, "PROCESSED_CSV", tmp_path / "c" / "d.csv")
    monkeypatch.setattr(storage, "PROCESSED_META", tmp_path / "c" / "d.meta.json")
    monkeypatch.setattr(config, "RAW_DIR", tmp_path / "empty_raw")
    monkeypatch.setattr(config, "FALLBACK_CSV", fb)
''', r'''    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "c")        # never read the real data/cache (live store!)
    monkeypatch.setattr(storage, "PROCESSED_CSV", tmp_path / "c" / "d.csv")
    monkeypatch.setattr(storage, "PROCESSED_META", tmp_path / "c" / "d.meta.json")
    monkeypatch.setattr(config, "RAW_DIR", tmp_path / "empty_raw")
    monkeypatch.setattr(config, "FALLBACK_CSV", fb)
'''),
]

CONFTEST_NEW = r'''"""Shared pytest configuration: the tests never touch your real data/cache, data/raw or .env settings."""
import pytest

from backend import config, storage


@pytest.fixture(scope="session", autouse=True)
def isolated_cache(tmp_path_factory):
    """Point every cache path at a temporary folder for the whole run.

    Files created by real use of the app (live_listings.csv, poi.json, refresh_state.json, the saved model, the
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
'''

EDITS["backend/service.py"] += [
(r'''        with self._refresh_lock:
            self._refresh_running = False
            state = storage.load_refresh_state()
            state["last_finished_at"] = self._clock()
            state["last_result"] = result
            storage.save_refresh_state(state)
        return result''', r'''        with self._refresh_lock:
            state = storage.load_refresh_state()
            state["last_finished_at"] = self._clock()
            state["last_result"] = result
            storage.save_refresh_state(state)
            self._refresh_running = False           # only after the result is saved: pollers never see 'idle' early
        return result'''),
(r'''    def refresh_status(self) -> dict:
        state, now = storage.load_refresh_state(), self._clock()
''', r'''    def refresh_status(self) -> dict:
        with self._refresh_lock:
            return self._refresh_status_locked()

    def _refresh_status_locked(self) -> dict:
        state, now = storage.load_refresh_state(), self._clock()
'''),
]

CONFTEST_OLD = r'''"""Shared pytest configuration."""
import importlib.util

import pytest

# Tests that exercise the real Google upload helper. They are only relevant for
# STORE_BACKEND=drive, which needs google-api-python-client; with the local
# (Google Drive for desktop) backend that package is not required.
_NEEDS_GOOGLE_CLIENT = {"test_drive_write_updates_in_place_without_duplicates"}


def pytest_collection_modifyitems(config, items):
    if importlib.util.find_spec("googleapiclient") is not None:
        return
    skip = pytest.mark.skip(reason="google-api-python-client not installed (only needed for STORE_BACKEND=drive)")
    for item in items:
        if item.name in _NEEDS_GOOGLE_CLIENT:
            item.add_marker(skip)
'''
EDITS["tests/conftest.py"] = [(CONFTEST_OLD, CONFTEST_NEW)]


def main() -> int:
    root = pathlib.Path(__file__).resolve().parent
    plan, problems = {}, []
    for rel, pairs in EDITS.items():
        path = root / rel
        if not path.exists():
            problems.append(f"missing file: {rel}")
            continue
        text, applied, skipped = path.read_text(encoding="utf-8"), 0, 0
        for old, new in pairs:
            if new in text and (old not in text or old in new):      # already applied
                skipped += 1
            elif text.count(old) == 1:
                text, applied = text.replace(old, new), applied + 1
            else:
                problems.append(f"{rel}: expected exactly 1 match, found {text.count(old)} for: {old.strip()[:70]!r}")
        plan[rel] = (text, applied, skipped)
    if problems:
        print("NOTHING WAS CHANGED. Problems found:")
        for p in problems:
            print("  -", p)
        print("The files differ from the versions this update was made for; send me this output.")
        return 1
    changed = 0
    for rel, (text, applied, skipped) in plan.items():
        path = root / rel
        if applied:
            shutil.copy2(path, path.with_name(path.name + ".before_update2"))
            path.write_text(text, encoding="utf-8")
            py_compile.compile(str(path), doraise=True)
            changed += 1
        print(f"{rel}: {applied} edits applied, {skipped} already present")
    print("Done." if changed else "Everything was already up to date.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
