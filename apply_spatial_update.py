#!/usr/bin/env python3
"""One-time update: apartment coordinates, spatial features (schools / hospitals / kindergartens within 2 km,
distance to the metro) and the binary renovation flag.

Run from the project root (the folder that contains backend/ and tests/):

    python3 apply_spatial_update.py

Safe by design:
  * every edit is checked first; if any anchor does not match exactly once, NOTHING is written;
  * re-running is harmless (already applied edits are skipped);
  * a backup <file>.before_spatial is saved next to each changed file;
  * every changed file is compiled afterwards to catch syntax errors.
New files (backend/geo.py, tests/test_geo.py, tests/test_coords.py, tests/test_spatial_model.py) are separate
uploads and are not touched here.
"""
import pathlib
import py_compile
import shutil
import sys

EDITS = {}


# ----------------------------------------------------------------------------- parser.py
EDITS["backend/parser.py"] = [
(r'''def fetch(url: str, referer: str | None = None, stats: dict | None = None) -> str:''',
 r'''def fetch(url: str, referer: str | None = None, stats: dict | None = None, use_cache: bool = True) -> str:'''),
(r'''    cached = storage.read_raw_cache(url)
''', r'''    cached = storage.read_raw_cache(url) if use_cache else None
'''),
(r'''        storage.write_raw_cache(url, html, resp.status_code)
''', r'''        if use_cache:
            storage.write_raw_cache(url, html, resp.status_code)
'''),
(r'''            for key in ADVERT_JSON_KEYS:
                if key in advert:
                    out[f"json_{key}"] = advert[key]
''', r'''            for key in ADVERT_JSON_KEYS:
                if key in advert:
                    out[f"json_{key}"] = advert[key]
            coords = advert.get("map")                      # only lat/lon are read from the map block
            if isinstance(coords, dict):
                out["json_lat"], out["json_lon"] = coords.get("lat"), coords.get("lon")
'''),
(r'''        merged["district_raw"] = detail.get("district_raw", "")
    return merged
''', r'''        merged["district_raw"] = detail.get("district_raw", "")
    if merged.get("latitude") is None and detail.get("json_lat") is not None:
        merged["latitude"], merged["longitude"] = detail["json_lat"], detail.get("json_lon")
    return merged
'''),
(r'''def normalize_listing(raw: dict) -> dict:
''', r'''LAT_RANGE, LON_RANGE = (42.8, 44.0), (76.3, 77.6)    # sanity box around Almaty; anything else is dropped


def _coord(value, lo: float, hi: float) -> float | None:
    """Coordinate as a float inside [lo, hi], else None."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return round(v, 6) if lo <= v <= hi else None


def _coord_pair(raw: dict) -> dict:
    """latitude/longitude together or not at all."""
    lat, lon = _coord(raw.get("latitude"), *LAT_RANGE), _coord(raw.get("longitude"), *LON_RANGE)
    ok = lat is not None and lon is not None
    return {"latitude": lat if ok else None, "longitude": lon if ok else None}


def normalize_listing(raw: dict) -> dict:
'''),
(r'''        "rooms": raw.get("rooms"),
''', r'''        "rooms": raw.get("rooms"),
        **_coord_pair(raw),
'''),
]

# ----------------------------------------------------------------------------- storage.py
EDITS["backend/storage.py"] = [
(r'''    "house_year", "building_type", "renovation", "rooms", "url", "scraped_at",
''', r'''    "house_year", "building_type", "renovation", "rooms", "latitude", "longitude", "url", "scraped_at",
'''),
(r'''def load_dataset(use_live: bool | None = None) -> DatasetResult:
''', r'''def _poi_path() -> Path:
    return config.CACHE_DIR / "poi.json"


def save_poi(poi: dict) -> Path:
    """Cache the points of interest (schools, hospitals, kindergartens, metro) downloaded by backend.geo."""
    path = _poi_path()
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(poi, ensure_ascii=False), encoding="utf-8")
    return path


def load_poi() -> dict | None:
    """Cached points of interest, or None if not downloaded yet / unreadable."""
    try:
        data = json.loads(_poi_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) and data.get("metro") else None


def load_dataset(use_live: bool | None = None) -> DatasetResult:
'''),
]

# ----------------------------------------------------------------------------- metrics.py
EDITS["backend/metrics.py"] = [
(r'''* ``renovation`` is deliberately NOT a feature: the seller-typed field is too
  inconsistent and noisy (free text, ~half missing). ``rooms`` is also not used:
  it is highly collinear with ``area_total`` and not asked for in the input form.''',
 r'''* ``renovation`` enters only as a binary flag: 0 = "Черновая отделка" (shell), 1 = anything else. The
  free-text field itself is too noisy. ``rooms`` is not used: it is collinear with ``area_total``.
* Spatial features (schools / hospitals / kindergartens within 2 km, distance to the metro) are used
  only when at least MIN_SPATIAL_COVERAGE of the rows have them; otherwise the model trains without
  them and says so in ``warnings``.'''),
(r'''NUMERIC_FEATURES = ["area_total", "floor", "floor_total", "floor_ratio", "house_age"]
CATEGORICAL_FEATURES = ["district", "building_type"]
BINARY_FEATURES = ["is_first_floor", "is_last_floor"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES
EXCLUDED_FEATURES = ("renovation", "rooms")''',
 r'''BASE_NUMERIC = ["area_total", "floor", "floor_total", "floor_ratio", "house_age"]
SPATIAL_FEATURES = ["schools_within_2km", "hospitals_within_2km", "kindergartens_within_2km", "dist_to_metro"]
NUMERIC_FEATURES = BASE_NUMERIC + SPATIAL_FEATURES        # superset; spatial ones are used only with enough coverage
CATEGORICAL_FEATURES = ["district", "building_type"]
BINARY_FEATURES = ["is_first_floor", "is_last_floor", "renovation"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES
EXCLUDED_FEATURES = ("rooms",)
MIN_SPATIAL_COVERAGE = 0.6'''),
(r'''def add_features(df: pd.DataFrame, ref_year: int | None = None) -> pd.DataFrame:
''', r'''def _renovation_flag(series: pd.Series) -> pd.Series:
    """0 = 'Черновая отделка' (shell), 1 = anything else; already-binary values pass through; missing stays NaN."""
    def one(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return np.nan
        if isinstance(v, (bool, np.bool_, int, float, np.integer, np.floating)):
            return 1.0 if v else 0.0
        t = str(v).strip().lower()
        if t in ("0", "1"):
            return float(t)
        return 0.0 if "черновая" in t else 1.0
    return series.map(one).astype(float)


def add_features(df: pd.DataFrame, ref_year: int | None = None) -> pd.DataFrame:
'''),
(r'''    for col in ["area_total", "floor", "floor_total", "house_year", "price_kzt"]:
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors="coerce")
''', r'''    for col in ["area_total", "floor", "floor_total", "house_year", "price_kzt"] + SPATIAL_FEATURES:
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors="coerce")
        elif col in SPATIAL_FEATURES:
            d[col] = np.nan
    d["renovation"] = _renovation_flag(d["renovation"]) if "renovation" in d.columns else np.nan
'''),
(r'''def build_pipeline(model: str = "hgb") -> Pipeline:''', r'''def build_pipeline(model: str = "hgb", spatial: bool = True) -> Pipeline:'''),
(r'''    num_steps = [("impute", SimpleImputer(strategy="median"))]
''', r'''    numeric = BASE_NUMERIC + (SPATIAL_FEATURES if spatial else [])
    num_steps = [("impute", SimpleImputer(strategy="median", keep_empty_features=True))]
'''),
(r'''        ("num", Pipeline(num_steps), NUMERIC_FEATURES),''', r'''        ("num", Pipeline(num_steps), numeric),'''),
(r'''        ("bin", "passthrough", BINARY_FEATURES),''',
 r'''        ("bin", SimpleImputer(strategy="most_frequent", keep_empty_features=True), BINARY_FEATURES),'''),
(r'''    n_out = len(NUMERIC_FEATURES) + sum(''', r'''    n_out = len(numeric) + sum('''),
(r'''    d = df if "floor_ratio" in df.columns else add_features(df)
''', r'''    d = df if all(c in df.columns for c in FEATURE_COLUMNS) else add_features(df)
'''),
(r'''    metrics: dict = {"n_rows": int(n), "features": FEATURE_COLUMNS, "excluded_features": list(EXCLUDED_FEATURES)}''',
 r'''    coverage = float(d[SPATIAL_FEATURES].notna().all(axis=1).mean()) if n else 0.0
    use_spatial = coverage >= MIN_SPATIAL_COVERAGE
    used = BASE_NUMERIC + (SPATIAL_FEATURES if use_spatial else []) + CATEGORICAL_FEATURES + BINARY_FEATURES
    metrics: dict = {"n_rows": int(n), "features": used, "excluded_features": list(EXCLUDED_FEATURES),
                     "spatial_features_used": use_spatial, "spatial_coverage": coverage}'''),
(r'''    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    estimators = {name: build_pipeline(name) for name in ("ridge", "rf", "hgb", "linrf", "linhgb")}
''', r'''    if not use_spatial:
        warnings.append(f"Пространственные признаки не используются: расположение известно только у {coverage:.0%} "
                        f"объявлений (нужно не менее {MIN_SPATIAL_COVERAGE:.0%}).")
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    estimators = {name: build_pipeline(name, spatial=use_spatial) for name in ("ridge", "rf", "hgb", "linrf", "linhgb")}
'''),
]

# ----------------------------------------------------------------------------- service.py
EDITS["backend/service.py"] = [
(r'''from backend import config, metrics, storage''', r'''from backend import config, geo, metrics, storage'''),
(r'''MODEL_VERSION = "2026-10-07-1"''', r'''MODEL_VERSION = "2026-10-09-1"'''),
(r'''STALE_DATA_DAYS = 30
''', r'''STALE_DATA_DAYS = 30
SPATIAL_LABELS = {"schools_within_2km": "школы в радиусе 2 км", "hospitals_within_2km": "больницы в радиусе 2 км",
                  "kindergartens_within_2km": "детские сады в радиусе 2 км", "dist_to_metro": "расстояние до метро"}
'''),
(r'''        self.collected_to: str | None = None
''', r'''        self.collected_to: str | None = None
        self.spatial_used = False
'''),
(r'''        ids = ",".join(sorted(df["listing_id"].astype(str)))
        text = "|".join([ids, MODEL_VERSION, sklearn.__version__, ",".join(metrics.FEATURE_COLUMNS)])''',
 r'''        ids = ",".join(sorted(df["listing_id"].astype(str)))
        cols = metrics.SPATIAL_FEATURES + ["renovation"]
        if set(cols) <= set(df.columns):
            ordered = df.assign(_k=df["listing_id"].astype(str)).sort_values("_k")[cols]
            sig = hashlib.sha1(ordered.round(3).fillna(-1).to_numpy().tobytes()).hexdigest()
        else:
            sig = ""                                  # retrain when coordinates / POI / renovation flags change
        text = "|".join([ids, sig, MODEL_VERSION, sklearn.__version__, ",".join(metrics.FEATURE_COLUMNS)])'''),
(r'''        clean, report = metrics.clean_dataset_with_report(result.df)
        feats = metrics.add_features(clean)
''', r'''        clean, report = metrics.clean_dataset_with_report(result.df)
        poi = storage.load_poi()
        if poi is not None and {"latitude", "longitude"} <= set(clean.columns):
            clean = geo.add_geo_features(clean, poi)          # schools / hospitals / kindergartens / metro
        feats = metrics.add_features(clean)
'''),
(r'''            "collected_from": None, "collected_to": None,
        }
''', r'''            "collected_from": None, "collected_to": None,
            "spatial_used": bool(m.get("spatial_features_used", False)),
        }
'''),
(r'''            "area_min": 10, "area_max": 400,
        }''', r'''            "area_min": 10, "area_max": 400,
            "limits": {"dist_to_metro": {"min": 0, "max": 60}, "schools_within_2km": {"min": 0, "max": 100},
                       "hospitals_within_2km": {"min": 0, "max": 50},
                       "kindergartens_within_2km": {"min": 0, "max": 100}},
        }'''),
(r'''        return w
''', r'''        if self.spatial_used:
            missing = [label for key, label in SPATIAL_LABELS.items() if inputs.get(key) is None]
            if missing:
                w.append("Не указано: " + ", ".join(missing) + ". Использованы типичные значения.")
        return w
'''),
(r'''            "rows_used": self.rows_used,
            "warnings": self._warnings(inputs),
''', r'''            "rows_used": self.rows_used,
            "spatial_features_used": self.spatial_used,
            "warnings": self._warnings(inputs),
'''),
]

# ----------------------------------------------------------------------------- main.py
EDITS["backend/main.py"] = [
(r'''    "floor_total": "Этажей в доме", "house_year": "Год постройки", "building_type": "Тип дома",
}''', r'''    "floor_total": "Этажей в доме", "house_year": "Год постройки", "building_type": "Тип дома",
    "renovation": "Состояние отделки", "dist_to_metro": "Расстояние до метро",
    "schools_within_2km": "Школ в радиусе 2 км", "hospitals_within_2km": "Больниц в радиусе 2 км",
    "kindergartens_within_2km": "Детских садов в радиусе 2 км",
}'''),
(r'''                               examples=["монолит"])
''', r'''                               examples=["монолит"])
    renovation: int | None = Field(default=None, description="Отделка: 0 = «Черновая отделка», 1 = «Другое»",
                                   examples=[1])
    schools_within_2km: int | None = Field(default=None, ge=0, le=100, description="Школ в радиусе 2 км")
    hospitals_within_2km: int | None = Field(default=None, ge=0, le=50, description="Больниц в радиусе 2 км")
    kindergartens_within_2km: int | None = Field(default=None, ge=0, le=100,
                                                 description="Детских садов в радиусе 2 км")
    dist_to_metro: float | None = Field(default=None, ge=0, le=60, allow_inf_nan=False,
                                        description="Расстояние до ближайшей станции метро по прямой, км")

    @field_validator("renovation", mode="before")
    @classmethod
    def _renovation(cls, v):
        if v is None or v == "":
            return None
        if isinstance(v, str):
            t = v.strip().lower()
            v = int(t) if t in ("0", "1") else (0 if "черновая" in t else 1)
        if isinstance(v, bool):
            v = int(v)
        if v in (0, 1):
            return int(v)
        raise ValueError("Состояние отделки: 0 — «Черновая отделка», 1 — «Другое».")
'''),
(r'''    rows_used: int
    warnings: list[str]
''', r'''    rows_used: int
    spatial_features_used: bool = False
    warnings: list[str]
'''),
]

# ----------------------------------------------------------------------------- batch.py
ENRICH = r'''def _http_status(exc: Exception) -> int | None:
    return getattr(getattr(exc, "response", None), "status_code", None)


def run_enrich(store, s: Settings, limit: int | None = None) -> int:
    """Add latitude/longitude to dataset rows from their detail pages (one polite request per listing).

    Resumable like a normal batch: rows that already have coordinates are skipped, listings that were
    removed (404/410) or show no map are remembered in the state file and never requested again.
    """
    config.MIN_DELAY, config.MAX_DELAY = s.min_delay, s.max_delay
    location = store.prepare()
    t = Tracker()
    t.pull(store)
    rows = _parse_csv(t.csv_path.read_bytes())
    skip = set(t.state.get("no_coords", []))
    todo = [r for r in rows if not r.get("latitude") and r["listing_id"] not in skip]
    limit = min(limit or s.batch_size, config.MAX_LISTINGS)
    print(f"Storage: {location}")
    print(f"{len(rows) - len(todo)} of {len(rows)} listings have coordinates or are known to have none; "
          f"{len(todo)} to go. This run: up to {min(limit, len(todo))} detail pages, {s.min_delay:g}-{s.max_delay:g} s apart.")
    stats: dict = {}
    found = missing = 0
    reason, code = "done", 0
    try:
        for r in todo[:limit]:
            try:
                html = parser.fetch(r["url"], referer=config.SOURCE_URL, stats=stats, use_cache=False)
            except parser.BlockedError:
                raise
            except Exception as exc:                       # removed listing or a network hiccup
                if _http_status(exc) in (404, 410):
                    skip.add(r["listing_id"])
                log.warning("detail page failed for %s: %s", r["listing_id"], exc)
                missing += 1
                continue
            d = parser.parse_listing_detail(html)
            lat = parser._coord(d.get("json_lat"), *parser.LAT_RANGE)
            lon = parser._coord(d.get("json_lon"), *parser.LON_RANGE)
            if lat is None or lon is None:
                skip.add(r["listing_id"])                  # no usable map on this listing
                missing += 1
                continue
            r["latitude"], r["longitude"] = lat, lon
            found += 1
            _write_csv(t.csv_path, rows)                   # crash-safe: the file is rewritten after every hit
            if found % 10 == 0:
                t.state["no_coords"] = sorted(skip)
                t.push(store)
        reason = "limit_reached" if len(todo) > limit else "all_done"
    except parser.BlockedError as exc:
        reason, code = "blocked", 2
        print(f"\nSTOPPED: krisha.kz blocked the request ({exc}). Progress is saved; wait several hours before retrying.")
    except KeyboardInterrupt:
        reason, code = "interrupted", 130
        print("\nInterrupted by user. Saving progress ...")
    except Exception as exc:
        reason, code = "error", 1
        log.exception("Enrichment failed: %s", exc)
    finally:
        _write_csv(t.csv_path, rows)
        t.state["no_coords"] = sorted(skip)
        runs = t.state.setdefault("runs", [])
        runs.append({"at": _now(), "kind": "enrich", "added": found, "missing": missing, "reason": reason})
        t.state["runs"] = runs[-30:]
        if not t.push(store) and code == 0:
            code = 3
    have = sum(1 for r in rows if r.get("latitude"))
    print(f"\n--- Enrichment summary ---\nCoordinates found: {found} | not available: {missing} | "
          f"total with coordinates: {have}/{len(rows)} ({100 * have / max(len(rows), 1):.0f} %) | stop reason: {reason}")
    if code == 0 and reason == "limit_reached":
        print("Run  python3 -m backend.batch --enrich  again later for the next group.")
    if code == 0 and reason == "all_done":
        print("Done. Next: copy krisha_dataset.csv to data/offline_fallback/almaty_flats_fallback.csv, then "
              "python3 -m backend.geo --fetch")
    return code


def show_status(store, s: Settings) -> int:
'''
EDITS["backend/batch.py"] = [
(r'''        rows, seen = [], set()
        for r in _parse_csv(store.read(DATASET_NAME)) + _parse_csv(local_csv):
            lid = (r.get("listing_id") or "").strip()
            if lid and lid not in seen:
                seen.add(lid)
                rows.append(r)
''', r'''        rows, index = [], {}
        for r in _parse_csv(store.read(DATASET_NAME)) + _parse_csv(local_csv):
            lid = (r.get("listing_id") or "").strip()
            if not lid:
                continue
            if lid not in index:
                index[lid] = len(rows)
                rows.append(r)
            else:                                           # same listing in both copies: keep the more complete data
                kept = rows[index[lid]]
                for key, value in r.items():
                    if value not in (None, "") and kept.get(key) in (None, ""):
                        kept[key] = value
        seen = set(index)
'''),
(r'''def show_status(store, s: Settings) -> int:
''', ENRICH),
(r'''    print(f"Next start page: {t.state.get('next_page', 1)}")
''', r'''    print(f"Next start page: {t.state.get('next_page', 1)}")
    rows = _parse_csv(t.csv_path.read_bytes())
    have = sum(1 for r in rows if r.get("latitude"))
    print(f"Listings with coordinates: {have}/{len(rows)} (python3 -m backend.batch --enrich adds the rest)")
'''),
(r'''    for r in t.state.get("runs", [])[-5:]:
        print(f"  run {r['at']}: pages {r['start_page']}+{r['pages']}, added {r['added']}, {r['reason']}")
''', r'''    for r in t.state.get("runs", [])[-5:]:
        if r.get("kind") == "enrich":
            print(f"  enrich {r['at']}: coordinates added {r['added']}, unavailable {r['missing']}, {r['reason']}")
        else:
            print(f"  run {r['at']}: pages {r['start_page']}+{r['pages']}, added {r['added']}, {r['reason']}")
'''),
(r'''    ap.add_argument("--backend", choices=["drive", "local"])
''', r'''    ap.add_argument("--backend", choices=["drive", "local"])
    ap.add_argument("--enrich", action="store_true", help="add latitude/longitude from detail pages (resumable)")
    ap.add_argument("--limit", type=int, help="max detail pages in an --enrich run (default BATCH_SIZE)")
'''),
(r'''        return show_status(store, s) if args.status else run_batch(store, s, args.start_page, args.details)
''', r'''        if args.status:
            return show_status(store, s)
        if args.enrich:
            return run_enrich(store, s, args.limit)
        return run_batch(store, s, args.start_page, args.details)
'''),
]

# ----------------------------------------------------------------------------- tests/test_metrics.py
EDITS["tests/test_metrics.py"] = [
(r'''def test_renovation_is_not_a_feature(trained):
    pipe, m = trained
    assert "renovation" not in metrics.FEATURE_COLUMNS and "rooms" not in metrics.FEATURE_COLUMNS
    assert "renovation" in m["excluded_features"]
    inputs = dict(district="Бостандыкский", area_total=54.0, floor=5, floor_total=9, house_year=2012,
                  building_type="монолит")
    a = metrics.predict_with_range(pipe, dict(inputs))
    b = metrics.predict_with_range(pipe, dict(inputs, renovation="свежий ремонт"))   # extra key is ignored
    assert a == b''', r'''def test_renovation_is_a_binary_feature_and_rooms_are_not_used(trained):
    pipe, m = trained
    assert "renovation" in metrics.FEATURE_COLUMNS and "rooms" not in metrics.FEATURE_COLUMNS
    assert m["excluded_features"] == ["rooms"] and "renovation" in m["features"]
    inputs = dict(district="Бостандыкский", area_total=54.0, floor=5, floor_total=9, house_year=2012,
                  building_type="монолит")
    a = metrics.predict_with_range(pipe, dict(inputs))
    assert a == metrics.predict_with_range(pipe, dict(inputs, rooms=3))              # rooms are ignored
    for flag in (0, 1, "Черновая отделка", "свежий ремонт"):
        assert metrics.predict_with_range(pipe, dict(inputs, renovation=flag))[0] > 0'''),
]


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
            if new in text:
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
            shutil.copy2(path, path.with_name(path.name + ".before_spatial"))
            path.write_text(text, encoding="utf-8")
            py_compile.compile(str(path), doraise=True)
            changed += 1
        print(f"{rel}: {applied} edits applied, {skipped} already present")
    print("Done." if changed else "Everything was already up to date.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
