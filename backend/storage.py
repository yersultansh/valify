"""File I/O: raw HTML cache, listing CSV files, dataset loading with fallback.

This module never performs HTTP requests itself. The only network path is the
lazy ``parser.collect`` call inside ``load_dataset`` (live mode, opt-in).
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import joblib
import pandas as pd

from backend import config

log = logging.getLogger(__name__)

# Legacy coordinate columns are retained for lossless CSV compatibility only; never used by the model.
COLUMNS = [
    "listing_id", "price_kzt", "district", "area_total", "floor", "floor_total",
    "house_year", "building_type", "renovation", "rooms", "latitude", "longitude", "url", "scraped_at",
    "is_new_build", "source",
]
PROCESSED_CSV = config.CACHE_DIR / "dataset.csv"
PROCESSED_META = config.CACHE_DIR / "dataset.meta.json"
MODEL_PATH = config.CACHE_DIR / "model.joblib"
MIN_USABLE_ROWS = 100            # smaller datasets never replace a bigger stored one


# --------------------------------------------------------------------------
# Raw HTML cache (data/raw/<sha1(url)>.html + .meta.json)
# --------------------------------------------------------------------------
def _raw_paths(url: str, raw_dir: Path | None = None) -> tuple[Path, Path]:
    raw_dir = raw_dir or config.RAW_DIR
    key = hashlib.sha1(url.encode("utf-8")).hexdigest()
    return raw_dir / f"{key}.html", raw_dir / f"{key}.meta.json"


def read_raw_cache(url: str, ttl_hours: float | None = None, raw_dir: Path | None = None) -> str | None:
    """Return cached HTML for ``url`` if present and younger than the TTL."""
    ttl = config.CACHE_TTL_HOURS if ttl_hours is None else ttl_hours
    html_path, meta_path = _raw_paths(url, raw_dir)
    if not (html_path.exists() and meta_path.exists()):
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        fetched = datetime.fromisoformat(meta["fetched_at"])
    except (ValueError, KeyError, OSError):
        return None
    if datetime.now(timezone.utc) - fetched > timedelta(hours=ttl):
        return None
    return html_path.read_text(encoding="utf-8")


def write_raw_cache(url: str, html: str, status_code: int, raw_dir: Path | None = None) -> Path:
    """Save a fetched page plus its small sidecar (url, status_code, fetched_at)."""
    html_path, meta_path = _raw_paths(url, raw_dir)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(html, encoding="utf-8")
    meta = {"url": url, "status_code": status_code,
            "fetched_at": datetime.now(timezone.utc).isoformat()}
    meta_path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    return html_path


def iter_cached_listing_pages(raw_dir: Path | None = None):
    """Yield (url, html) for every cached listing page (any age)."""
    raw_dir = raw_dir or config.RAW_DIR
    for meta_path in sorted(raw_dir.glob("*.meta.json")):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            continue
        url = meta.get("url", "")
        if not url.startswith(config.SOURCE_URL) or meta.get("status_code") != 200:
            continue
        html_path = meta_path.with_name(meta_path.name.replace(".meta.json", ".html"))
        if html_path.exists():
            yield url, html_path.read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# Listing tables
# --------------------------------------------------------------------------
def rows_to_frame(rows: list[dict]) -> pd.DataFrame:
    """Build a DataFrame with the canonical column order."""
    df = pd.DataFrame(rows)
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = None
    return df[COLUMNS]


def save_listings(rows: list[dict], path: Path) -> Path:
    """Write normalized listings to CSV (UTF-8 with BOM so Excel shows Cyrillic)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows_to_frame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    return path


def read_listings(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def missing_report(df: pd.DataFrame) -> dict[str, float]:
    """Percent of missing values per modelling field."""
    fields = ["price_kzt", "district", "area_total", "floor", "floor_total",
              "house_year", "building_type", "renovation", "rooms"]
    out = {}
    for f in fields:
        s = df[f] if f in df.columns else pd.Series(dtype=object)
        miss = s.isna().sum() + (s == "не указано").sum()
        out[f] = round(float(100.0 * miss / max(len(df), 1)), 1)
    return out


# --------------------------------------------------------------------------
# Dataset loading with precedence (PROJECT_INSTRUCTIONS.md, section 7)
# --------------------------------------------------------------------------
@dataclass
class DatasetResult:
    df: pd.DataFrame
    data_source: str            # live | raw_cache | offline_fallback
    rows: int
    warnings: list[str] = field(default_factory=list)


def _stored_signature() -> dict:
    """Invalidate processed data when either authoritative CSV changes.

    The version also retires caches built with the former raw-HTML priority.
    Only file content is inspected; source datasets are never modified.
    """
    return {
        "version": 2,
        "fallback": hashlib.sha256(config.FALLBACK_CSV.read_bytes()).hexdigest()
        if config.FALLBACK_CSV.exists() else None,
        "live": hashlib.sha256(_live_path().read_bytes()).hexdigest()
        if _live_path().exists() else None,
    }


def _save_processed(df: pd.DataFrame, data_source: str) -> None:
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_CSV, index=False, encoding="utf-8-sig")
    PROCESSED_META.write_text(json.dumps({
        "data_source": data_source,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "rows": int(len(df)),
        "stored_signature": _stored_signature(),
    }), encoding="utf-8")


def _load_processed() -> DatasetResult | None:
    if not (PROCESSED_CSV.exists() and PROCESSED_META.exists()):
        return None
    try:
        meta = json.loads(PROCESSED_META.read_text(encoding="utf-8"))
        saved = datetime.fromisoformat(meta["saved_at"])
        if meta.get("stored_signature") != _stored_signature():
            return None
    except (ValueError, KeyError, OSError):
        return None
    if datetime.now(timezone.utc) - saved > timedelta(hours=config.DATA_TTL_HOURS):
        return None
    df = read_listings(PROCESSED_CSV)
    if len(df) < MIN_USABLE_ROWS:
        return None
    return DatasetResult(df, meta.get("data_source", "raw_cache"), len(df))


def _from_raw_cache() -> list[dict]:
    from backend import parser  # lazy: parser imports storage
    rows: dict[str, dict] = {}
    for url, html in iter_cached_listing_pages():
        for card in parser.parse_listing_cards(html):
            norm = parser.normalize_listing(card)
            if norm["is_new_build"]:
                continue
            rows.setdefault(norm["listing_id"], norm)
    return list(rows.values())


def _read_fallback() -> pd.DataFrame | None:
    if not config.FALLBACK_CSV.exists():
        return None
    df = read_listings(config.FALLBACK_CSV)
    if df.empty:
        return None
    df["listing_id"] = df["listing_id"].astype(str)
    return df


def _merge_new(base: pd.DataFrame | None, rows: list[dict]) -> tuple[pd.DataFrame, int]:
    """Add listings that are not in ``base`` yet (by listing_id). Returns (merged, number added)."""
    new = rows_to_frame(rows)
    new["listing_id"] = new["listing_id"].astype(str)
    if base is None or base.empty:
        return new, len(new)
    fresh = new[~new["listing_id"].isin(set(base["listing_id"]))]
    return pd.concat([base, fresh], ignore_index=True), len(fresh)


def _live_path() -> Path:
    return config.CACHE_DIR / "live_listings.csv"


def _refresh_state_path() -> Path:
    return config.CACHE_DIR / "refresh_state.json"


def read_live_store() -> pd.DataFrame | None:
    """Listings collected by earlier refreshes (data/cache/live_listings.csv), or None."""
    path = _live_path()
    if not path.exists():
        return None
    df = read_listings(path)
    if df.empty:
        return None
    df["listing_id"] = df["listing_id"].astype(str)
    return df


def _read_stored() -> pd.DataFrame | None:
    """Collected dataset = offline fallback CSV + everything refreshes added (union by listing_id)."""
    parts = [d for d in (_read_fallback(), read_live_store()) if d is not None]
    if not parts:
        return None
    return pd.concat(parts, ignore_index=True).drop_duplicates("listing_id", keep="first")


def append_live_rows(rows: list[dict]) -> int:
    """Store listings that are new (not in fallback or live store). Returns how many were added.

    Also drops the processed-data cache so the next load sees the new rows.
    """
    if not rows:
        return 0
    stored = _read_stored()
    new = rows_to_frame(rows)
    new["listing_id"] = new["listing_id"].astype(str)
    new = new.drop_duplicates("listing_id")
    if stored is not None:
        new = new[~new["listing_id"].isin(set(stored["listing_id"]))]
    if new.empty:
        return 0
    live = read_live_store()
    out = new if live is None else pd.concat([live, new], ignore_index=True)
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(_live_path(), index=False, encoding="utf-8-sig")
    for path in (PROCESSED_CSV, PROCESSED_META):
        path.unlink(missing_ok=True)
    return int(len(new))


def load_refresh_state() -> dict:
    """Persisted refresh bookkeeping (last start/finish, last result); {} if none."""
    path = _refresh_state_path()
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except (ValueError, OSError):
        return {}


def save_refresh_state(state: dict) -> None:
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _refresh_state_path().write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")








def load_dataset(use_live: bool | None = None) -> DatasetResult:
    """Return the best available dataset.

    Order: valid processed cache -> optional live parse -> stored fallback/live
    CSV union -> raw HTML recovery only when no stored rows are available.
    Processed caches must match the current source CSV content signatures.
    Batch files are still promoted to fallback manually by the user.
    """
    use_live = config.USE_LIVE_PARSER if use_live is None else use_live
    warnings: list[str] = []

    cached = _load_processed()
    if cached is not None:
        log.info("Using processed cache (%d rows, origin=%s)", cached.rows, cached.data_source)
        return cached

    base = _read_stored()

    if use_live:
        try:
            from backend import parser  # lazy: parser imports storage
            stats: dict = {}
            rows = parser.collect(stats=stats)
            if rows and not stats.get("blocked"):
                added = append_live_rows(rows)
                df = _read_stored()
                _save_processed(df, "live")
                log.info("Live parse: %d new listings merged into %d stored rows", added, len(df) - added)
                return DatasetResult(df, "live", len(df), warnings)
            warnings.append("Живой парсинг не дал данных или был заблокирован; используются сохранённые данные.")
        except Exception as exc:  # never crash the server because of the parser
            log.exception("Live parsing failed: %s", exc)
            warnings.append("Ошибка живого парсинга; используются сохранённые данные.")

    if base is not None:
        if "source" in base.columns and (base["source"] == "synthetic").any():
            n = int((base["source"] == "synthetic").sum())
            msg = f"Используются демонстрационные данные ({n} синтетических строк)."
            log.warning(msg)
            warnings.append(msg)
        source = "live" if read_live_store() is not None else "offline_fallback"
        return DatasetResult(base, source, len(base), warnings)

    raw_rows = _from_raw_cache()
    if len(raw_rows) >= MIN_USABLE_ROWS:
        df = rows_to_frame(raw_rows)
        _save_processed(df, "raw_cache")
        return DatasetResult(df, "raw_cache", len(df), warnings)

    if raw_rows:                                    # last resort: a small cache is better than nothing
        warnings.append(f"Мало данных ({len(raw_rows)} объявлений): оценка будет грубой.")
        return DatasetResult(rows_to_frame(raw_rows), "raw_cache", len(raw_rows), warnings)

    raise FileNotFoundError(f"No data available: fallback CSV missing at {config.FALLBACK_CSV}")


# --------------------------------------------------------------------------
# Trained model persistence (joblib)
# --------------------------------------------------------------------------
def save_model(payload: dict, path: Path | None = None) -> Path:
    """Persist {model, metrics, fingerprint, trained_at} to data/cache/model.joblib."""
    path = path or MODEL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, path)
    return path


def load_model(path: Path | None = None) -> dict | None:
    """Load a persisted model payload; None if missing or unreadable (caller retrains)."""
    path = path or MODEL_PATH
    if not path.exists():
        return None
    try:
        payload = joblib.load(path)
        return payload if isinstance(payload, dict) else None
    except Exception as exc:                          # corrupt file, incompatible library version, ...
        log.warning("Could not load saved model (%s); it will be retrained.", exc)
        return None
