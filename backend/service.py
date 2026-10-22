"""Business logic: load data, train once, predict, find comparables, stats.

No FastAPI imports (see PROJECT_INSTRUCTIONS.md, section 4). ``main.py`` only
validates input and calls ``get_service()``.

* The model trains lazily on first use (or via ``warmup()``) and is cached in
  memory; a joblib copy in data/cache/ is reused while the data and the code
  version are unchanged.
* ``renovation`` is a binary input (shell / other); ``rooms`` is excluded.
"""
from __future__ import annotations

import hashlib
import inspect
import logging
import math
import os
import threading
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import sklearn

from backend import config, metrics, storage

log = logging.getLogger(__name__)

MODEL_VERSION = "2026-10-13-no-location-v1"        # bump when features / models change: invalidates the saved model
COMPARABLES_K = 5
AREA_TOLERANCE = 0.20                 # comparables: area within +-20 %
SMALL_DISTRICT_ROWS = 30
STALE_DATA_DAYS = 30
REQUIRED_INPUTS = ("district", "area_total", "floor", "floor_total", "house_year", "building_type")

# Comparables distance weights (lower distance = more similar)
W_AREA, W_FLOOR_RATIO, W_YEAR, W_BTYPE, W_DISTRICT = 2.0, 0.5, 1.0, 0.5, 1.0
REFRESH_MIN_INTERVAL_S = 600          # /api/refresh: at most one run per 10 minutes
REFRESH_CACHE_TTL_HOURS = 0.1         # refresh always reads fresh pages (cache <= 6 min)
NEUTRAL_MISSING = 0.4                 # penalty fraction when a value is unknown on either side


class ServiceError(RuntimeError):
    """Raised when data or model are unavailable."""


def _py(value):
    """numpy / NaN -> plain JSON-safe Python value."""
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _round(x: float, step: int) -> int:
    return int(round(x / step) * step)


# --------------------------------------------------------------------------
# Comparables (pure function, easy to test)
# --------------------------------------------------------------------------
def find_comparables(df: pd.DataFrame, inputs: dict, k: int = COMPARABLES_K) -> list[dict]:
    """Up to ``k`` most similar listings.

    Pool: same district with area within +-20 %; if that gives fewer than k, the
    whole city (same area window); if still fewer, the area window is widened
    step by step so the user always sees k examples when the data has them.
    Distance = weighted area difference, floor-ratio, build-year and building-type
    differences, plus a penalty for another district.
    """
    area = float(inputs["area_total"])
    district = inputs.get("district")
    d = df.copy()
    rel_area = (d["area_total"] - area).abs() / area

    pool = d[(d["district"] == district) & (rel_area <= AREA_TOLERANCE)]
    if len(pool) < k:
        pool = d[rel_area <= AREA_TOLERANCE]
    tol = AREA_TOLERANCE
    while len(pool) < k and tol < 1.0:
        tol += 0.2
        pool = d[rel_area <= tol]
    if pool.empty:
        return []

    floor, total = inputs.get("floor"), inputs.get("floor_total")
    ratio = floor / total if floor and total else np.nan
    year, btype = inputs.get("house_year"), inputs.get("building_type")

    dist = W_AREA * (pool["area_total"] - area).abs() / area
    r = (pool["floor_ratio"] - ratio).abs() if not np.isnan(ratio) else pd.Series(np.nan, index=pool.index)
    dist = dist + W_FLOOR_RATIO * r.fillna(NEUTRAL_MISSING)
    y = (pool["house_year"] - year).abs() / 30.0 if year else pd.Series(np.nan, index=pool.index)
    dist = dist + W_YEAR * y.clip(upper=1.0).fillna(NEUTRAL_MISSING)
    if btype:
        mism = (pool["building_type"] != btype).astype(float)
        mism = mism.where(pool["building_type"].notna(), NEUTRAL_MISSING)
    else:
        mism = pd.Series(NEUTRAL_MISSING, index=pool.index)
    dist = dist + W_BTYPE * mism
    dist = dist + W_DISTRICT * (pool["district"] != district).astype(float)

    best = pool.assign(_dist=dist).sort_values("_dist").head(k)
    out = []
    for _, row in best.iterrows():
        out.append({
            "url": row.get("url"),
            "price_kzt": _py(int(row["price_kzt"])),
            "area_total": _py(row["area_total"]),
            "floor": _py(None if pd.isna(row["floor"]) else int(row["floor"])),
            "floor_total": _py(None if pd.isna(row["floor_total"]) else int(row["floor_total"])),
            "house_year": _py(None if pd.isna(row["house_year"]) else int(row["house_year"])),
            "district": row["district"],
            "building_type": None if pd.isna(row["building_type"]) else row["building_type"],
        })
    return out


# --------------------------------------------------------------------------
# Service
# --------------------------------------------------------------------------
class PriceService:
    def __init__(self, loader=None, use_saved_model: bool = True, collector=None,
                 live_enabled: bool | None = None, clock=None):
        self._loader = loader or storage.load_dataset
        self._collector = collector or self._collect_live       # callable(stats) -> list[dict]
        self._live_enabled = live_enabled                        # None -> config.USE_LIVE_PARSER at call time
        self._clock = clock or time.time
        self._refresh_lock = threading.Lock()
        self._refresh_running = False
        self._stale = False                                      # data stored but model not rebuilt yet
        self._use_saved_model = use_saved_model
        self._lock = threading.RLock()
        self._ready = False
        self._error: str | None = None
        self.model = None
        self.metrics: dict = {}
        self.data: pd.DataFrame | None = None
        self.data_source = "unknown"
        self.rows_used = 0
        self.trained_at: str | None = None
        self.cleaning_report: dict = {}
        self.base_warnings: list[str] = []
        self.district_stats: dict = {}
        self.area_bounds = (10.0, 400.0)
        self.collected_from: str | None = None
        self.collected_to: str | None = None

    # ---- lifecycle -----------------------------------------------------
    @staticmethod
    def _fingerprint(df: pd.DataFrame) -> str:
        # Hash every training input and target; legacy extra CSV columns are ignored.
        columns = ["listing_id", "price_kzt"] + metrics.FEATURE_COLUMNS
        ordered = df[columns].assign(listing_id=df["listing_id"].astype(str)).sort_values("listing_id")
        values = ordered.to_json(orient="split", index=False, double_precision=15)
        text = "|".join([values, MODEL_VERSION, sklearn.__version__])
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def ensure_ready(self, force: bool = False) -> None:
        """Load data and train (or load the saved model) once; thread-safe."""
        with self._lock:
            if self._ready and not force:
                return
            try:
                self._apply(self._build())
                self._error = None
                self._ready = True
            except Exception as exc:
                self._ready = False
                self._error = str(exc)
                log.exception("Service could not start: %s", exc)
                raise ServiceError(str(exc)) from exc

    warmup = ensure_ready

    def _call_loader(self, use_live: bool | None):
        """Call the data loader; ``use_live=False`` forbids network access (used by the refresh rebuild)."""
        if use_live is not None and "use_live" in inspect.signature(self._loader).parameters:
            return self._loader(use_live=use_live)
        return self._loader()

    def _build(self, use_live: bool | None = None) -> dict:
        """Load data, clean, train (or load the saved model). Returns the new state without touching self."""
        result = self._call_loader(use_live)
        clean, report = metrics.clean_dataset_with_report(result.df)
        feats = metrics.add_features(clean)
        fingerprint = self._fingerprint(feats)

        saved = storage.load_model() if self._use_saved_model else None
        if saved and saved.get("fingerprint") == fingerprint:
            model, m, trained_at = saved["model"], saved["metrics"], saved["trained_at"]
            log.info("Loaded saved model (%s)", m.get("chosen_model"))
        else:
            model, m = metrics.train_and_evaluate(feats)
            trained_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
            try:
                storage.save_model({"model": model, "metrics": m, "fingerprint": fingerprint,
                                    "trained_at": trained_at})
            except OSError as exc:
                log.warning("Could not save the trained model: %s", exc)

        state = {
            "model": model, "metrics": m, "trained_at": trained_at, "data": feats, "cleaning_report": report,
            "data_source": result.data_source, "rows_used": int(len(feats)),
            "base_warnings": list(dict.fromkeys(list(result.warnings) + list(m.get("warnings", [])))),
            "district_stats": metrics.district_stats(feats),
            "area_bounds": (float(feats["area_total"].quantile(0.01)), float(feats["area_total"].quantile(0.99))),
            "collected_from": None, "collected_to": None,
        }
        if "scraped_at" in feats.columns:
            ts = pd.to_datetime(feats["scraped_at"], errors="coerce", utc=True).dropna()
            if len(ts):
                state["collected_from"], state["collected_to"] = ts.min().date().isoformat(), ts.max().date().isoformat()
        return state


    def _apply(self, state: dict) -> None:
        """Swap in a freshly built state; predictions (which hold the lock) never see a half-updated one."""
        with self._lock:
            for key, value in state.items():
                setattr(self, key, value)
            self._ready, self._error, self._stale = True, None, False

    # ---- public API ----------------------------------------------------
    def options(self) -> dict:
        return {
            "districts": list(config.DISTRICTS),
            "building_types": list(config.BUILDING_TYPES),
            "year_min": 1950, "year_max": datetime.now().year + 5,
            "area_min": 10, "area_max": 400,

        }

    def health(self) -> dict:
        """Cheap status; never triggers training."""
        if self._ready:
            status = "ok"
        else:
            status = "error" if self._error else "not_ready"
        return {"status": status, "data_source": self.data_source, "rows": self.rows_used,
                "model_trained_at": self.trained_at, "error": self._error}

    def _model_metrics(self) -> dict:
        m = self.metrics
        chosen = m.get("chosen_model", "naive")
        if "cv" in m:
            src, label = m["cv"][chosen], "5-fold CV"
        else:
            src, label = m, "in-sample (small dataset)"
        return {"mae": _py(src["mae"]), "rmse": _py(src["rmse"]), "mape": _py(src["mape"]), "r2": _py(src["r2"]),
                "evaluation": label, "model": chosen,
                "naive_mape": _py(m["cv"]["naive"]["mape"]) if "cv" in m else None,
                "beats_baseline": bool(m.get("beats_baseline", False))}

    def _warnings(self, inputs: dict) -> list[str]:
        w = list(self.base_warnings)
        if self.data_source != "live":
            span = f" (собраны {self.collected_from} – {self.collected_to})" if self.collected_from else ""
            w.append(f"Используются сохранённые данные{span}, а не свежий парсинг.")
        if self.collected_to:
            age = (datetime.now(timezone.utc).date() - datetime.fromisoformat(self.collected_to).date()).days
            if age > STALE_DATA_DAYS:
                w.append(f"Данные собраны более {STALE_DATA_DAYS} дней назад: оценка может быть устаревшей.")
        n = self.district_stats.get(inputs["district"], {}).get("count", 0)
        if n < SMALL_DISTRICT_ROWS:
            w.append(f"В районе «{inputs['district']}» мало объявлений ({n}): оценка менее точна.")
        lo, hi = self.area_bounds
        if not (lo <= float(inputs["area_total"]) <= hi):
            w.append(f"Площадь вне основного диапазона данных ({lo:.0f}–{hi:.0f} м²): оценка менее надёжна.")
        return w

    def predict(self, inputs: dict) -> dict:
        """Estimate for one flat. ``inputs``: district, area_total, floor, floor_total, house_year, building_type."""
        self.ensure_ready()
        with self._lock:
            return self._predict_locked(inputs)

    def _predict_locked(self, inputs: dict) -> dict:
        missing = [k for k in REQUIRED_INPUTS if k not in inputs]
        if missing:
            raise ValueError(f"missing inputs: {', '.join(missing)}")
        if inputs["district"] not in config.DISTRICTS:
            raise ValueError(f"unknown district: {inputs['district']}")
        est, low, high = metrics.predict_with_range(self.model, inputs)
        area = float(inputs["area_total"])
        ppm2 = est / area
        median_ppm2 = self.district_stats.get(inputs["district"], {}).get("median_price_per_m2_kzt")
        diff = (ppm2 / median_ppm2 - 1.0) * 100.0 if median_ppm2 else None
        return {
            "price_estimate_kzt": _round(est, 100_000),
            "price_low_kzt": _round(low, 100_000),
            "price_high_kzt": _round(high, 100_000),
            "price_per_m2_kzt": _round(ppm2, 1_000),
            "district_median_price_per_m2_kzt": _round(median_ppm2, 1_000) if median_ppm2 else None,
            "vs_district_median_pct": round(diff, 1) if diff is not None else None,
            "comparables": find_comparables(self.data, inputs),
            "model_metrics": self._model_metrics(),
            "data_source": self.data_source,
            "data_collected_from": self.collected_from,
            "data_collected_to": self.collected_to,
            "rows_used": self.rows_used,
            "warnings": self._warnings(inputs),
        }

    def stats(self, district: str | None = None) -> dict:
        self.ensure_ready()
        with self._lock:
            return self._stats_locked(district)

    def _stats_locked(self, district: str | None) -> dict:
        if district is not None and district not in self.district_stats:
            raise KeyError(district)
        m = self.metrics
        out = {
            "district": district,
            "stats": self.district_stats[district] if district else None,
            "all_districts": None if district else self.district_stats,
            "model_metrics": self._model_metrics(),
            "model_details": {k: m.get(k) for k in ("chosen_model", "holdout", "cv", "range_coverage_oof",
                                                    "permutation_importance_mae", "features", "excluded_features",
                                                    "n_train", "n_test")},
            "cleaning_report": self.cleaning_report,
            "data_source": self.data_source, "rows_used": self.rows_used,
            "data_collected_from": self.collected_from, "data_collected_to": self.collected_to,
        }
        return out

    # ---- refresh (live collection + retrain) ----------------------------
    @staticmethod
    def _collect_live(stats: dict) -> list[dict]:
        """One polite collection run: configured page/listing limits, batch-style 8-12 s delays, fresh pages."""
        from backend import parser
        saved = (config.MIN_DELAY, config.MAX_DELAY, config.CACHE_TTL_HOURS)
        lo, hi = sorted((float(os.getenv("BATCH_MIN_DELAY", "8.0")), float(os.getenv("BATCH_MAX_DELAY", "12.0"))))
        config.MIN_DELAY, config.MAX_DELAY, config.CACHE_TTL_HOURS = lo, hi, REFRESH_CACHE_TTL_HOURS
        try:
            return parser.collect(stats=stats)
        finally:
            config.MIN_DELAY, config.MAX_DELAY, config.CACHE_TTL_HOURS = saved

    def _enabled(self) -> bool:
        return config.USE_LIVE_PARSER if self._live_enabled is None else self._live_enabled

    def start_refresh(self) -> dict:
        """Start a background refresh if allowed. Outcome: started | running | rate_limited | disabled."""
        with self._refresh_lock:
            if not self._enabled():
                return {"outcome": "disabled"}
            if self._refresh_running:
                return {"outcome": "running"}
            state, now = storage.load_refresh_state(), self._clock()
            last = state.get("last_started_at")
            if last is not None and 0 <= now - last < REFRESH_MIN_INTERVAL_S:
                return {"outcome": "rate_limited", "retry_after": math.ceil(REFRESH_MIN_INTERVAL_S - (now - last))}
            state["last_started_at"] = now          # counted even if the run fails or is blocked
            storage.save_refresh_state(state)
            self._refresh_running = True
        threading.Thread(target=self.run_refresh, name="data-refresh", daemon=True).start()
        return {"outcome": "started"}

    def run_refresh(self) -> dict:
        """Collect new listings, store them, retrain, swap the model. Never raises; keeps the old model on failure."""
        try:
            result = self._do_refresh()
        except Exception as exc:
            log.exception("Refresh failed: %s", exc)
            result = {"status": "error", "added": 0,
                      "message": f"Не удалось обновить данные ({type(exc).__name__}). Используются прежние данные."}
        with self._refresh_lock:
            state = storage.load_refresh_state()
            state["last_finished_at"] = self._clock()
            state["last_result"] = result
            storage.save_refresh_state(state)
            self._refresh_running = False           # only after the result is saved: pollers never see 'idle' early
        return result

    def _do_refresh(self) -> dict:
        stats: dict = {}
        rows = self._collector(stats)
        blocked = bool(stats.get("blocked"))
        added = storage.append_live_rows(rows) if rows else 0
        block_note = " Сайт ограничил доступ (403/429/captcha): сбор остановлен, повторите позже."
        if added == 0 and not self._stale:
            if blocked:
                return {"status": "blocked", "added": 0, "collected": len(rows),
                        "message": "Сайт ограничил доступ (403/429/captcha). Сбор остановлен; используются прежние данные."}
            return {"status": "no_new", "added": 0, "collected": len(rows),
                    "message": "Новых объявлений не найдено; данные и модель не изменились."}
        self._stale = True                           # data changed; stays set if the rebuild below fails
        state = self._build(use_live=False)          # the collection already happened: no second crawl
        self._apply(state)
        msg = f"Добавлено новых объявлений: {added}. Модель переобучена на {self.rows_used} объявлениях."
        return {"status": "ok", "added": added, "collected": len(rows), "blocked": blocked,
                "rows_used": self.rows_used, "model_trained_at": self.trained_at,
                "message": msg + (block_note if blocked else "")}

    def refresh_status(self) -> dict:
        with self._refresh_lock:
            return self._refresh_status_locked()

    def _refresh_status_locked(self) -> dict:
        state, now = storage.load_refresh_state(), self._clock()
        last = state.get("last_started_at")
        wait = 0 if last is None or not 0 <= now - last < REFRESH_MIN_INTERVAL_S \
            else math.ceil(REFRESH_MIN_INTERVAL_S - (now - last))
        iso = lambda t: datetime.fromtimestamp(t, timezone.utc).isoformat(timespec="seconds") if t else None
        return {"enabled": self._enabled(), "running": self._refresh_running,
                "min_interval_seconds": REFRESH_MIN_INTERVAL_S, "next_allowed_in_seconds": wait,
                "last_started_at": iso(last), "last_finished_at": iso(state.get("last_finished_at")),
                "last_result": state.get("last_result")}


_service: PriceService | None = None
_service_lock = threading.Lock()


def get_service() -> PriceService:
    """Process-wide singleton used by the API."""
    global _service
    with _service_lock:
        if _service is None:
            _service = PriceService()
        return _service
