"""Cleaning, feature engineering, model and evaluation metrics (no file I/O, no HTTP).

Design decisions
----------------
* Target: ``log(price_kzt)``. Flat prices are right-skewed and errors are
  proportional, so a log target makes errors relative and the model never
  predicts negative prices. Predictions are converted back with ``exp``.
* ``renovation`` enters only as a binary flag: 0 = "Черновая отделка" (shell), 1 = anything else. The
  free-text field itself is too noisy. ``rooms`` is not used: it is collinear with ``area_total``.
* The main models (RandomForest / HistGradientBoosting, plain or stacked on a log-linear
  stage) are fitted with a
  monotonic constraint on ``area_total``: for otherwise identical inputs a
  larger area can never give a lower price.
* Price range = 10th-90th percentile of out-of-fold log residuals.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from backend import config

RANDOM_STATE = 42
MIN_ROWS = 100                       # below this the naive district baseline is used
MISSING_CAT = "не указано"

NUMERIC_FEATURES = ["area_total", "floor", "floor_total", "floor_ratio", "house_age"]
CATEGORICAL_FEATURES = ["district", "building_type"]
BINARY_FEATURES = ["is_first_floor", "is_last_floor", "renovation"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES
EXCLUDED_FEATURES = ("rooms",)
AREA_COL_INDEX = NUMERIC_FEATURES.index("area_total")     # numeric block comes first in the transformed matrix

_REQUIRED_RAW = ["price_kzt", "area_total", "district"]
_NUMERIC_RAW = ["price_kzt", "area_total", "floor", "floor_total", "house_year"]


# --------------------------------------------------------------------------
# Cleaning
# --------------------------------------------------------------------------
def clean_dataset_with_report(df: pd.DataFrame, ref_year: int | None = None,
                              min_group: int = 30) -> tuple[pd.DataFrame, dict]:
    """Apply the section 3 cleaning rules and return (clean_df, report of dropped rows)."""
    ref_year = ref_year or datetime.now().year
    d = df.copy()
    rep: dict = {"rows_in": int(len(d))}
    for col in _NUMERIC_RAW:
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors="coerce")

    def drop(mask: pd.Series, key: str) -> None:
        nonlocal d
        mask = mask.reindex(d.index).fillna(False).astype(bool)
        rep[key] = int(mask.sum())
        d = d[~mask]

    if "is_new_build" in d.columns:
        drop(d["is_new_build"].astype(str).str.lower().isin(["true", "1"]), "dropped_new_build")
    if "listing_id" in d.columns:
        drop(d["listing_id"].duplicated(keep="first"), "dropped_duplicate_id")
    drop(d[_REQUIRED_RAW].isna().any(axis=1) | (d["district"].astype(str).str.strip() == ""), "dropped_missing_required")
    drop(d["price_kzt"] <= 0, "dropped_bad_price")
    drop((d["area_total"] < 10) | (d["area_total"] > 400), "dropped_bad_area")
    drop((d["floor"] < 1) | (d["floor"] > d["floor_total"]), "dropped_bad_floor")

    bad_year = d["house_year"].notna() & ((d["house_year"] < 1950) | (d["house_year"] > ref_year + 5))
    rep["house_year_set_missing"] = int(bad_year.sum())              # kept, year -> missing
    d.loc[bad_year, "house_year"] = np.nan

    key = [c for c in ["price_kzt", "area_total", "floor", "floor_total", "district", "house_year"] if c in d.columns]
    drop(d.duplicated(subset=key, keep="first"), "dropped_repost_duplicate")

    ppm2 = d["price_kzt"] / d["area_total"]
    g_lo, g_hi = ppm2.quantile(0.01), ppm2.quantile(0.99)
    lo = pd.Series(g_lo, index=d.index)
    hi = pd.Series(g_hi, index=d.index)
    big = d["district"].value_counts()
    for dist in big[big >= min_group].index:                         # own percentiles only for big districts
        m = d["district"] == dist
        lo[m], hi[m] = ppm2[m].quantile(0.01), ppm2[m].quantile(0.99)
    drop((ppm2 < lo) | (ppm2 > hi), "dropped_price_per_m2_outlier")
    rep["districts_with_own_percentiles"] = sorted(big[big >= min_group].index.tolist())

    rep["rows_out"] = int(len(d))
    return d.reset_index(drop=True), rep


def clean_dataset(df: pd.DataFrame, ref_year: int | None = None) -> pd.DataFrame:
    """Clean a raw listings frame (see ``clean_dataset_with_report`` for the rules)."""
    return clean_dataset_with_report(df, ref_year)[0]


# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------
def _renovation_flag(series: pd.Series) -> pd.Series:
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
    """Add floor_ratio, is_first_floor, is_last_floor, house_age (and price_per_m2 if price exists)."""
    ref_year = ref_year or datetime.now().year
    d = df.copy()
    for col in ["area_total", "floor", "floor_total", "house_year", "price_kzt"]:
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors="coerce")
    d["renovation"] = _renovation_flag(d["renovation"]) if "renovation" in d.columns else np.nan
    d["floor_ratio"] = (d["floor"] / d["floor_total"]).where(d["floor_total"] > 0)
    d["is_first_floor"] = (d["floor"] == 1).astype(int)
    d["is_last_floor"] = ((d["floor"] == d["floor_total"]) & d["floor_total"].notna()).astype(int)
    d["house_age"] = (ref_year - d["house_year"]).clip(lower=0)
    if "price_kzt" in d.columns:
        d["price_per_m2"] = d["price_kzt"] / d["area_total"]
    return d


def make_features(inputs: dict | pd.DataFrame, ref_year: int | None = None) -> pd.DataFrame:
    """Turn user inputs (district, area_total, floor, floor_total, house_year, building_type) into model features."""
    raw = inputs.copy() if isinstance(inputs, pd.DataFrame) else pd.DataFrame([inputs])
    return add_features(raw, ref_year)[FEATURE_COLUMNS]


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
class DistrictMedianBaseline(RegressorMixin, BaseEstimator):
    """Naive baseline: district median price per m2 x area (global median for unseen districts)."""

    def fit(self, X, y):
        ppm2 = np.asarray(y, float) / X["area_total"].to_numpy(float)
        frame = pd.DataFrame({"district": X["district"].to_numpy(), "ppm2": ppm2})
        self.medians_ = frame.groupby("district")["ppm2"].median().to_dict()
        self.global_median_ = float(np.median(ppm2))
        return self

    def predict(self, X):
        ppm2 = X["district"].map(self.medians_).fillna(self.global_median_)
        return (ppm2 * X["area_total"]).to_numpy(float)


class LinearPlusTrees(RegressorMixin, BaseEstimator):
    """Log-linear stage (with log(area)) + a tree model fitted on its residuals.

    Prices scale almost proportionally with area, which trees alone only learn
    step by step. The linear stage captures that scaling; the trees add
    interactions (district x age, floors, ...). Operates on the preprocessed
    matrix, where column ``area_idx`` is the raw area. Price stays non-decreasing
    in area: the linear coefficient of log(area) is kept >= 0 and the residual
    trees carry a monotonic constraint on area.
    """

    def __init__(self, trees=None, alpha: float = 3.0, area_idx: int = AREA_COL_INDEX):
        self.trees = trees
        self.alpha = alpha
        self.area_idx = area_idx

    def _z(self, X, with_area: bool = True):
        X = np.asarray(X, float)
        z = np.delete(X, self.area_idx, axis=1)
        if with_area:
            z = np.column_stack([z, np.log(np.clip(X[:, self.area_idx], 1.0, None))])
        return z

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y, float)
        self.use_area_ = True
        self.linear_ = make_pipeline(StandardScaler(), Ridge(alpha=self.alpha)).fit(self._z(X), y)
        if self.linear_[-1].coef_[-1] < 0:                         # never let area lower the price
            self.use_area_ = False
            self.linear_ = make_pipeline(StandardScaler(), Ridge(alpha=self.alpha)).fit(self._z(X, False), y)
        self.trees_ = clone(self.trees).fit(X, y - self.linear_.predict(self._z(X, self.use_area_)))
        return self

    def predict(self, X):
        X = np.asarray(X, float)
        return self.linear_.predict(self._z(X, self.use_area_)) + self.trees_.predict(X)


def build_baseline() -> DistrictMedianBaseline:
    return DistrictMedianBaseline()


def build_pipeline(model: str = "hgb") -> Pipeline:
    """Preprocessing + regressor on a log-price target. model: ridge | rf | hgb | linhgb | linrf."""
    categories = [config.DISTRICTS + [MISSING_CAT], config.BUILDING_TYPES + [MISSING_CAT]]
    numeric = NUMERIC_FEATURES
    num_steps = [("impute", SimpleImputer(strategy="median", keep_empty_features=True))]
    pre = ColumnTransformer([
        ("num", Pipeline(num_steps), numeric),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="constant", fill_value=MISSING_CAT)),
                          ("onehot", OneHotEncoder(categories=categories, handle_unknown="ignore",
                                                   sparse_output=False))]), CATEGORICAL_FEATURES),
        ("bin", SimpleImputer(strategy="most_frequent", keep_empty_features=True), BINARY_FEATURES),
    ])
    n_out = len(numeric) + sum(len(c) for c in categories) + len(BINARY_FEATURES)
    cst = [0] * n_out
    cst[AREA_COL_INDEX] = 1                                           # price non-decreasing in area
    def hgb(max_iter=300, depth=4):
        return HistGradientBoostingRegressor(max_iter=max_iter, learning_rate=0.05, max_depth=depth, min_samples_leaf=10,
                                             l2_regularization=1.0, monotonic_cst=cst, random_state=RANDOM_STATE)

    def rf(leaf=2):
        return RandomForestRegressor(n_estimators=400, min_samples_leaf=leaf, max_features=0.7, monotonic_cst=cst,
                                     random_state=RANDOM_STATE, n_jobs=-1)

    if model == "ridge":                                              # log-linear baseline (log(area) inside)
        reg = LinearPlusTrees(DummyRegressor(strategy="constant", constant=0.0))
    elif model == "rf":
        reg = rf()
    elif model == "hgb":
        reg = hgb()
    elif model == "linhgb":                                           # log-linear stage + boosted residuals
        reg = LinearPlusTrees(hgb(max_iter=150, depth=3))
    elif model == "linrf":
        reg = LinearPlusTrees(rf(leaf=5))
    else:
        raise ValueError(f"unknown model {model!r}")
    return Pipeline([("prep", pre),
                     ("model", TransformedTargetRegressor(regressor=reg, func=np.log, inverse_func=np.exp))])


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
def evaluate(y_true, y_pred) -> dict:
    """MAE, RMSE (KZT), MAPE (percent) and R2."""
    yt, yp = np.asarray(y_true, float), np.asarray(y_pred, float)
    err = yp - yt
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mape": float(np.mean(np.abs(err) / np.abs(yt)) * 100.0),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
    }


def train_and_evaluate(df: pd.DataFrame) -> tuple[Pipeline | DistrictMedianBaseline, dict]:
    """Compare naive / ridge / random forest / gradient boosting and return (final model, metrics).

    ``df`` is a cleaned listings frame. Holdout = 80/20 split (random_state=42);
    5-fold out-of-fold predictions on all rows give the CV metrics and the price
    range. The returned model is refitted on all rows. Fewer than MIN_ROWS rows:
    the naive baseline is returned and a warning is added.
    """
    d = df if all(c in df.columns for c in FEATURE_COLUMNS) else add_features(df)
    d = d.dropna(subset=_REQUIRED_RAW)
    X, y = d[FEATURE_COLUMNS], d["price_kzt"].astype(float)
    n = len(d)
    warnings: list[str] = []
    metrics: dict = {"n_rows": int(n), "features": list(FEATURE_COLUMNS),
                     "excluded_features": list(EXCLUDED_FEATURES)}

    if n < MIN_ROWS:
        base = build_baseline().fit(X, y)
        pred = base.predict(X)
        q10, q90 = np.quantile(np.log(y / pred), [0.1, 0.9])
        base.residual_quantiles_ = (float(q10), float(q90))
        warnings.append(f"Недостаточно данных ({n} < {MIN_ROWS} объявлений): используется простая оценка "
                        "по медиане цены за м² в районе.")
        metrics.update({"chosen_model": "naive", "in_sample": evaluate(y, pred), "warnings": warnings,
                        "beats_baseline": False, **{k: v for k, v in evaluate(y, pred).items()}})
        return base, metrics

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    estimators = {name: build_pipeline(name) for name in ("ridge", "rf", "hgb", "linrf", "linhgb")}
    holdout = {"naive": evaluate(y_te, build_baseline().fit(X_tr, y_tr).predict(X_te))}
    fitted = {}
    for name, est in estimators.items():
        fitted[name] = clone(est).fit(X_tr, y_tr)
        holdout[name] = evaluate(y_te, fitted[name].predict(X_te))

    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof = {"naive": cross_val_predict(build_baseline(), X, y, cv=kf)}
    for name, est in estimators.items():
        oof[name] = cross_val_predict(clone(est), X, y, cv=kf)
    cv = {name: evaluate(y, p) for name, p in oof.items()}

    chosen = min(("rf", "hgb", "linrf", "linhgb"), key=lambda m: cv[m]["mape"])
    log_res = np.log(y.to_numpy() / oof[chosen])
    q10, q90 = (float(v) for v in np.quantile(log_res, [0.1, 0.9]))
    lo, hi = oof[chosen] * np.exp(q10), oof[chosen] * np.exp(q90)
    coverage = float(np.mean((y.to_numpy() >= lo) & (y.to_numpy() <= hi)))

    imp = permutation_importance(fitted[chosen], X_te, y_te, n_repeats=20, random_state=RANDOM_STATE,
                                 scoring="neg_mean_absolute_error")
    importance = dict(sorted(zip(X_te.columns, (float(v) for v in imp.importances_mean)), key=lambda kv: -kv[1]))

    final = clone(estimators[chosen]).fit(X, y)
    final.residual_quantiles_ = (q10, q90)
    beats = bool(cv[chosen]["mape"] < cv["naive"]["mape"] and holdout[chosen]["mape"] < holdout["naive"]["mape"])
    if not beats:
        warnings.append("Модель не превосходит наивную оценку по медиане района; доверяйте результату осторожно.")
    metrics.update({
        "n_train": int(len(X_tr)), "n_test": int(len(X_te)), "chosen_model": chosen,
        "holdout": holdout, "cv": cv, **holdout[chosen],           # mae / rmse / mape / r2 of the chosen model
        "beats_baseline": beats, "residual_log_q10": q10, "residual_log_q90": q90,
        "range_coverage_oof": coverage, "permutation_importance_mae": importance, "warnings": warnings,
    })
    return final, metrics


def predict_with_range(pipeline, features: dict | pd.DataFrame, ref_year: int | None = None) -> tuple[float, float, float]:
    """Return (estimate, low, high) in KZT for one flat."""
    X = features if isinstance(features, pd.DataFrame) and "floor_ratio" in features.columns \
        else make_features(features, ref_year)
    est = float(pipeline.predict(X[FEATURE_COLUMNS])[0])
    q10, q90 = getattr(pipeline, "residual_quantiles_", (float(np.log(0.85)), float(np.log(1.15))))
    return est, est * float(np.exp(min(q10, 0.0))), est * float(np.exp(max(q90, 0.0)))


def district_stats(df: pd.DataFrame) -> dict:
    """Per district: listing count, median price and median price per m2 (KZT)."""
    d = df if "price_per_m2" in df.columns else add_features(df)
    out = {}
    for dist, g in d.groupby("district"):
        out[str(dist)] = {"count": int(len(g)),
                          "median_price_kzt": float(g["price_kzt"].median()),
                          "median_price_per_m2_kzt": float(g["price_per_m2"].median())}
    return out
