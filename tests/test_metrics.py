"""Tests for cleaning, features, model and metrics.

The synthetic frame below exists ONLY to unit-test the code. The product itself
never trains on synthetic data: real-data checks use tests/fixtures/krisha_dataset.csv
(or data/collected/) and are skipped when that file is absent.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backend import config, metrics

ROOT = Path(__file__).resolve().parent.parent
BASE_PPM2 = dict(zip(config.DISTRICTS, [650_000, 950_000, 800_000, 1_050_000, 750_000, 1_250_000, 700_000, 600_000]))


def synthetic_frame(n=400, seed=0):
    """Test-only data with a known structure (price ~ district x area x age x noise)."""
    rng = np.random.default_rng(seed)
    dist = rng.choice(config.DISTRICTS, n)
    area = rng.uniform(25, 140, n).round(1)
    total = rng.integers(3, 25, n)
    floor = np.minimum(rng.integers(1, 26, n), total)
    year = rng.integers(1960, 2025, n)
    btype = rng.choice(config.BUILDING_TYPES, n)
    ppm2 = np.array([BASE_PPM2[x] for x in dist]) * (1 + (year - 1990) / 300) * rng.lognormal(0, 0.08, n)
    return pd.DataFrame({
        "listing_id": [str(100_000 + i) for i in range(n)], "price_kzt": (ppm2 * area).round(-3),
        "district": dist, "area_total": area, "floor": floor, "floor_total": total, "house_year": year,
        "building_type": btype, "renovation": rng.choice(config.RENOVATION_STATES, n),
        "rooms": (area // 25 + 1).astype(int), "is_new_build": False, "source": "krisha",
    })


# ------------------------------- evaluate -------------------------------
def test_evaluate_hand_computed():
    m = metrics.evaluate([100, 200, 300], [110, 190, 330])
    assert m["mae"] == pytest.approx(50 / 3)
    assert m["rmse"] == pytest.approx(np.sqrt(1100 / 3))
    assert m["mape"] == pytest.approx((0.10 + 0.05 + 0.10) / 3 * 100)
    assert m["r2"] == pytest.approx(1 - 1100 / 20000)


# ------------------------------- cleaning -------------------------------
def _row(**kw):
    base = dict(listing_id="1", price_kzt=50_000_000, district="Бостандыкский", area_total=50.0, floor=3,
                floor_total=9, house_year=2010, building_type="монолит", renovation="не указано", is_new_build=False)
    base.update(kw)
    return base


def test_clean_dataset_rules():
    rows = [_row(listing_id=str(i), price_kzt=50_000_000 + i * 100_000, area_total=50.0 + i) for i in range(1, 41)]
    rows += [
        _row(listing_id="1"),                                              # duplicate id of row 1
        _row(listing_id="d1", price_kzt=None),                              # missing price
        _row(listing_id="d2", district=None),                               # missing district
        _row(listing_id="d3", area_total=5.0),                              # area < 10
        _row(listing_id="d4", area_total=450.0),                            # area > 400
        _row(listing_id="d5", floor=12, floor_total=9),                     # floor > floor_total
        _row(listing_id="d6", price_kzt=0),                                 # bad price
        _row(listing_id="d7", is_new_build=True),                           # new build
        _row(listing_id="d8", price_kzt=900_000_000, area_total=50.0),      # extreme price per m2
        _row(listing_id="d9", house_year=1850),                             # bad year: kept, year -> NaN
    ]
    clean, rep = metrics.clean_dataset_with_report(pd.DataFrame(rows), ref_year=2026)
    ids = set(clean["listing_id"])
    assert not ({"d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8"} & ids)
    assert clean["listing_id"].is_unique
    assert "d9" in ids and clean.loc[clean.listing_id == "d9", "house_year"].isna().all()
    assert rep["dropped_duplicate_id"] == 1 and rep["dropped_new_build"] == 1
    assert rep["dropped_bad_floor"] == 1 and rep["house_year_set_missing"] == 1
    assert rep["rows_in"] - sum(v for k, v in rep.items() if k.startswith("dropped_")) == rep["rows_out"]
    assert metrics.clean_dataset(pd.DataFrame(rows), ref_year=2026).shape == clean.shape


def test_repost_duplicates_dropped():
    rows = [_row(listing_id="a"), _row(listing_id="b")]                     # same flat posted twice
    assert len(metrics.clean_dataset(pd.DataFrame(rows))) == 1


# ------------------------------- features -------------------------------
def test_add_features():
    df = pd.DataFrame([_row(floor=1, floor_total=9, house_year=2016),
                       _row(floor=9, floor_total=9, house_year=1990),
                       _row(floor=4, floor_total=8, house_year=2030),
                       _row(floor=None, floor_total=None, house_year=None)])
    f = metrics.add_features(df, ref_year=2026)
    assert f["floor_ratio"].iloc[:3].tolist() == pytest.approx([1 / 9, 1.0, 0.5])
    assert f["is_first_floor"].tolist() == [1, 0, 0, 0]
    assert f["is_last_floor"].tolist() == [0, 1, 0, 0]
    assert f["house_age"].iloc[:3].tolist() == [10, 36, 0]                  # future year clipped to 0
    assert f["floor_ratio"].iloc[3] != f["floor_ratio"].iloc[3]             # NaN stays NaN
    assert f["price_per_m2"].iloc[0] == pytest.approx(1_000_000)


# --------------------------------- model ---------------------------------
@pytest.fixture(scope="module")
def trained():
    df = metrics.clean_dataset(synthetic_frame())
    return metrics.train_and_evaluate(df)


def test_renovation_is_a_binary_feature_and_rooms_are_not_used(trained):
    pipe, m = trained
    assert "renovation" in metrics.FEATURE_COLUMNS and "rooms" not in metrics.FEATURE_COLUMNS
    assert m["excluded_features"] == ["rooms"] and "renovation" in m["features"]
    inputs = dict(district="Бостандыкский", area_total=54.0, floor=5, floor_total=9, house_year=2012,
                  building_type="монолит")
    a = metrics.predict_with_range(pipe, dict(inputs))
    assert a == metrics.predict_with_range(pipe, dict(inputs, rooms=3))              # rooms are ignored
    for flag in (0, 1, "Черновая отделка", "свежий ремонт"):
        assert metrics.predict_with_range(pipe, dict(inputs, renovation=flag))[0] > 0


def test_prediction_positive_ordered_and_range(trained):
    pipe, m = trained
    est, low, high = metrics.predict_with_range(pipe, dict(district="Медеуский", area_total=60.0, floor=3,
                                                          floor_total=9, house_year=2005, building_type="кирпич"))
    assert 0 < low <= est <= high
    assert 0.6 <= m["range_coverage_oof"] <= 0.95
    assert m["chosen_model"] in ("rf", "hgb", "linrf", "linhgb") and m["n_test"] > 0


@pytest.mark.parametrize("model_name", ["ridge", "rf", "hgb", "linrf", "linhgb"])
def test_price_never_decreases_with_area(model_name):
    df = metrics.add_features(metrics.clean_dataset(synthetic_frame(seed=3)))
    pipe = metrics.build_pipeline(model_name).fit(df[metrics.FEATURE_COLUMNS], df["price_kzt"])
    for dist in config.DISTRICTS:
        for btype in config.BUILDING_TYPES + [None]:
            base = dict(district=dist, floor=4, floor_total=9, house_year=2008, building_type=btype)
            frame = pd.DataFrame([dict(base, area_total=a) for a in np.arange(15, 300, 2.5)])
            preds = pipe.predict(metrics.make_features(frame))
            assert np.all(np.diff(preds) >= -1e-6), (model_name, dist, btype)


def test_stacked_model_drops_negative_area_effect():
    """If the linear stage would give log(area) a negative coefficient it is removed."""
    rng = np.random.default_rng(1)
    X = np.column_stack([rng.uniform(20, 100, 200), rng.integers(0, 3, 200)])
    y = 15 - 0.02 * X[:, 0] + rng.normal(0, 0.01, 200)                      # price falls with area
    m = metrics.LinearPlusTrees(metrics.HistGradientBoostingRegressor(max_iter=20, monotonic_cst=[1, 0]),
                                area_idx=0).fit(X, y)
    assert m.use_area_ is False
    grid = np.column_stack([np.arange(20, 100, 1.0), np.ones(80)])
    assert np.all(np.diff(m.predict(grid)) >= -1e-9)


def test_model_beats_naive_baseline_on_structured_data(trained):
    _, m = trained
    assert m["beats_baseline"]
    assert m["cv"][m["chosen_model"]]["mape"] < m["cv"]["naive"]["mape"]
    assert m["r2"] > 0.8


def test_small_dataset_falls_back_to_baseline():
    small = metrics.clean_dataset(synthetic_frame(n=60))
    model, m = metrics.train_and_evaluate(small)
    assert isinstance(model, metrics.DistrictMedianBaseline) and m["chosen_model"] == "naive"
    assert any("Недостаточно данных" in w for w in m["warnings"])
    est, low, high = metrics.predict_with_range(model, dict(district="Алмалинский", area_total=50.0, floor=2,
                                                           floor_total=5, house_year=1980, building_type="панель"))
    assert 0 < low <= est <= high


def test_unknown_values_do_not_crash(trained):
    pipe, _ = trained
    est, _, _ = metrics.predict_with_range(pipe, dict(district="Неизвестный", area_total=50.0, floor=2,
                                                      floor_total=5, house_year=None, building_type=None))
    assert est > 0


def test_district_stats():
    s = metrics.district_stats(metrics.clean_dataset(synthetic_frame()))
    assert set(s) == set(config.DISTRICTS)
    assert all(v["median_price_per_m2_kzt"] > 0 and v["count"] > 0 for v in s.values())


# ------------------------- real data (skipped if absent) -------------------------
def _real_csv():
    for p in (ROOT / "tests" / "fixtures" / "krisha_dataset.csv", ROOT / "data" / "collected" / "krisha_dataset.csv"):
        if p.exists():
            return p
    return None


def test_real_dataset_pipeline_runs_and_reports():
    path = _real_csv()
    if path is None:
        pytest.skip("real krisha_dataset.csv not available")
    raw = pd.read_csv(path, encoding="utf-8-sig")
    clean, rep = metrics.clean_dataset_with_report(raw)
    assert rep["rows_out"] >= metrics.MIN_ROWS
    assert (raw["source"] == "krisha").all()                    # 100 % real rows
    model, m = metrics.train_and_evaluate(clean)
    assert m["mape"] < 100 and np.isfinite(m["r2"])
    # The model must beat the naive district baseline; if it does not on this data, the
    # model reports it via m["warnings"] instead of hiding it.
    assert m["beats_baseline"] or any("не превосходит" in w for w in m["warnings"])
