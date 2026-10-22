# Almaty Flat Price Calculator

A local web app that estimates the **asking price of a flat in Almaty** from real listings collected on krisha.kz.
Enter district, area, floor, floors in the house, build year, building type, the finish (shell or finished); get an estimate in KZT, a price range, the price per m², a comparison with the district median and the five most similar listings. Runs on `localhost` only; no cloud, no accounts. The UI is in Russian.

> The estimate is indicative, based on asking prices, and is not an official valuation. Current cross-validation metrics are available at `/api/stats`.

## Quick start

Requires **Python 3.11 or newer** (the pinned pandas 3 / NumPy 2.4 / scikit-learn 1.8 do not run on 3.10).

```bash
cd almaty-flat-calculator
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
test -f .env || cp .env.example .env  # preserve your existing settings
USE_LIVE_PARSER=false python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000. The server answers at once; the model trains in the background on the first start (usually tens of seconds, depending on the computer) and is saved to `data/cache/model.joblib`, so later starts take a few seconds. While it trains, `/api/health` shows `training` and the form retries automatically.

It works **without internet and without scraping**, as long as `data/offline_fallback/almaty_flats_fallback.csv` exists (a copy of the collected `krisha_dataset.csv`). `--reload` restarts when Python files change; bump `MODEL_VERSION` in `backend/service.py` after editing the model code.

## How it works

```
krisha.kz -> parser.py -> data/raw (HTML cache) -> storage.py -> dataset
dataset = offline_fallback CSV + data/cache/live_listings.csv (rows added by refresh)
dataset -> metrics.py (clean, features, model) -> service.py -> main.py (FastAPI) -> frontend/
batch.py -> krisha_dataset.csv + visited_urls.txt + batch_state.json (Google Drive folder or local folder)
```

| File | Responsibility |
|---|---|
| `backend/config.py` | paths, `.env` variables, 10 User-Agent strings, domain constants |
| `backend/parser.py` | polite fetching and HTML parsing of krisha.kz (cards and optional basic detail attributes) |
| `backend/storage.py` | raw cache, CSV/model persistence, dataset loading and fallback order |
| `backend/metrics.py` | cleaning, features, models, MAE/RMSE/MAPE/R² |
| `backend/service.py` | train once, predict, comparables, stats, refresh job |
| `backend/main.py` | FastAPI routes, validation (Russian messages), static files |
| `backend/batch.py`, `backend/drive_store.py` | resumable batch collector, Drive/local storage |
| `frontend/` | `index.html`, `style.css`, `app.js` (plain HTML/CSS/JS, no build step) |

Dataset order: processed cache (younger than `DATA_TTL_HOURS` and matching current fallback/live CSV hashes) -> optional live parse -> fallback + live CSV union -> raw HTML recovery only when stored CSVs have no rows. Old processed caches without source hashes are ignored. The API always reports `data_source` (`live`, `raw_cache`, `offline_fallback`) and `rows_used`.

## Data

Application datasets contain real collected listings. Synthetic examples are used only in automated tests. Listings are collected in resumable batches:

```bash
python3 -m backend.batch --status      # progress; no krisha.kz requests
python3 -m backend.batch               # one batch (BATCH_SIZE listings, up to BATCH_MAX_PAGES pages), then exit
```

Each run skips every listing already in `visited_urls.txt`, appends new ones to `krisha_dataset.csv`, saves after every page, and stops at once on a block. With `STORE_BACKEND=local` the files live in `LOCAL_STORE_DIR` (e.g. a Google Drive for desktop folder). `STORE_BACKEND=drive` uses the Drive API instead; the full setup is in `BATCH_RUNBOOK.md`.

Fields: `listing_id, price_kzt, district, area_total, floor, floor_total, house_year, building_type, renovation, rooms, latitude, longitude, url, scraped_at, is_new_build, source`. New-build cards ("от X ₸" prices) are excluded. Seller names and phone numbers are never read.

**Refreshing from the web UI/API:** with `USE_LIVE_PARSER=true`, the "Обновить данные" button (or `POST /api/refresh`) collects new listings in the background (same limits and 8-12 s delays as the batch collector), stores them in `data/cache/live_listings.csv`, retrains and swaps the model without interrupting predictions. At most one run per 10 minutes. A block, error or failed retrain keeps the previous model. Collected batches and web refreshes remain separate stores; there is no automatic sync command.

Current change report and startup instructions: `../reports/CHANGES_2026-10-13.md`. Current architecture: `../reports/PROJECT_MAP_2026-10-13.md`.

## Changes on 2026-10-13

Location inputs, coordinate enrichment, OpenStreetMap/POI processing and spatial imputations have been removed. District remains a required model feature. The renovation radio remains active: **0 = «Черновая отделка», 1 = «Другое»**. Legacy latitude/longitude CSV columns are retained only to avoid losing historical data; the parser and model do not use them.

The model version and dataset fingerprint changed. Restarting automatically retrains an old cached model. `backend.geo` and `batch --enrich` are retired; old files are kept only in `archive/removed_location_2026-10-13/`. Do not run old `apply_*` migration scripts: they describe older versions and may restore removed functionality.

Work only in **project for forecasting/almaty-flat-calculator**. The `.env` collection path now resolves to this application's `data/collected`, not the other project. No folder rename is needed.

## API

| Method and path | Purpose |
|---|---|
| `GET /` , `/static/...` | web interface |
| `GET /api/health` | `status` (`ok`, `training`, `not_ready`, `error`), `data_source`, `rows`, `model_trained_at` |
| `GET /api/options` | districts, building types, year and area limits for the form |
| `POST /api/predict` | estimate (below) |
| `GET /api/stats?district=...` | district medians, model metrics, cleaning report |
| `POST /api/refresh` | start a background refresh: 202, 403 (live parsing disabled), 409 (running), 429 (rate limit) |
| `GET /api/refresh/status` | running flag, next allowed time, last result |
| `GET /docs` | interactive Swagger UI |

Request fields: required `district, area_total, floor, floor_total, house_year, building_type`; optional `renovation` (0 = «Черновая отделка», 1 = «Другое»; labels and existing textual repair values are accepted too). Missing renovation is mode-imputed inside the model. Unknown extra request fields are ignored for compatibility; they cannot affect predictions.

```bash
curl -s -X POST http://127.0.0.1:8000/api/predict -H 'Content-Type: application/json' \
  -d '{"district":"Бостандыкский","area_total":54,"floor":5,"floor_total":9,"house_year":2012,"building_type":"монолит","renovation":1}'
```

The response has `price_estimate_kzt`, `price_low_kzt`, `price_high_kzt` (rounded to 100 000), `price_per_m2_kzt`, `district_median_price_per_m2_kzt`, `vs_district_median_pct`, five `comparables`, `model_metrics`, `data_source`, `data_collected_from/to`, `rows_used` and Russian `warnings`. Validation errors are 422 with a Russian `message`. URL-encode Cyrillic query values (`curl -G --data-urlencode "district=Медеуский" .../api/stats`).

## Web interface

Plain HTML/CSS/JS served by FastAPI (no Streamlit, no build step). The form has the six main fields, a **radio for the finish** («Черновая отделка» / «Другое»), and the result block (price, range bar, price per m², comparison with the district median, comparables table, warnings). The **«Обновить данные»** button starts `POST /api/refresh`, shows a spinner and the live status from `GET /api/refresh/status`, and displays the result message (or why it is not allowed). It continues to show a running refresh after a page reload.

## Model

* Target `log(price)`. Features: area, floor, floors in the house, floor ratio, house age, district, building type, first-floor and last-floor flags, the **binary finish flag** (0 = «Черновая отделка», 1 = anything else, including "не указано"). `rooms` is not used (collinear with area).
* Candidates: naive district baseline (district median price per m² × area), log-linear baseline, RandomForest, HistGradientBoosting, and two stacked models (log-linear stage with log(area) + trees on the residuals). The best by 5-fold CV is used. Price never decreases when area grows (monotonic constraint, tested).
* Range = 10th-90th percentile of out-of-fold errors (about x0.78 ... x1.32, 80 % coverage).
* Fewer than 100 clean rows: the naive baseline is used and the response says so.

### Accuracy

Historical results (468 cleaned listings, renovation excluded) were CV MAPE 16.2 % and R² 0.81. Those numbers describe a different model and must not be presented as the current result. The current model uses renovation and no micro-location features; `/api/stats` reports its actual metrics. The small, newest-first sample and asking prices limit how well it represents completed market transactions.

## Tests

```bash
python -m pytest -q
```

Tests never touch the network and redirect runtime writes away from your real `data/cache` and `data/raw` (a session fixture redirects them to temporary folders), so results do not depend on what you ran before. They use saved pages in `tests/fixtures/`, a fake Drive API and test-only synthetic frames (used solely to unit-test code, never for training). The real-data test is skipped when `krisha_dataset.csv` is absent.

## Legal, ethics and safety

* **robots.txt** (checked 2026-10-02): `/prodazha/kvartiry/almaty/`, `?page=N` and `/a/show/<id>` are allowed, no crawl-delay declared. The parser refuses disallowed paths and parameters (`/ajax/`, `/a/show-map/`, `raion=`, `sub_cat`, `utm_source`, `till=`, ...).
* **Terms of use:** krisha.kz's terms restrict automated collection. This project is for **personal, non-commercial, educational use**, at the operator's own responsibility; being a student/academic project does not grant permission. Keep collections small and slow (8-12 s between requests, 5 pages and 200 listings per run at most, 10 minutes between refreshes), keep the collected data **private** and do not redistribute it (see `.gitignore`).
* On 403/429/468/captcha everything stops and the saved data is used; there is no proxy rotation, captcha solving or any other evasion, and there never will be.
* No personal data (phones, names) is collected. `.env`, `credentials.json` and `token.json` must never be committed or uploaded.
* The UI always shows: "Оценка ориентировочная и не является официальной оценкой стоимости недвижимости."

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `/api/predict` returns 503 "Модель обучается" | first start; wait usually tens of seconds, depending on the computer (the web form retries by itself) |
| `/api/refresh` returns 403 | set `USE_LIVE_PARSER=true` in `.env` and restart |
| Refresh finds "Новых объявлений не найдено" | normal: krisha lists newest first, so a recent batch leaves little new |
| `ModuleNotFoundError: googleapiclient` | only for `STORE_BACKEND=drive`; use `local` or install the optional google-* packages |
| Everything reports `offline_fallback` | normal until a refresh adds rows; the data is real, just not collected today |

## Known limitations

About 500 asking prices, one city, two collection days; year and building type are missing for about 28 % of listings (imputed); micro-location is not used; Жетысуский has only about 26 listings; flats built before 1950 are out of range.

## Manual data update (no sync command)

The current Drive snapshot has 500 batch rows and 637 live-store rows; these counts include duplicates/rows subsequently removed by cleaning. Its canonical fallback CSV contains only a header. The live store currently permits offline startup, but for an independent baseline copy the real batch file manually while the server is stopped:

```bash
cp data/collected/krisha_dataset.csv data/offline_fallback/almaty_flats_fallback.csv
```

Restart the server after copying the batch file: source-content hashes invalidate the processed cache automatically and changed training data invalidate the model cache. No cache deletion is required. Keep `live_listings.csv`, raw pages and batch state. This operation does not merge stores or mark live IDs as visited. See `BATCH_RUNBOOK.md` for adding batches.

## Data loading fix (2026-10-13)

Old raw HTML could replace a larger stored dataset. The observed 540-row raw cache cleaned to 509 training rows, while the existing 637-row live store cleaned to 600. Stored CSVs now take priority over raw HTML. Batch output is still imported manually: stop the server, copy data/collected/krisha_dataset.csv to data/offline_fallback/almaty_flats_fallback.csv, and restart. The running in-memory model does not watch file changes. The UI count reports rows after cleaning. See ../reports/DATA_LOADING_FIX_2026-10-13.md.
