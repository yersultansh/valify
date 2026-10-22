# PROJECT_INSTRUCTIONS.md — Almaty Flat Price Calculator

> **Read this file fully before writing any code.** It is the single source of truth for the project.
> The user gives short instructions ("build the parser", "add a district filter"); you (the AI agent) generate, run and fix the code to match this spec.

---

## 1. Role of the AI agent

You are a **code assistant ("vibe-coding" tool)**. The user describes what they want in plain language; you produce working, tested code.

Rules of behaviour:

1. **Generate complete files**, not fragments, unless the user asks for a patch.
2. **Follow the directory structure in section 4 exactly.** Do not rename or add top-level folders without asking.
3. **Run what you write** (tests, the server, the parser in offline mode) and fix errors before reporting "done".
4. **Never invent selectors or data.** Inspect real HTML from krisha.kz (saved to `data/raw/`) before writing parsing code. If a field cannot be found, store `None` and log a warning.
5. **Small, reviewable steps.** After each milestone (section 12) summarize what changed, how to run it, and what is next.
6. **Ask only when blocked.** If a requirement is ambiguous, pick the most reasonable option, state the assumption in one line, and continue.
7. Code comments and docstrings in **English**. UI labels in **Russian** (the audience is in Almaty), with English variable names.
8. Do not add dependencies that are not listed in section 5 without telling the user why.
9. **All files (code and outputs) are stored in Google Drive, folder `project for forecasting`** — see section 13.

---

## 2. Goal

A local web application that estimates the **sale price of a flat in Almaty** based on real listings from krisha.kz.

The user enters flat parameters in an HTML form → the backend returns:

- an **estimated price in KZT** and a **price range** (low / high),
- **price per m²** for that flat,
- a comparison with the **district median** and a list of the **most similar listings** (comparables),
- data freshness info (live / cached / offline fallback).

Runs entirely on **localhost**. No cloud hosting, no authentication.

---

## 3. Data model (listing fields)

Every listing is normalized to this schema (`backend/storage.py` + `backend/parser.py`):

| Field | Type | Description (RU) | Notes / normalization |
|---|---|---|---|
| `listing_id` | str | ID объявления | From krisha URL, used for de-duplication |
| `price_kzt` | int | Цена в тенге | Strip spaces, `〒`, `₸`; reject ≤ 0 |
| `district` | str | Район | Canonical names, e.g. `Алмалинский`, `Бостандыкский`, `Ауэзовский`, `Медеуский`, `Турксибский`, `Жетысуский`, `Алатауский`, `Наурызбайский`. Strip the word "р-н" |
| `area_total` | float | Общая площадь, м² | Parse "52.4 м²" → 52.4 |
| `floor` | int | Этаж | From "3 из 9" → 3 |
| `floor_total` | int | Этажность дома | From "3 из 9" → 9 |
| `house_year` | int | Год постройки | Valid range 1950–current year + 5 |
| `building_type` | str | Тип дома | Canonical: `монолит`, `кирпич`, `панель`, `иное` |
| `renovation` | str | Состояние/ремонт | Canonical set, see below |
| `rooms` | int (optional) | Число комнат | Stored for reference, excluded from model |
| `url` | str | Ссылка на объявление | |
| `scraped_at` | datetime | Когда собрано | UTC ISO-8601 |

Canonical `renovation` values (map krisha wording to these):
`свежий ремонт`, `хорошее`, `среднее`, `требует ремонта`, `черновая отделка`, `не указано`.

Derived field: `price_per_m2 = price_kzt / area_total`.
Derived features for the model: `floor_ratio = floor / floor_total`, `is_first_floor`, `is_last_floor`, `house_age`.

### Data cleaning rules (`backend/metrics.py` or `service.py`)

- Drop rows missing `price_kzt`, `area_total` or `district`.
- Drop obvious outliers: `price_per_m2` outside the 1st–99th percentile **per district**, area < 10 or > 400 m², `floor > floor_total`.
- De-duplicate by `listing_id`.
- Impute missing numeric fields with medians and missing categorical/binary fields with their modes only inside the model pipeline, never overwrite raw data.

---

## 4. Project structure (mandatory)

```
almaty-flat-calculator/
├── PROJECT_INSTRUCTIONS.md      # this file
├── README.md                    # how to install & run (user-facing)
├── .env                         # local secrets/config (NOT committed)
├── .env.example                 # template with all variables, no secrets
├── .gitignore
├── requirements.txt
├── backend/
│   ├── __init__.py
│   ├── config.py                # loads .env, constants, paths
│   ├── parser.py                # krisha.kz scraping + HTML parsing
│   ├── metrics.py               # cleaning, features, model, evaluation metrics
│   ├── service.py               # business logic: load data, train, predict, comparables
│   ├── storage.py               # read/write raw cache, CSV/Parquet, fallback loading
│   └── main.py                  # FastAPI app, routes, static frontend serving
├── frontend/
│   ├── index.html               # form + results
│   ├── style.css
│   └── app.js                   # fetch() to the API, render results
├── data/
│   ├── raw/                     # raw HTML/JSON cache from krisha.kz
│   │   └── .gitkeep
│   ├── cache/                   # processed dataset, trained model
│   │   └── .gitkeep
│   └── offline_fallback/
│       └── almaty_flats_fallback.csv   # pre-collected dataset
└── tests/
    ├── __init__.py
    └── test_metrics.py
```

Note: the files must be named `__init__.py` (double underscores). You may add more test files (`test_parser.py`, `test_service.py`) inside `tests/`.

Responsibilities:

| File | Responsibility | Must NOT contain |
|---|---|---|
| `config.py` | Paths, env variables, defaults | Business logic |
| `parser.py` | HTTP fetching, HTML → dict parsing | Model code |
| `storage.py` | File I/O, caching, fallback logic | HTTP requests |
| `metrics.py` | Cleaning, feature engineering, model, MAE/MAPE/R² | File I/O, HTTP |
| `service.py` | Orchestrates the above for the API | FastAPI imports |
| `main.py` | HTTP routes, validation (Pydantic), static files | Parsing or ML logic |

---

## 5. Tech stack

- **Python 3.11+**
- **FastAPI + Uvicorn** (backend, serves the API and the static frontend)
- **requests** + **BeautifulSoup4** (+ `lxml`) for parsing
- **pandas**, **numpy**, **scikit-learn** (model: `GradientBoostingRegressor` or `RandomForestRegressor`, with `Ridge` as a baseline)
- **python-dotenv**, **pydantic**
- **pytest** for tests
- **joblib** to persist the trained model in `data/cache/`
- Frontend: **plain HTML + CSS + vanilla JavaScript**. No React, no build step. Chart.js via CDN is allowed for one optional chart.

Pin versions in `requirements.txt`.

---

## 6. Parser specification (`backend/parser.py`)

**Source:** `https://krisha.kz/prodazha/kvartiry/almaty/` (listing pages, pagination via `?page=N`), plus individual listing pages when a field is only available there.

### 6.1 Politeness and anti-ban requirements (all mandatory)

1. **Check `https://krisha.kz/robots.txt` and the site's terms before the first run**, and respect disallowed paths. Mention the result in the README. This tool is for personal, non-commercial analysis.
2. **User-Agent rotation:** keep a list of at least 8 realistic, current desktop and mobile browser User-Agent strings in `config.py`; pick a random one per request.
3. **Proper HTTP headers:** send `Accept`, `Accept-Language` (`ru-RU,ru;q=0.9,en;q=0.8`), `Accept-Encoding`, `Referer` (the previous page of the listing), `Connection: keep-alive`. Use a `requests.Session`.
4. **Random delays:** `time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))` between every request. Defaults: `MIN_DELAY=2.0`, `MAX_DELAY=6.0` seconds, configurable in `.env`.
5. **Caching to `data/raw/`:** before any request, check whether the page is already cached and younger than `CACHE_TTL_HOURS` (default 24). Save every fetched page as `data/raw/<sha1(url)>.html` (or `.json` for embedded JSON) plus a small sidecar `.meta.json` with `url`, `status_code`, `fetched_at`.
6. **Retries with exponential backoff** (max 3) on timeouts / 5xx. On **HTTP 403 / 429 / 468 / captcha page**: stop immediately, do **not** retry aggressively, log clearly, and switch to the offline fallback (section 7).
7. **Hard limits:** `MAX_PAGES` (default 5) and `MAX_LISTINGS` (default 200) per run, both configurable. Never crawl the whole site.
8. **No concurrency.** Sequential requests only.
9. **Timeout** on every request (`timeout=15`).

### 6.2 Parsing approach

1. First fetch one listing page and one detail page, save them in `data/raw/`, and **inspect the real markup** (also look for embedded JSON such as `<script>` blocks with listing data — prefer JSON over fragile CSS selectors when it exists).
2. Write small pure functions that are easy to unit-test offline using saved HTML:
   - `fetch(url) -> str` (with cache, headers, delay),
   - `parse_listing_cards(html) -> list[dict]` (listing id, URL, price, short info),
   - `parse_listing_detail(html) -> dict` (year, building type, renovation, floor, area, district),
   - `normalize_listing(raw: dict) -> dict` (apply section 3 mappings),
   - `collect(max_pages, max_listings) -> list[dict]` (orchestrator).
3. Selectors and field labels live in one clearly marked constants block at the top of the file, so they are easy to update when krisha changes its markup.
4. Log (use `logging`, not `print`) pages fetched, cache hits, parse failures per field.
5. A `python -m backend.parser --pages 2` CLI entry point is required for manual runs.

---

## 7. Offline fallback dataset

- File: `data/offline_fallback/almaty_flats_fallback.csv`
- Columns exactly as in section 3 (minus `scraped_at` if not available), plus a `source` column.
- **Size:** at least **300 rows**, covering all 8 districts, all building types and renovation states, years 1960–2024, realistic prices.
- **How to create it:**
  1. Preferred: run the parser once (politely, within limits) and export the cleaned result.
  2. If live collection is unavailable, use the existing real batch dataset. Do not generate replacement market data. Synthetic fixtures are allowed only in tests.
- `storage.load_dataset()` order of precedence:
  1. Fresh processed data in `data/cache/` (younger than `DATA_TTL_HOURS`),
  2. Live parse (only if `USE_LIVE_PARSER=true` and not blocked),
  3. Cached raw HTML in `data/raw/` re-parsed,
  4. Offline fallback CSV.
- The API response must always include `data_source`: `live | raw_cache | offline_fallback` and `rows_used`.

---

## 8. Model and metrics (`backend/metrics.py`)

### 8.1 Features

Numeric: `area_total`, `floor`, `floor_total`, `floor_ratio`, `house_age`.
Categorical (one-hot): `district`, `building_type`.
Binary: `is_first_floor`, `is_last_floor`, `renovation`.

Renovation must remain functional: 0 = «Черновая отделка», 1 = «Другое». Existing textual values other than shell (including «не указано») map to 1; absent values remain missing and are mode-imputed in the pipeline. Never add metro, schools, hospitals, kindergartens, coordinates, POI features or spatial imputations. District remains supported. Rooms are not model inputs.

Target: train on `log(price_kzt)` (or `price_per_m2`) and convert back. Document the choice in a docstring.

### 8.2 Required functions

- `clean_dataset(df) -> df`
- `add_features(df) -> df`
- `build_pipeline() -> sklearn.Pipeline` (preprocessing + model)
- `train_and_evaluate(df) -> (pipeline, metrics_dict)` — 5-fold cross-validation with fixed random seed; keep holdout diagnostics separate
- `evaluate(y_true, y_pred) -> dict` returning **MAE, RMSE, MAPE, R²**
- `predict_with_range(pipeline, features) -> (estimate, low, high)` — range from residual quantiles (e.g. 10th–90th percentile) or from the spread of forest trees
- `district_stats(df) -> dict` — median price, median price per m², count per district

### 8.3 Quality bar

- Beat a naive baseline (district median price per m² × area). Report both in the metrics output.
- If the dataset has fewer than 100 clean rows, fall back to the baseline and tell the user.
- Predictions must be monotonic-sane: larger area must not give a lower total price for otherwise identical inputs (add a unit test).

### 8.4 Comparables (`service.py`)

Return the **5 most similar listings**: same district (relax to the whole city if fewer than 5), area within ±20 %, and closest by a simple distance on area, floor ratio, year and building type. Include `url`, `price_kzt`, `area_total`, `floor`/`floor_total`, `house_year`.

---

## 9. Backend API (`backend/main.py`)

Run with `uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000`.

| Method & path | Purpose |
|---|---|
| `GET /` | Serves `frontend/index.html` |
| `GET /static/...` | Serves `style.css`, `app.js` |
| `GET /api/health` | `{status, data_source, rows, model_trained_at}` |
| `GET /api/options` | Allowed values for dropdowns: districts, building types, renovation states, year range |
| `POST /api/predict` | Body below → estimate |
| `GET /api/stats?district=...` | Median price, median price/m², listing count, metrics of the model |
| `POST /api/refresh` | Re-run parser (rate limited: 1 call per 10 minutes) and retrain; falls back safely |

`POST /api/predict` request:

```json
{
  "district": "Бостандыкский",
  "area_total": 54.0,
  "floor": 5,
  "floor_total": 9,
  "house_year": 2012,
  "building_type": "монолит",
  "renovation": 1
}
```

Response:

```json
{
  "price_estimate_kzt": 52300000,
  "price_low_kzt": 47800000,
  "price_high_kzt": 57100000,
  "price_per_m2_kzt": 968500,
  "district_median_price_per_m2_kzt": 905000,
  "comparables": [ { "url": "...", "price_kzt": 51000000, "area_total": 52.0, "floor": 4, "floor_total": 9, "house_year": 2011 } ],
  "model_metrics": { "mae": 0, "mape": 0, "r2": 0 },
  "data_source": "offline_fallback",
  "rows_used": 312,
  "warnings": []
}
```

Validation (Pydantic): `area_total` 10–400, `floor` ≥ 1, `floor ≤ floor_total`, `house_year` 1950–(current year + 5), enums for `district`, `building_type`; renovation accepts 0/1 or supported text labels. Return clear 422 messages in Russian.

Other requirements: CORS restricted to `http://127.0.0.1:8000` and `http://localhost:8000`; load/train the model once at startup (lazy and cached); never crash the server because of a parser failure.

---

## 10. Frontend (`frontend/`)

Plain HTML/CSS/JS, responsive, Russian UI.

**Form fields** (labels in Russian):

| Label | Field | Control |
|---|---|---|
| Район | `district` | `<select>` filled from `/api/options` |
| Общая площадь, м² | `area_total` | number input |
| Этаж | `floor` | number input |
| Этажей в доме | `floor_total` | number input |
| Год постройки | `house_year` | number input |
| Тип дома | `building_type` | `<select>` (монолит / кирпич / панель / иное) |
| Состояние / ремонт | `renovation` | Radio: «Черновая отделка» = 0 / «Другое» = 1 |

**Result block:** big estimated price (formatted `52 300 000 ₸`), range, price per m², comparison with the district median (± %), table of comparables with links opening in a new tab, a badge showing the data source, and a warning banner when `data_source != live` or data is synthetic.

**UX rules:** client-side validation mirroring the API rules, loading state while waiting, readable error messages, no page reload (use `fetch`), keyboard-accessible, no external libraries required.

---

## 11. Configuration (`.env` / `.env.example`)

```
HOST=127.0.0.1
PORT=8000
USE_LIVE_PARSER=false
MIN_DELAY=2.0
MAX_DELAY=6.0
MAX_PAGES=5
MAX_LISTINGS=200
CACHE_TTL_HOURS=24
DATA_TTL_HOURS=24
REQUEST_TIMEOUT=15
LOG_LEVEL=INFO
GOOGLE_DRIVE_OUTPUT_FOLDER=project for forecasting
```

- `.env.example` is committed; `.env` is **not** (listed in `.gitignore`).
- `.gitignore` must exclude: `.env`, `__pycache__/`, `.venv/`, `data/raw/*` and `data/cache/*` (keep `.gitkeep`), `.pytest_cache/`, `*.joblib`. Keep collected datasets and real-page fixtures private; do not publish them.
- Default is `USE_LIVE_PARSER=false`, so the app works immediately offline and never hits krisha.kz unless the user opts in.

---

## 12. Build order (milestones)

For initial implementation, work through these milestones. For an existing project, change only the requested functionality and report progress without restarting the build:

1. **Scaffold:** create the full directory tree, `requirements.txt`, `.gitignore`, `.env.example`, `config.py`, placeholder README.
2. **Fallback dataset:** create `almaty_flats_fallback.csv` (section 7) and `storage.py` loading logic.
3. **Metrics & model:** `metrics.py` + `tests/test_metrics.py`. Tests must pass.
4. **Service layer:** `service.py` (train on startup, predict, comparables, stats).
5. **API:** `main.py`, check with `curl` or the Swagger UI at `/docs`.
6. **Frontend:** `index.html`, `style.css`, `app.js` wired to the API.
7. **Parser:** `parser.py` with caching, UA rotation, delays, fallback on block; unit tests with saved HTML.
8. **Integration:** `/api/refresh`, data-source badges, warnings.
9. **Docs & polish:** full `README.md`, final run of all tests, reports (section 13).

Definition of done: `pip install -r requirements.txt` → `uvicorn backend.main:app` → open `http://127.0.0.1:8000` → fill in the form → get a price estimate, **without** internet access and without live scraping.

---

## 13. Storage location: Google Drive "project for forecasting" folder (mandatory)

**Everything the agent produces lives in the Google Drive folder `project for forecasting`** — the application source code (the whole tree from section 4), the README, tests, the fallback dataset, and also all reports, charts, exports and intermediate outputs. Nothing is stored anywhere else.

Drive layout:

```
My Drive/
└── project for forecasting/                      # created if missing; name match is case-insensitive
    ├── almaty-flat-calculator/       # the project tree from section 4, same folders & file names
    │   ├── PROJECT_INSTRUCTIONS.md
    │   ├── backend/ ... frontend/ ... data/ ... tests/
    └── reports/                      # reports, charts, exports, intermediate outputs
```

Procedure the agent must follow **before saving any file**:

1. Search Google Drive for a folder named `project for forecasting` (case-insensitive, not in trash).
2. Use the existing authorized root ID `1yiaa_qpfXMrOtrDBIopcFsa4_zbCYuv6`. If inaccessible, stop and report the problem; do not select another project.
3. Never read or modify the separate `Forecasting` project. Folder names alone do not override the authorized root ID.
4. Inside it, use (or create) `almaty-flat-calculator/` for project files and `reports/` for reports, charts and exports. Recreate the sub-folders from section 4 as needed (`backend`, `frontend`, `data/raw`, `data/cache`, `data/offline_fallback`, `tests`).
5. When creating files in Drive, upload them as **plain files** (keep the original type: `.py`, `.html`, `.css`, `.js`, `.md`, `.csv`, `.txt`) and do **not** convert them to Google Docs / Sheets. Before creating a file, check whether one with the same name already exists in that folder; if it does, update the existing file by ID to preserve its history.
6. Use dated, descriptive names for reports, e.g. `almaty_model_report_2026-10-02.md`.
7. After saving, tell the user the file name and its Drive link.
8. If Google Drive is not connected or a save fails, **do not save anywhere else silently** — tell the user, keep the content in the chat, and ask them to connect Drive.
9. Move obsolete modules into a dated `archive/` within this project rather than deleting them; exclude the archive from pytest discovery. Never trash files in the folder, and never read or modify files that are not part of this project (the folder may contain unrelated files such as keys or notes).

How the app runs on localhost:

- The Drive copy is the **source of truth**. To run the app, the user syncs the `almaty-flat-calculator` folder to their computer (Google Drive for desktop) or downloads it, then follows the README (`pip install -r requirements.txt` → `uvicorn backend.main:app`).
- The agent may use a temporary scratch working copy **only** to run tests and the server while developing. Every finished or changed file must then be written back to Drive, and the agent must say so explicitly.
- Files the app creates while running on the user's machine (`data/raw/`, `data/cache/`, trained model) stay in that local copy and are not uploaded unless the user asks.
- If the user wants the folder name changed, they will say so explicitly; the name lives in `GOOGLE_DRIVE_OUTPUT_FOLDER`.

---

## 14. Testing requirements (`tests/`)

Minimum for `test_metrics.py`:

- `evaluate()` returns correct MAE / RMSE / MAPE / R² on a small hand-computed example.
- `clean_dataset()` removes outliers, duplicates and invalid floors.
- `add_features()` creates `floor_ratio`, `is_first_floor`, `is_last_floor`, `house_age` correctly.
- A trained pipeline predicts a positive price and, for identical inputs, price grows with area.
- The model beats the naive district baseline on the fallback dataset (or the test documents why not).

Additional (recommended): `test_parser.py` that parses saved HTML from `data/raw/` fixtures and never touches the network; an API test with FastAPI's `TestClient` for `/api/predict` and `/api/health`.

Run with `pytest -q` from the project root.

---

## 15. Security, legal and ethics

- Personal, non-commercial use. Respect krisha.kz robots.txt and terms; do not scrape phone numbers, names or other personal data of sellers — collect only the fields in section 3.
- Do not bypass captchas, rotate proxies, or use any technique to evade blocking. If blocked, stop and use the fallback.
- Never commit `.env`. Never log secrets.
- Display a disclaimer in the UI: "Оценка ориентировочная и не является официальной оценкой стоимости недвижимости."

---

## 16. How the user will talk to you (examples)

| User says | You do |
|---|---|
| "Start the project" | Run milestones 1–2 and report |
| "Build the model" | Milestone 3, show metrics table |
| "Add a parameter: number of rooms" | Update section 3 schema, parser, features, form, API, tests, docs |
| "Parser stopped working" | Re-inspect saved HTML in `data/raw/`, update the selectors block, add a regression test |
| "Make a report on prices by district" | Generate the report and chart, save to Drive `project for forecasting/reports` (section 13) |
| "Run everything" | Install deps, run tests, start the server, give the localhost URL |

When you finish a task, always reply with: **what changed → how to run/verify → what's next**.

---

## 17. Example file (code style reference)

If the user attaches an example project or file, treat it as the **style reference**: mirror its naming conventions, module layout, logging style and docstring format, but keep the structure and requirements of this document. If no example is attached, use standard PEP 8, type hints on all public functions, and short docstrings.

## 18. Current maintenance constraints (2026-10-13)

- No dataset sync command; the user collects batches and copies the fallback manually.
- Batch defaults: `STORE_BACKEND=local`, `LOCAL_STORE_DIR=data/collected` relative to the application root. Never use a path into the other agent's project.
- Retired commands: `python -m backend.geo` and `python -m backend.batch --enrich`. Do not restore them or run older `apply_*` migrations.
- Historical CSV latitude/longitude columns may remain solely for lossless storage compatibility; no active extraction, inference, imputation, form controls, API options or model features use them.
- Validate renovation from frontend payload through Pydantic normalization into the fitted feature matrix. Assert no retired features remain in model columns or OpenAPI schema.
- Bump `MODEL_VERSION` when changing feature semantics. Cache fingerprints include model version, library version, target and actual training features.
- Reports belong to the sibling `project for forecasting/reports/`; keep dated historical reports as history and clearly distinguish their model metrics from the current model.
