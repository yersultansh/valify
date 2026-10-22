# BATCH_RUNBOOK.md: collecting ~500 real listings in resumable batches

Real data only: synthetic examples are used only in tests. Each run collects ONE batch, saves to the selected store and exits. The default is the local data/collected folder inside this application.

## Recommended setup: local collection inside this project

The existing `.env` uses `STORE_BACKEND=local` and `LOCAL_STORE_DIR=data/collected`. The relative path is resolved against the application root, not the terminal directory. If this project is synced by Google Drive for desktop, these files sync normally. No OAuth setup is needed.

For another ~500 listings, change only `TARGET_TOTAL=1000` in `.env`, then:

```bash
python -m backend.batch --status
python -m backend.batch --start-page 1
python -m backend.batch
```

Run further batches manually, spaced apart, and stop on a block. `--start-page` can move deeper in the feed. No website requests were made as part of the 2026-10-13 code update. Do not run `--enrich` or `backend.geo`: coordinate enrichment was removed.

After collecting, stop the web server, copy manually and restart:

```bash
cp data/collected/krisha_dataset.csv data/offline_fallback/almaty_flats_fallback.csv
```

The application also reads `data/cache/live_listings.csv`. This command does not merge stores or synchronize `visited_urls.txt`. Clear only processed `dataset.csv` / `dataset.meta.json` if you need the new baseline immediately rather than after cache expiry. Do not remove live listings or batch state.

## Optional alternative: Drive API setup (macOS)

1. Install the extra packages (they are not in section 5 of the spec; they are needed for the Drive API):
   `pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib`
   Add the same three names to `requirements.txt`.
2. Google Cloud Console (https://console.cloud.google.com):
   - Create a project (any name).
   - APIs & Services -> Library -> enable **Google Drive API**.
   - APIs & Services -> OAuth consent screen (Google Auth Platform): user type **External**, fill app name and your e-mail, add **your own Google account as a Test user**.
   - Credentials -> Create credentials -> **OAuth client ID** -> Application type **Desktop app** -> download the JSON.
   - Save it as `credentials.json` in the project root (NOT inside Drive).
3. Add to `.gitignore`: `credentials.json` and `token.json`. Never upload them anywhere.
4. Append to `.env` (keep your existing lines; `GOOGLE_DRIVE_OUTPUT_FOLDER=project for forecasting` is reused as the parent folder):

```
STORE_BACKEND=drive
DRIVE_OUTPUT_PATH=almaty-flat-calculator/data/collected
GOOGLE_CREDENTIALS_FILE=credentials.json
GOOGLE_TOKEN_FILE=token.json
BATCH_SIZE=70
BATCH_MAX_PAGES=5
BATCH_MIN_DELAY=8.0
BATCH_MAX_DELAY=12.0
TARGET_TOTAL=500
RESUME_OVERLAP_PAGES=1
INCLUDE_NEW_BUILDS=false
```

5. First sign-in (makes NO requests to krisha.kz):
   `python3 -m backend.batch --status`
   A browser opens. Because the app is in Testing mode Google shows "Google hasn't verified this app": Advanced -> Continue -> allow. The script then finds the Drive folder `project for forecasting` (case-insensitive) or creates it, creates `almaty-flat-calculator/data/collected/` inside it, and prints the folder link.

## Every batch

```
python3 -m backend.batch
```

About 1 minute (up to 5 pages, 8-12 s between requests). It prints a summary with rows added, total, stop reason and the next start page. Run it again whenever you like: in 30 minutes, 2 hours or tomorrow. About 8 runs give ~500 rows. Do not run two copies at the same time.

Useful options: `--status`, `--batch-size 50`, `--max-pages 4`, `--start-page 40` (jump deeper), `--details 10` (fetch up to 10 detail pages to fill floor/year/district), `--backend local`.

## Files in Drive (`project for forecasting/almaty-flat-calculator/data/collected/`)

| File | Content |
|---|---|
| `krisha_dataset.csv` | normalized listings (section 3 schema + `is_new_build`, `source=krisha`) |
| `visited_urls.txt` | one listing URL per line; anything listed is never requested/parsed again |
| `batch_state.json` | resume position (`next_page`) and the last 30 runs |

Progress is appended to a local working copy (`data/cache/batch/`) immediately for every listing, and mirrored to Drive after every page and at exit (also after Ctrl+C or a block). If an upload fails, the next run merges the local copy automatically.

## If krisha.kz blocks you (403 / 429 / 468 / captcha)

The run stops at once, saves everything, prints a `STOPPED` banner and exits with code 2. Wait several hours or a day before the next run, and consider `BATCH_MIN_DELAY=12`, `BATCH_MAX_DELAY=20`, `BATCH_SIZE=40`. There is no bypass logic and there never will be.

## Notes

- The Drive scope is full Drive access (needed to find the existing `project for forecasting` folder). The code only searches the parent for folders and only touches the three files above inside the output folder; it never deletes anything.
- In Testing mode Google expires the saved login after 7 days; the script just opens the browser again.
- No OAuth alternative: install Google Drive for desktop and use `STORE_BACKEND=local` with `LOCAL_STORE_DIR=data/collected`.
- Sampling: listings come newest-first, so consecutive batches are consecutive slices of the feed. Use `--start-page` now and then to mix in older ads.
- Terms of use: automated collection is restricted by krisha.kz's terms; this is a small personal, non-commercial, rate-limited collection at the owner's responsibility. Keep the collected data private.
