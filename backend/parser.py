"""krisha.kz scraping and HTML parsing for Almaty flat sale listings.

Politeness rules (PROJECT_INSTRUCTIONS.md, section 6.1) are implemented in
``fetch``/``collect``: robots.txt-aware URL filter, User-Agent rotation, random
delays, on-disk cache, retries with backoff, hard limits, sequential requests,
and an immediate stop on 403 / 429 / captcha. No personal data (seller names,
phones) is ever read or stored.

CLI:  python -m backend.parser --pages 2 --max-listings 40
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import re
import time
from datetime import datetime, timezone

import requests
from bs4 import BeautifulSoup

from backend import config, storage

log = logging.getLogger(__name__)

# ==========================================================================
# SELECTORS AND LABELS  (verified on saved HTML, 2026-10-02; edit here only)
# ==========================================================================
BASE_URL = "https://krisha.kz"
CARD_SEL = "div.a-card"
CARD_TITLE_SEL = ".a-card__title"
CARD_PRICE_SEL = ".a-card__price"
CARD_SUBTITLE_SEL = ".a-card__subtitle"
CARD_PREVIEW_SEL = ".a-card__text-preview"
CARD_COMPLEX_SEL = ".a-card__complex-label"          # present on new-build cards
CARD_ID_ATTR = "data-id"

DETAIL_INFO_ITEM_SEL = ".offer__info-item"
DETAIL_INFO_VALUE_SEL = ".offer__advert-short-info"
DETAIL_LOCATION_SEL = ".offer__location"
DETAIL_PRICE_SEL = ".offer__price"
DETAIL_DATA_NAMES = {                                # data-name -> raw field
    "flat.building": "building_raw",
    "house.year": "year_raw",
    "flat.floor": "floor_raw",
    "live.square": "area_raw",
    "flat.renovation": "renovation_raw",
}
JSDATA_SCRIPT_ID = "jsdata"
# Whitelist of keys read from window.data.advert. Seller fields are NOT listed.
ADVERT_JSON_KEYS = ("id", "price", "square", "rooms", "title")

RENOVATION_PHRASES = [                               # (phrase on site, canonical)
    ("не новый, но аккуратный ремонт", "хорошее"),
    ("свежий ремонт", "свежий ремонт"),
    ("требует ремонта", "требует ремонта"),
    ("черновая отделка", "черновая отделка"),
]
BUILDING_MAP = {"кирпичный": "кирпич", "панельный": "панель",
                "монолитный": "монолит", "иной": "иное"}

# robots.txt (checked 2026-10-02): URLs containing these are never requested.
ROBOTS_DISALLOWED_SUBSTRINGS = (
    "raion=", "sub_cat", "utm_source", "gclid=", "till=", "from=adwords",
    "poll_answer=", "actionpay=", "_openstat", "claim", "comments", "sendmess",
    "incorrect-data",
)
ROBOTS_DISALLOWED_PREFIXES = ("/ajax/", "/a/show-map/", "/captcha", "/ms/", "/index/")
BLOCK_STATUS_CODES = (403, 429, 468)       # 468 is the Qrator anti-bot block page
GONE_STATUS_CODES = (404, 410)              # a removed listing is not a block
# ==========================================================================


class BlockedError(RuntimeError):
    """Raised on HTTP 403 / 429 or a captcha page. Never retried or bypassed."""


# --------------------------------------------------------------------------
# HTTP layer
# --------------------------------------------------------------------------
_session: requests.Session | None = None
_last_request_ts: float | None = None


def _get_session() -> requests.Session:
    global _session
    if _session is None:
        _session = requests.Session()
    return _session


def is_allowed(url: str) -> bool:
    """Return False for URLs that krisha.kz robots.txt disallows."""
    path = url.replace(BASE_URL, "", 1)
    if any(path.startswith(p) for p in ROBOTS_DISALLOWED_PREFIXES):
        return False
    return not any(s in url for s in ROBOTS_DISALLOWED_SUBSTRINGS)


def looks_like_captcha(html: str, final_url: str = "") -> bool:
    """Detect a captcha / block page (the footer reCAPTCHA notice is NOT a signal)."""
    if "/captcha" in final_url:
        return True
    head = html[:3000].lower()
    return ("captcha" in head or "подтвердите, что вы не робот" in head) and CARD_SEL not in html[:200000]


def _headers(referer: str | None) -> dict[str, str]:
    h = {
        "User-Agent": random.choice(config.USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.8",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive",
    }
    if referer:
        h["Referer"] = referer
    return h


def fetch(url: str, referer: str | None = None, stats: dict | None = None, use_cache: bool = True) -> str:
    """Return page HTML: from the raw cache if fresh, else from the network."""
    global _last_request_ts
    stats = stats if stats is not None else {}
    if not is_allowed(url):
        raise ValueError(f"URL disallowed by robots.txt rules: {url}")

    cached = storage.read_raw_cache(url) if use_cache else None
    if cached is not None:
        stats["cache_hits"] = stats.get("cache_hits", 0) + 1
        log.info("cache hit: %s", url)
        return cached

    last_exc: Exception | None = None
    for attempt in range(3):
        if _last_request_ts is not None:           # delay between every request
            time.sleep(random.uniform(config.MIN_DELAY, config.MAX_DELAY))
        _last_request_ts = time.monotonic()
        try:
            resp = _get_session().get(url, headers=_headers(referer), timeout=config.REQUEST_TIMEOUT)
        except (requests.Timeout, requests.ConnectionError) as exc:
            last_exc = exc
            log.warning("request error (%s), attempt %d/3: %s", type(exc).__name__, attempt + 1, url)
            time.sleep(2 ** attempt)
            continue
        if resp.status_code in BLOCK_STATUS_CODES or (400 <= resp.status_code < 500
                                                      and resp.status_code not in GONE_STATUS_CODES):
            raise BlockedError(f"HTTP {resp.status_code} from {url}")   # any other 4xx is treated as a block too
        if resp.status_code >= 500:
            last_exc = RuntimeError(f"HTTP {resp.status_code}")
            log.warning("server error %d, attempt %d/3: %s", resp.status_code, attempt + 1, url)
            time.sleep(2 ** attempt)
            continue
        resp.raise_for_status()
        resp.encoding = resp.encoding or "utf-8"
        html = resp.text
        if looks_like_captcha(html, resp.url):
            raise BlockedError(f"captcha page at {url}")
        if use_cache:
            storage.write_raw_cache(url, html, resp.status_code)
        stats["pages_fetched"] = stats.get("pages_fetched", 0) + 1
        log.info("fetched (%d): %s", resp.status_code, url)
        return html
    raise RuntimeError(f"giving up on {url} after 3 attempts: {last_exc}")


# --------------------------------------------------------------------------
# Small text helpers
# --------------------------------------------------------------------------
def _text(el) -> str:
    return re.sub(r"\s+", " ", el.get_text(" ", strip=True).replace("\xa0", " ")) if el else ""


def _to_int(text: str) -> int | None:
    digits = re.sub(r"\D", "", text or "")
    return int(digits) if digits else None


def _to_float(text: str) -> float | None:
    m = re.search(r"(\d+(?:[.,]\d+)?)", (text or "").replace(" ", ""))
    return float(m.group(1).replace(",", ".")) if m else None


def _district_from(text: str) -> str | None:
    m = re.search(r"([А-Яа-яЁё-]+)\s+р-н", text or "")
    if m and m.group(1) in config.DISTRICTS:
        return m.group(1)
    return None


def _building_from(text: str) -> str | None:
    m = re.search(r"\b(кирпичный|панельный|монолитный|иной)(?:\s+дом)?\b", (text or "").lower())
    return m.group(1) if m else None


def _renovation_from_preview(preview: str) -> str | None:
    """Return the text after 'состояние:' (commas inside phrases are expected)."""
    m = re.search(r"состояние:\s*(.*)", preview)
    return m.group(1) if m else None


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------
def parse_listing_cards(html: str) -> list[dict]:
    """Parse a listing page into raw card dicts (one per ``div.a-card``)."""
    soup = BeautifulSoup(html, "lxml")
    out: list[dict] = []
    for card in soup.select(CARD_SEL):
        lid = card.get(CARD_ID_ATTR)
        if not lid:
            continue
        title = _text(card.select_one(CARD_TITLE_SEL))
        price_text = _text(card.select_one(CARD_PRICE_SEL))
        subtitle = _text(card.select_one(CARD_SUBTITLE_SEL))
        preview = _text(card.select_one(CARD_PREVIEW_SEL))
        fm = re.search(r"(\d+)\s*/\s*(\d+)\s*этаж", title)
        year = re.search(r"(\d{4})\s*г\.п\.", preview)
        out.append({
            "listing_id": str(lid),
            "url": f"{BASE_URL}/a/show/{lid}",
            "title": title,
            "price_text": price_text,
            "price_is_from": price_text.startswith("от"),
            "is_new_build": bool(card.select_one(CARD_COMPLEX_SEL)) or price_text.startswith("от"),
            "subtitle": subtitle,
            "rooms": _to_int((re.search(r"(\d+)-комн", title) or [None, ""])[1]),
            "area_text": (re.search(r"([\d.,]+)\s*м²", title) or [None, ""])[1],
            "floor": int(fm.group(1)) if fm else None,
            "floor_total": int(fm.group(2)) if fm else None,
            "year_raw": year.group(1) if year else None,
            "building_raw": _building_from(preview),
            "renovation_raw": _renovation_from_preview(preview),
        })
    return out


def parse_listing_detail(html: str) -> dict:
    """Parse a detail page. Reads only whitelisted advert JSON keys and labelled info items."""
    soup = BeautifulSoup(html, "lxml")
    out: dict = {}
    script = soup.find("script", id=JSDATA_SCRIPT_ID)
    if script and script.string and "window.data" in script.string:
        try:
            data = json.loads(script.string.split("window.data =", 1)[1].strip().rstrip(";"))
            advert = data.get("advert", {})
            for key in ADVERT_JSON_KEYS:
                if key in advert:
                    out[f"json_{key}"] = advert[key]
        except (ValueError, IndexError):
            log.warning("detail: could not parse embedded JSON")
    for item in soup.select(DETAIL_INFO_ITEM_SEL):
        name = item.get("data-name")
        if name in DETAIL_DATA_NAMES:
            out[DETAIL_DATA_NAMES[name]] = _text(item.select_one(DETAIL_INFO_VALUE_SEL))
    loc = _text(soup.select_one(DETAIL_LOCATION_SEL))
    out["district_raw"] = loc
    out["price_text"] = _text(soup.select_one(DETAIL_PRICE_SEL))
    return out


def _canon_building(raw: str | None) -> str | None:
    """Missing -> None (never guessed); unknown wording -> 'иное'."""
    if not raw or not raw.strip():
        return None
    return BUILDING_MAP.get(raw.strip().lower(), "иное")


def _canon_renovation(raw: str | None) -> str:
    if not raw:
        return "не указано"
    low = raw.strip().lower()
    for phrase, canonical in RENOVATION_PHRASES:
        if low.startswith(phrase):
            return canonical
    log.warning("unknown renovation wording, mapped to 'не указано': %r", raw[:60])
    return "не указано"


def normalize_listing(raw: dict) -> dict:
    """Apply the section 3 normalization rules; unknown values become None."""
    price = _to_int(raw.get("price_text", ""))
    if price is not None and price <= 0:
        price = None
    area = raw.get("area") or _to_float(raw.get("area_text", ""))
    year = _to_int(str(raw.get("year_raw") or ""))
    if year is not None and not (1950 <= year <= datetime.now().year + 5):
        log.warning("house_year out of range, dropped: %s", year)
        year = None
    district = raw.get("district") or _district_from(raw.get("subtitle", "")) \
        or _district_from(raw.get("district_raw", ""))
    return {
        "listing_id": raw["listing_id"],
        "price_kzt": price,
        "district": district,
        "area_total": area,
        "floor": raw.get("floor"),
        "floor_total": raw.get("floor_total"),
        "house_year": year,
        "building_type": _canon_building(raw.get("building_raw")),
        "renovation": _canon_renovation(raw.get("renovation_raw")),
        "rooms": raw.get("rooms"),
        "url": raw.get("url") or f"{BASE_URL}/a/show/{raw['listing_id']}",
        "scraped_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "is_new_build": bool(raw.get("is_new_build", False)),
        "source": "krisha",
    }


def merge_detail(raw: dict, detail: dict) -> dict:
    """Fill gaps in a card dict from a parsed detail page (never overwrites known values)."""
    merged = dict(raw)
    fm = re.search(r"(\d+)\s*из\s*(\d+)", detail.get("floor_raw", "") or "")
    if fm and merged.get("floor") is None:
        merged["floor"], merged["floor_total"] = int(fm.group(1)), int(fm.group(2))
    for key in ("year_raw", "building_raw", "renovation_raw"):
        if not merged.get(key) and detail.get(key):
            merged[key] = detail[key]
    if not merged.get("area_text") and detail.get("area_raw"):
        merged["area_text"] = detail["area_raw"]
    if not merged.get("district_raw"):
        merged["district_raw"] = detail.get("district_raw", "")
    return merged


# --------------------------------------------------------------------------
# Orchestrator
# --------------------------------------------------------------------------
def collect(max_pages: int | None = None, max_listings: int | None = None,
            include_new_builds: bool = False, max_details: int = 0,
            stats: dict | None = None) -> list[dict]:
    """Collect normalized listings, stopping at the limits or on any block.

    ``max_details`` > 0 enables detail-page requests for cards that lack
    floor / year / district (at most that many per run).
    """
    max_pages = config.MAX_PAGES if max_pages is None else max_pages
    max_listings = config.MAX_LISTINGS if max_listings is None else max_listings
    stats = stats if stats is not None else {}
    stats.update({"blocked": False, "new_build_skipped": 0, "cards_seen": 0})
    seen: set[str] = set()
    rows: list[dict] = []
    referer: str | None = None
    details_left = max_details

    for page in range(1, max_pages + 1):
        url = config.SOURCE_URL if page == 1 else f"{config.SOURCE_URL}?page={page}"
        try:
            html = fetch(url, referer=referer, stats=stats)
        except BlockedError as exc:
            log.error("BLOCKED: %s. Stopping; use the cached data / offline fallback.", exc)
            stats["blocked"] = True
            break
        except Exception as exc:
            log.error("page %d failed: %s", page, exc)
            stats["errors"] = stats.get("errors", 0) + 1
            break
        referer = url
        cards = parse_listing_cards(html)
        if not cards:
            log.warning("page %d: no cards found (markup changed?)", page)
            break
        for raw in cards:
            stats["cards_seen"] += 1
            if raw["listing_id"] in seen:
                continue
            seen.add(raw["listing_id"])
            if raw["is_new_build"] and not include_new_builds:
                stats["new_build_skipped"] += 1
                continue
            needs_detail = raw["floor"] is None or not raw["year_raw"] or not _district_from(raw["subtitle"])
            if needs_detail and details_left > 0 and not raw["is_new_build"]:
                try:
                    raw = merge_detail(raw, parse_listing_detail(fetch(raw["url"], referer=url, stats=stats)))
                    details_left -= 1
                except BlockedError as exc:
                    log.error("BLOCKED on detail page: %s", exc)
                    stats["blocked"] = True
                    break
                except Exception as exc:
                    log.warning("detail failed for %s: %s", raw["listing_id"], exc)
            rows.append(normalize_listing(raw))
            if len(rows) >= max_listings:
                break
        if stats["blocked"] or len(rows) >= max_listings:
            break
    log.info("collected %d listings (%s)", len(rows), stats)
    return rows


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Collect Almaty flat listings from krisha.kz (polite, limited).")
    ap.add_argument("--pages", type=int, default=config.MAX_PAGES)
    ap.add_argument("--max-listings", type=int, default=config.MAX_LISTINGS)
    ap.add_argument("--include-new-builds", action="store_true")
    ap.add_argument("--details", type=int, default=0, help="max detail-page requests (default 0)")
    ap.add_argument("--out", default=str(config.CACHE_DIR / "pilot_listings.csv"))
    args = ap.parse_args(argv)
    logging.basicConfig(level=config.LOG_LEVEL, format="%(asctime)s %(levelname)s %(message)s")

    stats: dict = {}
    rows = collect(args.pages, args.max_listings, args.include_new_builds, args.details, stats)
    if not rows:
        print("No listings collected. blocked =", stats.get("blocked"), "| stats:", stats)
        return 1
    from pathlib import Path
    path = storage.save_listings(rows, Path(args.out))
    df = storage.rows_to_frame(rows)
    print(f"\nListings parsed: {len(df)}  (new builds skipped: {stats.get('new_build_skipped', 0)}, "
          f"pages fetched: {stats.get('pages_fetched', 0)}, cache hits: {stats.get('cache_hits', 0)}, "
          f"blocked: {stats.get('blocked')})")
    print("Missing % per field (None or 'не указано'):")
    for k, v in storage.missing_report(df).items():
        print(f"  {k:14s} {v:5.1f}%")
    print("\n3 sample normalized rows:")
    for r in rows[:3]:
        print(json.dumps(r, ensure_ascii=False))
    print(f"\nSaved: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
