"""Configuration: paths, .env variables, constants. No business logic here."""
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# --- Paths ---
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
CACHE_DIR = DATA_DIR / "cache"
FALLBACK_CSV = DATA_DIR / "offline_fallback" / "almaty_flats_fallback.csv"
FRONTEND_DIR = BASE_DIR / "frontend"

# --- Server ---
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "8000"))

# --- Parser ---
SOURCE_URL = "https://krisha.kz/prodazha/kvartiry/almaty/"
USE_LIVE_PARSER = os.getenv("USE_LIVE_PARSER", "false").lower() == "true"
MIN_DELAY = float(os.getenv("MIN_DELAY", "2.0"))
MAX_DELAY = float(os.getenv("MAX_DELAY", "6.0"))
MAX_PAGES = int(os.getenv("MAX_PAGES", "5"))
MAX_LISTINGS = int(os.getenv("MAX_LISTINGS", "200"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "15"))
CACHE_TTL_HOURS = int(os.getenv("CACHE_TTL_HOURS", "24"))
DATA_TTL_HOURS = int(os.getenv("DATA_TTL_HOURS", "24"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
GOOGLE_DRIVE_OUTPUT_FOLDER = os.getenv("GOOGLE_DRIVE_OUTPUT_FOLDER", "project for forecasting")

# Realistic desktop and mobile browser User-Agent strings; one is picked at
# random per request (PROJECT_INSTRUCTIONS.md, section 6.1). Refresh them
# from time to time so they stay current.
USER_AGENTS: list[str] = [
    # Desktop Chrome (Windows, macOS, Linux)
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36",
    # Desktop Firefox
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:143.0) Gecko/20100101 Firefox/143.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:143.0) Gecko/20100101 Firefox/143.0",
    # Desktop Safari and Edge
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.6 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36 Edg/141.0.0.0",
    # Mobile
    "Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Mobile Safari/537.36",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 18_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.6 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (Linux; Android 14; SM-S918B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Mobile Safari/537.36",
]

# --- Domain constants (PROJECT_INSTRUCTIONS.md, section 3) ---
DISTRICTS = [
    "Алатауский", "Алмалинский", "Ауэзовский", "Бостандыкский",
    "Жетысуский", "Медеуский", "Наурызбайский", "Турксибский",
]
BUILDING_TYPES = ["монолит", "кирпич", "панель", "иное"]
RENOVATION_STATES = [
    "свежий ремонт", "хорошее", "среднее",
    "требует ремонта", "черновая отделка", "не указано",
]
