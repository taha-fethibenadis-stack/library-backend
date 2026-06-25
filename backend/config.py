"""
Centralized environment configuration.
Every variable here must be set in Railway's service Variables tab
(or in a local .env file for local testing — see .env.example).
"""
import os
from dotenv import load_dotenv

load_dotenv()


def _require(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"Missing required environment variable: {name}. "
            f"Set it in Railway > your service > Variables."
        )
    return value


TELEGRAM_BOT_TOKEN = _require("TELEGRAM_BOT_TOKEN")
TELEGRAM_WEBHOOK_SECRET = _require("TELEGRAM_WEBHOOK_SECRET")
PUBLIC_BASE_URL = _require("PUBLIC_BASE_URL").rstrip("/")

SUPABASE_URL = _require("SUPABASE_URL")
SUPABASE_KEY = _require("SUPABASE_KEY")

GEMINI_API_KEY = _require("GEMINI_API_KEY")

# Optional — restricts the bot's auto-save behaviour to one specific group.
# Leave unset to allow any group the bot is added to.
TELEGRAM_GROUP_ID = os.environ.get("TELEGRAM_GROUP_ID")

# Comma-separated list of allowed frontend origins for CORS
FRONTEND_ORIGIN = os.environ.get("FRONTEND_ORIGIN", "*")
ALLOWED_ORIGINS = (
    ["*"] if FRONTEND_ORIGIN == "*" else [o.strip() for o in FRONTEND_ORIGIN.split(",")]
)

WEBHOOK_PATH = f"/webhook/{TELEGRAM_WEBHOOK_SECRET}"
WEBHOOK_URL = f"{PUBLIC_BASE_URL}{WEBHOOK_PATH}"
