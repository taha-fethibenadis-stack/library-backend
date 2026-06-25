"""
Validates Telegram Mini App `initData` strings.

Every request the Mini App frontend sends to our API includes the raw
`initData` string provided by window.Telegram.WebApp.initData. We must verify
its HMAC signature so random people can't call our API pretending to be a
specific Telegram user. See:
https://core.telegram.org/bots/webapps#validating-data-received-via-the-mini-app
"""
import hashlib
import hmac
import json
import time
from urllib.parse import parse_qsl

from config import TELEGRAM_BOT_TOKEN

# How long an initData payload stays valid after Telegram issued it (seconds).
# Telegram re-issues initData every time the Mini App is opened, so this just
# protects against very old/replayed payloads.
MAX_AUTH_AGE_SECONDS = 24 * 60 * 60


def validate_init_data(init_data: str) -> dict | None:
    """
    Returns the parsed Telegram user dict if `init_data` is genuine and fresh,
    otherwise None.
    """
    if not init_data:
        return None

    try:
        pairs = parse_qsl(init_data, strict_parsing=True, keep_blank_values=True)
    except ValueError:
        return None

    data = dict(pairs)
    received_hash = data.pop("hash", None)
    if not received_hash:
        return None

    data_check_string = "\n".join(f"{k}={v}" for k, v in sorted(data.items()))

    secret_key = hmac.new(b"WebAppData", TELEGRAM_BOT_TOKEN.encode(), hashlib.sha256).digest()
    computed_hash = hmac.new(
        secret_key, data_check_string.encode(), hashlib.sha256
    ).hexdigest()

    if not hmac.compare_digest(computed_hash, received_hash):
        return None

    auth_date = data.get("auth_date")
    if auth_date and (time.time() - int(auth_date)) > MAX_AUTH_AGE_SECONDS:
        return None

    user_raw = data.get("user")
    if not user_raw:
        return None

    try:
        return json.loads(user_raw)
    except json.JSONDecodeError:
        return None
