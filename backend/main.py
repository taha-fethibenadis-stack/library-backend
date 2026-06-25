"""
Promo Library backend — single process running on Railway.

Combines:
  - A Telegram bot (webhook mode) that watches the group chat for uploaded
    files, classifies them with Gemini, and stores them in Supabase.
  - A small public API the Vercel-hosted Mini App calls to list/search files
    and to ask the bot to DM a file to the requesting student.

Run locally:
    uvicorn main:app --reload --port 8000
    (you'll also need a tunnel like ngrok if you want Telegram to reach your
    webhook locally — see README.md)
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from telegram import Update, Bot
from telegram.ext import Application, MessageHandler, ContextTypes, filters

import config
import db
from gemini_classify import classify_file, rank_search
from telegram_auth import validate_init_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("promo_library")

telegram_app: Application = Application.builder().token(config.TELEGRAM_BOT_TOKEN).build()
bot: Bot = telegram_app.bot


# ---------------------------------------------------------------------------
# Telegram handlers
# ---------------------------------------------------------------------------

async def handle_upload(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Fires on any document or photo posted in the group."""
    message = update.effective_message
    if message is None:
        return

    # Optional: restrict auto-save to one specific configured group
    if config.TELEGRAM_GROUP_ID and str(message.chat_id) != str(config.TELEGRAM_GROUP_ID):
        return

    document = message.document
    photo = message.photo[-1] if message.photo else None  # largest size

    if not document and not photo:
        return

    if document:
        file_id = document.file_id
        file_unique_id = document.file_unique_id
        file_name = document.file_name or "Untitled file"
        mime_type = document.mime_type
    else:
        file_id = photo.file_id
        file_unique_id = photo.file_unique_id
        file_name = f"photo_{photo.file_unique_id}.jpg"
        mime_type = "image/jpeg"

    if db.file_exists(file_unique_id):
        logger.info("Duplicate file skipped: %s", file_unique_id)
        return

    sender = message.from_user
    caption = message.caption

    # Try to download the actual bytes so Gemini can read real content
    # (works great for PDFs and images; harmless no-op fallback otherwise).
    file_bytes = None
    try:
        tg_file = await context.bot.get_file(file_id)
        # Guard against very large files (free Gemini inline-data limit, Telegram bot 20MB cap)
        if tg_file.file_size and tg_file.file_size <= 15 * 1024 * 1024:
            file_bytes = bytes(await tg_file.download_as_bytearray())
    except Exception:
        logger.exception("Could not download file %s for Gemini analysis", file_name)

    classification = classify_file(
        file_name=file_name,
        caption=caption,
        mime_type=mime_type,
        file_bytes=file_bytes,
    )

    record = {
        "file_id": file_id,
        "file_unique_id": file_unique_id,
        "file_name": file_name,
        "is_photo": photo is not None,
        "mime_type": mime_type,
        "sender_username": sender.username if sender else None,
        "sender_user_id": sender.id if sender else None,
        "subject": classification["subject"],
        "tag": classification["tag"],
        "summary": classification["summary"],
    }
    db.insert_file(record)

    await message.reply_text(f"📥 Saved {file_name} to the Promo Library!")


telegram_app.add_handler(
    MessageHandler(
        (filters.Document.ALL | filters.PHOTO) & filters.ChatType.GROUPS,
        handle_upload,
    )
)


# ---------------------------------------------------------------------------
# FastAPI app + lifecycle (sets the Telegram webhook on startup)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    await telegram_app.initialize()
    await bot.set_webhook(
        url=config.WEBHOOK_URL,
        secret_token=config.TELEGRAM_WEBHOOK_SECRET,
        allowed_updates=Update.ALL_TYPES,
    )
    await telegram_app.start()
    logger.info("Telegram webhook set to %s", config.WEBHOOK_URL)
    yield
    await telegram_app.stop()
    await telegram_app.shutdown()


app = FastAPI(title="Promo Library API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.post(config.WEBHOOK_PATH)
async def telegram_webhook(request: Request):
    secret = request.headers.get("X-Telegram-Bot-Api-Secret-Token")
    if secret != config.TELEGRAM_WEBHOOK_SECRET:
        raise HTTPException(status_code=403, detail="Invalid webhook secret")

    data = await request.json()
    update = Update.de_json(data, bot)
    await telegram_app.process_update(update)
    return {"ok": True}


@app.get("/health")
async def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Public API for the Mini App frontend
# ---------------------------------------------------------------------------

@app.get("/api/files")
async def api_files():
    """Returns every categorized file — the frontend groups them client-side."""
    return {"files": db.get_all_files()}


class SearchRequest(BaseModel):
    init_data: str
    query: str


@app.post("/api/search")
async def api_search(payload: SearchRequest):
    user = validate_init_data(payload.init_data)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid Telegram authentication")

    query = payload.query.strip()
    if not query:
        return {"results": []}

    all_files = db.get_all_files()
    matched_ids = rank_search(query, all_files)
    matched_files = db.get_files_by_ids(matched_ids)

    # Preserve Gemini's best-match-first ordering
    order = {fid: i for i, fid in enumerate(matched_ids)}
    matched_files.sort(key=lambda f: order.get(f["id"], 999))

    return {"results": matched_files}


class SendFileRequest(BaseModel):
    init_data: str
    file_id: str
    file_name: str
    is_photo: bool = False


@app.post("/api/send_file")
async def api_send_file(payload: SendFileRequest):
    """
    Called when a student taps a search/library result. Sends the file
    directly to that student's private chat with the bot.
    Note: the student must have started a private chat with the bot at least
    once (Telegram requirement), otherwise this will fail with "chat not found".
    """
    user = validate_init_data(payload.init_data)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid Telegram authentication")

    chat_id = user["id"]
    try:
        if payload.is_photo:
            await bot.send_photo(chat_id=chat_id, photo=payload.file_id)
        else:
            await bot.send_document(chat_id=chat_id, document=payload.file_id)
    except Exception as e:
        logger.exception("Failed to send file to user %s", chat_id)
        raise HTTPException(
            status_code=400,
            detail=(
                "Couldn't send the file. Make sure you've opened a private chat "
                "with the bot at least once, then try again. "
                f"({e})"
            ),
        )

    return {"ok": True, "message": f"Sent {payload.file_name} to your private chat."}
