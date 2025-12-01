# stockd/notify.py
from __future__ import annotations

import os
import requests

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")


def send_telegram_message(text: str) -> None:
    """Trimite un mesaj simplu pe Telegram.

    Dacă nu există token / chat id, doar loghează și iese.
    """
    if not BOT_TOKEN or not CHAT_ID:
        print("[TG] Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID, skipping.")
        return

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": text,
    }

    try:
        resp = requests.post(url, json=payload, timeout=10)
        resp.raise_for_status()
        print("[TG] Telegram message sent.")
    except Exception as exc:
        print(f"[TG] Error sending Telegram message: {exc}")
