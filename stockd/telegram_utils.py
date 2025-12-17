# stockd/telegram_utils.py
from __future__ import annotations

import os
from typing import Optional
import requests


def _get_creds() -> tuple[str, str]:
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not bot_token or not chat_id:
        raise RuntimeError("Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID in environment.")
    return bot_token, chat_id


def send_telegram_message(text: str, parse_mode: str = "Markdown") -> None:
    bot_token, chat_id = _get_creds()
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text}
    if parse_mode:
        payload["parse_mode"] = parse_mode
    r = requests.post(url, json=payload, timeout=30)
    r.raise_for_status()


def send_telegram_document(path: str, caption: Optional[str] = None, parse_mode: str = "Markdown") -> None:
    bot_token, chat_id = _get_creds()
    url = f"https://api.telegram.org/bot{bot_token}/sendDocument"
    with open(path, "rb") as f:
        files = {"document": f}
        data = {"chat_id": chat_id}
        if caption:
            data["caption"] = caption
            if parse_mode:
                data["parse_mode"] = parse_mode
        r = requests.post(url, data=data, files=files, timeout=90)
        r.raise_for_status()


def send_telegram_photo(path: str, caption: Optional[str] = None, parse_mode: str = "Markdown") -> None:
    bot_token, chat_id = _get_creds()
    url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
    with open(path, "rb") as f:
        files = {"photo": f}
        data = {"chat_id": chat_id}
        if caption:
            data["caption"] = caption
            if parse_mode:
                data["parse_mode"] = parse_mode
        r = requests.post(url, data=data, files=files, timeout=90)
        r.raise_for_status()
