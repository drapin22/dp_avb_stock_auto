from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import requests

from stockd import settings


def _has_telegram() -> bool:
    return bool(settings.TELEGRAM_BOT_TOKEN and settings.TELEGRAM_CHAT_ID)


def _api_url(method: str) -> str:
    return f"https://api.telegram.org/bot{settings.TELEGRAM_BOT_TOKEN}/{method}"


def send_telegram_message(text: str, parse_mode: Optional[str] = None, disable_preview: bool = True) -> bool:
    if not _has_telegram():
        return False

    payload = {
        "chat_id": settings.TELEGRAM_CHAT_ID,
        "text": text,
        "disable_web_page_preview": disable_preview,
    }
    if parse_mode:
        payload["parse_mode"] = parse_mode

    r = requests.post(_api_url("sendMessage"), json=payload, timeout=30)
    return r.ok


def send_telegram_document(file_path: str | Path, caption: Optional[str] = None) -> bool:
    if not _has_telegram():
        return False

    p = Path(file_path)
    if not p.exists():
        return False

    data = {"chat_id": settings.TELEGRAM_CHAT_ID}
    if caption:
        data["caption"] = caption

    with p.open("rb") as f:
        files = {"document": (p.name, f)}
        r = requests.post(_api_url("sendDocument"), data=data, files=files, timeout=60)
    return r.ok


def send_telegram_photo(image_path: str | Path, caption: Optional[str] = None) -> bool:
    if not _has_telegram():
        return False

    p = Path(image_path)
    if not p.exists():
        return False

    data = {"chat_id": settings.TELEGRAM_CHAT_ID}
    if caption:
        data["caption"] = caption

    with p.open("rb") as f:
        files = {"photo": (p.name, f)}
        r = requests.post(_api_url("sendPhoto"), data=data, files=files, timeout=60)
    return r.ok


def send_chunked_message(text: str, parse_mode: Optional[str] = None) -> None:
    if not _has_telegram():
        return

    max_len = settings.TELEGRAM_MAX_CHARS
    chunks = []
    cur = []
    cur_len = 0

    for line in text.splitlines():
        add_len = len(line) + 1
        if cur_len + add_len > max_len and cur:
            chunks.append("\n".join(cur))
            cur = [line]
            cur_len = len(line) + 1
        else:
            cur.append(line)
            cur_len += add_len

    if cur:
        chunks.append("\n".join(cur))

    for i, c in enumerate(chunks):
        send_telegram_message(c, parse_mode=parse_mode)
        if i < len(chunks) - 1:
            time.sleep(0.7)
