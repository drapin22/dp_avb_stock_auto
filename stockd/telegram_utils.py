from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import requests

from stockd import settings


def _has_creds() -> bool:
    return bool(settings.TELEGRAM_BOT_TOKEN and settings.TELEGRAM_CHAT_ID)


def _api(method: str) -> str:
    return f"https://api.telegram.org/bot{settings.TELEGRAM_BOT_TOKEN}/{method}"


def send_telegram_message(text: str, parse_mode: Optional[str] = None) -> bool:
    if not _has_creds():
        print("[TG] Missing TELEGRAM creds, skipping message.")
        return False

    payload = {"chat_id": settings.TELEGRAM_CHAT_ID, "text": text}
    if parse_mode:
        payload["parse_mode"] = parse_mode

    try:
        r = requests.post(_api("sendMessage"), json=payload, timeout=30)
        r.raise_for_status()
        return True
    except Exception as e:
        print(f"[TG] sendMessage error: {e}")
        return False


def send_chunked_message(text: str, parse_mode: Optional[str] = None) -> None:
    if not _has_creds():
        print("[TG] Missing TELEGRAM creds, skipping chunked message.")
        return

    max_len = settings.TELEGRAM_MAX_CHARS
    lines = text.splitlines()
    chunks = []
    cur = []
    cur_len = 0

    for line in lines:
        add_len = len(line) + 1
        if cur and (cur_len + add_len > max_len):
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


def send_telegram_document(path: str | Path, caption: str | None = None) -> bool:
    if not _has_creds():
        print("[TG] Missing TELEGRAM creds, skipping document.")
        return False

    p = Path(path)
    if not p.exists():
        print(f"[TG] Document not found: {p}")
        return False

    try:
        with p.open("rb") as f:
            files = {"document": (p.name, f)}
            data = {"chat_id": settings.TELEGRAM_CHAT_ID, "caption": caption or ""}
            r = requests.post(_api("sendDocument"), data=data, files=files, timeout=60)
            r.raise_for_status()
            return True
    except Exception as e:
        print(f"[TG] sendDocument error: {e}")
        return False


def send_telegram_photo(path: str | Path, caption: str | None = None) -> bool:
    if not _has_creds():
        print("[TG] Missing TELEGRAM creds, skipping photo.")
        return False

    p = Path(path)
    if not p.exists():
        print(f"[TG] Photo not found: {p}")
        return False

    try:
        with p.open("rb") as f:
            files = {"photo": (p.name, f)}
            data = {"chat_id": settings.TELEGRAM_CHAT_ID, "caption": caption or ""}
            r = requests.post(_api("sendPhoto"), data=data, files=files, timeout=60)
            r.raise_for_status()
            return True
    except Exception as e:
        print(f"[TG] sendPhoto error: {e}")
        return False
