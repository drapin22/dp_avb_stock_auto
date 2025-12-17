# stockd/telegram_utils.py
from __future__ import annotations

import os
import time
from typing import Optional

import requests


TELEGRAM_MAX_CHARS = 3900  # sub 4096 ca să evităm erori


def _get_token_chat():
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    return token, chat_id


def send_telegram_message(text: str, parse_mode: Optional[str] = None) -> None:
    token, chat_id = _get_token_chat()
    if not token or not chat_id:
        print("[TELEGRAM] Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID. Skipping message.")
        return

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "disable_web_page_preview": True,
    }
    if parse_mode:
        payload["parse_mode"] = parse_mode

    r = requests.post(url, data=payload, timeout=25)
    if not r.ok:
        print(f"[TELEGRAM] sendMessage failed: {r.status_code} {r.text[:300]}")


def send_telegram_long_message(text: str, parse_mode: Optional[str] = None, sleep_s: float = 0.4) -> None:
    """
    Trimite automat mesaje lungi împărțindu-le pe linii, ca să nu depășească limita Telegram.
    """
    if not text:
        return

    lines = text.splitlines()
    chunks = []
    cur = ""

    for line in lines:
        candidate = (cur + "\n" + line) if cur else line
        if len(candidate) <= TELEGRAM_MAX_CHARS:
            cur = candidate
        else:
            if cur:
                chunks.append(cur)

            if len(line) > TELEGRAM_MAX_CHARS:
                start = 0
                while start < len(line):
                    chunks.append(line[start:start + TELEGRAM_MAX_CHARS])
                    start += TELEGRAM_MAX_CHARS
                cur = ""
            else:
                cur = line

    if cur:
        chunks.append(cur)

    for c in chunks:
        send_telegram_message(c, parse_mode=parse_mode)
        time.sleep(sleep_s)


def send_telegram_document(file_path: str, caption: str = "") -> None:
    token, chat_id = _get_token_chat()
    if not token or not chat_id:
        print("[TELEGRAM] Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID. Skipping document.")
        return

    url = f"https://api.telegram.org/bot{token}/sendDocument"
    with open(file_path, "rb") as f:
        files = {"document": f}
        data = {"chat_id": chat_id, "caption": caption[:1024]}
        r = requests.post(url, data=data, files=files, timeout=60)

    if not r.ok:
        print(f"[TELEGRAM] sendDocument failed: {r.status_code} {r.text[:300]}")


def send_telegram_photo(photo_path: str, caption: str = "") -> None:
    """
    Pentru weekly_report: trimite poze (grafice) generate local.
    """
    token, chat_id = _get_token_chat()
    if not token or not chat_id:
        print("[TELEGRAM] Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID. Skipping photo.")
        return

    url = f"https://api.telegram.org/bot{token}/sendPhoto"
    with open(photo_path, "rb") as f:
        files = {"photo": f}
        data = {"chat_id": chat_id, "caption": caption[:1024]}
        r = requests.post(url, data=data, files=files, timeout=60)

    if not r.ok:
        print(f"[TELEGRAM] sendPhoto failed: {r.status_code} {r.text[:300]}")
