import os
import requests


def _get_creds():
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not bot_token or not chat_id:
        raise RuntimeError(
            "Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID in environment."
        )
    return bot_token, chat_id


def send_telegram_message(text: str) -> None:
    bot_token, chat_id = _get_creds()
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text, "parse_mode": "Markdown"}
    r = requests.post(url, json=payload, timeout=30)
    r.raise_for_status()


def send_telegram_document(path: str, caption: str | None = None) -> None:
    bot_token, chat_id = _get_creds()
    url = f"https://api.telegram.org/bot{bot_token}/sendDocument"
    with open(path, "rb") as f:
        files = {"document": f}
        data = {"chat_id": chat_id, "caption": caption or ""}
        r = requests.post(url, data=data, files=files, timeout=60)
        r.raise_for_status()


def send_telegram_photo(path: str, caption: str | None = None) -> None:
    bot_token, chat_id = _get_creds()
    url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
    with open(path, "rb") as f:
        files = {"photo": f}
        data = {"chat_id": chat_id, "caption": caption or ""}
        r = requests.post(url, data=data, files=files, timeout=60)
        r.raise_for_status()
