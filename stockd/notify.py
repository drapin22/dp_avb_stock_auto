# stockd/notify.py
from __future__ import annotations

def send_telegram_message(text: str) -> None:
    """
    Compatibilitate: în codul existent ai import din stockd.notify.
    De acum delegăm totul către stockd.telegram_utils.
    """
    from stockd.telegram_utils import send_telegram_message as _send
    try:
        _send(text)
    except Exception as exc:
        print(f"[TG] Could not send Telegram message: {exc}")
