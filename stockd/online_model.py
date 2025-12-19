from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from stockd import settings


DEFAULT_FEATURE_COLS = [
    "ret_20d",
    "ret_60d",
    "vol_20d",
    "max_dd_60d",
    "beta",
    "vix_level",
    "dxy_ret_5d",
    "oil_ret_5d",
    "gold_ret_5d",
    "bench_ret_5d",
]


def load_state(path: Path | None = None) -> Dict:
    """
    Loads model state from JSON. If missing, returns a safe default state.
    """
    path = Path(path) if path is not None else Path(settings.MODEL_STATE_JSON)

    if not path.exists():
        return {
            "feature_cols": DEFAULT_FEATURE_COLS,
            "coef": [0.0] * len(DEFAULT_FEATURE_COLS),
            "intercept": 0.0,
            "trained_on_target_date": None,
            "n_samples": 0,
        }

    try:
        state = json.loads(path.read_text(encoding="utf-8"))
        # minimal validation
        feature_cols = state.get("feature_cols") or DEFAULT_FEATURE_COLS
        coef = state.get("coef") or [0.0] * len(feature_cols)
        if len(coef) != len(feature_cols):
            coef = [0.0] * len(feature_cols)

        return {
            "feature_cols": feature_cols,
            "coef": coef,
            "intercept": float(state.get("intercept") or 0.0),
            "trained_on_target_date": state.get("trained_on_target_date"),
            "n_samples": int(state.get("n_samples") or 0),
        }
    except Exception:
        # corrupted file -> fall back
        return {
            "feature_cols": DEFAULT_FEATURE_COLS,
            "coef": [0.0] * len(DEFAULT_FEATURE_COLS),
            "intercept": 0.0,
            "trained_on_target_date": None,
            "n_samples": 0,
        }


def save_state(state: Dict, path: Path | None = None) -> None:
    """
    Saves model state to JSON.
    """
    path = Path(path) if path is not None else Path(settings.MODEL_STATE_JSON)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def predict(df: pd.DataFrame, state: Dict | None = None) -> np.ndarray:
    """
    Linear model prediction: y = intercept + X @ coef
    """
    state = state or load_state()
    cols: List[str] = list(state.get("feature_cols") or DEFAULT_FEATURE_COLS)
    coef = np.asarray(state.get("coef") or [0.0] * len(cols), dtype=float)
    intercept = float(state.get("intercept") or 0.0)

    X = df.copy()
    for c in cols:
        if c not in X.columns:
            X[c] = 0.0
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0.0)

    mat = X[cols].to_numpy(dtype=float)
    if mat.shape[1] != coef.shape[0]:
        # mismatch -> return intercept only
        return np.full((mat.shape[0],), intercept, dtype=float)

    return intercept + mat @ coef
