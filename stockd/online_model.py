from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class RidgeConfig:
    l2: float = 10.0  # regularizare, mare = stabil
    feature_names: Tuple[str, ...] = (
        "bias",
        "ret_20d",
        "ret_60d",
        "vol_20d",
        "max_dd_60d",
        "beta",
        "bench_ret_5d",
        "vix_level",
        "dxy_ret_5d",
        "oil_ret_5d",
        "gold_ret_5d",
        "ro_proxy_ret_5d",
    )


def _to_float(x, default=0.0) -> float:
    try:
        v = float(x)
        if np.isfinite(v):
            return v
        return default
    except Exception:
        return default


def build_X(df: pd.DataFrame, cfg: RidgeConfig) -> np.ndarray:
    cols = list(cfg.feature_names)
    X = np.zeros((len(df), len(cols)), dtype=float)
    for i, c in enumerate(cols):
        if c == "bias":
            X[:, i] = 1.0
        else:
            X[:, i] = pd.to_numeric(df.get(c), errors="coerce").fillna(0.0).astype(float).values
    return X


def fit_ridge(df: pd.DataFrame, y_col: str, cfg: RidgeConfig) -> Dict:
    """
    Returnează dict cu coeficienți.
    """
    if df is None or df.empty:
        return {"ok": False, "reason": "empty_df"}

    y = pd.to_numeric(df[y_col], errors="coerce").fillna(0.0).astype(float).values
    X = build_X(df, cfg)

    # ridge closed-form: (X'X + l2 I)^-1 X'y
    XtX = X.T @ X
    I = np.eye(XtX.shape[0])
    XtX_reg = XtX + cfg.l2 * I
    Xty = X.T @ y

    try:
        w = np.linalg.solve(XtX_reg, Xty)
    except Exception:
        w = np.linalg.lstsq(XtX_reg, Xty, rcond=None)[0]

    return {
        "ok": True,
        "l2": cfg.l2,
        "feature_names": list(cfg.feature_names),
        "weights": [float(v) for v in w],
        "n": int(len(df)),
        "y_mean": float(np.mean(y)) if len(y) else 0.0,
    }


def predict(df: pd.DataFrame, model_state: Dict) -> np.ndarray:
    if not model_state or not model_state.get("ok"):
        return np.zeros(len(df), dtype=float)

    fn = model_state.get("feature_names", [])
    w = np.array(model_state.get("weights", []), dtype=float)
    if not fn or w.size != len(fn):
        return np.zeros(len(df), dtype=float)

    cfg = RidgeConfig(l2=float(model_state.get("l2", 10.0)), feature_names=tuple(fn))
    X = build_X(df, cfg)
    return X @ w


def save_state(path: Path, state: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def load_state(path: Path) -> Dict:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return {}
