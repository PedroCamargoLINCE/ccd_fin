"""Métricas pontuais e distribucionais."""
from __future__ import annotations

import numpy as np
import pandas as pd


def mae(y, yhat):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(yhat))))


def rmse(y, yhat):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(yhat)) ** 2)))


def smape(y, yhat):
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    denom = (np.abs(y) + np.abs(yhat)) / 2
    mask = denom > 0
    if not mask.any():
        return 0.0
    return float(np.mean(np.abs(y[mask] - yhat[mask]) / denom[mask])) * 100


def mape(y, yhat, eps: float = 1.0):
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    return float(np.mean(np.abs(y - yhat) / np.maximum(y, eps))) * 100


def weighted_quantile_loss(y, quantiles: np.ndarray, quantile_levels: tuple[float, ...]) -> float:
    """
    quantiles: shape (n, len(quantile_levels))
    quantile_levels: ex (0.1, 0.5, 0.9)
    """
    y = np.asarray(y, dtype=float).reshape(-1, 1)
    q = np.asarray(quantiles, dtype=float)
    taus = np.array(quantile_levels).reshape(1, -1)
    diff = y - q
    loss = np.maximum(taus * diff, (taus - 1) * diff)
    return float(2 * loss.sum() / np.abs(y).sum()) if np.abs(y).sum() > 0 else float(loss.mean())


def evaluate(y_true, y_pred, name: str = "model", disease: str | None = None, horizon: int | None = None) -> dict:
    y = np.asarray(y_true, dtype=float)
    yh = np.asarray(y_pred, dtype=float)
    mask = ~(np.isnan(y) | np.isnan(yh))
    y, yh = y[mask], yh[mask]
    return {
        "model": name,
        "disease": disease,
        "horizon": horizon,
        "n": int(len(y)),
        "mae": mae(y, yh),
        "rmse": rmse(y, yh),
        "smape": smape(y, yh),
        "mape": mape(y, yh),
    }


def aggregate_results(records: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    return df
