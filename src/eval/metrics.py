"""Métricas pontuais e distribucionais."""
from __future__ import annotations

import numpy as np
import pandas as pd


def mae(y, yhat):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(yhat))))


def rmse(y, yhat):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(yhat)) ** 2)))


def r2(y, yhat):
    """Coeficiente de determinação. NaN quando o alvo é constante (var=0),
    caso comum em municípios com quase-tudo-zero (ex.: hanseníase)."""
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot <= 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


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
    if len(y) == 0:
        return {
            "model": name, "disease": disease, "horizon": horizon, "n": 0,
            "mae": float("nan"), "rmse": float("nan"), "r2": float("nan"),
            "smape": float("nan"), "mape": float("nan"),
        }
    return {
        "model": name,
        "disease": disease,
        "horizon": horizon,
        "n": int(len(y)),
        "mae": mae(y, yh),
        "rmse": rmse(y, yh),
        "r2": r2(y, yh),
        "smape": smape(y, yh),
        "mape": mape(y, yh),
    }


def aggregate_results(records: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    return df


# ============================================================================
# MÉTRICAS ADICIONAIS (revisão): DTW e recall/precisão de eventos não-zero.
# ============================================================================
def recall_nonzero(y_true, y_pred, thr: float = 0.5) -> dict:
    """Detecção de eventos (caso > 0) numa série zero-inflada.

    evento_real  = y_true > 0
    evento_previsto = round(y_pred) >= 1  <=>  y_pred >= thr (thr=0.5)

    Retorna recall, precisão, F1 e contagens. Recall é a métrica-chave: dos
    meses-município com casos reais, quantos o modelo sinalizou.
    """
    y = np.asarray(y_true, dtype=float)
    yh = np.asarray(y_pred, dtype=float)
    mask = ~(np.isnan(y) | np.isnan(yh))
    y, yh = y[mask], yh[mask]
    ev_t = y > 0
    ev_p = yh >= thr
    tp = int(np.sum(ev_t & ev_p))
    fn = int(np.sum(ev_t & ~ev_p))
    fp = int(np.sum(~ev_t & ev_p))
    rec = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    f1 = (2 * prec * rec / (prec + rec)) if (prec and rec and (prec + rec) > 0) else float("nan")
    return {"n_events": tp + fn, "recall_nz": rec, "precision_nz": prec, "f1_nz": f1}


def dtw_distance(y_true_seq, y_pred_seq, normalize: bool = True) -> float:
    """DTW (fastdtw) entre a trajetória observada e a prevista de UMA série.

    Mede aderência de FORMA da curva, complementando o erro pontual (MAE).
    normalize=True divide pela quantidade de passos, tornando comparável entre
    séries de comprimentos diferentes. Requer `fastdtw` (pip install fastdtw).
    """
    from fastdtw import fastdtw

    a = np.asarray(y_true_seq, dtype=float)
    b = np.asarray(y_pred_seq, dtype=float)
    m = ~(np.isnan(a) | np.isnan(b))
    a, b = a[m], b[m]
    if len(a) < 2:
        return float("nan")
    dist, _ = fastdtw(a, b, dist=lambda x, y: abs(x - y))
    return dist / len(a) if normalize else dist


def dtw_by_group(df, group_col: str = "cd_mun", time_col: str = "time_idx",
                 y_true_col: str = "y_true", y_pred_col: str = "y_pred",
                 normalize: bool = True) -> float:
    """DTW médio entre grupos (ex.: municípios). Ordena cada série por tempo,
    calcula o DTW normalizado e retorna a média entre grupos."""
    vals = []
    for _, g in df.groupby(group_col):
        g = g.sort_values(time_col)
        d = dtw_distance(g[y_true_col].values, g[y_pred_col].values, normalize=normalize)
        if not np.isnan(d):
            vals.append(d)
    return float(np.mean(vals)) if vals else float("nan")
