"""Common scoring (protocol v1.0 sections 8-9): trajectory MAE (log10) and trend HSS (Theil-Sen, ln scale)."""

from __future__ import annotations

import numpy as np

LN10 = np.log(10.0)
DELTA = 0.02  # ln units per day (D-025, D-039)
_DAYS = np.arange(1, 8, dtype=float)
_I, _J = np.triu_indices(7, k=1)


def theil_sen(v: np.ndarray) -> np.ndarray:
    """Row-wise Theil-Sen slope of (n,7) values against days 1..7 (pairs with a missing value ignored)."""
    s = (v[:, _J] - v[:, _I]) / (_DAYS[_J] - _DAYS[_I])
    if np.isfinite(s).all():
        return np.median(s, axis=1)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(s, axis=1)


def trend_class(y_log10: np.ndarray) -> np.ndarray:
    """-1 decrease, 0 no change, +1 increase, from the ln-scale slope."""
    s = theil_sen(y_log10 * LN10)
    return np.where(s > DELTA, 1, np.where(s < -DELTA, -1, 0))


def hss(obs: np.ndarray, pred: np.ndarray) -> float:
    """Multi-category Heidke skill score; 0 = no better than chance, 1 = perfect."""
    n = len(obs)
    cats = (-1, 0, 1)
    pc = np.mean(obs == pred)
    pe = sum(np.mean(obs == c) * np.mean(pred == c) for c in cats)
    return float((pc - pe) / (1 - pe)) if pe < 1 else float("nan")


def score(pred: np.ndarray, obs: np.ndarray) -> dict:
    """pred, obs: (n,7) log10. Failed origins must already be replaced (section 12).
    Missing obs (v2: interpolated days excluded) are skipped; the predicted trend is computed on the same days."""
    pm = np.where(np.isfinite(obs), pred, np.nan)
    err = np.abs(pm - obs)
    mae_per_origin = np.nanmean(err, axis=1)
    co, cp = trend_class(obs), trend_class(pm)
    return {"n": int(len(obs)), "mae": float(mae_per_origin.mean()), "hss": hss(co, cp),
            "trend_accuracy": float(np.mean(co == cp)), "mae_by_lead": np.round(np.nanmean(err, axis=0), 4).tolist(),
            "rmse": float(np.sqrt(np.nanmean(err ** 2))),
            "nse": float(1 - np.nansum(err ** 2) / np.nansum((obs - np.nanmean(obs)) ** 2))}


def persistence(y_t: np.ndarray) -> np.ndarray:
    return np.repeat(y_t[:, None], 7, axis=1)


def climatology_days(daily, train_days):
    """Training days with a real algal-density observation (the days the monthly climatology is built from)."""
    t = daily.index.intersection(train_days)
    return t[daily.loc[t, "y_obs"].to_numpy(bool)] if "y_obs" in daily else t


def climatology(daily, train_days, origins) -> np.ndarray:
    """Median y of the same calendar month over training days (D-041)."""
    d = daily.loc[daily.index.intersection(train_days)]
    med = d.loc[d["y_valid"], "y"].groupby(d.index[d["y_valid"]].month).median()
    out = np.empty((len(origins), 7))
    for h in range(1, 8):
        months = (origins + np.timedelta64(h, "D")).dt.month
        out[:, h - 1] = med.reindex(months).to_numpy()
    return out


def fill_failures(pred: np.ndarray, y_t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Replace origins without 7 finite values by persistence; return (pred, failed_mask)."""
    bad = ~np.isfinite(pred).all(axis=1)
    p = pred.copy()
    p[bad] = persistence(y_t[bad])
    return p, bad
