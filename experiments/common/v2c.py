"""v2c (user decision 2026-10-20): the same 2023-2024 daily data as v2/v2b, but gaps are interpolated over the
WHOLE record in one piece, so the forecasts and figures have no break at period boundaries. Models are NOT
refitted: the frozen v2b models (saved parameters) re-forecast every origin (experiments/reforecast.py).

Periods for reporting: training 2023-01-01..2024-04-30 (model fitting, unchanged) and evaluation
2024-05-01..2024-12-31 (the former test and forecast periods pooled, option A).
Cost of one-piece interpolation, stated in the paper: a training-period gap that ends after 30 April can be filled
from an early-May observation (a few days at most); this does not change any fitted parameter because the models
were fitted on the v2b data.
Origin t: t-6..t+7 complete. Main scoring (scheme 2, D-054): t >= 2024-05-01, day t observed (y, water
temperature, TN, TP real), >= 4 observed labels, accuracy on observed days only.
"""
from __future__ import annotations

import hashlib

import pandas as pd

from .v2 import CV_BLOCKS, HISTORY, HORIZON, START, END, VARS

TRAIN = ("train", "2023-01-01", "2024-04-30")
EVAL = ("eval", "2024-05-01", "2024-12-31")
SEGMENTS = (TRAIN, EVAL)


def build_daily(qc_daily: pd.DataFrame) -> pd.DataFrame:
    """qc_daily: v1 QC'd daily frame (index date). One-piece linear interpolation of y, wt, tn, tp."""
    d = qc_daily.loc[START:END].copy()
    obs = {"y": d["y_valid"].astype(bool) & d["y"].notna()}
    for v in ("wt", "tn", "tp"):
        obs[v] = d[v].notna() & ~d[f"{v}_filled"].astype(bool)
    out = pd.DataFrame(index=d.index)
    for v in VARS:
        raw = d[v].where(obs[v])
        filled = raw.interpolate(method="time", limit_area="inside")
        out[v] = filled
        out[f"{v}_filled"] = filled.notna() & ~obs[v]
    out["y_obs"] = obs["y"]
    out["y_valid"] = out["y"].notna()
    out["segment"] = None
    for name, a, b in SEGMENTS:
        out.loc[a:b, "segment"] = name
    return out


def origins(daily: pd.DataFrame) -> pd.DataFrame:
    """Every day t with t-6..t+7 complete. split_name: cv_blockN for training-period origins, 'eval' from 1 May 2024."""
    comp = daily[list(VARS)].notna().all(axis=1)
    hist_ok = comp.rolling(HISTORY, min_periods=HISTORY).sum() == HISTORY
    real_t = daily["y_obs"] & ~daily[["wt_filled", "tn_filled", "tp_filled"]].any(axis=1)
    yobs = daily["y_obs"].to_numpy()
    rows = []
    for i, t in enumerate(daily.index):
        if i + HORIZON >= len(daily) or not hist_ok.iloc[i] or not comp.iloc[i + 1:i + HORIZON + 1].all():
            continue
        if t >= pd.Timestamp(EVAL[1]):
            name = "eval"
        else:
            name = next(n for n, a, b in CV_BLOCKS if pd.Timestamp(a) <= t <= pd.Timestamp(b))
        rows.append((t, name, t + pd.Timedelta(days=HORIZON), int(yobs[i + 1:i + HORIZON + 1].sum()), bool(real_t.iloc[i])))
    return pd.DataFrame(rows, columns=["origin", "split_name", "last_label_date", "n_obs_labels", "real_t"])


def sha256_text(o: pd.DataFrame) -> str:
    text = "\n".join(f"{t:%Y-%m-%d},{s},{n},{r}" for t, s, n, r in o[["origin", "split_name", "n_obs_labels", "real_t"]].itertuples(index=False))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
