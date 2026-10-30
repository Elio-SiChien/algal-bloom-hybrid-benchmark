"""v2 dataset (D-050..D-053): 2023-01-01..2024-12-31, gaps linearly interpolated within each segment,
split train / test / forecast as in Saravani et al. (2025).

Segments (interpolation never crosses a boundary):
    train    2023-01-01 .. 2024-04-30   (4 CV blocks of 4 months for configuration selection)
    test     2024-05-01 .. 2024-08-31
    forecast 2024-09-01 .. 2024-12-31
Interpolation: y (log10 algal density), water temperature, TN, TP; linear in time, interior gaps of any
length; leading/trailing gaps of a segment stay missing. `<var>_filled` marks interpolated values,
`y_obs` marks real observations.
Origin t (all methods, common list): y, water temperature, TN and TP are REAL observations on day t (so no
interpolated value on t-6..t can depend on data after t); t-6..t complete; t+1..t+7 inside t's segment.
`n_obs_labels` = number of real observations among t+1..t+7. Main scoring uses origins with
n_obs_labels >= 4 and real-observation days only; scoring on all days (interpolated labels) is reported
as a supplement.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

START, END = "2023-01-01", "2024-12-31"
SEGMENTS = (("train", "2023-01-01", "2024-04-30"), ("test", "2024-05-01", "2024-08-31"),
            ("forecast", "2024-09-01", "2024-12-31"))
CV_BLOCKS = (("cv_block1", "2023-01-01", "2023-04-30"), ("cv_block2", "2023-05-01", "2023-08-31"),
             ("cv_block3", "2023-09-01", "2023-12-31"), ("cv_block4", "2024-01-01", "2024-04-30"))
EVAL_SPLITS = ("test", "forecast")
VARS = ("y", "wt", "tn", "tp")
MIN_OBS_LABELS = 4
HISTORY, HORIZON = 7, 7


def interpolate_segment(x: pd.Series) -> pd.Series:
    return x.interpolate(method="time", limit_area="inside")


def build_daily(qc_daily: pd.DataFrame) -> pd.DataFrame:
    """qc_daily: v1 QC'd daily frame (index date). Returns the v2 daily frame."""
    d = qc_daily.loc[START:END].copy()
    obs = {"y": d["y_valid"].astype(bool) & d["y"].notna()}
    for v in ("wt", "tn", "tp"):
        obs[v] = d[v].notna() & ~d[f"{v}_filled"].astype(bool)
    out = pd.DataFrame(index=d.index)
    for v in VARS:
        raw = d[v].where(obs[v])
        filled = pd.concat([interpolate_segment(raw.loc[a:b]) for _, a, b in SEGMENTS])
        out[v] = filled
        out[f"{v}_filled"] = filled.notna() & ~obs[v]
    out["y_obs"] = obs["y"]
    out["y_valid"] = out["y"].notna()
    out["segment"] = None
    for name, a, b in SEGMENTS:
        out.loc[a:b, "segment"] = name
    return out


def origins(daily: pd.DataFrame) -> pd.DataFrame:
    real_t = daily["y_obs"] & ~daily[["wt_filled", "tn_filled", "tp_filled"]].any(axis=1) & daily[["wt", "tn", "tp"]].notna().all(axis=1)
    comp = daily[list(VARS)].notna().all(axis=1)
    hist_ok = comp.rolling(HISTORY, min_periods=HISTORY).sum() == HISTORY
    rows = []
    idx = daily.index
    seg = daily["segment"].to_numpy()
    yobs = daily["y_obs"].to_numpy()
    for i, t in enumerate(idx):
        if i + HORIZON >= len(idx) or not (real_t.iloc[i] and hist_ok.iloc[i]):
            continue
        if seg[i + HORIZON] != seg[i] or not comp.iloc[i + 1:i + HORIZON + 1].all():
            continue
        name = seg[i]
        if name == "train":
            name = next(n for n, a, b in CV_BLOCKS if pd.Timestamp(a) <= t <= pd.Timestamp(b))
        rows.append((t, name, t + pd.Timedelta(days=HORIZON), int(yobs[i + 1:i + HORIZON + 1].sum())))
    return pd.DataFrame(rows, columns=["origin", "split_name", "last_label_date", "n_obs_labels"])


def origin_list_sha256(o: pd.DataFrame) -> str:
    text = "\n".join(f"{t:%Y-%m-%d},{s},{n}" for t, s, n in zip(o["origin"], o["split_name"], o["n_obs_labels"]))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def origins_all_days(daily: pd.DataFrame) -> pd.DataFrame:
    """v2b (D-054): an origin on EVERY day whose t-6..t+7 window is complete and whose labels stay in t's
    segment, whether or not day t itself is a real observation. Scheme 1 scores all of them on all days;
    scheme 2 (main) keeps origins whose day t is real and that have >= MIN_OBS_LABELS real labels, real days only."""
    comp = daily[list(VARS)].notna().all(axis=1)
    hist_ok = comp.rolling(HISTORY, min_periods=HISTORY).sum() == HISTORY
    real_t = daily["y_obs"] & ~daily[["wt_filled", "tn_filled", "tp_filled"]].any(axis=1)
    rows = []
    idx = daily.index
    seg = daily["segment"].to_numpy()
    yobs = daily["y_obs"].to_numpy()
    for i, t in enumerate(idx):
        if i + HORIZON >= len(idx) or not hist_ok.iloc[i]:
            continue
        if seg[i + HORIZON] != seg[i] or not comp.iloc[i + 1:i + HORIZON + 1].all():
            continue
        name = seg[i]
        if name == "train":
            name = next(n for n, a, b in CV_BLOCKS if pd.Timestamp(a) <= t <= pd.Timestamp(b))
        rows.append((t, name, t + pd.Timedelta(days=HORIZON), int(yobs[i + 1:i + HORIZON + 1].sum()), bool(real_t.iloc[i])))
    return pd.DataFrame(rows, columns=["origin", "split_name", "last_label_date", "n_obs_labels", "real_t"])
