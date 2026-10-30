"""v3 dataset (D-059, D-060): 10 waterbodies, 2023 training / 2024 independent evaluation.

Data: CNEMC 4-hourly readings 2023-01-01..2024-12-28 -> QC (qc.py, D-034/D-041/D-043/D-044) -> daily ->
y (log10 algal density), water temperature, TN, TP linearly interpolated over the WHOLE record in one piece.
`<var>_filled` = not a real daily observation (interpolated, incl. the <=2-day driver fills of qc.py).

Origins t (t-6..t+7 complete):
  training  t in 2023, and t+7 <= L, where L = the earliest of the four variables' last real observation in
            2023 (no training input or label can be interpolated from a 2024 observation). CV blocks = quarters.
  eval      2024-01-01 <= t <= 2024-12-21 (t+7 <= last data day). Scored when day t is real for y, water
            temperature, TN and TP (no interpolated value on t-6..t can use data after t) and >= 4 of t+1..t+7 are
            real; accuracy on real days only. The interpolated-label scoring of v2 (scheme 1) is dropped.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

START, END = "2023-01-01", "2024-12-28"
TRAIN_YEAR_END, EVAL_START, EVAL_LAST_ORIGIN = "2023-12-31", "2024-01-01", "2024-12-21"
CV_BLOCKS = (("cv_block1", "2023-01-01", "2023-03-31"), ("cv_block2", "2023-04-01", "2023-06-30"),
             ("cv_block3", "2023-07-01", "2023-09-30"), ("cv_block4", "2023-10-01", "2023-12-31"))
VARS = ("y", "wt", "tn", "tp")
QC_VERSION = "qc-v1.3"   # qc-v1.1 (qc.py) + detection-floor rule (D-061) + TN/TP outlier rule (D-062)
ALGAE_FLOOR = 1e4        # D-061: algal density <= 1e4 cells/L is the instrument floor, not a measurement -> removed
MIN_OBS_LABELS, HISTORY, HORIZON = 4, 7, 7

# site_id: (basin, station, waterbody, English name, station lon, station lat) -- coordinates D-060
SITES = {
    "dianchi": ("滇池流域", "观音山西", "滇池", "Lake Dianchi", 102.6756, 24.8372),
    "chaohu": ("巢湖流域", "龟山", "巢湖", "Lake Chaohu", 117.8069, 31.6111),
    "taihu": ("太湖流域", "锡东水厂", "太湖", "Lake Taihu", 120.3722, 31.4483),
    "erhai": ("西南诸河", "洱海湖心", "洱海", "Lake Erhai", 100.1875, 25.7633),
    "fuxian": ("珠江流域", "抚仙湖心", "抚仙湖", "Lake Fuxian", 102.8931, 24.5265),
    "liangzi": ("长江流域", "西梁子湖湖南", "梁子湖", "Lake Liangzi", 114.4466, 30.1954),
    "hongze": ("淮河流域", "临淮乡", "洪泽湖", "Lake Hongze", 118.3691, 33.2730),
    "shahu": ("黄河流域", "沙湖", "沙湖", "Lake Shahu", 106.3728, 38.8170),
    "hedi": ("珠江流域", "渠首", "鹤地水库", "Hedi Reservoir", 110.3126, 21.7088),
    "yinghu": ("长江流域", "瀛湖坝前", "瀛湖", "Yinghu Reservoir", 108.8361, 32.6041),
}


def remove_floor(readings: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """D-061: set algal density <= ALGAE_FLOOR to missing; return (readings, removal list)."""
    r = readings.copy()
    m = r["algae_cells_l"] <= ALGAE_FLOOR
    rem = pd.DataFrame({"time": r.loc[m, "time"], "variable": "algae_cells_l", "reason": "detection_floor",
                        "value": r.loc[m, "algae_cells_l"]})
    r.loc[m, "algae_cells_l"] = np.nan
    return r, rem


def nutrient_outliers(q: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """D-062: Hampel check on daily log10 TN and log10 TP (7-day centred window, >= 3 days, flag when the deviation
    from the window median exceeds both 5*1.4826*MAD and 0.5 log10). Unlike algal density there is NO
    "two-day deviation = real event" exception. Flagged days are removed; the <= 2-day driver fill of qc.py is redone."""
    from . import qc
    q = q.copy()
    rows = []
    for c in ("tn", "tp"):
        x = q[c].where(~q[f"{c}_filled"].astype(bool))
        v = np.log10(x.to_numpy(dtype=float))
        flags = np.zeros(len(v), dtype=bool)
        for i in range(len(v)):
            if np.isnan(v[i]):
                continue
            w = v[max(0, i - qc.HAMPEL_HALF):i + qc.HAMPEL_HALF + 1]
            w = w[~np.isnan(w)]
            if len(w) < 3:
                continue
            m = np.median(w)
            mad = np.median(np.abs(w - m))
            flags[i] = abs(v[i] - m) > qc.HAMPEL_K * 1.4826 * mad and abs(v[i] - m) > qc.HAMPEL_MIN_LOG10
        f = pd.Series(flags, index=q.index)
        rows += [(t, c, "hampel_nutrient", x[t]) for t in q.index[f]]
        q[c], q[f"{c}_filled"] = qc.fill_short_gaps(x.where(~f))
    return q, pd.DataFrame(rows, columns=["time", "variable", "reason", "value"])


def build_daily(qc_daily: pd.DataFrame) -> pd.DataFrame:
    """qc_daily: qc.daily_series output (index date). One-piece linear interpolation of y, wt, tn, tp."""
    d = qc_daily.loc[START:END].copy()
    obs = {"y": d["y_valid"].astype(bool) & d["y"].notna()}
    for v in ("wt", "tn", "tp"):
        obs[v] = d[v].notna() & ~d[f"{v}_filled"].astype(bool)
    out = pd.DataFrame(index=d.index)
    for v in VARS:
        filled = d[v].where(obs[v]).interpolate(method="time", limit_area="inside")
        out[v] = filled
        out[f"{v}_filled"] = filled.notna() & ~obs[v]
    out["y_obs"] = obs["y"]
    out["y_valid"] = out["y"].notna()
    out["segment"] = np.where(out.index <= pd.Timestamp(TRAIN_YEAR_END), "train", "eval")
    return out


def last_real_2023(daily: pd.DataFrame) -> pd.Timestamp:
    d = daily.loc[:TRAIN_YEAR_END]
    lasts = [d.index[d[v].notna() & ~d[f"{v}_filled"]].max() for v in VARS]
    return min(lasts)


def origins(daily: pd.DataFrame) -> pd.DataFrame:
    comp = daily[list(VARS)].notna().all(axis=1)
    hist_ok = comp.rolling(HISTORY, min_periods=HISTORY).sum() == HISTORY
    real_t = daily["y_obs"] & ~daily[["wt_filled", "tn_filled", "tp_filled"]].any(axis=1)
    yobs = daily["y_obs"].to_numpy()
    L = last_real_2023(daily)
    rows = []
    for i, t in enumerate(daily.index):
        if i + HORIZON >= len(daily) or not hist_ok.iloc[i] or not comp.iloc[i + 1:i + HORIZON + 1].all():
            continue
        last = t + pd.Timedelta(days=HORIZON)
        if t <= pd.Timestamp(TRAIN_YEAR_END):
            if last > L:
                continue
            name = next(n for n, a, b in CV_BLOCKS if pd.Timestamp(a) <= t <= pd.Timestamp(b))
        elif t <= pd.Timestamp(EVAL_LAST_ORIGIN):
            name = "eval"
        else:
            continue
        n_obs = int(yobs[i + 1:i + HORIZON + 1].sum())
        rows.append((t, name, last, n_obs, bool(real_t.iloc[i]), bool(name == "eval" and real_t.iloc[i] and n_obs >= MIN_OBS_LABELS)))
    return pd.DataFrame(rows, columns=["origin", "split_name", "last_label_date", "n_obs_labels", "real_t", "scored"])


def sha256_text(o: pd.DataFrame) -> str:
    text = "\n".join(f"{t:%Y-%m-%d},{s},{n},{r}" for t, s, n, r in o[["origin", "split_name", "n_obs_labels", "real_t"]].itertuples(index=False))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
