"""Quality control for CNEMC 4-hourly readings -> daily series (protocol v1.0, section 4).

Steps (D-034, D-041, D-043):
1. keep readings with status == "正常";
2. algal density / TN / TP <= 0 -> missing;
3. stuck sensor: >= 6 consecutive identical algal-density readings -> all removed (not applied to TN/TP);
4. daily target y = median of log10(algal density) over the Beijing day; drivers (water temp, TN, TP) = daily median;
5. Hampel on daily y: 7-day centred window, flag if |y - median| > 5*1.4826*MAD AND > 0.5 log10,
   but keep the day if an adjacent day deviates the same way by > 0.5 log10 (D-044);
6. drivers only: fill interior gaps of <= 2 days by linear interpolation (flag column *_filled).
Every removed reading/day is written to a removal list.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

QC_VERSION = "qc-v1.1"
STATIONS = {"dianchi": ("滇池流域", "观音山西"), "chaohu": ("巢湖流域", "龟山")}
DATA_START, DATA_END = "2021-06-17", "2025-02-13"
STUCK_RUN = 6
HAMPEL_HALF = 3          # 7-day centred window
HAMPEL_K = 5.0
HAMPEL_MIN_LOG10 = 0.5
MAX_FILL_GAP = 2
DRIVERS = ("wt", "tn", "tp")


def stuck_mask(values: pd.Series, run: int = STUCK_RUN) -> pd.Series:
    """True for readings belonging to a run of >= `run` identical consecutive non-missing values."""
    v = values.dropna()
    grp = (v != v.shift()).cumsum()
    size = v.groupby(grp).transform("size")
    out = pd.Series(False, index=values.index)
    out.loc[v.index] = size.to_numpy() >= run
    return out


def qc_readings(raw: pd.DataFrame, basin: str, station: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filter one station's 4-hourly readings. Returns (clean readings, removal list)."""
    r = raw[(raw["basin"] == basin) & (raw["station"] == station)].copy()
    r["time"] = pd.to_datetime(r["time"])
    r = r.sort_values("time").drop_duplicates("time", keep="last").reset_index(drop=True)
    removed = []
    bad_status = r["status"] != "正常"
    for _, row in r[bad_status].iterrows():
        removed.append((row["time"], "all", "status_not_normal", row["status"]))
    r = r[~bad_status].copy()
    for col in ("algae_cells_l", "tn", "tp"):
        m = r[col] <= 0
        for _, row in r[m].iterrows():
            removed.append((row["time"], col, "non_positive", row[col]))
        r.loc[m, col] = np.nan
    m = stuck_mask(r["algae_cells_l"])
    for _, row in r[m].iterrows():
        removed.append((row["time"], "algae_cells_l", "stuck_run", row["algae_cells_l"]))
    r.loc[m, "algae_cells_l"] = np.nan
    rem = pd.DataFrame(removed, columns=["time", "variable", "reason", "value"])
    return r, rem


def hampel_flags(y: pd.Series) -> pd.Series:
    """y: daily series on a complete calendar index. True = isolated outlier.

    Day i is flagged when |y_i - m_i| > 5*1.4826*MAD_i and > 0.5 log10, where m_i / MAD_i are the
    median / MAD of the non-missing days in the centred 7-day window (needs >= 3 days), UNLESS an
    adjacent day (i-1 or i+1) is also > 0.5 log10 away from m_i in the same direction (D-044):
    a deviation that lasts two or more days is treated as a real event, not a sensor glitch.
    """
    v = y.to_numpy(dtype=float)
    n = len(v)
    flags = np.zeros(n, dtype=bool)
    for i in range(n):
        if np.isnan(v[i]):
            continue
        w = v[max(0, i - HAMPEL_HALF):min(n, i + HAMPEL_HALF + 1)]
        w = w[~np.isnan(w)]
        if len(w) < 3:
            continue
        m = np.median(w)
        mad = np.median(np.abs(w - m))
        dev = v[i] - m
        if not (abs(dev) > HAMPEL_K * 1.4826 * mad and abs(dev) > HAMPEL_MIN_LOG10):
            continue
        sustained = any(0 <= j < n and not np.isnan(v[j]) and (v[j] - m) * np.sign(dev) > HAMPEL_MIN_LOG10
                        for j in (i - 1, i + 1))
        flags[i] = not sustained
    return pd.Series(flags, index=y.index)


def fill_short_gaps(x: pd.Series, max_gap: int = MAX_FILL_GAP) -> tuple[pd.Series, pd.Series]:
    """Linear fill of interior NaN runs with length <= max_gap. Returns (filled, was_filled)."""
    isna = x.isna()
    run_id = (isna != isna.shift()).cumsum()
    run_len = isna.groupby(run_id).transform("sum")
    has_left = x.ffill().notna()
    has_right = x.bfill().notna()
    fillable = isna & (run_len <= max_gap) & has_left & has_right
    filled = x.interpolate(method="linear", limit_area="inside")
    out = x.copy()
    out[fillable] = filled[fillable]
    return out, fillable


def daily_series(readings: pd.DataFrame, start: str = DATA_START, end: str = DATA_END) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clean readings -> one row per Beijing calendar day. Returns (daily, hampel removals)."""
    r = readings.copy()
    r["date"] = r["time"].dt.floor("D")
    r["log10_algae"] = np.log10(r["algae_cells_l"])
    g = r.groupby("date")
    days = pd.date_range(start, end, name="date")
    out = pd.DataFrame(index=days)
    out["y"] = g["log10_algae"].median().reindex(days)
    out["n_readings_y"] = g["log10_algae"].count().reindex(days).fillna(0).astype(int)
    flags = hampel_flags(out["y"])
    ham = pd.DataFrame({"time": out.index[flags], "variable": "y_daily", "reason": "hampel",
                        "value": out["y"][flags].to_numpy()})
    out.loc[flags, "y"] = np.nan
    out["y_valid"] = out["y"].notna()
    for c in DRIVERS:
        raw = g[c].median().reindex(days)
        out[c], out[f"{c}_filled"] = fill_short_gaps(raw)
    return out, ham


def run_station(raw: pd.DataFrame, site: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    basin, station = STATIONS[site]
    r, rem = qc_readings(raw, basin, station)
    daily, ham = daily_series(r)
    removed = pd.concat([rem, ham], ignore_index=True)
    removed.insert(0, "site", site)
    daily.insert(0, "site", site)
    return daily, removed


def sha256_of_frame(df: pd.DataFrame) -> str:
    return hashlib.sha256(df.to_csv(index=True, lineterminator="\n").encode("utf-8")).hexdigest()
