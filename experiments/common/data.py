"""Load the locked daily data, origins and ERA5 drivers for one site (protocol v1.0 sections 2-6)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "data" / "study_lakes"
LN10 = np.log(10.0)
VERSION = "v1"  # "v1": protocol v1.0; "v2": 2023-2024 interpolated, real-day origins (D-050);
# "v2b": same daily data, origins on every day, two scoring schemes (D-054)
# "v2c": one-piece interpolation 2023-2024, frozen v2b models re-forecast every day (2026-10-20)
# "v3": 10 waterbodies, one-piece interpolation, 2023 training / 2024 evaluation, ERA5-Land at the station (D-059..D-062)


def set_version(v: str) -> None:
    global VERSION
    if v not in ("v1", "v2", "v2b", "v2c", "v3"):
        raise ValueError(v)
    VERSION = v


def _paths():
    if VERSION == "v1":
        return STUDY, "qc_daily_{}.csv", "origins.csv", "origins_manifest.json"
    if VERSION == "v2b":
        return STUDY / "v2", "qc_daily_v2_{}.csv", "origins_v2b.csv", "manifest_v2b.json"
    if VERSION == "v3":
        return STUDY / "v3", "qc_daily_v3_{}.csv", "origins_v3.csv", "manifest_v3.json"
    if VERSION == "v2c":  # one-piece interpolation, frozen-model re-forecast (2026-10-20)
        return STUDY / "v2c", "qc_daily_v2c_{}.csv", "origins_v2c.csv", "manifest_v2c.json"
    return STUDY / "v2", "qc_daily_v2_{}.csv", "origins_v2.csv", "manifest_v2.json"


def is_v2() -> bool:
    return VERSION in ("v2", "v2b", "v2c", "v3")  # interpolated daily data, scoring on real observations


def real_origin_mask(daily: pd.DataFrame, origins) -> np.ndarray:
    """Day t has real (not interpolated) algal density, water temperature, TN and TP (D-052)."""
    i = daily.index.get_indexer(origins)
    ok = daily["y_obs"].to_numpy(bool)
    for c in ("wt_filled", "tn_filled", "tp_filled"):
        if c in daily:
            ok = ok & ~daily[c].to_numpy(bool)
    return ok[i]


def manifest() -> dict:
    import json
    base, _, _, man = _paths()
    return json.loads((base / man).read_text(encoding="utf-8"))


def origins_file():
    base, _, org, _ = _paths()
    return base / org


def eval_splits() -> tuple:
    if VERSION in ("v2c", "v3"):
        return ("eval",)
    return ("test",) if VERSION == "v1" else ("test", "forecast")


def load_site(site: str) -> pd.DataFrame:
    """Daily frame indexed by date: y, y_valid, wt, tn, tp (QC'd, short gaps filled) + ERA5 drivers."""
    base, daily_name, _, _ = _paths()
    d = pd.read_csv(base / daily_name.format(site), index_col=0, parse_dates=True)
    era = (STUDY / "v3" if VERSION == "v3" else STUDY) / f"era5land_{site}.csv"  # v3: grid point nearest the station
    e = pd.read_csv(era, index_col=0, parse_dates=True, encoding="utf-8-sig")
    e = e[["air_temp_c", "precip_mm_day", "shortwave_w_m2_daymean", "par_umol_m2_s_daymean"]]
    e = e.rename(columns={"shortwave_w_m2_daymean": "sw", "par_umol_m2_s_daymean": "par", "precip_mm_day": "precip"})
    out = d.join(e, how="left")
    out["y_valid"] = out["y_valid"].astype(bool)
    if "y_obs" not in out:
        out["y_obs"] = out["y_valid"]
    out["y_obs"] = out["y_obs"].astype(bool)
    return out


def load_origins(site: str) -> pd.DataFrame:
    o = pd.read_csv(origins_file(), parse_dates=["origin", "last_label_date"])
    o = o[o["site"] == site].reset_index(drop=True)
    if "n_obs_labels" not in o:
        o["n_obs_labels"] = 7
    return o


def labels(daily: pd.DataFrame, origins: pd.Series, observed_only: bool = False) -> np.ndarray:
    """(n_origins, 7) y at t+1..t+7 (log10 cells/L). observed_only: interpolated days -> NaN (v2 scoring)."""
    idx = daily.index.get_indexer(origins)
    y = daily["y"].to_numpy(dtype=float)
    if observed_only:
        y = np.where(daily["y_obs"].to_numpy(bool), y, np.nan)
    return np.stack([y[idx + h] for h in range(1, 8)], axis=1)


def fit_wt_regression(daily: pd.DataFrame, train_days: pd.DatetimeIndex) -> tuple[float, float]:
    """OLS water temperature = alpha + beta * air temperature on observed (not filled) training days (D-024)."""
    d = daily.loc[daily.index.intersection(train_days)]
    m = d["wt"].notna() & ~d["wt_filled"].astype(bool) & d["air_temp_c"].notna()
    beta, alpha = np.polyfit(d.loc[m, "air_temp_c"], d.loc[m, "wt"], 1)
    return float(alpha), float(beta)
