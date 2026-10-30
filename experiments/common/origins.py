"""Common forecast origins and block split (protocol v1.0, section 5).

Origin t is kept for ALL methods only if
- y is valid (QC-passed, not filled) on t and on each of t+1..t+7;
- on t-6..t, algal density, water temperature, TN and TP are complete after the <=2-day
  interior gap fill of section 4 item 5 (y itself is never filled on t or in the label window).
Each origin is assigned to the block that contains t.
"""

from __future__ import annotations

import hashlib

import pandas as pd

from .qc import DRIVERS, fill_short_gaps

HORIZON = 7
HISTORY = 7  # t-6..t
BLOCKS = (
    ("cv_block1", "2021-06-17", "2022-12-09"),
    ("cv_block2", "2022-12-10", "2023-03-24"),
    ("cv_block3", "2023-03-25", "2023-06-26"),
    ("cv_block4", "2023-06-27", "2023-09-06"),
    ("cv_block5", "2023-09-07", "2023-12-31"),
    ("test", "2024-01-01", "2025-02-13"),
)


def common_origins(daily: pd.DataFrame) -> pd.DataFrame:
    """daily: output of qc.daily_series on a complete calendar index. Returns one row per origin."""
    valid = daily["y_valid"].astype(bool)
    y_hist, _ = fill_short_gaps(daily["y"])
    hist_ok = pd.concat([y_hist.notna()] + [daily[c].notna() for c in DRIVERS], axis=1).all(axis=1)
    hist_ok = hist_ok.rolling(HISTORY, min_periods=HISTORY).sum() == HISTORY
    future_ok = valid[::-1].rolling(HORIZON, min_periods=HORIZON).sum()[::-1].shift(-1) == HORIZON
    ok = hist_ok & valid & future_ok.fillna(False).astype(bool)
    t = daily.index[ok]
    out = pd.DataFrame({"origin": t})
    out["split_name"] = None
    for name, a, b in BLOCKS:
        out.loc[(out.origin >= a) & (out.origin <= b), "split_name"] = name
    out = out.dropna(subset=["split_name"]).reset_index(drop=True)
    out["last_label_date"] = out["origin"] + pd.Timedelta(days=HORIZON)
    return out


def origin_counts(origins: pd.DataFrame) -> dict:
    c = origins["split_name"].value_counts()
    return {name: int(c.get(name, 0)) for name, _, _ in BLOCKS}


def origin_list_sha256(origins: pd.DataFrame) -> str:
    text = "\n".join(f"{t:%Y-%m-%d},{s}" for t, s in zip(origins["origin"], origins["split_name"]))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
