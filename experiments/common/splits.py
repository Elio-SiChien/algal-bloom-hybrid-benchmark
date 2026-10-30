"""Blocked cross-validation folds with 7-day buffers (v1: 5 blocks, D-037; v2: 4 blocks; v3: the 4 quarters of 2023, D-059)."""

from __future__ import annotations

import pandas as pd

from . import data
from .origins import BLOCKS, HISTORY, HORIZON

BUFFER = 7


def cv_blocks():
    if data.VERSION == "v3":
        from .v3 import CV_BLOCKS as B3
        return list(B3)
    if data.is_v2():
        from .v2 import CV_BLOCKS as B2
        return list(B2)
    return [b for b in BLOCKS if b[0].startswith("cv_")]


CV_BLOCKS = cv_blocks()  # v1 default; code paths call cv_blocks() so the version switch is honoured


def window(t: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Days touched by origin t: inputs t-6..t and labels t+1..t+7."""
    return t - pd.Timedelta(days=HISTORY - 1), t + pd.Timedelta(days=HORIZON)


def _filled_any(daily, day) -> bool:
    if daily is None or day not in daily.index:
        return False
    cols = [c for c in ("y_filled", "wt_filled", "tn_filled", "tp_filled") if c in daily]
    return bool(daily.loc[day, cols].astype(bool).any()) if cols else False


def cv_folds(origins: pd.DataFrame, daily: pd.DataFrame | None = None):
    """Yield (fold_name, train_mask, val_mask, excluded_span) for each held-out block.
    v2 (R4-01): if the buffer edge falls inside an interpolated run (whose values depend on observations
    in the held-out block), the excluded span is widened to the end of that run."""
    for name, a, b in cv_blocks():
        va = origins["split_name"] == name
        lo = pd.Timestamp(a) - pd.Timedelta(days=BUFFER)
        hi = pd.Timestamp(b) + pd.Timedelta(days=BUFFER)
        while _filled_any(daily, hi + pd.Timedelta(days=1)):
            hi += pd.Timedelta(days=1)
        while _filled_any(daily, lo - pd.Timedelta(days=1)):
            lo -= pd.Timedelta(days=1)
        w0, w1 = window(origins["origin"])
        overlaps = (w0 <= hi) & (w1 >= lo)
        tr = origins["split_name"].str.startswith("cv_") & ~va & ~overlaps
        yield name, tr.to_numpy(), va.to_numpy(), (lo, hi)


def train_days(daily_index: pd.DatetimeIndex, exclude=None) -> pd.DatetimeIndex:
    """Calendar days of the CV blocks (training period), minus excluded spans (a (lo, hi) tuple or a list of them)."""
    cb = cv_blocks()
    d = daily_index[(daily_index >= cb[0][1]) & (daily_index <= cb[-1][2])]
    spans = [] if exclude is None else ([exclude] if isinstance(exclude, tuple) else list(exclude))
    for lo, hi in spans:
        d = d[(d < lo) | (d > hi)]
    return d
