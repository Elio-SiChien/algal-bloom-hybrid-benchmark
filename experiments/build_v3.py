"""Build the v3 daily data and origin lists for the 10 waterbodies (D-059, D-060).

    python -X utf8 experiments/build_v3.py [--national data/candidates/cnemc_algae_national_4h.csv.gz] [--lock]

Writes data/study_lakes/v3/: intermediate/readings_4h_raw_<site>.csv.gz (all readings, any status),
intermediate/daily_after_qc_<site>.csv (daily medians after QC, before interpolation), qc_daily_v3_<site>.csv, qc_removed_v3.csv, origins_v3.csv, summary_v3.csv,
manifest_v3.json (status 'draft'; with --lock status 'locked' and later runs refuse to overwrite).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import qc, v3  # noqa: E402

ROOT = HERE.parent
OUT = ROOT / "data" / "study_lakes" / "v3"


def max_gap(mask: pd.Series) -> int:
    """Longest run of False (no real observation)."""
    run = best = 0
    for m in mask.to_numpy():
        run = 0 if m else run + 1
        best = max(best, run)
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--national", default=str(ROOT / "data" / "candidates" / "cnemc_algae_national_4h.csv.gz"))
    ap.add_argument("--lock", action="store_true")
    a = ap.parse_args()
    man_f = OUT / "manifest_v3.json"
    if man_f.exists() and json.loads(man_f.read_text(encoding="utf-8")).get("status") == "locked":
        raise SystemExit("v3 data are locked; refusing to overwrite")
    (OUT / "intermediate").mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(a.national, usecols=["basin", "station", "time", "status", "wt", "tn", "tp", "algae_cells_l"])
    man = {"version": "v3", "status": "locked" if a.lock else "draft", "created": datetime.now().isoformat(timespec="seconds"),
           "qc_version": v3.QC_VERSION, "source": Path(a.national).name,
           "source_sha256": hashlib.sha256(Path(a.national).read_bytes()).hexdigest(), "sites": {}}
    allo, allrem, summ = [], [], []
    for site, (basin, station, lake, eng, lon, lat) in v3.SITES.items():
        r, rem = qc.qc_readings(raw, basin, station)
        if r.empty:
            raise SystemExit(f"{site}: no readings for {basin}/{station}")
        r, flo = v3.remove_floor(r)
        rem = pd.concat([rem, flo], ignore_index=True)
        q, ham = qc.daily_series(r, v3.START, v3.END)
        q, nut = v3.nutrient_outliers(q)
        ham = pd.concat([ham, nut], ignore_index=True)
        # intermediate products kept for the supporting information
        q.to_csv(OUT / "intermediate" / f"daily_after_qc_{site}.csv", date_format="%Y-%m-%d", lineterminator="\n")
        rr = raw[(raw["basin"] == basin) & (raw["station"] == station)].copy()
        rr["time"] = pd.to_datetime(rr["time"])
        rr.sort_values("time").to_csv(OUT / "intermediate" / f"readings_4h_raw_{site}.csv.gz", index=False)
        d = v3.build_daily(q)
        f = OUT / f"qc_daily_v3_{site}.csv"
        d.to_csv(f, date_format="%Y-%m-%d", lineterminator="\n")
        removed = pd.concat([rem, ham], ignore_index=True)
        removed.insert(0, "site", site)
        allrem.append(removed)
        o = v3.origins(d)
        o.insert(0, "site", site)
        allo.append(o)
        n_read = int(r["algae_cells_l"].notna().sum() + (rem["variable"] == "algae_cells_l").sum())
        row = {"site": site, "waterbody": lake, "station": station}
        for per, (s0, s1) in {"2023": ("2023-01-01", "2023-12-31"), "2024": ("2024-01-01", v3.END)}.items():
            x = d.loc[s0:s1]
            row[f"y_obs_{per}"] = round(float(x["y_obs"].mean()), 3)
            for v in ("wt", "tn", "tp"):
                row[f"{v}_obs_{per}"] = round(float((x[v].notna() & ~x[f"{v}_filled"]).mean()), 3)
            row[f"max_y_gap_days_{per}"] = max_gap(x["y_obs"])
        row["algae_readings_removed_floor"] = len(flo)
        row["algae_readings_removed_stuck"] = int((rem["reason"] == "stuck_run").sum())
        row["algae_readings_removed_share"] = round(row["algae_readings_removed_stuck"] / max(n_read, 1), 3)
        row["days_removed_hampel_algae"] = int((ham["reason"] == "hampel").sum())
        row["days_removed_hampel_tn_tp"] = len(nut)
        row["train_last_label_limit"] = f"{v3.last_real_2023(d):%Y-%m-%d}"
        row["train_origins"] = int(o["split_name"].str.startswith("cv_").sum())
        for b, *_ in v3.CV_BLOCKS:
            row[b] = int((o["split_name"] == b).sum())
        ev = o[o["split_name"] == "eval"]
        row["eval_origins"] = len(ev)
        row["eval_scored"] = int(ev["scored"].sum())
        sc = ev[ev["scored"]]
        row["eval_scored_may_oct"] = int(sc["origin"].dt.month.between(5, 10).sum())
        summ.append(row)
        man["sites"][site] = {"basin": basin, "station": station, "lon": lon, "lat": lat,
                              "daily_sha256": hashlib.sha256(f.read_bytes()).hexdigest(),
                              "origin_list_sha256": v3.sha256_text(o)}
        print(f"{site}: train {row['train_origins']}, eval {row['eval_origins']} (scored {row['eval_scored']})", flush=True)
    pd.concat(allo).to_csv(OUT / "origins_v3.csv", index=False, date_format="%Y-%m-%d", lineterminator="\n")
    pd.concat(allrem).to_csv(OUT / "qc_removed_v3.csv", index=False, encoding="utf-8-sig")
    s = pd.DataFrame(summ)
    s.to_csv(OUT / "summary_v3.csv", index=False, encoding="utf-8-sig")
    man["origins_csv_sha256"] = hashlib.sha256((OUT / "origins_v3.csv").read_bytes()).hexdigest()
    man_f.write_text(json.dumps(man, indent=2, ensure_ascii=False), encoding="utf-8")
    with pd.option_context("display.width", 250, "display.max_columns", 50):
        print(s.to_string())


if __name__ == "__main__":
    main()
