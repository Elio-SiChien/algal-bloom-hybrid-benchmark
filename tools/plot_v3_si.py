"""Supporting-information figures and tables for the v3 data processing (D-059..D-061).
python -X utf8 tools/plot_v3_si.py
Reads data/study_lakes/v3/, writes results/si_data_processing/:
  FigS_v3_<var>.png/.pdf  one figure per variable (algal density, water temperature, TN, TP), 10 panels:
                          real daily observations (black), interpolated days (grey), values removed by QC (red x)
  TableS_qc_removals.csv  readings / days removed, by site, variable and reason
  TableS_data_summary.csv per-site coverage, gaps and origin counts (copy of summary_v3.csv)
  TableS_sites.csv        stations and verified coordinates (D-060)
"""
import shutil
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from common import v3  # noqa: E402
D = ROOT / "data" / "study_lakes" / "v3"
OUT = ROOT / "results" / "si_data_processing"
MM = 1 / 25.4
VARS = {  # column: (label, axis label, raw reading column, log10?)
    "y": ("Algal density", "log$_{10}$ cells L$^{-1}$", "algae_cells_l", True),
    "wt": ("Water temperature", "°C", "wt", False),
    "tn": ("Total nitrogen", "mg L$^{-1}$", "tn", False),
    "tp": ("Total phosphorus", "mg L$^{-1}$", "tp", False),
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rem = pd.read_csv(D / "qc_removed_v3.csv", parse_dates=["time"])
    rem["value"] = pd.to_numeric(rem["value"], errors="coerce")
    summ = pd.read_csv(D / "summary_v3.csv")
    for col, (title, unit, rawcol, logy) in VARS.items():
        fig, axs = plt.subplots(5, 2, figsize=(180 * MM, 200 * MM), sharex=True)
        for ax, site in zip(axs.T.ravel(), v3.SITES):
            d = pd.read_csv(D / f"qc_daily_v3_{site}.csv", index_col=0, parse_dates=True)
            obs = d["y_obs"] if col == "y" else ~d[f"{col}_filled"] & d[col].notna()
            ax.axvspan(pd.Timestamp(v3.EVAL_START), d.index[-1], color="#EEF3FA", zorder=0)
            ax.plot(d.index, d[col], color="0.65", lw=0.8, label="interpolated")
            ax.plot(d.index, d[col].where(obs), color="k", lw=0.8, label="observed")
            if col == "y":
                r = rem[(rem.site == site) & rem.variable.isin(["y_daily", "algae_cells_l"]) & (rem.reason != "status_not_normal")]
                vals = np.where(r.variable == "algae_cells_l", np.log10(r.value.clip(lower=1)), r.value)
            else:
                r = rem[(rem.site == site) & (rem.variable == rawcol)]
                vals = r.value.to_numpy()
            lo, hi = np.nanmin(d[col]), np.nanmax(d[col])
            pad = 0.08 * (hi - lo)
            lo, hi = lo - pad, hi + pad
            if len(r):
                vals = np.asarray(vals, float)
                inside = (vals >= lo) & (vals <= hi)
                ax.plot(r.time[inside], vals[inside], "x", color="#D62728", ms=3.5, mew=0.9, label="removed by QC")
                off_t, off_v = list(r.time[~inside]), list(vals[~inside])  # off-scale removed values: marker at the axis edge
                for t, v in zip(off_t, off_v):
                    ax.plot([t], [hi if v > hi else lo], "^" if v > hi else "v", color="#D62728", ms=3.5, clip_on=False)
            ax.set_ylim(lo, hi)
            ax.set_title(v3.SITES[site][3], loc="left", fontsize=8, fontweight="bold")
            ax.set_title(f"observed {obs.loc[:'2023'].mean():.0%} (2023) / {obs.loc['2024':].mean():.0%} (2024)",
                         loc="right", fontsize=6.3, color="0.35")
            ax.tick_params(labelsize=6.5)
            ax.set_ylabel(unit, fontsize=6.5)
        for ax in axs[-1]:
            ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
        from matplotlib.lines import Line2D
        h = [Line2D([], [], color="k", lw=0.8), Line2D([], [], color="0.65", lw=0.8),
             Line2D([], [], color="#D62728", marker="x", ls="none", ms=3.5),
             Line2D([], [], color="#D62728", marker="^", ls="none", ms=3.5)]
        l = ["observed", "interpolated", "removed by QC", "removed by QC, off scale"]
        fig.legend(h, l, loc="upper left", ncol=4, fontsize=6.5, frameon=False, bbox_to_anchor=(0.01, 0.975))
        fig.suptitle(f"{title}: daily series after QC and interpolation (2024 evaluation period shaded)",
                     fontsize=7.5, x=0.02, ha="left", y=0.985)
        fig.tight_layout(rect=(0, 0, 1, 0.955))
        for ext in ("png", "pdf"):
            fig.savefig(OUT / f"FigS_v3_{col}.{ext}", dpi=300)
        plt.close(fig)
    # R6-09/R6-10: 2023 vs 2024 levels (real observations only) and interpolated share per site
    yr = []
    for site in v3.SITES:
        d = pd.read_csv(D / f"qc_daily_v3_{site}.csv", index_col=0, parse_dates=True)
        r = {"site": site, "name": v3.SITES[site][3]}
        for col in ("y", "wt", "tn", "tp"):
            real = d["y_obs"] if col == "y" else ~d[f"{col}_filled"] & d[col].notna()
            for y in ("2023", "2024"):
                x = d.loc[y]
                r[f"{col}_median_{y}"] = round(float(x[col].where(real.loc[y]).median()), 4)
        for y in ("2023", "2024"):
            r[f"y_interpolated_share_{y}"] = round(float(1 - d.loc[y, "y_obs"].mean()), 3)
        yr.append(r)
    pd.DataFrame(yr).to_csv(OUT / "TableS_year_comparison.csv", index=False, encoding="utf-8-sig")
    t = rem.groupby(["site", "variable", "reason"]).size().rename("n").reset_index()
    t.to_csv(OUT / "TableS_qc_removals.csv", index=False, encoding="utf-8-sig")
    shutil.copy(D / "summary_v3.csv", OUT / "TableS_data_summary.csv")
    pd.DataFrame([{"site": s, "waterbody": v[2], "name": v[3], "basin": v[0], "station": v[1], "lon": v[4], "lat": v[5]}
                  for s, v in v3.SITES.items()]).to_csv(OUT / "TableS_sites.csv", index=False, encoding="utf-8-sig")
    print(t.pivot_table(index="site", columns=["variable", "reason"], values="n", fill_value=0).to_string())


if __name__ == "__main__":
    main()
