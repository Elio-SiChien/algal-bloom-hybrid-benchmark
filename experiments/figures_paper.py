"""Paper figures and tables from the formal runs (read-only; never refits anything).

    python -X utf8 experiments/figures_paper.py                 # v3: 10 waterbodies (results/runs_v3), default
    python -X utf8 experiments/figures_paper.py --version v2c   # earlier two-lake pilot (frozen re-forecast)

v3 (D-059): training 2023, evaluation 2024. Scoring: origin day t observed (algal density, water temperature, TN,
TP), >= 4 observed labels, accuracy on observed days only. Fig. 5: one figure per waterbody, one panel per method,
the 7-day-ahead forecast drawn as one line over training (in-sample fit) and evaluation.
Statistics (D-064, R6-04): the main test is lake-level -- per method, the skill vs persistence of each waterbody
(1 - MAE_method / MAE_persistence, n = 10) is tested with a one-sided Wilcoxon signed-rank test (median > 0), Holm
over the methods (Table3_lake_level_tests.csv). Per-lake Diebold-Mariano tests are descriptive only. Lead-time
confidence intervals use a moving-block bootstrap with blocks of 14 calendar days.
Outputs: PNG/PDF pairs and CSV tables in results/figures_paper_<version>/.
v3: Fig3_algal_density, Fig4_radar, Fig5_dianchi/shahu, Fig6_rises_falls,
Fig7_ranking, FigS_days_ahead and FigS_timeseries_<site> for the other waterbodies.
Earlier versions retain their historical figure names and layouts.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import data, evaluate, splits  # noqa: E402
from common.v2 import MIN_OBS_LABELS, SEGMENTS as SEGMENTS_V2  # noqa: E402
from common import v2c, v3  # noqa: E402
from analyze_results import dm_test, holm  # noqa: E402

# ----------------------------------------------------------------------------- style
MM = 1 / 25.4
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "lines.linewidth": 1.0, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
    "savefig.dpi": 600})

# code id -> (display name, entry point, colour). Okabe-Ito colours, colour-blind safe.
METHODS = [("M2", "HAPP", "Before", "#E0474C"),     # colours follow Saravani et al. (2025): red, blue, gold, green
           ("M3", "NGR", "Inside", "#3A78C9"),
           ("M1", "T-PGNN", "Alongside", "#4FA350"),
           ("M4r", "PM-GP", "After", "#D9A400")]
FALLBACK = {"M4r": "M4"}  # v2 runs have the original M4 only
REFS = [("persistence", "Persistence", "#555555", (0, (3, 2))), ("climatology", "Climatology", "#A0A0A0", (0, (1, 1.5)))]
SITES = [("dianchi", "Lake Dianchi"), ("chaohu", "Lake Chaohu")]
SPLITS = ("test", "forecast")           # pooled into one evaluation period (option A); v2c: ("eval",)
SEGMENTS = SEGMENTS_V2
PRED_FILE = "test_predictions.csv"
EVAL_LABEL = "May to Dec 2024"
EVAL_START = "2024-05-01"
METRICS = [("mae", "1/MAE", False), ("rmse", "1/RMSE", False), ("nse", "NSE", True),
           ("trend_accuracy", "Rise/fall\ncorrect", True), ("hss", "Rise/fall\nskill (HSS)", True)]


def save(fig, out: Path, name: str):
    fig.savefig(out / f"{name}.png", bbox_inches="tight")
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------- data
def load_preds(runs: Path, site: str):
    out, missing = {}, []
    for code, name, _, _ in METHODS:
        for c in (code, FALLBACK.get(code)):
            if c and (runs / f"{c}_{site}" / PRED_FILE).exists():
                t = pd.read_csv(runs / f"{c}_{site}" / PRED_FILE)
                t = t[t["series"] == "final"]
                out[name] = {sp: g.pivot(index="origin", columns="lead", values="pred_log10")
                             for sp, g in t.groupby("split_name")}
                break
        else:
            missing.append(name)
    return out, missing


def scheme2(d, oo):
    obs = data.labels(d, oo["origin"], observed_only=True)
    keep = (np.isfinite(obs).sum(axis=1) >= MIN_OBS_LABELS) & data.real_origin_mask(d, oo["origin"])
    return obs[keep], oo["origin"][keep].reset_index(drop=True)


def ref_preds(d, origins):
    yt = d["y"].to_numpy()[d.index.get_indexer(origins)]
    train_days = evaluate.climatology_days(d, splits.train_days(d.index, None))  # same function as run_method.references
    return {"Persistence": evaluate.persistence(yt), "Climatology": evaluate.climatology(d, train_days, origins)}


def block_boot_skill(em, ep, dates=None, block=14, n=2000, seed=0):
    """95% interval of 1 - mean(em)/mean(ep) by a moving-block bootstrap. With `dates` (origin dates), blocks are
    14 CALENDAR days (R6-04): a block = all origins in [start, start + 14 d); origins missing in a block are skipped."""
    rng = np.random.default_rng(seed)
    m = len(em)
    if m < block:
        return np.nan, np.nan
    if dates is None:
        starts = np.arange(m - block + 1)
        k = int(np.ceil(m / block))
        groups = None
    else:
        day = ((pd.to_datetime(pd.Series(dates)) - pd.to_datetime(pd.Series(dates)).min()).dt.days).to_numpy()
        span = day.max() + 1
        k = int(np.ceil(span / block))
        groups = [np.flatnonzero((day >= s0) & (day < s0 + block)) for s0 in range(span - block + 1)]
    vals = np.empty(n)
    for i in range(n):
        if groups is None:
            idx = (rng.choice(starts, k)[:, None] + np.arange(block)[None, :]).ravel()[:m]
        else:
            idx = np.concatenate([groups[j] for j in rng.integers(0, len(groups), k)])
        vals[i] = 1 - np.nanmean(em[idx]) / np.nanmean(ep[idx])
    return np.nanpercentile(vals, [2.5, 97.5])


def period_block(d, o, preds, splits):
    """Scheme-2 labels, origins and all forecasts for the given split(s), concatenated in time order."""
    obs_l, org_l, pr_l = [], [], []
    for sp in splits:
        oo = o[o["split_name"] == sp].reset_index(drop=True)
        obs, origins = scheme2(d, oo)
        key = origins.dt.strftime("%Y-%m-%d")
        allp = dict(ref_preds(d, origins))
        for name, by_split in preds.items():
            if sp in by_split:
                p = by_split[sp].reindex(key).to_numpy()
                p, _ = evaluate.fill_failures(p, allp["Persistence"][:, 0])
                allp[name] = p
        obs_l.append(obs), org_l.append(origins), pr_l.append(allp)
    names = set.intersection(*[set(p) for p in pr_l])
    return (np.vstack(obs_l), pd.concat(org_l, ignore_index=True),
            {n: np.vstack([p[n] for p in pr_l]) for n in names})


def per_origin_mae(p, obs):
    with np.errstate(all="ignore"):
        return np.nanmean(np.abs(np.where(np.isfinite(obs), p, np.nan) - obs), axis=1)


def score_rows(site, period, obs, allp):
    rows = []
    co = evaluate.trend_class(obs)
    e_pers = per_origin_mae(allp["Persistence"], obs)
    pvals = {}
    for name, p in allp.items():
        s = evaluate.score(p, obs)
        cp = evaluate.trend_class(np.where(np.isfinite(obs), p, np.nan))
        r = {"lake": site, "period": period, "method": name, **{k: s[k] for k in ("n", "mae", "rmse", "nse", "hss", "trend_accuracy")},
             "rise_hit": float(np.mean(cp[co == 1] == 1)) if (co == 1).any() else np.nan,
             "fall_hit": float(np.mean(cp[co == -1] == -1)) if (co == -1).any() else np.nan}
        if name not in ("Persistence", "Climatology"):
            stat, pv = dm_test(per_origin_mae(p, obs) - e_pers)
            r.update(dm_stat=stat, dm_p_raw=pv)
            pvals[name] = pv
        rows.append(r)
    adj = holm(pvals) if pvals else {}
    for r in rows:
        if r["method"] in adj:
            r["dm_p_holm"] = adj[r["method"]]
    chance = float(sum(np.mean(co == c) ** 2 for c in (-1, 0, 1)))  # random guessing with the observed class frequencies
    return rows, chance


def collect(runs: Path):
    rows, season_rows, lead_rows, store, missing_all, chance = [], [], [], {}, set(), {}
    for site, _ in SITES:
        d, o = data.load_site(site), data.load_origins(site)
        preds, missing = load_preds(runs, site)
        missing_all |= set(missing)
        obs, origins, allp = period_block(d, o, preds, SPLITS)
        store[site] = {"daily": d, "preds": preds, "obs": obs, "origins": origins, "allp": allp}
        r, chance[site] = score_rows(site, "eval", obs, allp)
        rows += r
        if len(SPLITS) > 1:  # supplement: old seasonal split
            for sp in SPLITS:
                ob, _, ap = period_block(d, o, preds, (sp,))
                season_rows += score_rows(site, sp, ob, ap)[0]
        err_p = np.abs(np.where(np.isfinite(obs), allp["Persistence"], np.nan) - obs)
        for name, p in allp.items():
            err = np.abs(np.where(np.isfinite(obs), p, np.nan) - obs)
            for h in range(7):
                em, ep = err[:, h], err_p[:, h]
                ok = np.isfinite(em) & np.isfinite(ep)
                lo, hi = (np.nan, np.nan) if name == "Persistence" else block_boot_skill(em[ok], ep[ok], origins[ok])
                lead_rows.append({"lake": site, "method": name, "lead": h + 1, "mae": em[ok].mean(),
                                  "skill": 1 - em[ok].mean() / ep[ok].mean(), "lo": lo, "hi": hi})
    tab, season = pd.DataFrame(rows), pd.DataFrame(season_rows)
    for t in ((tab, season) if len(season) else (tab,)):
        mp = t[t["method"] == "Persistence"].set_index(["lake", "period"])["mae"]
        t["skill_vs_persistence"] = [1 - r.mae / mp[(r.lake, r.period)] for r in t.itertuples()]
    return tab, season, pd.DataFrame(lead_rows), store, sorted(missing_all), chance


def lake_level_tests(tab: pd.DataFrame) -> pd.DataFrame:
    """D-064 (R6-04): per method, skill vs persistence of each waterbody; one-sided Wilcoxon signed-rank test that the
    median skill > 0 (each waterbody = one sample); Holm over the methods. The main conclusion rests on this table."""
    from scipy.stats import wilcoxon
    rows, p = [], {}
    for name in [n for _, n, _, _ in METHODS if n in set(tab["method"])]:
        sk = tab[(tab["method"] == name) & (tab["period"] == "eval")]["skill_vs_persistence"].dropna()
        r = {"method": name, "n_lakes": len(sk), "median_skill": float(sk.median()) if len(sk) else np.nan,
             "lakes_better_than_persistence": int((sk > 0).sum())}
        if len(sk) >= 5:
            r["wilcoxon_p_raw"] = float(wilcoxon(sk, alternative="greater").pvalue)
            p[name] = r["wilcoxon_p_raw"]
        rows.append(r)
    out = pd.DataFrame(rows)
    if p:
        adj = holm(p)
        out["wilcoxon_p_holm"] = out["method"].map(adj)
    return out


def order(present):
    return [n for _, n, _, _ in METHODS if n in present] + [r[1] for r in REFS]


def style(name):
    for _, n, _, c in METHODS:
        if n == name:
            return {"color": c, "ls": "-"}
    for _, n, c, ls in REFS:
        if n == name:
            return {"color": c, "ls": ls}
    return {"color": "k", "ls": "-"}


def shade_periods(ax, labels=False, y=0.93):
    ax.axvspan(pd.Timestamp(SEGMENTS[0][1]), pd.Timestamp(SEGMENTS[0][2]), color="#F2F2F2", lw=0)
    ax.axvline(pd.Timestamp(EVAL_START), color="#888888", lw=0.6, ls=(0, (4, 3)))
    if labels:
        mid = lambda a, b: pd.Timestamp(a) + (pd.Timestamp(b) - pd.Timestamp(a)) / 2
        for x, lab in ((mid(SEGMENTS[0][1], SEGMENTS[0][2]), "Training"), (mid(EVAL_START, SEGMENTS[-1][2]), "Evaluation")):
            ax.text(x, y, lab, transform=ax.get_xaxis_transform(), ha="center", fontsize=6.5, color="#555555")


# ----------------------------------------------------------------------------- Fig. 3 data overview
def fig3(store, out):
    fig = plt.figure(figsize=(180 * MM, 105 * MM))
    gs = fig.add_gridspec(2, 2, width_ratios=[3.2, 1], hspace=0.45, wspace=0.18)
    lake_col = {"dianchi": "#3B6E8F", "chaohu": "#8F5B3B"}
    for i, (site, sname) in enumerate(SITES):
        d = store[site]["daily"]
        ax = fig.add_subplot(gs[i, 0])
        shade_periods(ax, labels=(i == 0))
        ax.plot(d.index, d["y"], color=lake_col[site], lw=0.8)
        ax.set_ylabel("log$_{10}$ cells L$^{-1}$")
        ax.set_title(f"({'ac'[i]}) {sname}", loc="left")
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %Y"))
        obs = d.loc[d["y_obs"], "y"]
        ax.set_title(f"median {obs.median():.2f}, range {obs.min():.2f} to {obs.max():.2f}, "
                     f"day-to-day autocorrelation {obs.autocorr(1):.2f}", loc="right", fontsize=6.2, color="#333333")
        ax2 = fig.add_subplot(gs[i, 1], sharey=ax)
        groups = [obs[obs.index.month == m].to_numpy() for m in range(1, 13)]
        bp = ax2.boxplot(groups, widths=0.6, patch_artist=True, showfliers=False,
                         medianprops={"color": "black", "lw": 0.8}, whiskerprops={"lw": 0.6}, capprops={"lw": 0.6})
        for b in bp["boxes"]:
            b.set(facecolor=lake_col[site], alpha=0.35, lw=0.6)
        ax2.set_xticks(range(1, 13), list("JFMAMJJASOND"))
        ax2.tick_params(axis="y", labelleft=False)
        ax2.set_title(f"({'bd'[i]}) By month", loc="left")
    save(fig, out, "Fig3_algal_density")


# ----------------------------------------------------------------------------- Fig. 4 radar
def fig4(tab, out):
    names = order(tab["method"].unique())
    labels = [m[1] for m in METRICS]
    keys = ["inv_mae", "inv_rmse", "nse", "trend_accuracy", "hss"]
    t = tab.copy()
    t["inv_mae"], t["inv_rmse"] = 1 / t["mae"], 1 / t["rmse"]
    ang = np.linspace(0, 2 * np.pi, len(keys), endpoint=False)
    fig, axes = plt.subplots(1, 2, figsize=(180 * MM, 88 * MM), subplot_kw={"projection": "polar"})
    for i, (site, sname) in enumerate(SITES):
        ax = axes[i]
        g = t[t["lake"] == site].set_index("method")
        lo, hi = g[keys].min(), g[keys].max()
        span = (hi - lo).replace(0, 1)
        base = lo - 0.15 * span  # multiscaled radar (each axis its own range), as Saravani et al. (2025)
        for n in names:
            if n not in g.index:
                continue
            v = ((g.loc[n, keys] - base) / (hi - base)).to_numpy(float)
            st = style(n)
            ax.plot(np.r_[ang, ang[0]], np.r_[v, v[0]], color=st["color"], ls=st["ls"], lw=1.1)
            if n not in [r[1] for r in REFS]:
                ax.fill(np.r_[ang, ang[0]], np.r_[v, v[0]], color=st["color"], alpha=0.06)
        ax.set_xticks(ang)
        fmt = {"trend_accuracy": "{:.0%}"}
        ax.set_xticklabels([f"{l}\n[{fmt.get(k, '{:.2f}').format(lo[k])}, {fmt.get(k, '{:.2f}').format(hi[k])}]"
                            for l, k in zip(labels, keys)], fontsize=6)
        ax.tick_params(axis="x", pad=7)
        ax.set_yticklabels([])
        ax.set_ylim(0, 1.05)
        ax.grid(lw=0.4, color="#DDDDDD")
        ax.spines["polar"].set_linewidth(0.4)
        ax.set_title(f"({'ab'[i]}) {sname}, {EVAL_LABEL}", pad=18, fontsize=7.5)
    handles = [Line2D([], [], color=style(n)["color"], ls=style(n)["ls"], lw=1.2, label=n) for n in names]
    fig.legend(handles=handles, loc="lower center", ncol=len(names), frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.text(0.5, -0.08, "Each axis has its own scale (range in brackets). Further out is better on every axis.",
             ha="center", fontsize=6.5, color="#555555")
    fig.subplots_adjust(wspace=0.55)
    save(fig, out, "Fig4_radar")


# ----------------------------------------------------------------------------- Fig. 5 time series (Saravani et al. 2025 style)
def fig5(store, tab, out):
    """Fig. 5, one figure per waterbody (D-057): one panel per method. Observed = black solid; the method's forecast
    issued 7 days earlier = one continuous dashed line in the method's colour over training (grey, in-sample fit) and
    evaluation (white, independent forecast). Panel title: evaluation MAE and share of rises/falls called correctly."""
    for site, sname in SITES:
        d, preds = store[site]["daily"], store[site]["preds"]
        names = [m for _, m, _, _ in METHODS if m in preds]
        if not names:
            continue
        fig, axes = plt.subplots(len(names), 1, figsize=(180 * MM, (30 * len(names) + 16) * MM), sharex=True, squeeze=False)
        for k, (ax, n) in enumerate(zip(axes[:, 0], names)):
            shade_periods(ax, labels=(k == len(names) - 1), y=0.04)
            ax.plot(d.index, d["y"], color="black", lw=0.8, zorder=5)
            ser = [pd.Series(p[7].to_numpy(), index=pd.to_datetime(p.index) + pd.Timedelta(days=7)) for p in preds[n].values()]
            sr = pd.concat(ser).sort_index()
            sr = sr[~sr.index.duplicated()]  # one continuous line (no break at the training/evaluation boundary)
            ax.plot(sr.index, sr.values, color=style(n)["color"], lw=0.9, ls=(0, (3, 1.5)), zorder=6)
            g = tab[(tab["lake"] == site) & (tab["method"] == n)].iloc[0]
            ax.set_title(n, loc="left", color=style(n)["color"], fontweight="bold")
            ax.set_title(f"evaluation: MAE {g['mae']:.3f}, rises/falls called correctly {g['trend_accuracy']:.0%}",
                         loc="right", fontsize=6.5, color="#444444")
            ax.set_ylabel("log$_{10}$ cells L$^{-1}$")
        axes[-1, 0].xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %Y"))
        h = [Line2D([], [], color="black", lw=0.8)] + [Line2D([], [], color=style(n)["color"], lw=0.9, ls=(0, (3, 1.5))) for n in names]
        fig.legend(h, ["Observed"] + [f"{n} (issued 7 days earlier)" for n in names], loc="upper center",
                   ncol=len(h), frameon=False, bbox_to_anchor=(0.5, 0.985), fontsize=6.5, handlelength=1.8, columnspacing=1.2)
        fig.suptitle(sname, x=0.01, y=1.035, ha="left", fontweight="bold", fontsize=8)
        fig.text(0.5, -0.01, "Grey: training period (in-sample fit). White: evaluation period (independent forecast). "
                 "MAE in log$_{10}$ cells L$^{-1}$, real observations only.", ha="center", fontsize=6.3, color="#555555")
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        name = f"FigS_timeseries_{site}" if data.VERSION == "v3" and site not in ("dianchi", "shahu") else f"Fig5_{site}"
        save(fig, out, name)


# ----------------------------------------------------------------------------- Fig. 6 days ahead
def fig6(lead, out):
    names = order(lead["method"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(180 * MM, 70 * MM), sharex=True)
    for i, (site, sname) in enumerate(SITES):
        ax = axes[i]
        g = lead[lead["lake"] == site]
        ax.axhline(0, color=style("Persistence")["color"], ls=style("Persistence")["ls"], lw=0.9)
        ax.text(7.25, 0, "Persistence", va="center", fontsize=6, color="#555555")
        for n in names:
            if n == "Persistence":
                continue
            s = g[g["method"] == n].sort_values("lead")
            if s.empty:
                continue
            st = style(n)
            ax.plot(s["lead"], 100 * s["skill"], color=st["color"], ls=st["ls"], marker="o", ms=2.5, lw=1.0)
            if n != "Climatology":
                ax.fill_between(s["lead"], 100 * s["lo"], 100 * s["hi"], color=st["color"], alpha=0.12, lw=0)
        ax.set_ylim(-100, max(100 * g["hi"].max() + 5, 20))
        ax.set_title(f"({'ab'[i]}) {sname}, {EVAL_LABEL}", loc="left")
        ax.set_xticks(range(1, 8))
        ax.set_xlabel("Days ahead")
        if i == 0:
            ax.set_ylabel("Error reduction relative to persistence (%)")
    handles = [Line2D([], [], color=style(n)["color"], ls=style(n)["ls"], lw=1.2, label=n) for n in names if n != "Persistence"]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, bbox_to_anchor=(0.5, -0.13))
    fig.text(0.5, -0.17, "Above zero: smaller error than simply repeating today's value. Shading: 95% moving-block "
             "bootstrap interval. Values below -100% are cut off.", ha="center", fontsize=6.5, color="#555555")
    fig.subplots_adjust(wspace=0.22)
    save(fig, out, "Fig6_days_ahead")


# ----------------------------------------------------------------------------- Fig. 7 rises and falls + events
def pick_cases(obs, origins, seed=0):
    full = np.isfinite(obs).sum(axis=1) >= 6
    s = np.where(full, evaluate.theil_sen(obs * evaluate.LN10), np.nan)
    i_up, i_dn, i_st = int(np.nanargmax(s)), int(np.nanargmin(s)), int(np.nanargmin(np.abs(s)))
    rest = [k for k in np.flatnonzero(full) if k not in (i_up, i_dn, i_st)]
    i_rd = int(np.random.default_rng(seed).choice(rest))
    return [("Largest rise", i_up), ("Largest fall", i_dn), ("Most stable", i_st), ("Random", i_rd)]


def fig7(tab, store, chance, out):
    names = order(tab["method"].unique())
    fig = plt.figure(figsize=(180 * MM, 150 * MM))
    gs = fig.add_gridspec(3, 4, height_ratios=[1.1, 1, 1], hspace=0.6, wspace=0.45)
    bar_names = [n for n in names if n != "Climatology"]
    for k, (site, sname) in enumerate(SITES):
        g = tab[tab["lake"] == site].set_index("method")
        # (a, b) share of weeks whose direction (rise / no change / fall) was called correctly
        ax = fig.add_subplot(gs[0, k])
        x = np.arange(len(bar_names))
        ax.bar(x, [100 * g.loc[n, "trend_accuracy"] for n in bar_names], 0.7,
               color=[style(n)["color"] for n in bar_names], alpha=0.85)
        ax.axhline(100 * chance[site], color="#555555", lw=0.7, ls=(0, (1, 1.5)))
        ax.text(-0.45, 96, "dotted line: random guess", fontsize=5.8, color="#555555", va="top")
        ax.set_xticks(x, bar_names, rotation=30, ha="right", rotation_mode="anchor")
        ax.set_ylim(0, 100)
        if k == 0:
            ax.set_ylabel("Weeks with direction\ncalled correctly (%)")
        ax.set_title(f"({'ab'[k]}) {sname.split()[-1]}: direction", loc="left", fontsize=7)
        # (c, d) hit rate for observed rises and falls
        ax = fig.add_subplot(gs[0, 2 + k])
        mn = [n for n in bar_names if n != "Persistence"]
        w = 0.38
        xx = np.arange(len(mn))
        ax.bar(xx - w / 2, [100 * g.loc[n, "rise_hit"] for n in mn], w, color=[style(n)["color"] for n in mn], alpha=0.9)
        ax.bar(xx + w / 2, [100 * g.loc[n, "fall_hit"] for n in mn], w, color=[style(n)["color"] for n in mn], alpha=0.4,
               hatch="////", edgecolor="white", lw=0)
        ax.set_xticks(xx, mn, rotation=30, ha="right", rotation_mode="anchor")
        ax.set_ylim(0, 100)
        if k == 0:
            ax.set_ylabel("Weeks caught (%)", labelpad=1)
        ax.set_title(f"({'cd'[k]}) {sname.split()[-1]}: rises vs falls", loc="left", fontsize=7)
        for xi, n in zip(xx, mn):
            for dx, key in ((-w / 2, "rise_hit"), (w / 2, "fall_hit")):
                if g.loc[n, key] == 0:
                    ax.text(xi + dx, 1.5, "0", ha="center", fontsize=5.5, color="#555555")
        if k == 1:
            ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color="#777777", alpha=0.9, label="Rising weeks"),
                               plt.Rectangle((0, 0), 1, 1, facecolor="#777777", alpha=0.4, hatch="////", edgecolor="white", label="Falling weeks")],
                      loc="upper right", frameon=False, fontsize=6)
    # (e-l) case weeks
    for i, (site, sname) in enumerate(SITES):
        obs, origins, allp = store[site]["obs"], store[site]["origins"], store[site]["allp"]
        for j, (lab, k) in enumerate(pick_cases(obs, origins)):
            ax = fig.add_subplot(gs[1 + i, j])
            yt = store[site]["daily"]["y"].loc[origins[k]]
            ax.plot(range(0, 8), np.r_[yt, obs[k]], color="black", lw=1.0, marker="o", ms=2.2, zorder=4)
            for n in names:
                st = style(n)
                ax.plot(range(0, 8), np.r_[yt, allp[n][k]], color=st["color"], ls=st["ls"], lw=0.9)
            ax.set_xticks(range(0, 8))
            ax.set_title(f"({'efghijkl'[4 * i + j]}) {sname.split()[-1]}: {lab}\nissued {origins[k]:%d %b %Y}", loc="left", fontsize=6.5)
            if j == 0:
                ax.set_ylabel("log$_{10}$ cells L$^{-1}$")
            if i == 1:
                ax.set_xlabel("Days after forecast issued")
    handles = [Line2D([], [], color="black", marker="o", ms=3, lw=1, label="Observed")] + \
              [Line2D([], [], color=style(n)["color"], ls=style(n)["ls"], lw=1.2, label=n) for n in names]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(bottom=0.1)
    save(fig, out, "Fig7_rises_falls")


# ----------------------------------------------------------------------------- Fig. 8 ranking heatmap
def fig8(tab, out):
    names = order(tab["method"].unique())
    rank_metrics = [("mae", "MAE", False), ("rmse", "RMSE", False), ("nse", "NSE", True),
                    ("trend_accuracy", "Rise/fall\ncorrect", True), ("hss", "Rise/fall\nskill", True)]
    cols, ranks = [], []
    for site, sname in SITES:
        g = tab[tab["lake"] == site].set_index("method").reindex(names)
        for key, lab, higher in rank_metrics:
            ranks.append(g[key].rank(ascending=not higher, method="min").to_numpy())
            cols.append((sname.split()[-1], lab))
    R = np.array(ranks).T
    avg = np.nanmean(R, axis=1)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(130 * MM, (14 + 6 * len(names)) * MM),
                                  gridspec_kw={"width_ratios": [len(cols), 1.4], "wspace": 0.05})
    cmap = plt.get_cmap("YlGnBu_r", len(names))
    ax.imshow(R, cmap=cmap, vmin=0.5, vmax=len(names) + 0.5, aspect="auto")
    for (a, b), v in np.ndenumerate(R):
        if np.isfinite(v):
            ax.text(b, a, f"{int(v)}", ha="center", va="center", fontsize=6.5, color="white" if v <= 2 else "black")
    ax.set_yticks(range(len(names)), names)
    ax.set_xticks(range(len(cols)), [c[1] for c in cols], fontsize=6)
    ax.axvline(len(rank_metrics) - 0.5, color="white", lw=2)
    for k, (site, sname) in enumerate(SITES):
        ax.text(k * len(rank_metrics) + (len(rank_metrics) - 1) / 2, -0.8, sname, ha="center", fontsize=7)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax2.imshow(avg[:, None], cmap=cmap, vmin=0.5, vmax=len(names) + 0.5, aspect="auto")
    for a, v in enumerate(avg):
        ax2.text(0, a, f"{v:.1f}", ha="center", va="center", fontsize=6.5, color="white" if v <= 2 else "black", fontweight="bold")
    ax2.set_xticks([0], ["Mean\nrank"], fontsize=6)
    ax2.set_yticks([])
    ax2.tick_params(length=0)
    for s in ax2.spines.values():
        s.set_visible(False)
    save(fig, out, "Fig8_ranking")


# ============================================================================ v3: ten-waterbody layouts (2026-10-21)
PENDING = "T-PGNN"  # methods without results yet are shown in legends as "(run pending)"


def legend_methods(present, with_refs=True, observed=False):
    h = [Line2D([], [], color="black", lw=0.9, label="Observed")] if observed else []
    for _, n, _, c in METHODS:
        lab = n if n in present else f"{n} (run pending)"
        h.append(Line2D([], [], color=c, lw=1.2, ls="-" if n in present else ":", label=lab))
    if with_refs:
        h += [Line2D([], [], color=c, ls=ls, lw=1.1, label=n) for _, n, c, ls in REFS]
    return h


def fig3_v3(store, out):
    """Daily algal density of the ten waterbodies (observed black, interpolated grey) with summary statistics."""
    fig, axes = plt.subplots(5, 2, figsize=(180 * MM, 190 * MM), sharex=True)
    for ax, (site, sname) in zip(axes.T.ravel(), SITES):
        d = store[site]["daily"]
        shade_periods(ax, labels=(site == SITES[0][0]), y=0.04)
        ax.plot(d.index, d["y"], color="#9A9A9A", lw=0.7)
        ax.plot(d.index, d["y"].where(d["y_obs"]), color="black", lw=0.7)
        obs = d.loc[d["y_obs"], "y"]
        peak = obs.groupby(obs.index.month).median().idxmax()
        ax.set_title(sname, loc="left", fontweight="bold")
        ax.set_title(f"median {obs.median():.1f}, peak month {peak}, lag-1 autocorr. {obs.autocorr(1):.2f}",
                     loc="right", fontsize=6, color="#444444")
        ax.set_ylabel("log$_{10}$ cells L$^{-1}$", fontsize=6.5)
    for ax in axes[-1]:
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b\n%Y"))
    fig.legend(handles=[Line2D([], [], color="black", lw=0.8, label="Observed"),
                        Line2D([], [], color="#9A9A9A", lw=0.8, label="Interpolated (not scored)")],
               loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, out, "Fig3_algal_density")


def fig7_radar_v3(tab, out):
    """Multiscaled radar per waterbody (2 x 5), as Saravani et al. (2025); further out is better on every axis."""
    present = set(tab["method"])
    names = [n for n in order(present) if n in present]
    keys, labels = ["inv_mae", "inv_rmse", "nse", "trend_accuracy", "hss"], ["1/MAE", "1/RMSE", "NSE", "Rise/fall\ncorrect", "HSS"]
    t = tab.copy()
    t["inv_mae"], t["inv_rmse"] = 1 / t["mae"], 1 / t["rmse"]
    ang = np.linspace(0, 2 * np.pi, len(keys), endpoint=False)
    fig, axes = plt.subplots(2, 5, figsize=(180 * MM, 95 * MM), subplot_kw={"projection": "polar"})
    for ax, (site, sname) in zip(axes.ravel(), SITES):
        g = t[t["lake"] == site].set_index("method")
        lo, hi = g[keys].min(), g[keys].max()
        base = lo - 0.15 * (hi - lo).replace(0, 1)
        for n in names:
            v = ((g.loc[n, keys] - base) / (hi - base)).to_numpy(float)
            st = style(n)
            ax.plot(np.r_[ang, ang[0]], np.r_[v, v[0]], color=st["color"], ls=st["ls"], lw=0.9)
            if n not in [r[1] for r in REFS]:
                ax.fill(np.r_[ang, ang[0]], np.r_[v, v[0]], color=st["color"], alpha=0.05)
        ax.set_xticks(ang, labels, fontsize=5)
        ax.tick_params(axis="x", pad=1)
        ax.set_yticklabels([])
        ax.set_ylim(0, 1.05)
        ax.grid(lw=0.3, color="#DDDDDD")
        ax.spines["polar"].set_linewidth(0.3)
        ax.set_title(sname, fontsize=7, pad=9)
    fig.legend(handles=legend_methods(present), loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.02), fontsize=6)
    fig.text(0.5, -0.06, "Each axis of each panel is scaled from the worst to the best model in that waterbody; further out is better. "
             "Evaluation year 2024.", ha="center", fontsize=6.2, color="#555555")
    fig.subplots_adjust(wspace=0.65, hspace=0.45, bottom=0.12)
    save(fig, out, "Fig4_radar")


def fig8_days_ahead_v3(lead, out):
    """Error reduction relative to persistence by days ahead: median over waterbodies (line), interquartile range
    (band) and each waterbody (thin lines)."""
    present = set(lead["method"])
    names = [n for n in order(present) if n in present and n not in ("Persistence",)]
    fig, ax = plt.subplots(figsize=(120 * MM, 75 * MM))
    ax.axhline(0, color="#555555", ls=(0, (3, 2)), lw=0.9)
    ax.text(1, 4, "Persistence (no change)", va="bottom", fontsize=6, color="#555555")
    for n in names:
        st = style(n)
        w = lead[lead["method"] == n].pivot(index="lead", columns="lake", values="skill") * 100
        for c in w:
            ax.plot(w.index, w[c].clip(lower=-1000), color=st["color"], lw=0.3, alpha=0.35)
        ax.fill_between(w.index, w.quantile(0.25, axis=1).clip(lower=-1000), w.quantile(0.75, axis=1).clip(lower=-1000),
                        color=st["color"], alpha=0.15, lw=0)
        ax.plot(w.index, w.median(axis=1).clip(lower=-1000), color=st["color"], ls=st["ls"], lw=1.6, marker="o", ms=3)
    ax.set_xticks(range(1, 8))
    ax.set_xlabel("Days ahead")
    ax.set_ylabel("Error reduction relative to persistence (%)")
    ax.set_yscale("symlog", linthresh=25, linscale=1.5)
    ax.set_ylim(-1000, 40)
    ax.set_yticks([-1000, -300, -100, -25, -10, 0, 10, 25], ["−1000", "−300", "−100", "−25", "−10", "0", "10", "25"])
    h = [x for x in legend_methods(present) if x.get_label() != "Persistence"]
    ax.legend(handles=h, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=6)
    fig.text(0.5, -0.04, "Thick line: median of the ten waterbodies; band: interquartile range; thin lines: individual "
             "waterbodies. Vertical axis is linear between −25% and 25% and logarithmic beyond.", ha="center", fontsize=6.2, color="#555555")
    save(fig, out, "FigS_days_ahead")


def fig4_rises_falls_v3(tab, store, chance, out):
    """(a) HSS per waterbody; (b) share of rising and falling weeks caught (median and waterbodies);
    (c-j) case weeks in the two waterbodies with the largest day-to-day variability."""
    present = set(tab["method"])
    hyb = [n for _, n, _, _ in METHODS if n in present]
    fig = plt.figure(figsize=(180 * MM, 165 * MM))
    gs = fig.add_gridspec(3, 4, height_ratios=[1.35, 1, 1], hspace=0.65, wspace=0.45)
    ax = fig.add_subplot(gs[0, :2])
    lakes = [s for s, _ in SITES][::-1]
    for j, n in enumerate(hyb):
        g = tab[tab["method"] == n].set_index("lake").reindex(lakes)
        ax.scatter(g["hss"], np.arange(len(lakes)) + (j - 1) * 0.18, color=style(n)["color"], s=12, label=n, zorder=3)
    ax.axvline(0, color="#555555", lw=0.8, ls=(0, (3, 2)))
    ax.text(0.005, -0.9, "persistence and random guessing = 0", fontsize=5.8, color="#555555", va="center")
    ax.set_ylim(-1.3, len(lakes) - 0.5)
    ax.set_yticks(range(len(lakes)), [dict(SITES)[s] for s in lakes], fontsize=6)
    ax.set_xlabel("Rise/fall skill (Heidke skill score)")
    ax.set_title("(a) Calling the direction of change", loc="left", fontsize=7)
    ax = fig.add_subplot(gs[0, 2:])
    w = 0.38
    for j, n in enumerate(hyb):
        g = tab[tab["method"] == n]
        for dx, key, a in ((-w / 2, "rise_hit", 0.9), (w / 2, "fall_hit", 0.4)):
            ax.bar(j + dx, 100 * g[key].median(), w, color=style(n)["color"], alpha=a, hatch=None if key == "rise_hit" else "////",
                   edgecolor="white", lw=0)
            ax.scatter(np.full(len(g), j + dx), 100 * g[key], color="black", s=3, zorder=3)
    ax.set_xticks(range(len(hyb)), hyb)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Weeks caught (%)")
    ax.set_title("(b) Rising vs falling weeks (bar: median; dots: waterbodies)", loc="left", fontsize=7)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color="#777777", alpha=0.9, label="Rising weeks"),
                       plt.Rectangle((0, 0), 1, 1, facecolor="#777777", alpha=0.4, hatch="////", edgecolor="white", label="Falling weeks")],
              loc="upper right", frameon=False, fontsize=6)
    var = {s: store[s]["daily"].loc[store[s]["daily"]["y_obs"], "y"].diff().abs().median() for s, _ in SITES}
    pick = sorted(var, key=var.get, reverse=True)[:2]
    names = order(present)
    for i, site in enumerate(pick):
        obs, origins, allp = store[site]["obs"], store[site]["origins"], store[site]["allp"]
        for j, (lab, k) in enumerate(pick_cases(obs, origins)):
            ax = fig.add_subplot(gs[1 + i, j])
            yt = store[site]["daily"]["y"].loc[origins[k]]
            ax.plot(range(0, 8), np.r_[yt, obs[k]], color="black", lw=1.0, marker="o", ms=2.2, zorder=4)
            for n in names:
                st = style(n)
                ax.plot(range(0, 8), np.r_[yt, allp[n][k]], color=st["color"], ls=st["ls"], lw=0.9)
            ax.set_xticks(range(0, 8))
            ax.set_title(f"({'cdefghij'[4 * i + j]}) {dict(SITES)[site]}: {lab}\nissued {origins[k]:%d %b %Y}", loc="left", fontsize=6.3)
            if j == 0:
                ax.set_ylabel("log$_{10}$ cells L$^{-1}$")
            if i == 1:
                ax.set_xlabel("Days after forecast issued")
    fig.legend(handles=legend_methods(present, observed=True), loc="lower center", ncol=7, frameon=False,
               bbox_to_anchor=(0.5, 0.0), fontsize=6)
    fig.subplots_adjust(bottom=0.1, top=0.96)
    save(fig, out, "Fig6_rises_falls")


def fig6_ranking_v3(tab, out):
    """HSS and MAE of every model in every waterbody (cell text = value; colour = rank within the waterbody,
    darker = better), with the median over the ten waterbodies (D-075, 2026-10-23: values instead of ranks)."""
    present = set(tab["method"])
    names = [n for _, n, _, _ in METHODS] + [r[1] for r in REFS]
    fig, axes = plt.subplots(2, 2, figsize=(180 * MM, 105 * MM),
                             gridspec_kw={"width_ratios": [len(SITES), 1.3], "wspace": 0.04, "hspace": 0.45})
    cmap = plt.get_cmap("YlGnBu_r", len(present))
    for row, (key, lab, higher, fmt) in enumerate((("hss", "(a) HSS (higher is better)", True, "{:.2f}"),
                                                    ("mae", "(b) MAE, log$_{10}$ cells L$^{-1}$ (lower is better)", False, "{:.2f}"))):
        R = np.full((len(names), len(SITES)), np.nan)
        V = np.full_like(R, np.nan)
        for c, (site, _) in enumerate(SITES):
            g = tab[tab["lake"] == site].set_index("method")[key]
            r = g.rank(ascending=not higher, method="min")
            for a, n in enumerate(names):
                if n in r:
                    R[a, c], V[a, c] = r[n], g[n]
        ax, ax2 = axes[row]
        ax.imshow(np.ma.masked_invalid(R), cmap=cmap, vmin=0.5, vmax=len(present) + 0.5, aspect="auto")
        for (a, b), v in np.ndenumerate(V):
            txt = "–" if not np.isfinite(v) else fmt.format(v).replace("-0.00", "0.00")
            ax.text(b, a, txt, ha="center", va="center", fontsize=5.6,
                    color="#999999" if not np.isfinite(v) else ("white" if R[a, b] <= 2 else "black"))
        ax.set_yticks(range(len(names)), [n if n in present else f"{n} (pending)" for n in names], fontsize=6.5)
        ax.set_xticks(range(len(SITES)), [s.replace("Lake ", "").replace(" Reservoir", " Res.") for _, s in SITES],
                      rotation=35, ha="right", fontsize=6)
        ax.set_title(lab, loc="left", fontsize=7.5)
        med = np.nanmedian(V, axis=1)
        mr = (pd.Series(med).rank(ascending=not higher, method="min")).to_numpy()
        ax2.imshow(np.ma.masked_invalid(mr[:, None]), cmap=cmap, vmin=0.5, vmax=len(present) + 0.5, aspect="auto")
        for a, v in enumerate(med):
            ax2.text(0, a, "–" if not np.isfinite(v) else fmt.format(v).replace("-0.00", "0.00"), ha="center", va="center",
                     fontsize=6.3, color="#999999" if not np.isfinite(v) else ("white" if mr[a] <= 2 else "black"), fontweight="bold")
        ax2.set_xticks([0], ["Median"], fontsize=6)
        ax2.set_yticks([])
        for a_ in (ax, ax2):
            a_.tick_params(length=0)
            for sp in a_.spines.values():
                sp.set_visible(False)
    fig.text(0.5, -0.02, "Darker cells: better value within the waterbody. HSS = 0: no better than chance (persistence is 0 by definition).",
             ha="center", fontsize=6.2, color="#555555")
    save(fig, out, "Fig7_ranking")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v3", choices=("v2", "v2b", "v2c", "v3"))
    ap.add_argument("--runs", default=None, help="override the run folder (testing)")
    ap.add_argument("--out", default=None, help="override the output folder (testing)")
    a = ap.parse_args()
    data.set_version(a.version)
    global SPLITS, SEGMENTS, PRED_FILE, SITES, EVAL_START, EVAL_LABEL
    if a.version == "v3":  # 10 waterbodies, 2023 training / 2024 evaluation (D-059)
        SPLITS, PRED_FILE = ("eval",), "test_predictions.csv"
        SEGMENTS = (("train", v3.START, v3.TRAIN_YEAR_END), ("eval", v3.EVAL_START, v3.END))
        EVAL_START, EVAL_LABEL = v3.EVAL_START, "2024"
        runs = HERE.parent / "results" / "runs_v3"
        if a.runs:
            runs = Path(a.runs)
        SITES = [(k, v[3]) for k, v in v3.SITES.items() if any((runs / f"{m}_{k}").exists() for m, *_ in METHODS)]
    elif a.version == "v2c":  # frozen-model re-forecast on one-piece data (experiments/reforecast.py)
        SPLITS, SEGMENTS, PRED_FILE = ("eval",), v2c.SEGMENTS, "predictions.csv"
        runs = HERE.parent / "results" / "reforecast_v2c"
    else:
        runs = HERE.parent / "results" / f"runs_{a.version}"
    out = Path(a.out) if a.out else HERE.parent / "results" / f"figures_paper_{a.version}"
    out.mkdir(parents=True, exist_ok=True)
    # Historical cleanup is not applicable to v3: Fig4_radar is now a current output.
    old_names = () if a.version == "v3" else (
                "Fig4_radar", "Fig6_days_ahead", "Fig7_rises_falls", "Fig8_ranking",  # historical names
                "Fig6_lead_time", "Fig7_direction_events", "Fig5_timeseries_dianchi", "Fig5_timeseries_chaohu")
    for old in old_names:
        for ext in (".png", ".pdf"):
            (out / f"{old}{ext}").unlink(missing_ok=True)
    tab, season, lead, store, missing, chance = collect(runs)
    tab.round(4).to_csv(out / "Table2_metrics.csv", index=False)
    if len(season):
        season.round(4).to_csv(out / "Table2_by_season.csv", index=False)
    lead.round(4).to_csv(out / "lead_time_skill.csv", index=False)
    if len(SITES) >= 5:
        lake_level_tests(tab).round(4).to_csv(out / "Table3_lake_level_tests.csv", index=False)
    fig5(store, tab, out)
    if a.version == "v3":
        fig3_v3(store, out)
        fig7_radar_v3(tab, out)
        fig8_days_ahead_v3(lead, out)
        fig4_rises_falls_v3(tab, store, chance, out)
        fig6_ranking_v3(tab, out)
    else:  # Figs 3, 4, 6-8 are laid out for two lakes; their 10-waterbody versions are still to be designed
        fig3(store, out)
        fig4(tab, out)
        fig6(lead, out)
        fig7(tab, store, chance, out)
        fig8(tab, out)
    print("written to", out)
    print("methods missing (skipped):", missing or "none")
    print("random-guess direction accuracy:", {k: round(v, 3) for k, v in chance.items()})
    cols = ["lake", "method", "n", "mae", "skill_vs_persistence", "dm_p_holm", "trend_accuracy", "hss", "rise_hit", "fall_hit"]
    print(tab[[c for c in cols if c in tab]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
