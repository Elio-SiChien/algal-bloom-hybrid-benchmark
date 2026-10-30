"""Summarise formal test results (protocol v1.0 sections 9-10).

    python -X utf8 experiments/analyze_results.py
Reads results/runs/<method>_<site>/ and results/runs/references_<site>/ and writes
results/runs/summary_<site>.csv and results/runs/summary.json:
- per method: trajectory MAE, HSS, raw trend accuracy, skill vs persistence (1 - MAE/MAE_persistence);
- main test: each method vs persistence, Diebold-Mariano on per-origin trajectory MAE differences,
  HAC (Newey-West, lag 6) variance, two-sided normal p-value, Holm correction over the methods present;
- between methods: MAE difference with a 95% moving-block bootstrap interval (block 14 origins, 2000
  resamples), descriptive only (no significance claim).
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import data, evaluate  # noqa: E402

RUNS = HERE.parent / "results" / "runs"
METHODS = ("M1", "M2", "M3", "M4")
HAC_LAG, BLOCK, NBOOT = 6, 14, 2000


def per_origin_mae(pred, obs):
    return np.mean(np.abs(pred - obs), axis=1)


def dm_test(d: np.ndarray, lag: int = HAC_LAG) -> tuple[float, float]:
    n = len(d)
    dc = d - d.mean()
    v = dc @ dc / n
    for k in range(1, lag + 1):
        v += 2 * (1 - k / (lag + 1)) * (dc[k:] @ dc[:-k]) / n
    stat = d.mean() / np.sqrt(v / n)
    return float(stat), float(2 * norm.sf(abs(stat)))


def holm(p: dict) -> dict:
    keys = sorted(p, key=p.get)
    m, out, run = len(keys), {}, 0.0
    for i, k in enumerate(keys):
        run = max(run, min(1.0, (m - i) * p[k]))
        out[k] = run
    return out


def block_bootstrap_ci(d: np.ndarray, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(d)
    nb = int(np.ceil(n / BLOCK))
    means = []
    for _ in range(NBOOT):
        starts = rng.integers(0, n - BLOCK + 1, nb)
        idx = np.concatenate([np.arange(s, s + BLOCK) for s in starts])[:n]
        means.append(d[idx].mean())
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def load_pred(path: Path, origins: pd.Series) -> np.ndarray:
    t = pd.read_csv(path / "test_predictions.csv")
    t = t[t["series"] == "final"].pivot(index="origin", columns="lead", values="pred_log10")
    return t.loc[origins.dt.strftime("%Y-%m-%d")].to_numpy()


def main() -> None:
    summary = {}
    for site in ("dianchi", "chaohu"):
        d, o = data.load_site(site), data.load_origins(site)
        test_o = o.loc[o["split_name"] == "test", "origin"].reset_index(drop=True)
        obs = data.labels(d, test_o)
        yt = d["y"].to_numpy()[d.index.get_indexer(test_o)]
        pers = per_origin_mae(evaluate.persistence(yt), obs)
        mae_p = float(pers.mean())
        rows, errs = [], {}
        ref = json.loads((RUNS / f"references_{site}" / "test_scores.json").read_text(encoding="utf-8"))
        for name in ("persistence", "climatology"):
            r = ref[name]
            rows.append({"method": name, "n": r["n"], "mae": r["mae"], "hss": r["hss"],
                         "trend_accuracy": r["trend_accuracy"], "skill_vs_persistence": 1 - r["mae"] / mae_p})
        pvals = {}
        for m in METHODS:
            path = RUNS / f"{m}_{site}"
            if not (path / "test_scores.json").exists():
                continue
            s = json.loads((path / "test_scores.json").read_text(encoding="utf-8"))
            pred = load_pred(path, test_o)
            e = per_origin_mae(pred, obs)
            errs[m] = e
            stat, p = dm_test(e - pers)
            pvals[m] = p
            rows.append({"method": m, "n": s["n"], "mae": s["mae"], "hss": s["hss"], "trend_accuracy": s["trend_accuracy"],
                         "skill_vs_persistence": 1 - s["mae"] / mae_p, "n_failed": s["n_failed"],
                         "dm_stat_vs_persistence": stat, "dm_p_raw": p})
        adj = holm(pvals) if pvals else {}
        for r in rows:
            if r["method"] in adj:
                r["dm_p_holm"] = adj[r["method"]]
        pairs = []
        for a, b in itertools.combinations(sorted(errs), 2):
            diff = errs[a] - errs[b]
            lo, hi = block_bootstrap_ci(diff)
            pairs.append({"a": a, "b": b, "mae_diff_a_minus_b": float(diff.mean()), "ci95": [lo, hi]})
        tab = pd.DataFrame(rows)
        tab.to_csv(RUNS / f"summary_{site}.csv", index=False)
        summary[site] = {"table": tab.round(4).to_dict(orient="records"), "pairwise_descriptive": pairs,
                         "methods_present": sorted(errs)}
    (RUNS / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
