"""Formal run for one method and one site.

    python -X utf8 experiments/run_method.py --method M2 --site dianchi --data v3 --formal
    python -X utf8 experiments/run_method.py --method M2 --site dianchi --data v3 --dry-run
Stages:
  1. Blocked cross-validation (v3: the four quarters of 2023, D-059; v1: 5 blocks) with 7-day buffers that are
     widened along interpolated runs; selects the configuration where the method has one (M1: Optuna TPE,
     seed 0; M4/M4r: GP window {14,30,60}); M2/M3 have none, CV is reported for the median-over-seeds forecast.
  2. Final fit on all training origins (5 seeds for M1/M2/M3, median over seeds - D-047). v3: no training day after
     L (last real observation of 2023) and no training origin whose window touches an excluded span (R6-05).
  3. One forecast of the evaluation period, scored once on real observations; failed origins are replaced by
     persistence and counted. v3: the final model also forecasts every training origin (in-sample fit, Fig. 5).
Outputs: results/runs_<data>/<method>_<site>/ (formal) or results/dry_runs_<data>/<method>_<site>/ (dry run).
A formal run (and `--method references`) reads the evaluation period: it requires --formal, i.e. the user's
authorisation. A dry run uses the last CV block as pseudo-test, tiny budgets, and never reads the evaluation period.
"""

from __future__ import annotations

import argparse
import faulthandler
import hashlib
import json
import pickle
import platform
import shutil
import sys
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# Windows/Anaconda: MKL-numpy and PyTorch each ship an OpenMP runtime. If PyTorch loads first, the first
# numpy linear-algebra call aborts the process (OMP Error #15). Initialising numpy's linear algebra before
# PyTorch is imported avoids this without the unsafe KMP_DUPLICATE_LIB_OK switch.
np.linalg.lstsq(np.eye(2), np.ones(2), rcond=None)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import data, evaluate, splits  # noqa: E402
from common.origins import origin_list_sha256  # noqa: E402

ROOT = HERE.parent
SEEDS = (0, 1, 2, 3, 4)
CODE_FILES = ["run_method.py", "search_space_v1.json", "common/data.py", "common/splits.py", "common/evaluate.py",
              "common/bayesopt.py", "common/pso.py", "common/v2.py", "common/qc.py", "common/v3.py"]
METHOD_FILES = {"M1": "methods/m1_tpgnn.py", "M2": "methods/m2_happ.py", "M3": "methods/m3_neural_growth.py",
                "M4": "methods/m4_cheng_gp.py", "M4r": "methods/m4r_cheng_gp_reset.py"}


def code_sha256(method: str | None = None) -> str:
    """Shared code + the method's own file (2026-10-21: editing one method no longer invalidates the others' caches)."""
    h = hashlib.sha256()
    for f in CODE_FILES + ([METHOD_FILES[method]] if method in METHOD_FILES else list(METHOD_FILES.values())):
        h.update((HERE / f).read_bytes())
    return h.hexdigest()


def jsonable(x):
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return str(x)


def median_over_seeds(preds: list[np.ndarray]) -> np.ndarray:
    """Per-origin, per-lead median over seeds, finite values only (D-047; R3-05)."""
    a = np.stack(preds).astype(float)
    a[~np.isfinite(a)] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(a, axis=0)


def scored(pred, daily, origins, observed=None):
    """Score forecasts. v2 main scoring (observed=True): real-observation days only, origins with
    >= MIN_OBS_LABELS observed labels; observed=False scores every day (v2: interpolated labels too)."""
    if observed is None:
        observed = data.is_v2()
    y = data.labels(daily, origins, observed_only=observed)
    yt = daily["y"].to_numpy()[daily.index.get_indexer(origins)]
    p, bad = evaluate.fill_failures(pred, yt)
    keep = np.ones(len(y), bool)
    if observed and data.is_v2():
        from common.v2 import MIN_OBS_LABELS
        keep = (np.isfinite(y).sum(axis=1) >= MIN_OBS_LABELS) & data.real_origin_mask(daily, origins)
    s = evaluate.score(p[keep], y[keep])
    s["n_failed"] = int(bad[keep].sum())
    return s, p


def scored_all_ways(pred, daily, origins) -> dict:
    if data.VERSION == "v1":
        return scored(pred, daily, origins)[0]
    if data.VERSION == "v3":  # D-059: real-observation scoring only (the all-days supplement is dropped)
        return {"observed_main": scored(pred, daily, origins, True)[0]}
    return {"observed_main": scored(pred, daily, origins, True)[0],
            "all_days_supplement": scored(pred, daily, origins, False)[0]}


def check_locked_inputs(site: str, o: pd.DataFrame, d: pd.DataFrame) -> dict:
    """R3-12 / R3-10: verify the locked origin list and that no origin's day-t driver is a filled value."""
    man = data.manifest()
    full = pd.read_csv(data.origins_file(), parse_dates=["origin", "last_label_date"])
    full = full[full["site"] == site]
    if data.VERSION == "v3":
        from common import v3
        if man.get("status") != "locked":
            raise SystemExit("v3 data are not locked (manifest_v3.json status)")
        sha = v3.sha256_text(full)
    elif data.is_v2():
        from common.v2 import origin_list_sha256 as sha_v2
        sha = sha_v2(full)
    else:
        sha = origin_list_sha256(full)
    if sha != man["sites"][site]["origin_list_sha256"]:
        raise SystemExit(f"origins.csv for {site} does not match the locked manifest")
    if data.is_v2():
        base, daily_name, _, _ = data._paths()
        got = hashlib.sha256((base / daily_name.format(site)).read_bytes()).hexdigest()
        want = man["sites"][site]["daily_sha256"] if data.VERSION == "v3" else man["daily_v2_sha256"][site]
        if got != want:
            raise SystemExit(f"qc_daily_v2_{site}.csv does not match the locked manifest (R4-03)")
    i = d.index.get_indexer(full["origin"])
    if data.VERSION == "v2" and not d["y_obs"].to_numpy(bool)[i].all():
        raise SystemExit("a v2 origin has an interpolated algal density on day t (R4-03)")
    extra = {}
    if data.VERSION == "v3":  # R6-06: whole origins file (incl. the scored flag), ERA5 files and search space
        base = data.origins_file().parent
        checks = {"origins_v3.csv": man["origins_csv_sha256"], f"era5land_{site}.csv": man["era5_sha256"][site],
                  "era5land_v3_points.csv": man["era5_points_sha256"]}
        for f, want in checks.items():
            if hashlib.sha256((base / f).read_bytes()).hexdigest() != want:
                raise SystemExit(f"{f} does not match the locked manifest")
        if hashlib.sha256((HERE / "search_space_v1.json").read_bytes()).hexdigest() != man["search_space_sha256"]:
            raise SystemExit("search_space_v1.json does not match the locked manifest")
        extra["era5_sha256"] = man["era5_sha256"][site]
    if data.VERSION == "v3":  # scored evaluation origins must be real on day t (D-059); others are not scored
        sc = full["scored"].to_numpy(bool)
        if not data.real_origin_mask(d, full["origin"])[sc].all():
            raise SystemExit("a scored v3 origin has an interpolated value on day t")
    elif data.VERSION != "v2b" and d[["wt_filled", "tn_filled", "tp_filled"]].to_numpy(bool)[i].any():
        raise SystemExit("an origin has a gap-filled water temperature/TN/TP on day t (would use later data)")
    return {"qc_version": man.get("qc_version", "qc-v1.1+" + data.VERSION), "data_version": data.VERSION,
            "origin_list_sha256": sha, **extra}


class Runner:
    def __init__(self, method, site, cfg):
        self.method, self.site, self.cfg = method, site, cfg
        self.d = data.load_site(site)
        self.o = data.load_origins(site)
        self.extra_exclude = []  # dry run: the pseudo-test block is never used for training
        if data.VERSION == "v3":  # D-059: no training day may be interpolated from a 2024 observation
            from common import v3
            self.extra_exclude.append((v3.last_real_2023(self.d) + pd.Timedelta(days=1), pd.Timestamp("2099-12-31")))

    def fold_context(self, exclude):
        spans = list(self.extra_exclude) + ([exclude] if exclude is not None else [])
        tdays = splits.train_days(self.d.index, spans)
        return tdays, data.fit_wt_regression(self.d, tdays)

    def fit_predict(self, train_mask, exclude, pred_origins, seed, conf=None):
        """Train on `train_mask` origins / training days, predict `pred_origins` -> (pred, info)."""
        # R6-05: drop every training origin whose window t-6..t+7 touches an excluded span (fold buffer, dry-run
        # pseudo-test, v3 cut-off L), so no label or input interpolated towards excluded observations is used.
        w0, w1 = splits.window(self.o["origin"])
        train_mask = np.asarray(train_mask, bool).copy()
        for lo, hi in list(self.extra_exclude) + ([exclude] if exclude is not None else []):
            train_mask &= ~((w0 <= hi) & (w1 >= lo)).to_numpy()
        tdays, reg = self.fold_context(exclude)
        tr_o = self.o.loc[train_mask, "origin"]
        if self.method == "M2":
            from methods import m2_happ
            m = m2_happ.M2(self.cfg)
            r = m.fit(m2_happ.env_matrix(self.d, tr_o), data.labels(self.d, tr_o), seed=seed)
            return m.predict(m2_happ.env_matrix(self.d, pred_origins)), {
                "params": m.params, "n_failed_evals": r["n_failed"], "history": r["history"], "seconds": r["seconds"]}
        if self.method == "M3":
            from methods import m3_neural_growth as m3
            m = m3.M3(self.cfg)
            z0, exo = m3.build_inputs(self.d, tr_o, reg)
            r = m.fit(z0, exo, data.labels(self.d, tr_o), seed=seed)
            z0p, exop = m3.build_inputs(self.d, pred_origins, reg)
            return m.predict(z0p, exop), {"iterations": r["iterations"], "train_mse": r["fun"],
                                          "sigma_max_beta": m.theta[36:].tolist(), "theta": m.theta.tolist(),
                                          "history": r["history"], "seconds": r["seconds"]}
        if self.method == "M4":
            from methods import m4_cheng_gp as m4
            m = m4.M4(self.cfg)
            info = m.fit(self.d, self.d.index.isin(tdays), seed=seed)
            preds = {w: m.predict(self.d, pred_origins, reg, w) for w in self.cfg["M4"]["gp"]["window_days"]["choice"]}
            return preds, {**info, "mech": m.mech, "gp": m.gp, "wt_reg": reg}
        if self.method == "M4r":  # D-055: re-initialised at y(t) every origin
            from methods import m4r_cheng_gp_reset as m4r
            m = m4r.M4r(self.cfg)
            info = m.fit(self.d, self.d.index.isin(tdays), tr_o, seed=seed)
            preds = {w: m.predict(self.d, pred_origins, reg, w) for w in self.cfg["M4"]["gp"]["window_days"]["choice"]}
            return preds, {**info, "mech": m.mech, "gp": m.gp, "wt_reg": reg}
        if self.method == "M1":
            from methods import m1_tpgnn as m1
            X, Y = m1.training_pairs(self.d, self.d.index.isin(tdays))
            m = m1.M1(self.cfg, conf, seed=seed)
            info = m.fit(X, Y)
            return m.predict(self.d, pred_origins, reg), {**info, "n_train_pairs": int(len(Y))}
        raise ValueError(self.method)


def dry_run_setup(R: Runner) -> None:
    """Code check only: the last CV block plays the test period and is excluded from all training (R3-06);
    the real evaluation periods are dropped."""
    last = splits.cv_blocks()[-1]
    o = R.o[R.o["split_name"].str.startswith("cv_")].copy()
    o.loc[o["split_name"] == last[0], "split_name"] = "test"
    R.o = o.reset_index(drop=True)
    b5 = last
    R.extra_exclude.append((pd.Timestamp(b5[1]) - pd.Timedelta(days=splits.BUFFER), pd.Timestamp("2099-12-31")))
    c = R.cfg
    c["M2"]["bayesopt_max_evals"] = 25
    c["M3"]["pso"]["max_iter"] = 15
    c["M4"]["mech_inits"] = c["M4"]["gp_inits"] = 1
    c["M1"]["tpe_max_trials"], c["M1"]["max_epochs"] = 2, 2


def run(method: str, site: str, out: Path, dry: bool) -> None:
    cfg = json.loads((HERE / "search_space_v1.json").read_text(encoding="utf-8"))
    if dry and out.exists():
        shutil.rmtree(out)  # R3-09: a dry run always starts clean
    if (out / "test_scores.json").exists():
        raise SystemExit(f"Refusing to overwrite an existing test result: {out}")
    out.mkdir(parents=True, exist_ok=True)
    R = Runner(method, site, cfg)
    log = {"method": method, "site": site, "dry_run": dry, "code_sha256": code_sha256(method),
           **check_locked_inputs(site, R.o, R.d),
           "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
           "machine": platform.node(), "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    if dry:
        dry_run_setup(R)
    o, d = R.o, R.d
    cache = out / "_resume_cache"  # long cloud runs: a killed run resumes from finished (fold, seed) fits

    def fit_predict(key, *args, **kw):
        if dry:
            return R.fit_predict(*args, **kw)
        cache.mkdir(exist_ok=True)
        tag = cache / "code_sha256.txt"
        if not tag.exists():
            tag.write_text(log["code_sha256"])
        elif tag.read_text() != log["code_sha256"]:
            raise SystemExit("resume cache was produced by different code; move it away first")
        f = cache / f"{key}.pkl"
        if f.exists():
            return pickle.loads(f.read_bytes())
        r = R.fit_predict(*args, **kw)
        f.with_suffix(".tmp").write_bytes(pickle.dumps(r))
        f.with_suffix(".tmp").replace(f)
        return r

    t0 = time.perf_counter()
    folds = [f for f in splits.cv_folds(o, d if data.is_v2() else None) if f[2].any()]
    conf, cv, cv_hist = None, {}, {}

    # ---------------- 1. cross-validation / selection
    if method == "M1":
        import optuna
        import torch
        log["torch"] = torch.__version__
        log["optuna"] = optuna.__version__
        hp = cfg["M1"]["hyper"]

        def objective(trial):
            c = {k: (trial.suggest_categorical(k, v["choice"]) if "choice" in v else
                     trial.suggest_float(k, *v["loguniform"], log=True) if "loguniform" in v else
                     trial.suggest_float(k, *v["uniform"])) for k, v in hp.items()}
            if c["d_model"] % c["n_heads"]:
                return float("inf")
            try:
                maes = []
                for name, tr, va, ex in folds:
                    p, info = R.fit_predict(tr, ex, o.loc[va, "origin"], seed=0, conf=c)
                    if not np.isfinite(p).all():
                        trial.set_user_attr("invalid", f"non-finite forecast in {name}")
                        return float("inf")  # section 12: invalid configuration
                    maes.append(scored(p, d, o.loc[va, "origin"])[0]["mae"])
                    trial.set_user_attr(f"epochs_{name}", info["epochs"])
                    trial.set_user_attr(f"loss_hist_{name}", [round(v, 6) for v in info["history"]])
                return float(np.mean(maes))
            except Exception:  # R3-09: one bad trial must not stop the night
                trial.set_user_attr("error", traceback.format_exc()[-2000:])
                return float("inf")

        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=0),
                                    study_name=f"M1_{site}", storage=f"sqlite:///{(out / 'optuna.db').as_posix()}",
                                    load_if_exists=True)
        old = study.user_attrs.get("code_sha256")
        if old is not None and old != log["code_sha256"]:
            raise SystemExit("optuna.db was produced by different code; move it away before resuming (R3-09)")
        study.set_user_attr("code_sha256", log["code_sha256"])
        done = len([t for t in study.trials if t.state.is_finished()])
        study.optimize(objective, n_trials=max(0, cfg["M1"]["tpe_max_trials"] - done))
        if not np.isfinite(study.best_value):
            raise SystemExit("no valid M1 configuration")
        conf = study.best_params
        study.trials_dataframe().to_csv(out / "cv_trials.csv", index=False)
        cv = {"best_mean_cv_mae_seed0": study.best_value, "best_params": conf, "n_trials": len(study.trials)}
    else:
        rows = []
        for name, tr, va, ex in folds:
            vo = o.loc[va, "origin"]
            if method in ("M4", "M4r"):
                preds, info = fit_predict(f"cv_{name}", tr, ex, vo, seed=0)
                cv_hist[name] = info["history"]
                for w, p in preds.items():
                    s, _ = scored(p, d, vo)
                    if s["n_failed"]:  # R3-07: any failed origin makes the configuration invalid
                        s["mae"] = float("inf")
                    rows.append({"fold": name, "window": w, **s})
            else:
                ps = []
                for seed in SEEDS:
                    p, info = fit_predict(f"cv_{name}_seed{seed}", tr, ex, vo, seed=seed)
                    ps.append(p)
                    cv_hist[f"{name}_seed{seed}"] = info["history"]
                    rows.append({"fold": name, "seed": seed, **scored(p, d, vo)[0]})
                rows.append({"fold": name, "seed": "median", **scored(median_over_seeds(ps), d, vo)[0]})  # R3-08
        cvdf = pd.DataFrame(rows)
        cvdf.drop(columns=["mae_by_lead"]).to_csv(out / "cv_folds.csv", index=False)
        if method in ("M4", "M4r"):
            means = cvdf.groupby("window")["mae"].mean()
            conf = int(means.idxmin())
            cv = {"mean_cv_mae_by_window": means.round(5).to_dict(), "selected_window": conf}
        else:
            med = cvdf[cvdf["seed"] == "median"]
            cv = {"mean_cv_mae_median_forecast": float(med["mae"].mean()),
                  "mean_cv_mae_by_seed": cvdf[cvdf["seed"] != "median"].groupby("seed")["mae"].mean().round(5).to_dict()}
    (out / "cv_summary.json").write_text(json.dumps(cv, indent=2, default=jsonable), encoding="utf-8")
    (out / "cv_histories.json").write_text(json.dumps(cv_hist, default=jsonable), encoding="utf-8")
    log["cv_seconds"] = round(time.perf_counter() - t0, 1)

    # ---------------- 2. final fit on blocks 1-5, 3. test once
    t1 = time.perf_counter()
    all_cv = o["split_name"].str.startswith("cv_").to_numpy()
    ev_splits = ("test",) if dry else data.eval_splits()
    if data.VERSION == "v3" and not dry:  # the final model also forecasts every training origin (in-sample fit, Fig. 5)
        ev = o.reset_index(drop=True)
    else:
        ev = o[o["split_name"].isin(ev_splits)].reset_index(drop=True)
    test_o = ev["origin"]
    final_info = {}
    if method in ("M4", "M4r"):
        preds, info = fit_predict("final", all_cv, None, test_o, seed=0)
        pred, final_info, seed_preds = preds[conf], info, {"seed0": preds[conf]}
    else:
        seed_preds = {}
        for seed in SEEDS:
            p, info = (fit_predict(f"final_seed{seed}", all_cv, None, test_o, seed=seed, conf=conf) if method != "M1"
                       else R.fit_predict(all_cv, None, test_o, seed=seed, conf=conf))
            seed_preds[f"seed{seed}"] = p
            final_info[f"seed{seed}"] = info
        pred = median_over_seeds(list(seed_preds.values()))
    filled = scored(pred, d, test_o)[1]
    s = {}
    for sp in ev_splits:
        m = (ev["split_name"] == sp).to_numpy()
        s[sp] = scored_all_ways(pred[m], d, test_o[m])
    if data.VERSION == "v1":
        s = s["test"]  # v1 file format unchanged
    log["final_and_test_seconds"] = round(time.perf_counter() - t1, 1)
    log["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")

    long = [(name, ev["split_name"].iloc[k], t.strftime("%Y-%m-%d"), h + 1, p[k, h])
            for name, p in {**seed_preds, "final": filled}.items() for k, t in enumerate(test_o) for h in range(7)]
    tab = pd.DataFrame(long, columns=["series", "split_name", "origin", "lead", "pred_log10"])
    if dry:
        tab["split_name"] = "pseudo_test_last_cv_block"
    tab.insert(0, "method", method); tab.insert(1, "lake_id", site)
    tab["forecast_target"] = "algal_density_log10"
    tab["qc_version"], tab["origin_list_sha256"] = log["qc_version"], log["origin_list_sha256"]
    tab.to_csv(out / "test_predictions.csv", index=False)
    (out / "final_fit.json").write_text(json.dumps(final_info, indent=1, default=jsonable), encoding="utf-8")
    (out / "run_log.json").write_text(json.dumps(log, indent=2, default=jsonable), encoding="utf-8")
    (out / "test_scores.json").write_text(json.dumps(s, indent=2), encoding="utf-8")
    print(json.dumps({"cv": cv, "test" if not dry else "pseudo_test": s, "log": log}, indent=2, default=jsonable))


def references(site: str, out: Path) -> None:
    """Persistence and monthly climatology on the test period (section 8). Formal runs only (R3-04)."""
    d, o = data.load_site(site), data.load_origins(site)
    if (out / "test_scores.json").exists():
        raise SystemExit(f"Refusing to overwrite: {out}")
    out.mkdir(parents=True, exist_ok=True)
    res = {}
    for sp in data.eval_splits():
        test_o = o.loc[o["split_name"] == sp, "origin"]
        yt = d["y"].to_numpy()[d.index.get_indexer(test_o)]
        clim_days = splits.train_days(d.index, None)
        if data.is_v2():
            clim_days = evaluate.climatology_days(d, clim_days)  # climatology from real observations
        r = {"persistence": scored_all_ways(evaluate.persistence(yt), d, test_o),
             "climatology": scored_all_ways(evaluate.climatology(d, clim_days, test_o), d, test_o)}
        res[sp] = r
    if data.VERSION == "v1":
        res = res["test"]
    res.update(check_locked_inputs(site, o, d))
    (out / "test_scores.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    faulthandler.enable()  # print a traceback even on a hard crash
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--method", required=True, choices=["M1", "M2", "M3", "M4", "M4r", "references"])
    from common import v3 as _v3
    ap.add_argument("--site", required=True, choices=list(_v3.SITES))
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--formal", action="store_true", help="formal run, reads the test period (needs Q-008)")
    g.add_argument("--dry-run", action="store_true", help="code check: last CV block as pseudo-test, tiny budgets")
    ap.add_argument("--data", default="v1", choices=["v1", "v2", "v2b", "v3"], help="v1: protocol v1.0; v2: 2023-2024 interpolated (D-050)")
    a = ap.parse_args()
    data.set_version(a.data)
    suffix = {"v1": "", "v2": "_v2", "v2b": "_v2b", "v3": "_v3"}[a.data]
    base = ROOT / "results" / (("runs" if a.formal else "dry_runs") + suffix)
    if a.method == "references":
        if not a.formal:
            raise SystemExit("references read the test period; only with --formal (R3-04)")
        references(a.site, base / f"references_{a.site}")
    else:
        run(a.method, a.site, base / f"{a.method}_{a.site}", dry=a.dry_run)
