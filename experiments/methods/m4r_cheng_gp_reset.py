"""M4 revised (M4r, D-055): Cheng et al. (2025) growth/loss equation re-initialised at every forecast origin,
plus a Gaussian process on the daily growth-rate residuals.

Why (first principles, user decision 2026-10-20): at every new 7-day origin each method may start from all
observations up to t. The original M4 ran the mechanistic model continuously from the first day and let the
GP shift that background line; the mechanistic core therefore never used the latest state (see results v2).

Forecast at origin t (ln algal density Z):
    Z(t+h) = Z(t) + sum_{k=1..h} [ g(t+k) + r_hat(t+k) ],  h = 1..7
    g = mu - sigma (same equation, bounds and temperature floor as M4); target-day ERA5 shortwave,
    regression water temperature (D-024), TN/TP held at the origin values.
    r(s) = Z(s) - Z(s-1) - g(s) is the daily growth-rate residual on past days (drivers of day s);
    the GP (constant mean + squared-exponential + quasi-periodic + white noise, as M4) is conditioned on the
    residuals of the last W days up to and including t and predicts r_hat(t+1..t+7); < min_errors -> r_hat = 0.
Calibration (training days only): mechanistic parameters minimise the mean squared ln error of 7-day
trajectories issued from the training origins with r_hat = 0 (observed water temperature, TN/TP held at t);
GP hyperparameters by marginal likelihood on the training residual series. W in {14, 30, 60} by CV.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from common.evaluate import LN10
from methods.m4_cheng_gp import GP_NAMES, MECH, gp_bounds, gp_nll, gp_predict, growth


class M4r:
    def __init__(self, cfg: dict):
        self.cfg = cfg["M4"]
        b = self.cfg["mech_bounds"]
        self.bounds = [tuple(b[k]) for k in MECH]
        self.mech = None
        self.gp = None

    @staticmethod
    def arrays(daily: pd.DataFrame) -> dict:
        return {"z": np.where(daily["y_valid"], daily["y"] * LN10, np.nan), "T": daily["wt"].to_numpy(float),
                "TN": daily["tn"].to_numpy(float), "TP": daily["tp"].to_numpy(float),
                "I": daily["sw"].to_numpy(float), "air": daily["air_temp_c"].to_numpy(float)}

    @staticmethod
    def residuals(p: dict, A: dict) -> np.ndarray:
        """r(s) = Z(s) - Z(s-1) - g(s) with the drivers of day s (NaN at s = 0 or where undefined)."""
        g = growth(p, A["T"], A["TN"], A["TP"], A["I"])
        r = np.full(len(g), np.nan)
        r[1:] = A["z"][1:] - A["z"][:-1] - g[1:]
        return r

    def fit(self, daily: pd.DataFrame, train_mask: np.ndarray, train_origins: pd.Series, seed: int = 0) -> dict:
        A = self.arrays(daily)
        for k in ("z", "T", "TN", "TP", "I"):  # nothing outside the training days enters the fit (as R3-01)
            A[k] = np.where(train_mask, A[k], np.nan)
        idx = daily.index.get_indexer(train_origins)
        fut = idx[:, None] + np.arange(1, 8)[None, :]
        z0, zlab = A["z"][idx], A["z"][fut]
        T, I = A["T"][fut], A["I"][fut]
        TN, TP = A["TN"][idx][:, None], A["TP"][idx][:, None]
        rng = np.random.default_rng(seed)
        lo, hi = np.array(self.bounds).T

        def sse(x):
            g = growth(dict(zip(MECH, x)), T, TN, TP, I)
            err = (z0[:, None] + np.cumsum(g, axis=1)) - zlab
            err = err[np.isfinite(err)]
            return float(np.mean(err ** 2)) if len(err) else 1e10

        best, hist = None, []
        for _ in range(self.cfg["mech_inits"]):
            x0 = lo + rng.random(len(lo)) * (hi - lo)
            r = minimize(sse, x0, method="L-BFGS-B", bounds=self.bounds, options={"maxiter": 200})
            hist.append({"fun": float(r.fun), "nit": int(r.nit)})
            if best is None or r.fun < best.fun:
                best = r
        self.mech = dict(zip(MECH, best.x))
        res = self.residuals(self.mech, A)
        m = np.isfinite(res) & train_mask
        t, ev = np.flatnonzero(m).astype(float), res[m]
        bnds = gp_bounds(self.cfg["gp"])
        bnds[0] = (float(ev.min()), float(ev.max()))
        gbest, ghist = None, []
        for _ in range(self.cfg["gp_inits"]):
            x0 = np.array([rng.uniform(*b) for b in bnds])
            r = minimize(gp_nll, x0, args=(t, ev), method="L-BFGS-B", bounds=bnds, options={"maxiter": 200})
            ghist.append({"fun": float(r.fun), "nit": int(r.nit)})
            if gbest is None or r.fun < gbest.fun:
                gbest = r
        self.gp = dict(zip(GP_NAMES, gbest.x))
        return {"mech_mse_7d": float(best.fun), "gp_nll": float(gbest.fun), "n_train_residuals": int(m.sum()),
                "history": {"mech_inits": hist, "gp_inits": ghist}}

    def predict(self, daily: pd.DataFrame, origins: pd.Series, wt_reg: tuple, window: int) -> np.ndarray:
        A = self.arrays(daily)
        res = self.residuals(self.mech, A)  # past residuals use observed drivers of past days
        idx = daily.index.get_indexer(origins)
        out = np.full((len(idx), 7), np.nan)
        min_e = self.cfg["gp"]["min_errors_in_window"]
        for k, i in enumerate(idx):
            fut = np.arange(i + 1, i + 8)
            T_hat = wt_reg[0] + wt_reg[1] * A["air"][fut]
            g = growth(self.mech, T_hat, A["TN"][i], A["TP"][i], A["I"][fut])
            past = np.arange(max(1, i - window + 1), i + 1)
            ok = past[np.isfinite(res[past])]
            r_hat = gp_predict(self.gp, ok.astype(float), res[ok], fut.astype(float)) if len(ok) >= min_e else 0.0
            out[k] = (A["z"][i] + np.cumsum(g + r_hat)) / LN10
        return out
