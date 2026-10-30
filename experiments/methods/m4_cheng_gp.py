"""M4: Cheng et al. (2025) growth/loss ODE + Gaussian-process error correction, adapted (protocol v1.0 section 7).

Mechanism (ln algal density Z, daily step, outflow term omitted):
    Z(d) = Z(d-1) + g(d),   g = mu - sigma
    mu    = mu_max * exp[beta_mu * (T-20)/T] * TP/(KP+TP) * TN/(KN+TN) * I/(KI+I)
    sigma = sigma_max * exp[beta_sigma * (T-20)/T]
T in degC, evaluated as max(T, 1) in the temperature term (D-045); I = daily-mean shortwave (W/m2).
The mechanistic run is continuous; it starts at the first valid day of each segment and is reset to the
observation after a gap (missing drivers, or > 2 days without a valid observation) (R2-09).
Training uses observed water temperature; forecasts use the air-temperature regression (D-024).
Within the forecast window TN/TP stay at their origin values; weather is the ERA5 value of the target day.

GP correction (Cheng Algorithm 2): errors e(s) = Z_obs(s) - Z_mech(s) for valid s in the last W days up
to and including t are conditioned on, and the GP mean is added to Z_mech(t+1..t+7). Kernel: constant
mean + squared-exponential + quasi-periodic (periodic x squared-exponential) + white noise. Hyperparameters
are fitted once on the training error series by marginal likelihood and frozen. Fewer than
`min_errors_in_window` errors -> no correction (mechanistic trajectory only).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from common.evaluate import LN10

MECH = ("mu_max_per_day", "beta_mu", "KP_mgL", "KN_mgL", "KI_W_m2", "sigma_max_per_day", "beta_sigma")
TREF = 20.0
T_FLOOR = 1.0


def growth(p: dict, T, TN, TP, I):
    Tt = np.maximum(T, T_FLOOR)
    tf = (Tt - TREF) / Tt
    mu = (p["mu_max_per_day"] * np.exp(p["beta_mu"] * tf) * TP / (p["KP_mgL"] + TP)
          * TN / (p["KN_mgL"] + TN) * I / (p["KI_W_m2"] + I))
    return mu - p["sigma_max_per_day"] * np.exp(p["beta_sigma"] * tf)


def mech_run(p: dict, z_obs: np.ndarray, T, TN, TP, I, max_gap: int = 2) -> np.ndarray:
    """Continuous mechanistic ln trajectory over the calendar (NaN where it cannot be computed)."""
    g = growth(p, T, TN, TP, I)
    n = len(z_obs)
    z = np.full(n, np.nan)
    since_obs = 10 ** 6
    for d in range(n):
        ok_obs = np.isfinite(z_obs[d])
        prev = z[d - 1] if d > 0 else np.nan
        if np.isfinite(prev) and np.isfinite(g[d]) and not (ok_obs and since_obs > max_gap):
            z[d] = prev + g[d]
        elif ok_obs and np.isfinite(g[d]):
            z[d] = z_obs[d]  # start or reset
        since_obs = 0 if ok_obs else since_obs + 1
    return z


# ---------------------------------------------------------------- GP
GP_NAMES = ("mean", "log_v_se", "log_l_se", "log_v_qp", "log_period", "log_l_qp", "log_noise")


def kernel(d: np.ndarray, h: dict) -> np.ndarray:
    se = np.exp(h["log_v_se"]) * np.exp(-0.5 * (d / np.exp(h["log_l_se"])) ** 2)
    per = np.exp(-2.0 * np.sin(np.pi * d / np.exp(h["log_period"])) ** 2)
    qp = np.exp(h["log_v_qp"]) * per * np.exp(-0.5 * (d / np.exp(h["log_l_qp"])) ** 2)
    return se + qp


def gp_nll(theta: np.ndarray, t: np.ndarray, e: np.ndarray) -> float:
    h = dict(zip(GP_NAMES, theta))
    K = kernel(np.abs(t[:, None] - t[None, :]), h) + (np.exp(h["log_noise"]) + 1e-8) * np.eye(len(t))
    try:
        L = np.linalg.cholesky(K)
    except np.linalg.LinAlgError:
        return 1e10
    r = e - h["mean"]
    a = np.linalg.solve(L.T, np.linalg.solve(L, r))
    return float(0.5 * r @ a + np.log(np.diag(L)).sum())


def gp_bounds(cfg_gp: dict) -> list:
    ls, per, nz = cfg_gp["length_scale_days"], cfg_gp["period_days"], cfg_gp["noise_var_ln"]
    v = (np.log(1e-3), np.log(10.0))
    return [(-20.0, 20.0), v, tuple(np.log(ls)), v, tuple(np.log(per)), tuple(np.log(ls)), tuple(np.log(nz))]


def gp_predict(h: dict, t_obs, e_obs, t_new) -> np.ndarray:
    K = kernel(np.abs(t_obs[:, None] - t_obs[None, :]), h) + (np.exp(h["log_noise"]) + 1e-8) * np.eye(len(t_obs))
    Ks = kernel(np.abs(t_new[:, None] - t_obs[None, :]), h)
    return h["mean"] + Ks @ np.linalg.solve(K, e_obs - h["mean"])


# ---------------------------------------------------------------- model
class M4:
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

    def fit(self, daily: pd.DataFrame, train_mask: np.ndarray, seed: int = 0) -> dict:
        """Mechanism by Gaussian maximum likelihood (= least squares on ln) then GP hyperparameters."""
        A = self.arrays(daily)
        for k in ("z", "T", "TN", "TP", "I"):  # R3-01: nothing outside the training days enters the fit
            A[k] = np.where(train_mask, A[k], np.nan)
        rng = np.random.default_rng(seed)
        lo, hi = np.array(self.bounds).T
        mech_hist, gp_hist = [], []

        def sse(x):
            z = mech_run(dict(zip(MECH, x)), A["z"], A["T"], A["TN"], A["TP"], A["I"])
            r = (A["z"] - z)[train_mask]
            r = r[np.isfinite(r)]
            return float(np.mean(r ** 2)) if len(r) else 1e10

        best = None
        for _ in range(self.cfg["mech_inits"]):
            x0 = lo + rng.random(len(lo)) * (hi - lo)
            r = minimize(sse, x0, method="L-BFGS-B", bounds=self.bounds, options={"maxiter": 200})
            mech_hist.append({"start": x0.tolist(), "fun": float(r.fun), "nit": int(r.nit)})
            if best is None or r.fun < best.fun:
                best = r
        self.mech = dict(zip(MECH, best.x))
        z = mech_run(self.mech, A["z"], A["T"], A["TN"], A["TP"], A["I"])
        e = A["z"] - z
        m = train_mask & np.isfinite(e)
        t, ev = np.flatnonzero(m).astype(float), e[m]
        bnds = gp_bounds(self.cfg["gp"])
        bnds[0] = (float(ev.min()), float(ev.max()))
        gbest = None
        for _ in range(self.cfg["gp_inits"]):
            x0 = np.array([rng.uniform(*b) for b in bnds])
            r = minimize(gp_nll, x0, args=(t, ev), method="L-BFGS-B", bounds=bnds, options={"maxiter": 200})
            gp_hist.append({"fun": float(r.fun), "nit": int(r.nit)})
            if gbest is None or r.fun < gbest.fun:
                gbest = r
        self.gp = dict(zip(GP_NAMES, gbest.x))
        return {"mech_mse": float(best.fun), "gp_nll": float(gbest.fun), "n_train_errors": int(m.sum()),
                "history": {"mech_inits": mech_hist, "gp_inits": gp_hist}}

    def predict(self, daily: pd.DataFrame, origins: pd.Series, wt_reg: tuple, window: int) -> np.ndarray:
        """(n,7) log10 forecasts. wt_reg = (alpha, beta) fitted on the training days only."""
        A = self.arrays(daily)
        z = mech_run(self.mech, A["z"], A["T"], A["TN"], A["TP"], A["I"])
        e = A["z"] - z
        idx = daily.index.get_indexer(origins)
        out = np.full((len(idx), 7), np.nan)
        min_e = self.cfg["gp"]["min_errors_in_window"]
        for k, i in enumerate(idx):
            fut = np.arange(i + 1, i + 8)
            T_hat = wt_reg[0] + wt_reg[1] * A["air"][fut]
            g = growth(self.mech, T_hat, A["TN"][i], A["TP"][i], A["I"][fut])
            zm = z[i] + np.cumsum(g)
            past = np.arange(max(0, i - window + 1), i + 1)
            ok = past[np.isfinite(e[past])]
            corr = gp_predict(self.gp, ok.astype(float), e[ok], fut.astype(float)) if len(ok) >= min_e else 0.0
            out[k] = (zm + corr) / LN10
        return out
