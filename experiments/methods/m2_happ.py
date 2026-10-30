"""M2: HAPP algal growth potential (AGP) + fixed 7-day lagged linear map (Xie et al. 2025, adapted).

Prediction for origin t and lead h (protocol v1.0 section 7, D-045):
    s = t + h - 7                   (inputs of days t-6..t only; no future weather)
    y_hat(t+h|t) = a * AGP(s) + b   (y = log10 algal density)
AGP is the positive logistic equilibrium of
    RG - R - Mmax*C/(C+KM) - S_const = 0,  RG = mu_max*fT*fI*fN*fP,  R = kR*thetaR**(T-20)
-> q = RG - R - S_const;  AGP = KM*q/(Mmax-q) for 0<q<Mmax, 0 for q<=0, invalid for q>=Mmax.
KM is fixed at 22.5 mg/L (Q-003). The 10 kinetic parameters are searched by Bayesian optimisation
(<=200 evaluations) minimising the training 7-day trajectory mean squared error; for every candidate
the linear map a, b is solved directly by bounded least squares (D-046; exact bounded solver, D-070), because given AGP the best
a, b have a closed form and guessing them wastes the evaluation budget.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common import bayesopt

KINETIC = ("mu_max_per_day", "kG", "Topt_degC", "Iopt_umol_m2_s", "KN_mgL", "KP_mgL",
           "Mmax_per_day", "kR_per_day", "thetaR", "S_const_per_day")
PARAMS = KINETIC + ("a", "b_log10")


def solve_ab(A: np.ndarray, y: np.ndarray, a_bounds, b_bounds) -> tuple[float, float]:
    """Bounded least squares y ~ a*A + b with a, b inside their bounds (D-046; exact solver since A-04, D-070)."""
    from scipy.optimize import lsq_linear
    A, y = A.ravel(), y.ravel()
    ok = np.isfinite(A) & np.isfinite(y)
    A, y = A[ok], y[ok]
    if len(y) == 0:
        return 0.0, float(np.clip(0.0, *b_bounds))
    if np.ptp(A) == 0:  # constant AGP: the slope is not identifiable -> a = 0, best bounded intercept
        return 0.0, float(np.clip(np.mean(y), *b_bounds))
    r = lsq_linear(np.c_[A, np.ones_like(A)], y, bounds=([a_bounds[0], b_bounds[0]], [a_bounds[1], b_bounds[1]]),
                   method="bvls")
    return float(r.x[0]), float(r.x[1])


def env_matrix(daily: pd.DataFrame, origins: pd.Series) -> dict:
    """(n,7) arrays of T, I, TN, TP on days s = t-6..t (column h-1 <-> lead h)."""
    idx = daily.index.get_indexer(origins)
    out = {}
    for key, col in (("T", "wt"), ("I", "par"), ("TN", "tn"), ("TP", "tp")):
        v = daily[col].to_numpy(dtype=float)
        out[key] = np.stack([v[idx + h - 7] for h in range(1, 8)], axis=1)
    return out


def agp(p: dict, env: dict, KM: float) -> np.ndarray:
    T, I, TN, TP = env["T"], env["I"], env["TN"], env["TP"]
    fT = np.exp(-p["kG"] * np.abs(T - p["Topt_degC"]))
    fI = (I / p["Iopt_umol_m2_s"]) * np.exp(1.0 - I / p["Iopt_umol_m2_s"])
    fN = TN / (TN + p["KN_mgL"])
    fP = TP / (TP + p["KP_mgL"])
    RG = p["mu_max_per_day"] * fT * fI * fN * fP
    R = p["kR_per_day"] * p["thetaR"] ** (T - 20.0)
    q = RG - R - p["S_const_per_day"]
    Mmax = p["Mmax_per_day"]
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(q <= 0, 0.0, KM * q / (Mmax - q))
    out[q >= Mmax] = np.nan  # no finite equilibrium
    return out


def predict(p: dict, env: dict, KM: float) -> np.ndarray:
    return p["a"] * agp(p, env, KM) + p["b_log10"]


class M2:
    def __init__(self, cfg: dict):
        self.cfg = cfg["M2"]
        bd = self.cfg["bounds"]
        self.lo = np.array([bd[k][0] for k in KINETIC])
        self.hi = np.array([bd[k][1] for k in KINETIC])
        self.a_bounds, self.b_bounds = tuple(bd["a"]), tuple(bd["b_log10"])
        self.KM = float(self.cfg["KM_mgL_fixed"])
        self.params = None

    def _full(self, x, env, y) -> dict | None:
        p = dict(zip(KINETIC, x))
        A = agp(p, env, self.KM)
        if not np.isfinite(A).all():
            return None  # section 12: any origin without 7 finite values -> invalid
        p["a"], p["b_log10"] = solve_ab(A, y, self.a_bounds, self.b_bounds)
        return p

    def fit(self, env: dict, y: np.ndarray, seed: int, n_evals: int | None = None) -> dict:
        def loss(x):
            p = self._full(x, env, y)
            return np.inf if p is None else float(np.mean((predict(p, env, self.KM) - y) ** 2))

        res = bayesopt.minimize(loss, self.lo, self.hi, n_evals=n_evals or self.cfg["bayesopt_max_evals"], seed=seed)
        self.params = self._full(res["x"], env, y)
        return res

    def predict(self, env: dict) -> np.ndarray:
        if self.params is None:  # R3-11: every evaluation failed -> section 12 failure handling
            return np.full(env["T"].shape, np.nan)
        return predict(self.params, env, self.KM)
