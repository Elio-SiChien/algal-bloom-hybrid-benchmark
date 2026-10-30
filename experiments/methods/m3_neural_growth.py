"""M3: Cheng growth/loss equation with the growth rate replaced by a small ANN (Mohanty-style), adapted.

    dZ/dtau = mu_NN(x) - sigma(T),   Z = ln(algal density),  integrated with RK4, step 1 day
    x = [log10 C, T_hat, TN(t), TP(t), I]  scaled to [-1, 1] with training min/max (no clipping)
    mu_NN: 5-5-1, tanh hidden layer, linear output (signed effective rate, day^-1)
    sigma(T) = sigma_max * exp[beta_sigma * (T-20)/T], T in degC evaluated as max(T, 1) (D-045)
Within each day the exogenous inputs are fixed at the target-day values: T_hat from the air-temperature
regression (D-024), I = daily-mean shortwave of the target day (ERA5), TN/TP held at the origin values.
The state starts from the observed value at origin t; the 7 steps never read observed future values.
36 network weights + sigma_max + beta_sigma are learned together by PSO on the 7-day training trajectories
(mean squared error in log10).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common import pso
from common.evaluate import LN10

N_IN, N_HID = 5, 5
N_NET = N_IN * N_HID + N_HID + N_HID + 1  # 36
TREF, T_FLOOR = 20.0, 1.0


def unpack(theta: np.ndarray):
    """theta: (P, 38) -> W1 (P,5,5), b1 (P,5), w2 (P,5), b2 (P,), sigma_max (P,), beta_sigma (P,)."""
    i = 0
    W1 = theta[:, i:i + N_IN * N_HID].reshape(-1, N_IN, N_HID); i += N_IN * N_HID
    b1 = theta[:, i:i + N_HID]; i += N_HID
    w2 = theta[:, i:i + N_HID]; i += N_HID
    b2 = theta[:, i]; i += 1
    return W1, b1, w2, b2, theta[:, i], theta[:, i + 1]


class Scaler:
    def __init__(self, lo, hi):
        self.lo, self.hi = np.asarray(lo, float), np.asarray(hi, float)
        self.hi = np.where(self.hi > self.lo, self.hi, self.lo + 1.0)  # constant input -> avoid 0/0

    def __call__(self, x):
        return 2.0 * (x - self.lo) / (self.hi - self.lo) - 1.0


def rhs(theta, z, exo_scaled, T, scaler):
    """dZ/dtau for P parameter sets x N origins. z: (P,N); exo_scaled: (N,4); T: (N,)."""
    W1, b1, w2, b2, smax, bsig = unpack(theta)
    xc = 2.0 * (z / LN10 - scaler.lo[0]) / (scaler.hi[0] - scaler.lo[0]) - 1.0          # (P,N)
    pre = xc[:, :, None] * W1[:, None, 0, :] + np.einsum("nk,pkh->pnh", exo_scaled, W1[:, 1:, :]) + b1[:, None, :]
    mu = np.einsum("pnh,ph->pn", np.tanh(pre), w2) + b2[:, None]
    Tt = np.maximum(T, T_FLOOR)
    sigma = smax[:, None] * np.exp(bsig[:, None] * ((Tt - TREF) / Tt)[None, :])
    return mu - sigma


def simulate(theta: np.ndarray, z0: np.ndarray, exo: np.ndarray, scaler: Scaler) -> np.ndarray:
    """theta (P,38); z0 (N,) ln start; exo (N,7,4) raw [T_hat, TN, TP, I] per step. Returns (P,N,7) log10."""
    P, N = theta.shape[0], len(z0)
    z = np.repeat(z0[None, :], P, axis=0)
    out = np.empty((P, N, 7))
    for h in range(7):
        ex = exo[:, h, :]
        exs = 2.0 * (ex - scaler.lo[1:]) / (scaler.hi[1:] - scaler.lo[1:]) - 1.0
        T = ex[:, 0]
        k1 = rhs(theta, z, exs, T, scaler)
        k2 = rhs(theta, z + 0.5 * k1, exs, T, scaler)
        k3 = rhs(theta, z + 0.5 * k2, exs, T, scaler)
        k4 = rhs(theta, z + k3, exs, T, scaler)
        with np.errstate(over="ignore", invalid="ignore"):
            z = z + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        out[:, :, h] = z / LN10
    return out


def build_inputs(daily: pd.DataFrame, origins: pd.Series, wt_reg: tuple) -> tuple[np.ndarray, np.ndarray]:
    """z0 (N,) and exo (N,7,4) = [T_hat(t+h), TN(t), TP(t), SW(t+h)]."""
    idx = daily.index.get_indexer(origins)
    y = daily["y"].to_numpy(float)
    air, sw = daily["air_temp_c"].to_numpy(float), daily["sw"].to_numpy(float)
    tn, tp = daily["tn"].to_numpy(float), daily["tp"].to_numpy(float)
    fut = idx[:, None] + np.arange(1, 8)[None, :]
    exo = np.stack([wt_reg[0] + wt_reg[1] * air[fut], np.repeat(tn[idx, None], 7, 1),
                    np.repeat(tp[idx, None], 7, 1), sw[fut]], axis=2)
    return y[idx] * LN10, exo


class M3:
    def __init__(self, cfg: dict):
        self.cfg = cfg["M3"]
        b = self.cfg["bounds"]
        self.lo = np.r_[np.full(N_NET, b["net_weights_and_biases"][0]), b["sigma_max_per_day"][0], b["beta_sigma"][0]]
        self.hi = np.r_[np.full(N_NET, b["net_weights_and_biases"][1]), b["sigma_max_per_day"][1], b["beta_sigma"][1]]
        self.theta = None
        self.scaler = None

    def fit(self, z0, exo, y, seed: int, max_iter: int | None = None) -> dict:
        """Scaler from the training inputs; PSO on training trajectories."""
        lo = np.r_[np.nanmin(y), exo.reshape(-1, 4).min(0)]
        hi = np.r_[np.nanmax(y), exo.reshape(-1, 4).max(0)]
        self.scaler = Scaler(lo, hi)

        def f(X):
            pred = simulate(X, z0, exo, self.scaler)
            with np.errstate(invalid="ignore", over="ignore"):
                return np.mean((pred - y[None]) ** 2, axis=(1, 2))

        p = self.cfg["pso"]
        res = pso.minimize(f, self.lo, self.hi, particles=p["particles"], max_iter=max_iter or p["max_iter"],
                           stall=p["stall_iter"], seed=seed)
        self.theta = res["x"]
        return res

    def predict(self, z0, exo) -> np.ndarray:
        return simulate(self.theta[None, :], z0, exo, self.scaler)[0]
