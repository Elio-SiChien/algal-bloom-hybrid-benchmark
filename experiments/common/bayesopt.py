"""Small Gaussian-process Bayesian optimiser on a box (used by M2; D-017: <=200 evaluations).

Minimises f(x) for x in [lo, hi]^d. Parameters are searched on [0,1]^d internally.
Random Latin-hypercube start, then expected improvement maximised over random candidates
plus local perturbations of the incumbent. Non-finite objective values are replaced by the
worst finite value seen (so the GP stays well-defined) and recorded as failures.
"""

from __future__ import annotations

import time
import warnings

import numpy as np
from scipy.stats import norm
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel


def _lhs(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    u = (rng.random((n, d)) + np.arange(n)[:, None]) / n
    for j in range(d):
        u[:, j] = u[rng.permutation(n), j]
    return u


def minimize(f, lo, hi, n_evals: int = 200, n_init: int = 20, seed: int = 0, n_cand: int = 2000) -> dict:
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    d = len(lo)
    rng = np.random.default_rng(seed)
    X, Y, raw = [], [], []
    t0 = time.perf_counter()

    def ev(u):
        val = f(lo + u * (hi - lo))
        raw.append(val)
        X.append(u)
        finite = [v for v in raw if np.isfinite(v)]
        Y.append(val if np.isfinite(val) else (max(finite) if finite else 1e6))

    for u in _lhs(min(n_init, n_evals), d, rng):
        ev(u)
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(np.full(d, 0.3), (1e-2, 10.0), nu=2.5) + WhiteKernel(1e-4, (1e-8, 1e-1))
    while len(X) < n_evals:
        Xa, Ya = np.array(X), np.array(Y)
        mu_y, sd_y = Ya.mean(), Ya.std() + 1e-12
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            gp = GaussianProcessRegressor(kernel, normalize_y=False, n_restarts_optimizer=1,
                                          random_state=int(rng.integers(1 << 31)))
            gp.fit(Xa, (Ya - mu_y) / sd_y)
        best = Xa[np.argmin(Ya)]
        cand = np.vstack([rng.random((n_cand, d)),
                          np.clip(best + rng.normal(0, 0.05, (n_cand // 2, d)), 0, 1)])
        m, s = gp.predict(cand, return_std=True)
        ybest = (Ya.min() - mu_y) / sd_y
        z = (ybest - m) / np.maximum(s, 1e-12)
        ei = (ybest - m) * norm.cdf(z) + s * norm.pdf(z)
        ev(cand[np.argmax(ei)])
    Ya = np.array(Y)
    i = int(np.argmin(Ya))
    return {"x": lo + np.array(X[i]) * (hi - lo), "fun": float(raw[i]), "n_evals": len(X),
            "n_failed": int(sum(not np.isfinite(v) for v in raw)), "history": [float(v) for v in raw],
            "seconds": time.perf_counter() - t0}
