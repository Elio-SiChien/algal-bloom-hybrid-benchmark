"""Global-best particle swarm optimiser (M3; Mohanty-style settings: 30 particles, <=500 iterations,
stop after 20 iterations without improvement). Standard constriction coefficients (Clerc & Kennedy 2002)."""

from __future__ import annotations

import time

import numpy as np

W, C1, C2 = 0.7298, 1.49618, 1.49618


def minimize(f_batch, lo, hi, particles: int = 30, max_iter: int = 500, stall: int = 20, seed: int = 0) -> dict:
    """f_batch(X) -> losses for an (n_particles, d) array; non-finite losses count as +inf."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    rng = np.random.default_rng(seed)
    d = len(lo)
    vmax = 0.2 * (hi - lo)
    X = lo + rng.random((particles, d)) * (hi - lo)
    V = rng.uniform(-vmax, vmax, (particles, d))
    F = np.where(np.isfinite(f := f_batch(X)), f, np.inf)
    P, PF = X.copy(), F.copy()
    g = int(np.argmin(PF))
    best, hist, since, t0 = PF[g], [float(PF[g])], 0, time.perf_counter()
    it = 0
    for it in range(1, max_iter + 1):
        r1, r2 = rng.random((particles, d)), rng.random((particles, d))
        V = np.clip(W * V + C1 * r1 * (P - X) + C2 * r2 * (P[g] - X), -vmax, vmax)
        X = np.clip(X + V, lo, hi)
        F = f_batch(X)
        F = np.where(np.isfinite(F), F, np.inf)
        imp = F < PF
        P[imp], PF[imp] = X[imp], F[imp]
        g = int(np.argmin(PF))
        if PF[g] < best - 1e-12:
            best, since = PF[g], 0
        else:
            since += 1
        hist.append(float(best))
        if since >= stall:
            break
    return {"x": P[g].copy(), "fun": float(best), "iterations": it, "history": hist,
            "seconds": time.perf_counter() - t0}
