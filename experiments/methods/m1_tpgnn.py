"""M1: T-PGNN (Jung et al. 2026) adapted - Transformer encoder-decoder + physics-residual loss, recursive 7-day forecast.

Architecture (restored to the original, D-071 / audit A-01): the 7-day input window X is split into two sequences,
    X_E (encoder): daily shortwave (MJ/m2/day), precipitation          [original: Chl-a, DO, TOC, precipitation, irradiance]
    X_D (decoder): log10 algal density, water temperature, TN, TP      [original: water temperature, TN, TP, biomass]
each passed through its own linear embedding plus sinusoidal positional encoding, then through a Transformer
encoder and decoder (nn.Transformer, causal mask on the decoder); the last decoder step gives the next-day forecast.
Chl-a, DO and TOC are not in the locked v3 data (chlorophyll-a is unreliable at these stations, D-028; TOC is not
measured), so the encoder keeps the two weather drivers. Inputs are Min-Max scaled to [0, 1] with training minima and
maxima (as the original). Tuned (TPE): numbers of encoder and decoder layers, embedding dimension, attention heads,
feed-forward dimension, learning rates of network and physical parameters, network weight decay, dropout and physics
weight lambda; batch size fixed at 366 (original).
Training: one-step-ahead target y(u+1), loss = L_data + lambda * L_physics.
    L_data    = mean (Pred(u+1|u) - y(u+1))^2                            (log10)
    R_(u,s)   = [ln Pred(u+1|u) - ln Y_s] / (u+1-s) - r(theta, X_s),  s = u-6..u     (ln-rate form, protocol v1.0)
    r(theta,X)= A*exp(B*WT)*min(TN/(Ksn+TN), TP/(Ksp+TP))*I/(KI+I) - D   (day^-1, ln scale)
    L_physics = mean R^2
Network weights and theta are updated by two Adam optimisers (lr_net with weight decay, lr_theta). theta is kept inside
its bounds by a sigmoid reparameterisation (starts at the middle of the box, as the original). Early stop when the
training loss has not improved for `early_stop_patience` epochs; at most `max_epochs`. Targets may be interpolated
days (D-070, same as the other methods).
Forecast: frozen model applied 7 times; each step appends a row with the model's own prediction, the
air-temperature-regression water temperature (D-024), origin TN/TP, and ERA5 shortwave/precipitation of
the target day. No observed future value enters the window.
Device: CUDA GPU (user rule 2026-10-21: PyTorch always on the GPU). Without a usable GPU the run stops with an error
instead of silently falling back to the CPU; set environment variable M1_DEVICE=cpu only for tests.
GPU arithmetic is not bit-for-bit reproducible; the device is written to the run log.
"""

from __future__ import annotations

import math
import os

import numpy as np
import pandas as pd
import torch
from torch import nn

from common.qc import fill_short_gaps

FEATS = ("y", "wt", "tn", "tp", "sw_mj", "precip")
THETA = ("A_per_day", "B_per_degC", "Ksn_mgL", "Ksp_mgL", "KI_MJ_m2_day", "D_per_day")
ENC_IDX = [4, 5]      # sw_mj, precip  -> encoder sequence X_E
DEC_IDX = [0, 1, 2, 3]  # y, wt, tn, tp -> decoder sequence X_D
LN10 = math.log(10.0)
WIN = 7


def feature_table(daily: pd.DataFrame) -> np.ndarray:
    """(n_days, 6) raw features; y history filled for gaps <= 2 days (same rule as the origin list)."""
    y_hist, _ = fill_short_gaps(daily["y"])
    return np.stack([y_hist.to_numpy(float), daily["wt"].to_numpy(float), daily["tn"].to_numpy(float),
                     daily["tp"].to_numpy(float), daily["sw"].to_numpy(float) * 0.0864,
                     daily["precip"].to_numpy(float)], axis=1)


def training_pairs(daily: pd.DataFrame, day_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Windows X (n,7,6) and next-day targets (n,) whose days u-6..u+1 all lie in day_mask."""
    F = feature_table(daily)
    yv = np.where(daily["y_valid"].to_numpy(bool), daily["y"].to_numpy(float), np.nan)
    X, Y = [], []
    for u in range(WIN - 1, len(F) - 1):
        if not day_mask[u - WIN + 1:u + 2].all():
            continue
        w = F[u - WIN + 1:u + 1]
        if np.isfinite(w).all() and np.isfinite(yv[u + 1]) and np.isfinite(yv[u]):  # R3-03
            X.append(w)
            Y.append(yv[u + 1])
    return np.array(X), np.array(Y)


def sinusoidal_pe(n: int, d: int) -> torch.Tensor:
    pos = torch.arange(n, dtype=torch.float32)[:, None]
    div = torch.exp(torch.arange(0, d, 2, dtype=torch.float32) * (-math.log(10000.0) / d))
    pe = torch.zeros(n, d)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)[:, : d // 2]
    return pe[None]


class Net(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_enc_layers: int, n_dec_layers: int, dim_ff: int, dropout: float,
                 theta_bounds: dict):
        super().__init__()
        self.emb_e = nn.Linear(len(ENC_IDX), d_model)
        self.emb_d = nn.Linear(len(DEC_IDX), d_model)
        self.register_buffer("pe", sinusoidal_pe(WIN, d_model))
        self.tf = nn.Transformer(d_model=d_model, nhead=n_heads, num_encoder_layers=n_enc_layers,
                                 num_decoder_layers=n_dec_layers, dim_feedforward=dim_ff, dropout=dropout,
                                 batch_first=True)
        self.register_buffer("tgt_mask", nn.Transformer.generate_square_subsequent_mask(WIN))
        self.head = nn.Linear(d_model, 1)
        lo = torch.tensor([theta_bounds[k][0] for k in THETA], dtype=torch.float32)
        hi = torch.tensor([theta_bounds[k][1] for k in THETA], dtype=torch.float32)
        self.register_buffer("t_lo", lo)
        self.register_buffer("t_hi", hi)
        self.theta_raw = nn.Parameter(torch.zeros(len(THETA)))  # sigmoid(0) = middle of the box (original)

    def theta(self) -> torch.Tensor:
        return self.t_lo + (self.t_hi - self.t_lo) * torch.sigmoid(self.theta_raw)

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        src = self.emb_e(xs[..., ENC_IDX]) + self.pe
        tgt = self.emb_d(xs[..., DEC_IDX]) + self.pe
        h = self.tf(src, tgt, tgt_mask=self.tgt_mask)
        return self.head(h[:, -1, :]).squeeze(-1)


def physics_rate(theta: torch.Tensor, Xraw: torch.Tensor) -> torch.Tensor:
    """r(theta, X_s) for raw windows (n,7,6) -> (n,7)."""
    A, B, Ksn, Ksp, KI, D = theta
    wt, tn, tp, I = Xraw[..., 1], Xraw[..., 2], Xraw[..., 3], Xraw[..., 4]
    lim = torch.minimum(tn / (Ksn + tn), tp / (Ksp + tp))
    return A * torch.exp(B * wt) * lim * I / (KI + I) - D


class M1:
    def __init__(self, cfg: dict, hp: dict, seed: int = 0, threads: int = 4):
        self.cfg, self.hp, self.seed = cfg["M1"], hp, seed
        torch.set_num_threads(threads)
        want = os.environ.get("M1_DEVICE", "cuda")
        if want.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("T-PGNN must run on the NVIDIA GPU, but this PyTorch cannot see one "
                               f"(torch {torch.__version__}, built for CUDA {torch.version.cuda}). Run tools\\gpu_check.bat.")
        self.dev = torch.device(want)

    def fit(self, X: np.ndarray, Y: np.ndarray, max_epochs: int | None = None) -> dict:
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        flat = X.reshape(-1, X.shape[-1])  # Min-Max scaling to [0, 1] with training minima/maxima (original)
        self.mu, self.sd = flat.min(0), (flat.max(0) - flat.min(0)) + 1e-8
        hp = self.hp
        if self.dev.type == "cuda":
            torch.cuda.manual_seed_all(self.seed)
        self.net = Net(hp["d_model"], hp["n_heads"], hp["n_enc_layers"], hp["n_dec_layers"], hp["dim_ff"], hp["dropout"],
                       self.cfg["theta_bounds"]).to(self.dev)
        opt_w = torch.optim.Adam([p for n, p in self.net.named_parameters() if n != "theta_raw"], lr=hp["lr_net"],
                                 weight_decay=hp.get("weight_decay", 0.0))
        opt_t = torch.optim.Adam([self.net.theta_raw], lr=hp["lr_theta"])
        dev = self.dev
        Xr = torch.tensor(X, dtype=torch.float32, device=dev)
        Xs = (Xr - torch.tensor(self.mu, dtype=torch.float32, device=dev)) / torch.tensor(self.sd, dtype=torch.float32, device=dev)
        Yt = torch.tensor(Y, dtype=torch.float32, device=dev)
        lag = torch.arange(WIN, 0, -1, dtype=torch.float32, device=dev)  # u+1-s for s = u-6..u
        n, bs = len(Y), int(hp.get("batch_size", self.cfg.get("batch_size", 366)))
        best, since, hist = float("inf"), 0, []
        gen = torch.Generator().manual_seed(self.seed)
        for epoch in range(max_epochs or self.cfg["max_epochs"]):
            self.net.train()
            perm = torch.randperm(n, generator=gen)
            tot = 0.0
            for i in range(0, n, bs):
                b = perm[i:i + bs].to(dev)
                pred = self.net(Xs[b]) * self.sd[0] + self.mu[0]  # log10
                l_data = ((pred - Yt[b]) ** 2).mean()
                R = (pred[:, None] - Xr[b][..., 0]) * LN10 / lag - physics_rate(self.net.theta(), Xr[b])
                loss = l_data + hp["lambda_phys"] * (R ** 2).mean()
                opt_w.zero_grad(); opt_t.zero_grad()
                loss.backward()
                opt_w.step(); opt_t.step()
                tot += loss.item() * len(b)
            ep_loss = tot / n
            hist.append(ep_loss)
            if not math.isfinite(ep_loss):
                break
            if ep_loss < best - 1e-7:
                best, since = ep_loss, 0
            else:
                since += 1
            if since >= self.cfg["early_stop_patience"]:
                break
        return {"epochs": len(hist), "final_loss": hist[-1], "best_loss": best, "history": hist, "device": str(dev),
                "theta_start": dict(zip(THETA, ((self.net.t_lo + self.net.t_hi) / 2).cpu().numpy().round(5).tolist())),
                "theta": dict(zip(THETA, self.net.theta().detach().cpu().numpy().round(5).tolist()))}

    @torch.no_grad()
    def predict(self, daily: pd.DataFrame, origins: pd.Series, wt_reg: tuple) -> np.ndarray:
        self.net.eval()
        F = feature_table(daily)
        air, swm, pr = daily["air_temp_c"].to_numpy(float), daily["sw"].to_numpy(float) * 0.0864, daily["precip"].to_numpy(float)
        idx = daily.index.get_indexer(origins)
        win = np.stack([F[i - WIN + 1:i + 1] for i in idx]).astype(np.float32)
        out = np.empty((len(idx), 7))
        mu, sd = self.mu.astype(np.float32), self.sd.astype(np.float32)
        for h in range(1, 8):
            pred = self.net(torch.from_numpy((win - mu) / sd).to(self.dev)).cpu().numpy() * sd[0] + mu[0]
            out[:, h - 1] = pred
            new = np.stack([pred, wt_reg[0] + wt_reg[1] * air[idx + h], F[idx, 2], F[idx, 3], swm[idx + h], pr[idx + h]], 1)
            win = np.concatenate([win[:, 1:], new[:, None, :].astype(np.float32)], axis=1)
        return out
