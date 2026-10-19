"""Neural split x = x_exo + r with adversarial (HSIC) independence (Section 4 of the paper).

The network f_theta maps (x_tilde[, W][, y_tilde]) to x_exo, r = x_tilde - x_exo,
and is trained on sample half A to minimise

    alpha * HSIC(x_exo, r) + gamma * HSIC(y_tilde - beta x_tilde, x_exo)
    + eta * [max(0, c_min - rho^2) + max(0, rho^2 - c_max)],

with rho = Corr(x_exo, x_tilde) and beta updated in alternation by the IV
formula with x_exo as instrument. The reconstruction term of the proposal is
zero by construction. Estimation and inference use half B only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from . import diagnostics as dg
from .estimators import Estimate, _const, design, linear_regression, partial_out, tsls

try:  # torch is optional; the rest of the package works without it
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None


@dataclass
class SplitConfig:
    hidden: int = 32
    depth: int = 2
    epochs: int = 150
    batch_size: int = 256
    lr: float = 3e-3
    weight_decay: float = 1e-4
    alpha: float = 1.0
    gamma: float = 1.0
    eta: float = 10.0
    relevance_floor: float = 0.1
    relevance_cap: float = 0.9
    use_w: bool = True
    use_outcome: bool = False
    partial: str = "linear"
    train_frac: float = 0.5
    seed: int = 0


def _require_torch():
    if torch is None:
        raise ImportError("the neural split needs PyTorch: pip install 'ivsplit[nn]'")


def _torch_hsic(a, b):
    """Biased HSIC with Gaussian kernels and (detached) median bandwidths."""
    def gram(v):
        d2 = torch.cdist(v, v) ** 2
        med = torch.median(d2.detach()[d2.detach() > 0]) if (d2 > 0).any() else torch.tensor(1.0)
        return torch.exp(-d2 / med.clamp_min(1e-6))

    n = a.shape[0]
    K, L = gram(a), gram(b)
    H = torch.eye(n) - 1.0 / n
    return torch.trace(K @ H @ L @ H) / n ** 2


def _corr2(a, b):
    a = a - a.mean()
    b = b - b.mean()
    return (a * b).mean() ** 2 / (a.pow(2).mean() * b.pow(2).mean() + 1e-12)


class NeuralSplit:
    """Learn the split on one sample, apply it to another."""

    def __init__(self, cfg: SplitConfig | None = None):
        _require_torch()
        self.cfg = cfg or SplitConfig()
        self.net = None
        self.beta_train = np.nan
        self.history: list[dict] = []

    def _features(self, xr, yr, W):
        cols = [xr]
        if self.cfg.use_w:
            cols.append(W)
        if self.cfg.use_outcome:
            cols.append(yr)
        F = np.column_stack(cols)
        return (F - self._mu) / self._sd

    def _residualise(self, y, x, W, rng):
        xr = partial_out(x, W, self.cfg.partial, rng=rng)
        yr = partial_out(y, W, self.cfg.partial, rng=rng)
        return xr, yr

    def fit(self, y, x, W) -> "NeuralSplit":
        cfg = self.cfg
        torch.manual_seed(cfg.seed)
        rng = np.random.default_rng(cfg.seed)
        xr, yr = self._residualise(y, x, W, rng)
        self._sx = xr.std()
        cols = [xr] + ([W] if cfg.use_w else []) + ([yr] if cfg.use_outcome else [])
        F = np.column_stack(cols)
        self._mu, self._sd = F.mean(axis=0), F.std(axis=0)
        Ft = torch.tensor((F - self._mu) / self._sd, dtype=torch.float32)
        xt = torch.tensor(xr / self._sx, dtype=torch.float32)
        yt = torch.tensor(yr / self._sx, dtype=torch.float32)

        layers, d_in = [], Ft.shape[1]
        for _ in range(cfg.depth):
            layers += [nn.Linear(d_in, cfg.hidden), nn.ELU()]
            d_in = cfg.hidden
        layers.append(nn.Linear(d_in, 1))
        self.net = nn.Sequential(*layers)
        opt = torch.optim.Adam(self.net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

        def iv_beta():
            with torch.no_grad():
                z = self.net(Ft).squeeze(1)
                z = z - z.mean()
                return float((z * yt).sum() / (z * xt).sum())

        beta = float((xt * yt).sum() / (xt * xt).sum())  # start from OLS
        n = len(xt)
        bs = min(cfg.batch_size, n)
        for epoch in range(cfg.epochs):
            perm = torch.randperm(n)
            for start in range(0, n - bs + 1, bs):
                idx = perm[start:start + bs]
                xe = self.net(Ft[idx]).squeeze(1)
                r = xt[idx] - xe
                eps = yt[idx] - beta * xt[idx]
                rel = _corr2(xe, xt[idx])
                loss = (cfg.alpha * _torch_hsic(xe[:, None], r[:, None])
                        + cfg.gamma * _torch_hsic(eps[:, None], xe[:, None])
                        + cfg.eta * (torch.relu(cfg.relevance_floor - rel)
                                     + torch.relu(rel - cfg.relevance_cap)))
                opt.zero_grad()
                loss.backward()
                opt.step()
            beta = iv_beta()
            if epoch % 10 == 0 or epoch == cfg.epochs - 1:
                self.history.append(dict(epoch=epoch, loss=float(loss.detach()), beta=beta, rel=float(rel.detach())))
        self.beta_train = beta
        return self

    def transform(self, y, x, W, rng=None):
        """Return (x_exo, r) in the units of the W-residualised x."""
        xr, yr = self._residualise(y, x, W, np.random.default_rng(rng))
        F = torch.tensor(self._features(xr, yr, W), dtype=torch.float32)
        with torch.no_grad():
            xe = self.net(F).squeeze(1).numpy().astype(float) * self._sx
        return xe, xr - xe


@dataclass
class SplitResult:
    iv: Estimate
    cf: Estimate
    x_exo: np.ndarray
    r: np.ndarray
    test_idx: np.ndarray
    diagnostics: dict = field(default_factory=dict)


def split_iv(y, x, W, cfg: SplitConfig | None = None, run_tests: bool = False,
             n_boot: int = 99, rng=None) -> SplitResult:
    """Full procedure: train on half A, estimate and test on half B.

    Returns the 2SLS estimate with x_exo as instrument, the control-function
    estimate with r as control, and (optionally) the Section 5 battery on B.
    """
    cfg = cfg or SplitConfig()
    rng = np.random.default_rng(cfg.seed if rng is None else rng)
    n = len(y)
    perm = rng.permutation(n)
    n_a = int(cfg.train_frac * n)
    A, B = perm[:n_a], perm[n_a:]

    model = NeuralSplit(cfg).fit(y[A], x[A], W[A])
    yB, xB, WB = y[B], x[B], W[B]
    x_exo, r = model.transform(yB, xB, WB, rng=rng)

    iv = tsls(yB, xB, WB, x_exo, name="Neural-split IV")
    X = design(_const(len(yB)), xB, WB, r)
    coef, cov, _ = linear_regression(yB, X)
    se = float(np.sqrt(cov[1, 1]))
    z = stats.norm.ppf(0.975)
    cf = Estimate("Neural-split CF", float(coef[1]), se, (coef[1] - z * se, coef[1] + z * se))

    F = dg.first_stage_F(xB, x_exo, WB)
    ar = dg.anderson_rubin_ci(yB, xB, WB, x_exo, center=iv.beta)
    xr = partial_out(xB, WB, cfg.partial, rng=rng)
    yr = partial_out(yB, WB, cfg.partial, rng=rng)
    D = dg.identification_strength(xr, yr, n_boot=n_boot, rng=rng)
    identified = bool(F.statistic > 10 and not ar["unbounded"] and D.pvalue < 0.05)
    for est in (iv, cf):
        est.extra.update(F=F.statistic, D=D.statistic, identified=identified,
                         beta_train=model.beta_train)
    iv.extra.update(ar_lo=ar["lo"], ar_hi=ar["hi"], ar_unbounded=ar["unbounded"])

    diag = {}
    if run_tests:
        diag = {
            "first_stage_F": F,
            "identification_D": D,
            "residual_hsic": dg.residual_independence(yB, xB, WB, x_exo, rng=rng),
            "residual_dcov": dg.residual_independence(yB, xB, WB, x_exo, method="dcov", rng=rng),
            "overid_powers": dg.overid_functions(yB, xB, WB, x_exo),
            "hausman": dg.hausman_ols_iv(yB, xB, WB, x_exo),
            "split_hsic": dg.split_independence(x_exo, r, rng=rng),
        }
    return SplitResult(iv, cf, x_exo, r, B, diag)
