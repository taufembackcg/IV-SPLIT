"""Neural split x = x_exo + r with adversarial (HSIC) independence (Section 4 of the paper).

The network f_theta maps (x_tilde[, W][, y_tilde]) to x_exo, r = x_tilde - x_exo,
and is trained on sample half A to minimise

    alpha * HSIC(x_exo, r) + gamma * HSIC(y_tilde - beta x_tilde, x_exo)
    + eta * [max(0, c_min - rho^2) + max(0, rho^2 - c_max)],

with rho = Corr(x_exo, x_tilde) and beta updated in alternation by the IV
formula with x_exo as instrument. The reconstruction term of the proposal is
zero by construction. Estimation and inference use half B only.

Anchored split (Section 4.4 of the paper): beta is not updated but fixed at the
Cumulant-ICA estimate from half A (roots labelled by the sign of lambda or by
an auxiliary variable), and an explicit moment penalty
kappa * Corr(x_exo, y_tilde - beta_A x_tilde)^2 is added, so the network learns
instrument functions that are valid at the identified value.
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
    # Anchored split: None (unanchored), "sign" (L1) or "aux" (L2, auxiliary = W[:, 0]).
    anchor: str | None = None
    lam_sign: int = 1
    kappa: float = 10.0
    crossfit: bool = False

    def __post_init__(self):
        # With an anchor, valid relevant instruments generally require y_tilde as an
        # input: when E[u | x_tilde] is linear no function of (x_tilde, w) is both
        # relevant and orthogonal to u (see the paper, Section 4.3).
        if self.anchor and not self.use_outcome:
            self.use_outcome = True


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

    def fit(self, y, x, W, beta_fixed: float | None = None) -> "NeuralSplit":
        """Train on (y, x, W). With ``beta_fixed`` beta is held at the anchor."""
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

        # start from OLS, or hold beta at the anchor (scale-free: x and y share _sx)
        beta = float((xt * yt).sum() / (xt * xt).sum()) if beta_fixed is None else float(beta_fixed)
        kappa = cfg.kappa if beta_fixed is not None else 0.0
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
                                     + torch.relu(rel - cfg.relevance_cap))
                        + kappa * _corr2(xe, eps))
                opt.zero_grad()
                loss.backward()
                opt.step()
            if beta_fixed is None:
                beta = iv_beta()
            if epoch % 10 == 0 or epoch == cfg.epochs - 1:
                self.history.append(dict(epoch=epoch, loss=float(loss.detach()), beta=beta, rel=float(rel.detach())))
        self.beta_train = beta if beta_fixed is None else iv_beta()
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


def anchor_beta(y, x, W, cfg: SplitConfig, n_boot: int = 99, rng=None) -> Estimate:
    """Cumulant-ICA estimate used as the anchor, with a bootstrap standard error."""
    from .estimators import cumulant_ica
    aux = W[:, 0] if cfg.anchor == "aux" else None
    return cumulant_ica(y, x, W, label=cfg.anchor, lam_sign=cfg.lam_sign, aux=aux,
                        partial=cfg.partial, n_boot=n_boot, rng=rng)


def _fit_and_estimate(y, x, W, A, B, cfg, rng, n_boot, run_tests, label):
    """Train on A (anchored or not), estimate and test on B."""
    anchor = anchor_beta(y[A], x[A], W[A], cfg, n_boot, rng) if cfg.anchor else None
    beta_A = None if anchor is None else anchor.beta
    model = NeuralSplit(cfg).fit(y[A], x[A], W[A], beta_fixed=beta_A)
    yB, xB, WB = y[B], x[B], W[B]
    x_exo, r = model.transform(yB, xB, WB, rng=rng)

    iv = tsls(yB, xB, WB, x_exo, name=f"{label} IV")
    X = design(_const(len(yB)), xB, WB, r)
    coef, cov, _ = linear_regression(yB, X)
    se = float(np.sqrt(cov[1, 1]))
    z = stats.norm.ppf(0.975)
    cf = Estimate(f"{label} CF", float(coef[1]), se, (coef[1] - z * se, coef[1] + z * se))
    if cfg.use_outcome:
        # r = x_tilde - f(x_tilde, y_tilde) contains the outcome: not a control function
        cf = Estimate(f"{label} CF", np.nan, extra=dict(note="undefined when the network uses y"))
    if anchor is not None:
        # The instrument is built to be valid at beta_A, so the B-sample estimate
        # inherits the anchor's error; add its bootstrap variance to the B variance.
        se_tot = float(np.sqrt(iv.se ** 2 + anchor.se ** 2))
        iv.extra.update(se_naive=iv.se, se_anchor=anchor.se)
        iv.se, iv.ci = se_tot, (iv.beta - z * se_tot, iv.beta + z * se_tot)

    F = dg.first_stage_F(xB, x_exo, WB)
    ar = dg.anderson_rubin_ci(yB, xB, WB, x_exo, center=iv.beta)
    xr = partial_out(xB, WB, cfg.partial, rng=rng)
    yr = partial_out(yB, WB, cfg.partial, rng=rng)
    D = dg.identification_strength(xr, yr, n_boot=n_boot, rng=rng)
    identified = bool(F.statistic > 10 and not ar["unbounded"] and D.pvalue < 0.05)
    for est in (iv, cf):
        est.extra.update(F=F.statistic, D=D.statistic, identified=identified,
                         beta_train=model.beta_train, beta_anchor=beta_A)
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


def _average(a: Estimate, b: Estimate) -> Estimate:
    """Cross-fitted average of the two half-sample estimates.

    The halves share no observations, but each anchor uses the other half, so
    the two estimates are not independent; se = sqrt((v_a + v_b) / 4) ignores
    that dependence and is a lower bound when the anchors' errors are positively
    correlated with the estimates.
    """
    beta = 0.5 * (a.beta + b.beta)
    se = float(np.sqrt((a.se ** 2 + b.se ** 2) / 4))
    z = stats.norm.ppf(0.975)
    extra = dict(a.extra)
    extra["identified"] = bool(a.identified and b.identified)
    extra["beta_halves"] = (a.beta, b.beta)
    return Estimate(a.name + " (cross-fit)", beta, se, (beta - z * se, beta + z * se), extra)


def split_iv(y, x, W, cfg: SplitConfig | None = None, run_tests: bool = False,
             n_boot: int = 99, rng=None) -> SplitResult:
    """Full procedure: train on half A, estimate and test on half B.

    Returns the 2SLS estimate with x_exo as instrument, the control-function
    estimate with r as control, and (optionally) the Section 5 battery on B.
    With ``cfg.anchor`` the split is anchored at the Cumulant-ICA estimate from
    A; with ``cfg.crossfit`` the roles of A and B are also swapped and the two
    estimates averaged (diagnostics and x_exo refer to the first fold).
    """
    cfg = cfg or SplitConfig()
    rng = np.random.default_rng(cfg.seed if rng is None else rng)
    n = len(y)
    perm = rng.permutation(n)
    n_a = int(cfg.train_frac * n)
    A, B = perm[:n_a], perm[n_a:]
    label = "Anchored split" if cfg.anchor else "Neural-split"

    res = _fit_and_estimate(y, x, W, A, B, cfg, rng, n_boot, run_tests, label)
    if cfg.crossfit:
        other = _fit_and_estimate(y, x, W, B, A, cfg, rng, n_boot, False, label)
        res.iv = _average(res.iv, other.iv)
        res.cf = _average(res.cf, other.cf)
    return res
