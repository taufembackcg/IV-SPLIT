"""The GAN-instrument proposal (Section 2 of the paper), implemented to test it.

Proposal: in a linear regression with p possibly endogenous regressors
x_1, ..., x_p, generate two candidate instruments per endogenous regressor
(z_j1, z_j2 for x_j) with a GAN trained so that, for every j,

* the weak-instrument test passes: the first-stage F of x_j on (z_j1, z_j2)
  alone, controlling for W, exceeds 10; and
* the Sargan test passes in the restricted equation of x_j:
  y = beta_j x_j + w'delta + e instrumented by (z_j1, z_j2) only (1 d.o.f.);

then filter the excess candidates with Hausman tests.

Implementation:

* Generator G_theta(X, W, noise) -> Z in R^{n x 2p}, columns ordered
  (z_11, z_12, z_21, z_22, ...).
* Generator loss: sum_j [relu(log F* - log F_j) + relu(S_j - c) / c]
  + zeta * critic term, with F_j and S_j the per-variable statistics above
  written as differentiable functions of Z (c = chi2_1 critical value).
* Critic (the adversarial "discriminator") D_phi(Z) maximises
  Corr(D(Z), eps_hat)^2 for the full-system 2SLS residual, i.e. searches for a
  moment violation; the generator minimises it.
* Filtering: while some x_j keeps two candidates and the largest Hausman
  statistic (full instrument set vs. set without candidate c, all p
  coefficients) is significant, drop that candidate. Each x_j keeps at least
  one instrument. ``filter="ctest"`` uses the difference-in-Sargan (C)
  statistic, the correct test for a subset of instruments.
* ``sargan_eq="partialled"`` replaces the restricted equation by
  y - sum_{k != j} beta_k x_k = beta_j x_j + ..., with beta_k from the full
  system, so that the other endogenous regressors do not enter the error.

Exogeneity, Cov(Z, u) = 0, appears nowhere in this objective, and that is the point
of the experiment.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from . import diagnostics as dg
from .estimators import Estimate, tsls_fit

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None

Z975 = stats.norm.ppf(0.975)


@dataclass
class GanIVConfig:
    per_endog: int = 2
    noise_dim: int = 2
    hidden: int = 32
    epochs: int = 300
    lr: float = 3e-3
    F_target: float = 100.0
    zeta: float = 1.0
    alpha: float = 0.05
    filter: str = "hausman"        # "hausman", "ctest" or "none"
    sargan_eq: str = "restricted"  # or "partialled"
    use_y: bool = False            # the proposal generates from the regressors only
    sample_split: bool = False
    seed: int = 0


def _require_torch():
    if torch is None:
        raise ImportError("the GAN-IV experiment needs PyTorch: pip install 'ivsplit[nn]'")


def _as_2d(a):
    a = np.asarray(a, dtype=float)
    return a[:, None] if a.ndim == 1 else a


def _corr2(a, b):
    a = a - a.mean()
    b = b - b.mean()
    return (a * b).mean() ** 2 / (a.pow(2).mean() * b.pow(2).mean() + 1e-12)


def _tests_torch(Z, xt, yt, ridge=1e-4):
    """Differentiable first-stage F of x on Z and Sargan statistic of y = b x + e
    instrumented by Z (all variables already W-residualised)."""
    n, m = Z.shape
    Z = Z - Z.mean(0)
    G = Z.T @ Z + ridge * n * torch.eye(m, dtype=Z.dtype)
    pi = torch.linalg.solve(G, Z.T @ xt)
    xhat = Z @ pi
    r2 = (xhat * xhat).sum() / (xt * xt).sum()
    F = (r2 / m) / ((1 - r2).clamp_min(1e-6) / (n - m - 1))
    beta = (xhat * yt).sum() / (xhat * xt).sum()
    eps = yt - beta * xt
    g = torch.linalg.solve(G, Z.T @ eps)
    S = n * (eps @ (Z @ g)) / (eps * eps).sum()
    return F, S, eps, beta


def _system_torch(Z, Xt, yt, ridge=1e-4):
    """Full-system 2SLS coefficients and residual (W-residualised data)."""
    n, m = Z.shape
    Z = Z - Z.mean(0)
    G = Z.T @ Z + ridge * n * torch.eye(m, dtype=Z.dtype)
    Xhat = Z @ torch.linalg.solve(G, Z.T @ Xt)
    beta = torch.linalg.solve(Xhat.T @ Xt, Xhat.T @ yt)
    return beta, yt - Xt @ beta


def _residualise_np(v, W):
    X = np.column_stack([np.ones(len(v)), W])
    return v - X @ np.linalg.lstsq(X, v, rcond=None)[0]


def groups_of(p: int, per: int = 2) -> np.ndarray:
    """Endogenous regressor that each candidate column belongs to."""
    return np.repeat(np.arange(p), per)


class GanIV:
    def __init__(self, cfg: GanIVConfig | None = None):
        _require_torch()
        self.cfg = cfg or GanIVConfig()

    def _inputs(self, X, W, y, rng):
        cols = [X, W] + ([y] if self.cfg.use_y else [])
        F = (np.column_stack(cols) - self._mu) / self._sd
        noise = rng.standard_normal((len(X), self.cfg.noise_dim))
        return torch.tensor(np.column_stack([F, noise]), dtype=torch.float32)

    def fit(self, y, X, W) -> "GanIV":
        cfg = self.cfg
        X = _as_2d(X)
        torch.manual_seed(cfg.seed)
        rng = np.random.default_rng(cfg.seed)
        cols = [X, W] + ([y] if cfg.use_y else [])
        F = np.column_stack(cols)
        self._mu, self._sd = F.mean(0), F.std(0)
        self.p = X.shape[1]
        self.m = cfg.per_endog * self.p
        grp = groups_of(self.p, cfg.per_endog)
        inp = self._inputs(X, W, y, rng)
        Xr = np.column_stack([_residualise_np(X[:, j], W) for j in range(self.p)])
        yr = _residualise_np(y, W)
        sy = yr.std()
        Xt = torch.tensor(Xr / Xr.std(0), dtype=torch.float32)
        yt = torch.tensor(yr / sy, dtype=torch.float32)
        Wq = torch.tensor(np.linalg.qr(np.column_stack([np.ones(len(y)), W]))[0], dtype=torch.float32)

        self.gen = nn.Sequential(nn.Linear(inp.shape[1], cfg.hidden), nn.ELU(),
                                 nn.Linear(cfg.hidden, cfg.hidden), nn.ELU(),
                                 nn.Linear(cfg.hidden, self.m))
        self.critic = nn.Sequential(nn.Linear(self.m, cfg.hidden), nn.ELU(),
                                    nn.Linear(cfg.hidden, 1))
        og = torch.optim.Adam(self.gen.parameters(), lr=cfg.lr)
        oc = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr)
        crit = float(stats.chi2.ppf(1 - cfg.alpha, cfg.per_endog - 1))
        logF_target = float(np.log(cfg.F_target))
        self.history = []
        for epoch in range(cfg.epochs):
            Z = self.gen(inp)
            Z = Z - Wq @ (Wq.T @ Z)  # partial W out of the candidates
            beta_sys, eps_sys = _system_torch(Z, Xt, yt)
            g_loss = 0.0
            Fs, Ss = [], []
            for j in range(self.p):
                target = yt
                if cfg.sargan_eq == "partialled" and self.p > 1:
                    others = [k for k in range(self.p) if k != j]
                    target = yt - Xt[:, others] @ beta_sys.detach()[others]
                Fj, Sj, _, _ = _tests_torch(Z[:, grp == j], Xt[:, j], target)
                g_loss = (g_loss + torch.relu(logF_target - torch.log(Fj.clamp_min(1e-6)))
                          + torch.relu(Sj - crit) / crit)
                Fs.append(float(Fj.detach()))
                Ss.append(float(Sj.detach()))
            # critic step: find a function of Z that predicts the system residual
            c_loss = -_corr2(self.critic(Z.detach()).squeeze(1), eps_sys.detach())
            oc.zero_grad()
            c_loss.backward()
            oc.step()
            # generator step: pass the per-variable tests and fool the critic
            g_loss = g_loss + cfg.zeta * _corr2(self.critic(Z).squeeze(1), eps_sys)
            og.zero_grad()
            g_loss.backward()
            og.step()
            if epoch % 25 == 0 or epoch == cfg.epochs - 1:
                self.history.append(dict(epoch=epoch, F=Fs, S=Ss, critic=float(-c_loss.detach())))
        return self

    def transform(self, y, X, W, rng=None):
        rng = np.random.default_rng(self.cfg.seed + 1 if rng is None else rng)
        with torch.no_grad():
            return self.gen(self._inputs(_as_2d(X), W, y, rng)).numpy().astype(float)


# ---------------------------------------------------------------------------
# Tests and filtering on the generated candidates
# ---------------------------------------------------------------------------

def _system(y, X, W, Z):
    fit = tsls_fit(y, X, W, Z)
    p = X.shape[1]
    return fit.coef[1:1 + p], fit.cov[1:1 + p, 1:1 + p]


def _hausman_drop_stats(y, X, W, Z, keep, droppable):
    b_full, V_full = _system(y, X, W, Z[:, keep])
    out = []
    for c in keep:
        if c not in droppable:
            out.append(-np.inf)
            continue
        b_sub, V_sub = _system(y, X, W, Z[:, [i for i in keep if i != c]])
        d = b_sub - b_full
        out.append(float(d @ np.linalg.pinv(V_sub - V_full) @ d))
    return np.array(out)


def _ctest_drop_stats(y, X, W, Z, keep, droppable):
    out = []
    for c in keep:
        if c not in droppable:
            out.append(-np.inf)
            continue
        base = [i for i in keep if i != c]
        out.append(dg.c_test(y, X, W, Z[:, base], Z[:, [c]]).statistic)
    return np.array(out)


def filter_candidates(y, X, W, Z, groups, method: str = "hausman", alpha: float = 0.05):
    """Sequentially drop the most suspicious candidate while its statistic is
    significant, keeping at least one instrument per endogenous regressor."""
    X = _as_2d(X)
    groups = np.asarray(groups)
    keep = list(range(Z.shape[1]))
    if method == "none":
        return keep
    df = X.shape[1] if method == "hausman" else 1
    crit = stats.chi2.ppf(1 - alpha, df)
    fn = _hausman_drop_stats if method == "hausman" else _ctest_drop_stats
    while True:
        counts = {g: sum(groups[i] == g for i in keep) for g in set(groups[keep])}
        droppable = {i for i in keep if counts[groups[i]] > 1}
        if not droppable:
            break
        st = fn(y, X, W, Z, keep, droppable)
        k = int(np.argmax(st))
        if not np.isfinite(st[k]) or st[k] < crit:
            break
        keep.pop(k)
    return keep


def per_variable_tests(y, X, W, Z, groups, keep, beta_sys=None, sargan_eq="restricted"):
    """Weak-instrument F and Sargan test for each endogenous regressor, each using
    only that regressor's own kept candidates.

    F_j: robust first-stage F of x_j on its own instruments, controlling for W.
    Sargan_j: Hansen/Sargan J of the restricted equation y = b x_j + w'd + e
    (or, with ``sargan_eq="partialled"``, of y - sum_{k != j} beta_k x_k) with
    the same instruments; undefined (NaN) when only one instrument is left.
    """
    X = _as_2d(X)
    groups = np.asarray(groups)
    out = []
    for j in range(X.shape[1]):
        cols = [c for c in keep if groups[c] == j]
        Zj = Z[:, cols]
        target = y
        if sargan_eq == "partialled" and X.shape[1] > 1:
            others = [k for k in range(X.shape[1]) if k != j]
            target = y - X[:, others] @ beta_sys[others]
        F = dg.first_stage_F(X[:, j], Zj, W).statistic
        J = dg.gmm_j(target, X[:, j], W, Zj)
        out.append(dict(F=float(F), sargan_p=float(J.pvalue), n_kept=len(cols)))
    return out


def gan_iv(y, X, W, cfg: GanIVConfig | None = None, rng=None) -> Estimate:
    """Run the full proposal; the returned Estimate is for the first endogenous
    coefficient, with all coefficients and the per-variable tests in ``extra``."""
    cfg = cfg or GanIVConfig()
    X = _as_2d(X)
    p = X.shape[1]
    rng = np.random.default_rng(cfg.seed if rng is None else rng)
    n = len(y)
    if cfg.sample_split:
        perm = rng.permutation(n)
        A, B = perm[: n // 2], perm[n // 2:]
    else:
        A = B = np.arange(n)
    model = GanIV(cfg).fit(y[A], X[A], W[A])
    yB, XB, WB = y[B], X[B], W[B]
    Z = model.transform(yB, XB, WB, rng)
    groups = groups_of(p, cfg.per_endog)
    keep = filter_candidates(yB, XB, WB, Z, groups, cfg.filter, cfg.alpha)
    b, V = _system(yB, XB, WB, Z[:, keep])
    tests = per_variable_tests(yB, XB, WB, Z, groups, keep, b, cfg.sargan_eq)
    passed = all(t["F"] > 10 and (np.isnan(t["sargan_p"]) or t["sargan_p"] > cfg.alpha)
                 for t in tests)
    sargan_all = dg.gmm_j(yB, XB, WB, Z[:, keep])
    # fitted first stage of x_1, to measure (outside the estimator) its correlation with u
    Xf = np.column_stack([np.ones(len(yB)), Z[:, keep], WB])
    xhat = Xf @ np.linalg.lstsq(Xf, XB[:, 0], rcond=None)[0]
    se = float(np.sqrt(V[0, 0]))
    sp = [t["sargan_p"] for t in tests if np.isfinite(t["sargan_p"])]
    return Estimate(
        f"GAN-IV ({cfg.filter})", float(b[0]), se, (b[0] - Z975 * se, b[0] + Z975 * se),
        extra=dict(
            beta_all=b.tolist(), se_all=np.sqrt(np.diag(V)).tolist(), per_variable=tests,
            F=min(t["F"] for t in tests),
            sargan_p=min(sp) if sp else np.nan,
            sargan_system_p=float(sargan_all.pvalue),
            n_kept=len(keep), n_candidates=Z.shape[1], identified=passed,
            eval_idx=B, xhat=_residualise_np(xhat, WB), history=model.history,
        ))
