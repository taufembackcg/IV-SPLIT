"""Data-generating processes for the Monte Carlo design (Section 7 of the paper).

Base DGP (no external instrument):

    x = g(w) + H + e_X
    u = lambda * H + e_Y
    y = beta * x + delta' w + u

with H, e_X, e_Y mutually independent. ``e_X`` is the exogenous variation the
split should recover; it is also returned as the "planted" oracle instrument.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

DISTRIBUTIONS = ("laplace", "exponential", "bimodal", "gaussian", "uniform")


def draw_source(rng: np.random.Generator, dist: str, n: int) -> np.ndarray:
    """Draw ``n`` i.i.d. draws with mean 0 and variance 1 from ``dist``.

    Excess kurtosis: laplace 3, exponential 6 (skewness 2), bimodal < 0,
    uniform -1.2, gaussian 0.
    """
    if dist == "laplace":
        return rng.laplace(0.0, 1.0 / np.sqrt(2.0), n)
    if dist == "exponential":
        return rng.exponential(1.0, n) - 1.0
    if dist == "bimodal":
        # equal mixture of N(-m, s^2) and N(m, s^2) with m^2 + s^2 = 1
        m, s = 0.9, np.sqrt(1.0 - 0.81)
        sign = rng.choice([-1.0, 1.0], n)
        return sign * m + s * rng.standard_normal(n)
    if dist == "uniform":
        return rng.uniform(-np.sqrt(3.0), np.sqrt(3.0), n)
    if dist == "gaussian":
        return rng.standard_normal(n)
    raise ValueError(f"unknown distribution {dist!r}; choose from {DISTRIBUTIONS}")


def g_function(w: np.ndarray, kind: str) -> np.ndarray:
    """Exogenous first-stage component g(w)."""
    if kind == "linear":
        return w @ np.linspace(1.0, 0.5, w.shape[1])
    if kind == "nonlinear":
        return np.log1p(np.exp(w[:, 0])) + np.sin(2.0 * w[:, 1 % w.shape[1]]) - np.log(2.0)
    raise ValueError(f"unknown g kind {kind!r}")


@dataclass
class DGPConfig:
    n: int = 2000
    beta: float = 1.0
    lam: float = 0.5
    dist_H: str = "laplace"
    dist_X: str = "laplace"
    dist_Y: str = "laplace"
    g: str = "linear"
    n_w: int = 2
    sd_H: float = 1.0
    sd_X: float = 1.0
    sd_Y: float = 1.0
    # Violations of H1 (scenario 4): correlation between H and e_X, and a
    # direct channel from e_X to u.
    rho: float = 0.0
    excl: float = 0.0
    # Scenario 2: nonlinear confounding with no linear part, u = lam*(v^2 - E v^2) + e_Y.
    nonlinear_confounding: bool = False
    # Auxiliary-variable heteroskedasticity of e_X in w[:, 0] (breaks the H <-> e_X
    # label symmetry; see Proposition 2 in the paper).
    hetero_X: float = 0.0
    delta: np.ndarray | None = field(default=None, repr=False)


@dataclass
class Sample:
    y: np.ndarray
    x: np.ndarray
    W: np.ndarray
    z_oracle: np.ndarray  # e_X, the planted exogenous variation
    u: np.ndarray
    H: np.ndarray
    beta: float

    def subset(self, idx: np.ndarray) -> "Sample":
        return Sample(self.y[idx], self.x[idx], self.W[idx], self.z_oracle[idx],
                      self.u[idx], self.H[idx], self.beta)


def simulate(cfg: DGPConfig, rng: np.random.Generator | int | None = None) -> Sample:
    rng = np.random.default_rng(rng)
    n = cfg.n
    W = rng.standard_normal((n, cfg.n_w))
    H = cfg.sd_H * draw_source(rng, cfg.dist_H, n)
    eX = draw_source(rng, cfg.dist_X, n)
    if cfg.hetero_X:
        eX = eX * np.exp(cfg.hetero_X * W[:, 0]) / np.exp(cfg.hetero_X ** 2)
    eX = cfg.sd_X * eX
    if cfg.rho:
        # Corr(H, e_X) = rho while keeping Var(e_X) roughly sd_X^2.
        eX = np.sqrt(1.0 - cfg.rho ** 2) * eX + cfg.rho * cfg.sd_X / cfg.sd_H * H
    eY = cfg.sd_Y * draw_source(rng, cfg.dist_Y, n)

    gw = g_function(W, cfg.g)
    if cfg.nonlinear_confounding:
        v = eX
        x = gw + v
        u = cfg.lam * (v ** 2 - np.mean(v ** 2)) + eY
    else:
        x = gw + H + eX
        u = cfg.lam * H + eY
    u = u + cfg.excl * eX

    delta = cfg.delta if cfg.delta is not None else np.full(cfg.n_w, 0.5)
    y = cfg.beta * x + W @ delta + u
    return Sample(y=y, x=x, W=W, z_oracle=eX, u=u, H=H, beta=cfg.beta)


# Named scenarios from Section 7.2 of the paper.
SCENARIOS: dict[str, dict] = {
    "s1_latent_nongaussian": dict(dist_H="laplace", dist_X="laplace", dist_Y="laplace"),
    "s1_latent_nonlinear_g": dict(dist_H="laplace", dist_X="laplace", dist_Y="laplace", g="nonlinear"),
    "s2_nonlinear_confounding": dict(nonlinear_confounding=True, dist_X="exponential", dist_Y="gaussian", lam=0.3),
    "s3_gaussian": dict(dist_H="gaussian", dist_X="gaussian", dist_Y="gaussian"),
    "s4_correlated_sources": dict(rho=0.3),
    "s4_exclusion_violation": dict(excl=0.3),
    "s5_placebo": dict(lam=0.0),
    "aux_heteroskedastic": dict(hetero_X=0.5),
}


def scenario(name: str, **overrides) -> DGPConfig:
    if name not in SCENARIOS:
        raise KeyError(f"unknown scenario {name!r}; choose from {sorted(SCENARIOS)}")
    params = {**SCENARIOS[name], **overrides}
    return DGPConfig(**params)


# ---------------------------------------------------------------------------
# Several endogenous regressors (used by the GAN-instrument experiment)
# ---------------------------------------------------------------------------

@dataclass
class MultiSample:
    y: np.ndarray
    X: np.ndarray          # (n, p) endogenous regressors
    W: np.ndarray
    Z_oracle: np.ndarray   # (n, p) planted exogenous variation e_Xj
    u: np.ndarray
    H: np.ndarray
    beta: np.ndarray


def simulate_multi(n: int = 2000, p: int = 2, lam: float = 1.0, beta=None,
                   loadings=None, dist: str = "laplace", g: str = "linear", n_w: int = 2,
                   rng: np.random.Generator | int | None = None) -> MultiSample:
    """p endogenous regressors sharing one latent confounder:

        x_j = g_j(w) + a_j H + e_Xj,   u = lam H + e_Y,   y = X beta + delta' w + u.
    """
    rng = np.random.default_rng(rng)
    beta = np.ones(p) if beta is None else np.asarray(beta, dtype=float)
    a = np.linspace(1.0, 0.5, p) if loadings is None else np.asarray(loadings, dtype=float)
    W = rng.standard_normal((n, n_w))
    H = draw_source(rng, dist, n)
    E = np.column_stack([draw_source(rng, dist, n) for _ in range(p)])
    G = np.column_stack([g_function(np.roll(W, j, axis=1), g) for j in range(p)])
    X = G + H[:, None] * a + E
    u = lam * H + draw_source(rng, dist, n)
    y = X @ beta + W @ np.full(n_w, 0.5) + u
    return MultiSample(y=y, X=X, W=W, Z_oracle=E, u=u, H=H, beta=beta)
