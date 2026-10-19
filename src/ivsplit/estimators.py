"""Estimators compared in the Monte Carlo (Section 7.4 of the paper).

All estimators take the outcome ``y``, the endogenous regressor ``x`` and the
exogenous covariates ``W`` (an intercept is always added) and return an
:class:`Estimate` for the coefficient on ``x``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats

Z975 = stats.norm.ppf(0.975)


@dataclass
class Estimate:
    name: str
    beta: float
    se: float = np.nan
    ci: tuple[float, float] = (np.nan, np.nan)
    extra: dict = field(default_factory=dict)

    @property
    def identified(self) -> bool:
        return bool(self.extra.get("identified", True))


def _const(n: int) -> np.ndarray:
    return np.ones((n, 1))


def design(*blocks) -> np.ndarray:
    cols = [np.asarray(b, dtype=float).reshape(len(b), -1) for b in blocks if b is not None]
    return np.column_stack(cols)


def _wald_ci(b, se):
    return (b - Z975 * se, b + Z975 * se)


def linear_regression(y: np.ndarray, X: np.ndarray):
    """OLS with HC1 covariance. Returns (coef, cov, resid)."""
    n, k = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    coef = XtX_inv @ X.T @ y
    e = y - X @ coef
    meat = (X * e[:, None] ** 2).T @ X
    cov = XtX_inv @ meat @ XtX_inv * n / max(n - k, 1)
    return coef, cov, e


def residualize(v: np.ndarray, W: np.ndarray | None) -> np.ndarray:
    """Residual of ``v`` after linear projection on [1, W]."""
    n = len(v)
    X = _const(n) if W is None else design(_const(n), W)
    coef, *_ = np.linalg.lstsq(X, v, rcond=None)
    return v - X @ coef


def partial_out(v: np.ndarray, W: np.ndarray, method: str = "linear",
                n_folds: int = 2, rng=None) -> np.ndarray:
    """Residual of ``v`` on ``W``: linear, polynomial (degree 3) or cross-fitted boosting."""
    if method == "linear":
        return residualize(v, W)
    if method == "poly":
        from sklearn.preprocessing import PolynomialFeatures
        P = PolynomialFeatures(degree=3, include_bias=False).fit_transform(W)
        return residualize(v, np.column_stack([P, np.sin(2 * W), np.cos(2 * W), np.logaddexp(0, W)]))
    if method == "gbm":
        from sklearn.ensemble import HistGradientBoostingRegressor
        rng = np.random.default_rng(rng)
        folds = rng.integers(0, n_folds, len(v))
        out = np.empty_like(v, dtype=float)
        for k in range(n_folds):
            tr, te = folds != k, folds == k
            m = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05).fit(W[tr], v[tr])
            out[te] = v[te] - m.predict(W[te])
        return out - out.mean()
    raise ValueError(f"unknown partialling-out method {method!r}")


# ---------------------------------------------------------------------------
# OLS and 2SLS
# ---------------------------------------------------------------------------

def ols(y, x, W=None) -> Estimate:
    X = design(_const(len(y)), x, W)
    coef, cov, _ = linear_regression(y, X)
    b, se = coef[1], np.sqrt(cov[1, 1])
    return Estimate("OLS", b, se, _wald_ci(b, se))


@dataclass
class TSLSFit:
    coef: np.ndarray
    cov: np.ndarray
    resid: np.ndarray
    X: np.ndarray
    Z: np.ndarray


def tsls_fit(y, x, W, Z) -> TSLSFit:
    """2SLS of y on [1, x, W] with instruments [1, Z, W]; HC0 sandwich covariance."""
    n = len(y)
    X = design(_const(n), x, W)
    Zf = design(_const(n), Z, W)
    ZtZ_inv = np.linalg.pinv(Zf.T @ Zf)
    Xhat = Zf @ (ZtZ_inv @ (Zf.T @ X))
    A_inv = np.linalg.pinv(Xhat.T @ X)
    coef = A_inv @ Xhat.T @ y
    e = y - X @ coef
    meat = (Xhat * e[:, None] ** 2).T @ Xhat
    cov = A_inv @ meat @ A_inv.T
    return TSLSFit(coef, cov, e, X, Zf)


def tsls(y, x, W, Z, name: str = "2SLS") -> Estimate:
    fit = tsls_fit(y, x, W, Z)
    b, se = fit.coef[1], np.sqrt(fit.cov[1, 1])
    return Estimate(name, b, se, _wald_ci(b, se))


def oracle_tsls(y, x, W, z_oracle) -> Estimate:
    """Ceiling: 2SLS with the planted exogenous variation e_X as instrument."""
    return tsls(y, x, W, z_oracle, name="2SLS-oracle")


# ---------------------------------------------------------------------------
# Internal-instrument competitors
# ---------------------------------------------------------------------------

def lewbel(y, x, W) -> Estimate:
    """Lewbel (2012): instruments (W - mean W) * e_hat, e_hat = residual of x on [1, W]."""
    e1 = residualize(x, W)
    Z = (W - W.mean(axis=0)) * e1[:, None]
    est = tsls(y, x, W, Z, name="Lewbel")
    return est


def copula_cf(y, x, W, n_boot: int = 0, rng=None) -> Estimate:
    """Park & Gupta (2012) Gaussian-copula control function.

    Adds x* = Phi^{-1}(F_hat(x)) as a regressor. Standard errors from the plain
    regression ignore the generated regressor; use ``n_boot`` > 0 for a
    bootstrap standard error.
    """
    def fit(yy, xx, WW):
        r = stats.rankdata(xx) / (len(xx) + 1.0)
        xstar = stats.norm.ppf(r)
        X = design(_const(len(yy)), xx, WW, xstar)
        coef, cov, _ = linear_regression(yy, X)
        return coef[1], np.sqrt(cov[1, 1])

    b, se = fit(y, x, W)
    if n_boot:
        rng = np.random.default_rng(rng)
        draws = []
        for _ in range(n_boot):
            i = rng.integers(0, len(y), len(y))
            draws.append(fit(y[i], x[i], W[i])[0])
        se = float(np.std(draws, ddof=1))
    return Estimate("Copula-CF", b, se, _wald_ci(b, se))


def power_basis(v: np.ndarray, J: int) -> np.ndarray:
    """Powers v^2, ..., v^J of the standardised regressor (no linear term)."""
    s = (v - v.mean()) / v.std()
    return np.column_stack([s ** j for j in range(2, J + 1)])


def sp_projection(y, x, W, J: int = 4) -> Estimate:
    """Simplified structural-error-projection estimator (after Dong, Gao, Linton & Peng).

    The confounding E[u | x] is approximated by a sieve without a linear term
    (powers 2..J of the W-residualised regressor); LASSO selects the terms S,
    and the internal instrument is z = x - P_S x. Equivalent to post-LASSO OLS
    of y on [1, x, W, phi_S]. Identification requires that the confounding has
    no linear component in x. The original paper uses a cosine basis; powers
    are used here because the rank-based cosine basis also spans x itself.
    """
    from sklearn.linear_model import LassoCV

    xr = residualize(x, W)
    yr = residualize(y, W)
    Phi = power_basis(xr, J)
    # Frisch-Waugh: partial x out of both sides so LASSO selects only the
    # nonlinear part of the confounding while x stays unpenalised.
    Phi_lin = np.column_stack([residualize(Phi[:, j], xr[:, None]) for j in range(Phi.shape[1])])
    yrr = residualize(yr, xr[:, None])
    scale = Phi_lin.std(axis=0)
    lasso = LassoCV(cv=5, random_state=0).fit(Phi_lin / scale, yrr)
    S = np.flatnonzero(np.abs(lasso.coef_) > 1e-10)
    z = xr if S.size == 0 else residualize(xr, Phi[:, S])
    est = tsls(y, x, W, z, name="SP")
    est.extra.update(selected=(S + 2).tolist(), first_stage_r2=float(np.corrcoef(z, xr)[0, 1] ** 2))
    return est


# ---------------------------------------------------------------------------
# Linear identification layer: over-complete ICA through fourth-order cumulants
# ---------------------------------------------------------------------------

def joint_cumulants4(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Fourth-order joint cumulants c_{4-k,k} = cum(a,..,a,b,..,b), k = 0..4."""
    a = a - a.mean()
    b = b - b.mean()
    saa, sab, sbb = np.mean(a * a), np.mean(a * b), np.mean(b * b)
    m = [np.mean(a ** (4 - k) * b ** k) for k in range(5)]
    return np.array([
        m[0] - 3 * saa ** 2,
        m[1] - 3 * saa * sab,
        m[2] - saa * sbb - 2 * sab ** 2,
        m[3] - 3 * sbb * sab,
        m[4] - 3 * sbb ** 2,
    ])


def joint_cumulants3(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a - a.mean()
    b = b - b.mean()
    return np.array([np.mean(a ** (3 - k) * b ** k) for k in range(4)])


def cumulant_roots(xr: np.ndarray, yr: np.ndarray, use_third: bool = False):
    """Candidate slopes {beta, beta + lambda} from the Prony recursion of the cumulants.

    With x = H + e_X and y = (beta+lam) H + beta e_X + e_Y, the cumulants
    c_k = c_{4-k,k} for k = 0..3 are a two-term exponential sum in the slopes
    s_H = beta + lam and s_X = beta, so c_{k+2} = p c_{k+1} + q c_k with
    p = s_H + s_X and q = -s_H s_X. Returns (roots, det, discriminant).
    """
    c = joint_cumulants4(xr, yr)
    A = [[c[1], c[0]], [c[2], c[1]]]
    rhs = [c[2], c[3]]
    if use_third:
        t = joint_cumulants3(xr, yr)
        A.append([t[1], t[0]])
        rhs.append(t[2])
    A, rhs = np.array(A), np.array(rhs)
    p, q = np.linalg.lstsq(A, rhs, rcond=None)[0]
    disc = p ** 2 + 4 * q
    root = np.sqrt(max(disc, 0.0))
    roots = np.array([(p - root) / 2, (p + root) / 2])
    det = c[1] ** 2 - c[2] * c[0]
    return roots, float(det), float(disc)


def heteroskedasticity_slope(xr, yr, aux, n_bins: int = 5) -> float:
    """Slope of Cov(x,y | bin) on Var(x | bin) across quantile bins of ``aux``.

    If Var(e_X) moves with ``aux`` while the law of H does not, the slope
    equals beta (Rigobon-type identification); used only to label the roots.
    """
    edges = np.quantile(aux, np.linspace(0, 1, n_bins + 1))
    bins = np.clip(np.searchsorted(edges, aux, side="right") - 1, 0, n_bins - 1)
    v, c = [], []
    for k in range(n_bins):
        m = bins == k
        if m.sum() < 10:
            continue
        v.append(np.var(xr[m]))
        c.append(np.cov(xr[m], yr[m])[0, 1])
    v, c = np.array(v), np.array(c)
    return float(np.polyfit(v, c, 1)[0])


def label_roots(roots, label: str, lam_sign: int = 1, xr=None, yr=None, aux=None) -> float:
    """Pick beta from {beta, beta + lambda} (the H <-> e_X label symmetry)."""
    if label == "sign":
        return float(roots[0] if lam_sign > 0 else roots[1])
    if label == "aux":
        slope = heteroskedasticity_slope(xr, yr, aux)
        return float(roots[np.argmin(np.abs(roots - slope))])
    raise ValueError(f"unknown labelling rule {label!r}")


def cumulant_ica(y, x, W, label: str = "sign", lam_sign: int = 1, aux=None,
                 partial: str = "linear", use_third: bool = False,
                 n_boot: int = 199, rng=None) -> Estimate:
    """Linear over-complete ICA estimator of beta through fourth-order cumulants.

    Reports the identification-strength statistic D (Wald test of det = 0,
    bootstrap variance); D below the chi-square(1) 95% critical value flags
    weak or failed identification.
    """
    rng = np.random.default_rng(rng)
    xr = partial_out(x, W, partial, rng=rng)
    yr = partial_out(y, W, partial, rng=rng)

    def one(idx):
        roots, det, disc = cumulant_roots(xr[idx], yr[idx], use_third)
        a = None if aux is None else aux[idx]
        b = label_roots(roots, label, lam_sign, xr[idx], yr[idx], a)
        return b, det, roots

    full = np.arange(len(y))
    b, det, roots = one(full)
    se, ci, D = np.nan, (np.nan, np.nan), np.nan
    if n_boot:
        draws, dets = [], []
        for _ in range(n_boot):
            i = rng.integers(0, len(y), len(y))
            bb, dd, _ = one(i)
            draws.append(bb)
            dets.append(dd)
        draws = np.array(draws)
        se = float(np.std(draws, ddof=1))
        ci = (float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975)))
        D = float(det ** 2 / np.var(dets, ddof=1))
    identified = bool(np.isnan(D) or D > stats.chi2.ppf(0.95, 1))
    return Estimate("Cumulant-ICA", b, se, ci,
                    extra=dict(roots=roots.tolist(), det=det, D=D, identified=identified))
