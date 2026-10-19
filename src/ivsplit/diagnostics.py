"""Test battery (Section 5 of the paper).

Pre-estimation (non-Gaussianity, identification strength, heteroskedasticity),
relevance (robust first-stage F, Anderson-Rubin), exogeneity of the learned
instrument (HSIC/dCov against the structural residual, over-identification with
functions of the instrument, C test, Hausman, estimator agreement, negative
controls), sensitivity (Conley-Hansen-Rossi union of intervals) and
multiple-testing corrections.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from .estimators import (_const, cumulant_roots, design, linear_regression,
                         partial_out, tsls, tsls_fit)
from .independence import permutation_test


@dataclass
class TestResult:
    name: str
    statistic: float
    pvalue: float
    df: float = np.nan
    extra: dict = field(default_factory=dict)

    def reject(self, alpha: float = 0.05) -> bool:
        return bool(self.pvalue < alpha)


# ---------------------------------------------------------------------------
# 5.1 Pre-estimation
# ---------------------------------------------------------------------------

def nongaussianity(v: np.ndarray) -> TestResult:
    """Skewness, excess kurtosis and the Jarque-Bera test (H2)."""
    jb = stats.jarque_bera(v)
    return TestResult("Jarque-Bera", float(jb.statistic), float(jb.pvalue), 2,
                      extra=dict(skew=float(stats.skew(v)), exkurt=float(stats.kurtosis(v))))


def identification_strength(xr: np.ndarray, yr: np.ndarray, n_boot: int = 199,
                            rng=None) -> TestResult:
    """Identification-strength statistic D (analogue of Delbianco's D).

    Wald statistic for H0: det = 0, where det = c31^2 - c22 c40 equals
    -kappa_H kappa_X lambda^2 in the base model. det = 0 when either latent
    source is mesokurtic (e.g. Gaussian) or when there is no confounding; in
    both cases the cumulant equations carry no information about the split.
    """
    rng = np.random.default_rng(rng)
    _, det, _ = cumulant_roots(xr, yr)
    n = len(xr)
    dets = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        dets.append(cumulant_roots(xr[i], yr[i])[1])
    D = det ** 2 / np.var(dets, ddof=1)
    return TestResult("Identification strength D", float(D), float(stats.chi2.sf(D, 1)), 1,
                      extra=dict(det=det))


def breusch_pagan(resid: np.ndarray, W: np.ndarray) -> TestResult:
    """Koenker's studentised Breusch-Pagan test (n R^2 of e^2 on [1, W])."""
    e2 = resid ** 2
    X = design(_const(len(e2)), W)
    coef, *_ = np.linalg.lstsq(X, e2, rcond=None)
    r2 = 1.0 - np.var(e2 - X @ coef) / np.var(e2)
    stat = len(e2) * r2
    df = X.shape[1] - 1
    return TestResult("Breusch-Pagan", float(stat), float(stats.chi2.sf(stat, df)), df)


# ---------------------------------------------------------------------------
# 5.2 Relevance
# ---------------------------------------------------------------------------

def _robust_wald(y, X, cols):
    coef, cov, _ = linear_regression(y, X)
    b = coef[cols]
    V = cov[np.ix_(cols, cols)]
    return float(b @ np.linalg.pinv(V) @ b)


def first_stage_F(x, Z, W=None) -> TestResult:
    """Heteroskedasticity-robust first-stage F (equals the Montiel Olea-Pflueger
    effective F for a single instrument)."""
    Z = np.asarray(Z, dtype=float).reshape(len(x), -1)
    X = design(_const(len(x)), Z, W)
    k = Z.shape[1]
    F = _robust_wald(x, X, list(range(1, 1 + k))) / k
    return TestResult("First-stage robust F", F, float(stats.f.sf(F, k, len(x) - X.shape[1])), k)


def anderson_rubin(y, x, W, Z, b0: float) -> TestResult:
    """Robust Anderson-Rubin test of H0: beta = b0."""
    Z = np.asarray(Z, dtype=float).reshape(len(y), -1)
    X = design(_const(len(y)), Z, W)
    k = Z.shape[1]
    stat = _robust_wald(y - b0 * x, X, list(range(1, 1 + k)))
    return TestResult("Anderson-Rubin", stat, float(stats.chi2.sf(stat, k)), k)


def anderson_rubin_ci(y, x, W, Z, center: float, width: float = 5.0,
                      n_grid: int = 401, alpha: float = 0.05) -> dict:
    """Weak-instrument-robust confidence set by test inversion over a grid.

    Returns the hull of the accepted grid points and flags whether the set
    reaches the grid boundary (unbounded set: weak identification).
    """
    grid = np.linspace(center - width, center + width, n_grid)
    acc = np.array([anderson_rubin(y, x, W, Z, b).pvalue >= alpha for b in grid])
    if not acc.any():
        return dict(lo=np.nan, hi=np.nan, empty=True, unbounded=False)
    lo, hi = grid[acc].min(), grid[acc].max()
    return dict(lo=float(lo), hi=float(hi), empty=False,
                unbounded=bool(acc[0] or acc[-1]))


# ---------------------------------------------------------------------------
# 5.3 Exogeneity of the learned instrument
# ---------------------------------------------------------------------------

def residual_independence(y, x, W, z, method: str = "hsic", n_perm: int = 200,
                          rng=None) -> TestResult:
    """HSIC / distance-covariance permutation test between z and eps_hat = y - x b_IV - W d.

    The linear correlation is zero by construction, so only nonlinear
    dependence is informative. The permutation null ignores estimation of
    beta (approximate); under Gaussianity the test is vacuous.
    """
    fit = tsls_fit(y, x, W, z)
    t = permutation_test(z, fit.resid, method=method, n_perm=n_perm, rng=rng)
    return TestResult(f"Residual independence ({method})", t.statistic, t.pvalue)


def gmm_j(y, x, W, Z) -> TestResult:
    """Hansen's J from two-step efficient GMM with a robust weight matrix."""
    n = len(y)
    fit = tsls_fit(y, x, W, Z)
    X, Zf = fit.X, fit.Z
    g = Zf * fit.resid[:, None]
    S = g.T @ g / n
    S_inv = np.linalg.pinv(S)
    A = X.T @ Zf @ S_inv
    coef = np.linalg.pinv(A @ Zf.T @ X) @ A @ Zf.T @ y
    e = y - X @ coef
    gbar = Zf.T @ e / n
    J = float(n * gbar @ S_inv @ gbar)
    df = Zf.shape[1] - X.shape[1]
    return TestResult("Hansen J", J, float(stats.chi2.sf(J, df)) if df > 0 else np.nan, df,
                      extra=dict(beta=float(coef[1])))


def instrument_functions(z: np.ndarray, degree: int = 3) -> np.ndarray:
    """Standardised powers z, z^2, ..., z^degree used as extra instruments."""
    s = (z - z.mean()) / z.std()
    cols = [s]
    for d in range(2, degree + 1):
        c = s ** d
        cols.append((c - c.mean()) / c.std())
    return np.column_stack(cols)


def overid_functions(y, x, W, z, degree: int = 3) -> TestResult:
    """J test with f_1(z), ..., f_d(z) as instruments: a contaminated z implies
    different betas across the functions and the J test rejects."""
    out = gmm_j(y, x, W, instrument_functions(z, degree))
    out.name = f"J with powers of z (degree {degree})"
    return out


def c_test(y, x, W, Z_base, Z_suspect) -> TestResult:
    """Difference-in-Hansen (C) test for the validity of ``Z_suspect`` given ``Z_base``."""
    Z_base = np.asarray(Z_base, dtype=float).reshape(len(y), -1)
    Z_suspect = np.asarray(Z_suspect, dtype=float).reshape(len(y), -1)
    full = gmm_j(y, x, W, np.column_stack([Z_base, Z_suspect]))
    base = gmm_j(y, x, W, Z_base)
    C = full.statistic - (0.0 if np.isnan(base.pvalue) else base.statistic)
    df = Z_suspect.shape[1]
    return TestResult("C test (difference in Hansen)", float(C), float(stats.chi2.sf(max(C, 0.0), df)), df)


def hausman_ols_iv(y, x, W, z) -> TestResult:
    """Robust Durbin-Wu-Hausman test (control-function form): t-test on the
    first-stage residual added to the structural equation."""
    Z = np.asarray(z, dtype=float).reshape(len(y), -1)
    Xf = design(_const(len(y)), Z, W)
    coef, *_ = np.linalg.lstsq(Xf, x, rcond=None)
    v = x - Xf @ coef
    stat = _robust_wald(y, design(_const(len(y)), x, W, v), [-1])
    return TestResult("Hausman (OLS vs IV)", stat, float(stats.chi2.sf(stat, 1)), 1)


def estimator_agreement(estimators: dict, y, x, W, n_boot: int = 99, rng=None) -> TestResult:
    """Hausman-type test of equality between two estimators of beta.

    ``estimators`` maps two names to callables ``f(y, x, W) -> float``; the
    variance of the difference is obtained by a joint nonparametric bootstrap.
    """
    (n1, f1), (n2, f2) = list(estimators.items())
    rng = np.random.default_rng(rng)
    d0 = f1(y, x, W) - f2(y, x, W)
    diffs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        diffs.append(f1(y[i], x[i], W[i]) - f2(y[i], x[i], W[i]))
    stat = d0 ** 2 / np.var(diffs, ddof=1)
    return TestResult(f"Agreement {n1} vs {n2}", float(stat), float(stats.chi2.sf(stat, 1)), 1,
                      extra=dict(diff=float(d0)))


def negative_control(y_placebo, x, W, z) -> TestResult:
    """IV effect of x on an outcome it is known not to affect (should be zero)."""
    est = tsls(y_placebo, x, W, z, name="negative control")
    t2 = (est.beta / est.se) ** 2
    return TestResult("Negative-control outcome", float(t2), float(stats.chi2.sf(t2, 1)), 1,
                      extra=dict(beta=est.beta, se=est.se))


def split_independence(x_exo, r, method: str = "hsic", n_perm: int = 200, rng=None) -> TestResult:
    """HSIC(x_exo, r): necessary (not sufficient) condition for the split."""
    t = permutation_test(x_exo, r, method=method, n_perm=n_perm, rng=rng)
    return TestResult(f"Split independence ({method})", t.statistic, t.pvalue)


# ---------------------------------------------------------------------------
# 5.4 Sensitivity and 5.5 multiple testing
# ---------------------------------------------------------------------------

def conley_union_ci(y, x, W, z, delta: float, n_grid: int = 21) -> dict:
    """Conley, Hansen & Rossi (2012) union-of-intervals under y = x b + z g + ..., |g| <= delta."""
    los, his, betas = [], [], []
    for g in np.linspace(-delta, delta, n_grid):
        est = tsls(y - g * z, x, W, z)
        los.append(est.ci[0])
        his.append(est.ci[1])
        betas.append(est.beta)
    return dict(lo=float(min(los)), hi=float(max(his)),
                beta_range=(float(min(betas)), float(max(betas))))


def holm(pvalues) -> np.ndarray:
    """Holm (1979) step-down adjusted p-values."""
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def bonferroni(pvalues) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float)
    return np.minimum(1.0, p * len(p))


def pre_estimation_battery(y, x, W, partial: str = "linear", n_boot: int = 199, rng=None) -> dict:
    """Run the Section 5.1 checks on the W-residualised data."""
    xr = partial_out(x, W, partial, rng=rng)
    yr = partial_out(y, W, partial, rng=rng)
    ols_resid = linear_regression(y, design(_const(len(y)), x, W))[2]
    return {
        "nongaussian_x": nongaussianity(xr),
        "nongaussian_ols_resid": nongaussianity(ols_resid),
        "identification_D": identification_strength(xr, yr, n_boot=n_boot, rng=rng),
        "breusch_pagan": breusch_pagan(linear_regression(x, design(_const(len(x)), W))[2], W),
    }
