"""Kernel and distance-based independence measures and permutation tests.

HSIC: Gretton et al. (2005, 2008). Distance covariance: Szekely, Rizzo & Bakirov (2007).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _as_2d(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    return a[:, None] if a.ndim == 1 else a


def _sq_dists(a: np.ndarray) -> np.ndarray:
    sq = np.sum(a ** 2, axis=1)
    return np.maximum(sq[:, None] + sq[None, :] - 2.0 * a @ a.T, 0.0)


def median_bandwidth(a: np.ndarray) -> float:
    d2 = _sq_dists(_as_2d(a))
    med = np.median(d2[np.triu_indices_from(d2, k=1)])
    return float(np.sqrt(0.5 * med)) if med > 0 else 1.0


def gaussian_gram(a: np.ndarray, bandwidth: float | None = None) -> np.ndarray:
    a = _as_2d(a)
    bw = median_bandwidth(a) if bandwidth is None else bandwidth
    return np.exp(-_sq_dists(a) / (2.0 * bw ** 2))


def _center(K: np.ndarray) -> np.ndarray:
    return K - K.mean(axis=0, keepdims=True) - K.mean(axis=1, keepdims=True) + K.mean()


def hsic(a: np.ndarray, b: np.ndarray) -> float:
    """Biased (V-statistic) HSIC with Gaussian kernels and median bandwidths."""
    n = len(a)
    K = _center(gaussian_gram(a))
    L = gaussian_gram(b)
    return float(np.sum(K * L) / n ** 2)


def dcov2(a: np.ndarray, b: np.ndarray) -> float:
    """Squared sample distance covariance (V-statistic)."""
    A = _center(np.sqrt(_sq_dists(_as_2d(a))))
    B = _center(np.sqrt(_sq_dists(_as_2d(b))))
    return float(np.mean(A * B))


@dataclass
class IndependenceTest:
    statistic: float
    pvalue: float
    method: str


def _subsample(rng, n, max_n):
    return np.arange(n) if n <= max_n else rng.choice(n, max_n, replace=False)


def permutation_test(a, b, method: str = "hsic", n_perm: int = 200,
                     max_n: int = 1500, rng=None) -> IndependenceTest:
    """Permutation test of H0: a independent of b.

    Gram matrices are computed once; permutations only reindex one of them.
    Samples larger than ``max_n`` are subsampled to keep memory bounded.
    """
    rng = np.random.default_rng(rng)
    a, b = _as_2d(a), _as_2d(b)
    idx = _subsample(rng, len(a), max_n)
    a, b = a[idx], b[idx]
    n = len(a)
    if method == "hsic":
        K = _center(gaussian_gram(a))
        L = gaussian_gram(b)
        stat_fn = lambda p: np.sum(K * L[np.ix_(p, p)]) / n ** 2  # noqa: E731
    elif method == "dcov":
        K = _center(np.sqrt(_sq_dists(a)))
        L = _center(np.sqrt(_sq_dists(b)))
        stat_fn = lambda p: np.mean(K * L[np.ix_(p, p)])  # noqa: E731
    else:
        raise ValueError(method)
    obs = stat_fn(np.arange(n))
    null = np.array([stat_fn(rng.permutation(n)) for _ in range(n_perm)])
    pval = (1.0 + np.sum(null >= obs)) / (1.0 + n_perm)
    return IndependenceTest(float(obs), float(pval), method)
