"""Monte Carlo runner (Section 7 of the paper).

Each cell of the grid (scenario x n x lambda) is replicated ``reps`` times;
every replication draws a sample and runs the requested estimators. The
summary reports bias, RMSE, coverage of the 95% interval, the rejection
rate of H0: beta = beta_true, and the share of replications in which the
estimator declares that beta is not identified.
"""

from __future__ import annotations

import itertools
import zlib
from typing import Callable

import numpy as np

from . import estimators as est
from .dgp import scenario, simulate

EstimatorFn = Callable[[object, np.random.Generator], est.Estimate]


def _neural(kind: str, **kw) -> EstimatorFn:
    def run(s, rng):
        from .split import SplitConfig, split_iv
        # neural_iv and neural_cf share one fit per sample
        cache = s.__dict__.setdefault("_split_cache", {})
        key = tuple(sorted(kw.items()))
        if key not in cache:
            cfg = SplitConfig(seed=int(rng.integers(1 << 31)), **kw)
            cache[key] = split_iv(s.y, s.x, s.W, cfg, n_boot=49, rng=rng)
        res = cache[key]
        return res.iv if kind == "iv" else res.cf
    return run


ESTIMATORS: dict[str, EstimatorFn] = {
    "ols": lambda s, rng: est.ols(s.y, s.x, s.W),
    "oracle": lambda s, rng: est.oracle_tsls(s.y, s.x, s.W, s.z_oracle),
    "sp": lambda s, rng: est.sp_projection(s.y, s.x, s.W),
    "lewbel": lambda s, rng: est.lewbel(s.y, s.x, s.W),
    "copula": lambda s, rng: est.copula_cf(s.y, s.x, s.W),
    "cumulant": lambda s, rng: est.cumulant_ica(s.y, s.x, s.W, n_boot=99, rng=rng),
    "cumulant_aux": lambda s, rng: est.cumulant_ica(s.y, s.x, s.W, label="aux", aux=s.W[:, 0],
                                                    n_boot=99, rng=rng),
    "neural_iv": _neural("iv", epochs=60),
    "neural_cf": _neural("cf", epochs=60),
}

FAST_ESTIMATORS = ["ols", "oracle", "sp", "lewbel", "copula", "cumulant"]


def run_cell(scenario_name: str, n: int, lam: float, reps: int, estimators: list[str],
             seed: int = 0, **overrides) -> list[dict]:
    """Run one grid cell and return one record per (replication, estimator)."""
    cfg = scenario(scenario_name, n=n, lam=lam, **overrides)
    ss = np.random.SeedSequence([seed, n, int(round(1000 * lam)), zlib.crc32(scenario_name.encode())])
    records = []
    for rep, child in enumerate(ss.spawn(reps)):
        rng = np.random.default_rng(child)
        sample = simulate(cfg, rng)
        for name in estimators:
            e = ESTIMATORS[name](sample, rng)
            lo, hi = e.ci
            records.append(dict(
                scenario=scenario_name, n=n, lam=lam, rep=rep, estimator=name,
                beta_hat=e.beta, se=e.se, lo=lo, hi=hi, beta=cfg.beta,
                identified=e.identified,
            ))
    return records


def summarise(records: list[dict]) -> list[dict]:
    """Aggregate records by (scenario, n, lam, estimator)."""
    keys = ("scenario", "n", "lam", "estimator")
    groups: dict[tuple, list[dict]] = {}
    for r in records:
        groups.setdefault(tuple(r[k] for k in keys), []).append(r)
    out = []
    for key, rs in groups.items():
        b = np.array([r["beta_hat"] for r in rs])
        beta = rs[0]["beta"]
        lo = np.array([r["lo"] for r in rs])
        hi = np.array([r["hi"] for r in rs])
        has_ci = np.isfinite(lo) & np.isfinite(hi)
        cover = (lo <= beta) & (beta <= hi)
        out.append(dict(
            zip(keys, key),
            reps=len(rs),
            bias=float(np.mean(b - beta)),
            median_bias=float(np.median(b - beta)),
            rmse=float(np.sqrt(np.mean((b - beta) ** 2))),
            coverage=float(np.mean(cover[has_ci])) if has_ci.any() else np.nan,
            rejection=float(1 - np.mean(cover[has_ci])) if has_ci.any() else np.nan,
            not_identified=float(np.mean([not r["identified"] for r in rs])),
        ))
    return out


def run_grid(scenarios, ns, lams, reps: int, estimators: list[str], seed: int = 0,
             progress: bool = False) -> tuple[list[dict], list[dict]]:
    records = []
    for sc, n, lam in itertools.product(scenarios, ns, lams):
        if progress:
            print(f"[mc] {sc} n={n} lam={lam}", flush=True)
        records += run_cell(sc, n, lam, reps, estimators, seed)
    return records, summarise(records)


def write_csv(rows: list[dict], path) -> None:
    import csv
    if not rows:
        return
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


__all__ = ["ESTIMATORS", "FAST_ESTIMATORS", "run_cell", "run_grid", "summarise", "write_csv"]
