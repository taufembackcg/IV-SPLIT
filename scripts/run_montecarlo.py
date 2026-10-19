#!/usr/bin/env python3
"""Run the Monte Carlo grid of Section 7 and write per-replication and summary CSVs.

Examples
--------
Quick pilot (fast estimators only):
    python scripts/run_montecarlo.py --scenarios s1_latent_nongaussian s3_gaussian --n 1000 2000 --reps 50

Full design of the paper (slow; the neural split dominates the cost):
    python scripts/run_montecarlo.py --all-scenarios --n 500 1000 2000 5000 10000 \
        --lam 0.2 0.5 1.0 --reps 500 --estimators all
"""

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from ivsplit.dgp import SCENARIOS  # noqa: E402
from ivsplit.montecarlo import ESTIMATORS, FAST_ESTIMATORS, run_grid, write_csv  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenarios", nargs="+", default=["s1_latent_nongaussian", "s3_gaussian"],
                    choices=sorted(SCENARIOS))
    ap.add_argument("--all-scenarios", action="store_true")
    ap.add_argument("--n", nargs="+", type=int, default=[1000, 2000])
    ap.add_argument("--lam", nargs="+", type=float, default=[0.5])
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--estimators", nargs="+", default=FAST_ESTIMATORS,
                    help=f"subset of {sorted(ESTIMATORS)} or 'all'")
    ap.add_argument("--seed", type=int, default=20261019)
    ap.add_argument("--out", default="results")
    args = ap.parse_args(argv)

    scenarios = sorted(SCENARIOS) if args.all_scenarios else args.scenarios
    estimators = sorted(ESTIMATORS) if args.estimators == ["all"] else args.estimators
    records, summary = run_grid(scenarios, args.n, args.lam, args.reps, estimators,
                                seed=args.seed, progress=True)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(records, out / "replications.csv")
    write_csv(summary, out / "summary.csv")
    cols = ["scenario", "n", "lam", "estimator", "bias", "rmse", "coverage", "not_identified"]
    print("\t".join(cols))
    for row in summary:
        print("\t".join(f"{row[c]:.3f}" if isinstance(row[c], float) else str(row[c]) for c in cols))


if __name__ == "__main__":
    main()
