# IV-SPLIT: synthetic instruments by adversarial splitting

This repository holds a research proposal and its Monte Carlo testbed. The idea is to split an endogenous regressor into an exogenous part and a remainder, `x = x_exo + r` with `x_exo ⟂ u`, using a neural network with an HSIC independence penalty. Identification comes from independent, non-Gaussian latent sources.

- `paper/main.tex`: the article (English, LaTeX). Build it with `cd paper && latexmk -pdf main.tex`. If latexmk isn't installed, run `pdflatex main && bibtex main && pdflatex main && pdflatex main`.
- `src/ivsplit/`: implementation.
  - `dgp.py`: data-generating processes and Monte Carlo scenarios S1–S5 plus an auxiliary heteroskedastic design.
  - `estimators.py`: OLS, 2SLS, oracle 2SLS, Lewbel (2012), Gaussian-copula control function, a simplified structural-error-projection (SP) estimator, and Cumulant-ICA (fourth-order cumulants with Prony roots and labelling).
  - `independence.py`: HSIC and distance covariance, with permutation tests.
  - `split.py`: the neural split, trained on one half of the sample and evaluated on the other (needs PyTorch).
  - `diagnostics.py`: the test battery. It covers Jarque–Bera, the identification-strength statistic D, Breusch–Pagan, the robust first-stage F, Anderson–Rubin, residual HSIC/dCov, Hansen J with powers of the instrument, the C test, Hausman, estimator agreement, negative controls, Conley–Hansen–Rossi bounds, and Holm/Bonferroni corrections.
  - `montecarlo.py`: the grid runner and summary metrics (bias, RMSE, coverage, share flagged as not identified).
- `tests/`: the pytest suite.
- `scripts/run_montecarlo.py`: command-line interface for the Monte Carlo.
- `scripts/make_tables.py`: turns the Monte Carlo summaries into the paper's LaTeX tables.

## Quick start

```bash
pip install -e ".[nn,test]"        # torch is optional; only split.py needs it
pytest -m "not slow"               # fast suite
pytest -m slow                     # Monte Carlo checks (about a minute)

# pilot Monte Carlo and paper tables
python scripts/run_montecarlo.py --all-scenarios --n 1000 5000 --lam 0.5 --reps 200 \
    --estimators ols oracle sp lewbel copula cumulant cumulant_aux --out results/fast
python scripts/run_montecarlo.py --scenarios s1_latent_nongaussian s3_gaussian --n 1000 5000 10000 \
    --lam 1.0 --reps 200 --estimators ols oracle cumulant --out results/lam1
python scripts/run_montecarlo.py --scenarios s1_latent_nongaussian s3_gaussian aux_heteroskedastic \
    --n 2000 --lam 0.5 --reps 25 --estimators ols cumulant neural_iv neural_cf --out results/nn
# anchored split (one process per scenario keeps each run short) and GAN instruments
python scripts/run_montecarlo.py --scenarios s1_latent_nongaussian --n 2000 5000 --lam 1.0 --reps 40 \
    --estimators ols oracle cumulant cumulant_aux anchored anchored_aux anchored_xfit --out results/anchored
python scripts/run_montecarlo.py --scenarios s1_latent_nongaussian s1_asymmetric s2_nonlinear_confounding \
    s3_gaussian s5_placebo m2_latent_nongaussian m2_gaussian --n 1000 5000 --lam 1.0 --reps 60 \
    --estimators ols oracle gan_hausman gan_ctest gan_split gan_partialled --jobs 2 --out results/gan
python scripts/make_tables.py --fast results/fast/summary.csv --lam1 results/lam1/summary.csv \
    --nn results/nn/summary.csv --gan results/gan/summary.csv --anchored results/anchored/summary.csv
```

## Main findings so far

1. In the canonical latent-confounder model, fourth-order cumulants identify β only up to the set {β, β+λ}. The two points are observationally equivalent, so picking one needs a labelling assumption: either the sign of the confounding or an auxiliary variable.
2. The identification-strength statistic D is zero in the population when the sources are Gaussian or when there is no confounding. Small values of D flag failed identification.
3. In the over-complete model, the exogenous source is generally not a measurable function of the data. An unanchored network split therefore drifts towards OLS. The anchored split, which fixes β at the cumulant estimate and feeds ỹ to the network, carries the anchor over to the held-out half. With cross-fitting it matches the cumulant estimator's precision but does not improve on it.
4. The overidentification test that uses powers of the instrument has no power when E[H | x̃] is linear. That happens when the sources are Gaussian or when the confounder and the exogenous source are exchangeable.
5. The GAN-instrument idea (two generated candidates per endogenous regressor, trained to pass per-variable weak-instrument and Sargan tests, then filtered by Hausman) passes its tests in almost every replication but returns the OLS estimate (`src/ivsplit/gan_iv.py`, Section 7 of the paper). Under linear confounding no overidentification, Hausman, C or critic test has power.
