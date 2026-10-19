import numpy as np
import pytest

from ivsplit.montecarlo import ESTIMATORS, FAST_ESTIMATORS, run_cell, run_grid, summarise, write_csv


def test_run_cell_records():
    recs = run_cell("s1_latent_nongaussian", n=300, lam=0.5, reps=3, estimators=["ols", "oracle"])
    assert len(recs) == 6
    assert {"beta_hat", "se", "lo", "hi", "identified", "rep"} <= set(recs[0])


def test_reproducible_cells():
    a = run_cell("s3_gaussian", 200, 0.5, 2, ["ols"], seed=1)
    b = run_cell("s3_gaussian", 200, 0.5, 2, ["ols"], seed=1)
    assert [r["beta_hat"] for r in a] == [r["beta_hat"] for r in b]


def test_summary_metrics():
    recs = [dict(scenario="s", n=1, lam=0.5, estimator="e", beta=1.0, beta_hat=b, lo=b - 0.1,
                 hi=b + 0.1, identified=i, rep=k, se=0.05)
            for k, (b, i) in enumerate([(1.05, True), (0.85, False), (1.0, True), (1.3, True)])]
    (row,) = summarise(recs)
    assert row["bias"] == pytest.approx(0.05)
    assert row["coverage"] == pytest.approx(0.5)
    assert row["not_identified"] == pytest.approx(0.25)
    assert row["rmse"] == pytest.approx(np.sqrt((0.05**2 + 0.15**2 + 0 + 0.3**2) / 4))


def test_grid_and_csv(tmp_path):
    recs, summ = run_grid(["s5_placebo"], [200], [0.0], 2, ["ols", "copula"])
    write_csv(summ, tmp_path / "s.csv")
    assert (tmp_path / "s.csv").read_text().startswith("scenario,n,lam,estimator")
    assert len(summ) == 2


def test_registry():
    assert set(FAST_ESTIMATORS) <= set(ESTIMATORS)


@pytest.mark.slow
def test_oracle_coverage_and_ols_failure():
    _, summ = run_grid(["s1_latent_nongaussian"], [1000], [0.5], 200, ["ols", "oracle"])
    by = {r["estimator"]: r for r in summ}
    assert 0.91 <= by["oracle"]["coverage"] <= 0.98
    assert by["ols"]["coverage"] < 0.05


@pytest.mark.slow
def test_cumulant_layer_separates_identifiable_from_gaussian():
    """Step 2 of the research plan: scenario 1 vs scenario 3."""
    _, summ = run_grid(["s1_latent_nongaussian", "s3_gaussian"], [5000], [1.0], 40, ["cumulant"])
    by = {r["scenario"]: r for r in summ}
    assert by["s1_latent_nongaussian"]["not_identified"] < 0.3
    assert by["s3_gaussian"]["not_identified"] > 0.7
    assert abs(by["s1_latent_nongaussian"]["median_bias"]) < 0.1


def test_neural_estimators_share_one_fit():
    pytest.importorskip("torch")
    from ivsplit.dgp import scenario, simulate
    s = simulate(scenario("s1_latent_nongaussian", n=400), 0)
    rng = np.random.default_rng(0)
    iv = ESTIMATORS["neural_iv"](s, rng)
    cf = ESTIMATORS["neural_cf"](s, rng)
    assert len(s._split_cache) == 1
    assert iv.name == "Neural-split IV" and cf.name == "Neural-split CF"


def test_scenario_lambda_wins_over_grid():
    recs = run_cell("s5_placebo", n=200, lam=0.5, reps=1, estimators=["ols"])
    assert recs[0]["lam"] == 0.0
