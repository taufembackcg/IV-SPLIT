import numpy as np
import pytest

from ivsplit import diagnostics as D
from ivsplit import estimators as E


def test_jarque_bera_detects_laplace_not_gaussian():
    rng = np.random.default_rng(0)
    assert D.nongaussianity(rng.laplace(size=2000)).reject()
    res = D.nongaussianity(rng.standard_normal(2000))
    assert res.pvalue > 0.01
    assert set(res.extra) == {"skew", "exkurt"}


def test_identification_strength_separates_scenarios(make_sample):
    s = make_sample("s1_latent_nongaussian", n=20_000, lam=1.0)
    xr, yr = E.residualize(s.x, s.W), E.residualize(s.y, s.W)
    assert D.identification_strength(xr, yr, n_boot=99, rng=0).reject()
    g = make_sample("s3_gaussian", n=20_000, lam=1.0)
    xr, yr = E.residualize(g.x, g.W), E.residualize(g.y, g.W)
    assert not D.identification_strength(xr, yr, n_boot=99, rng=0).reject(0.01)


def test_identification_strength_vanishes_without_confounding(make_sample):
    s = make_sample("s5_placebo", n=20_000)
    xr, yr = E.residualize(s.x, s.W), E.residualize(s.y, s.W)
    assert not D.identification_strength(xr, yr, n_boot=99, rng=0).reject(0.01)


def test_breusch_pagan(make_sample):
    s = make_sample("aux_heteroskedastic", n=5000)
    e = E.residualize(s.x, s.W)
    assert D.breusch_pagan(e, s.W).statistic > 0
    rng = np.random.default_rng(0)
    W = rng.standard_normal((3000, 2))
    assert D.breusch_pagan(rng.standard_normal(3000) * np.exp(W[:, 0]), W).reject()


def test_first_stage_F_strong_and_noise(make_sample):
    s = make_sample("s1_latent_nongaussian", n=5000)
    assert D.first_stage_F(s.x, s.z_oracle, s.W).statistic > 100
    noise = np.random.default_rng(9).standard_normal(5000)
    assert D.first_stage_F(s.x, noise, s.W).statistic < 10


def test_anderson_rubin_ci(make_sample):
    s = make_sample("s1_latent_nongaussian", n=5000)
    ci = D.anderson_rubin_ci(s.y, s.x, s.W, s.z_oracle, center=1.0, width=1.0)
    assert ci["lo"] < 1.0 < ci["hi"] and not ci["unbounded"]
    noise = np.random.default_rng(1).standard_normal(5000)
    ci = D.anderson_rubin_ci(s.y, s.x, s.W, noise, center=1.0, width=1.0)
    assert ci["unbounded"] or ci["empty"]


def test_anderson_rubin_size(make_sample):
    rej = [D.anderson_rubin(*(lambda s: (s.y, s.x, s.W, s.z_oracle))(make_sample(
        "s1_latent_nongaussian", n=500, seed=k)), b0=1.0).reject() for k in range(100)]
    assert np.mean(rej) < 0.12


def test_overid_with_powers_valid_vs_contaminated(make_sample):
    s = make_sample("s1_latent_nongaussian", n=5000, lam=1.0, dist_H="exponential")
    valid = D.overid_functions(s.y, s.x, s.W, s.z_oracle)
    assert valid.df == 2 and valid.pvalue > 0.01
    contaminated = s.z_oracle + s.H  # = x - g(w), correlated with u
    assert D.overid_functions(s.y, s.x, s.W, contaminated).reject()


def test_overid_is_blind_with_exchangeable_sources(make_sample):
    # If H and e_X are i.i.d., E[H | H + e_X] is linear, every function of the
    # contaminated instrument implies the same (wrong) beta and J has no power.
    s = make_sample("s1_latent_nongaussian", n=20_000, lam=1.0)
    contaminated = s.z_oracle + s.H
    assert not D.overid_functions(s.y, s.x, s.W, contaminated).reject(0.01)


def test_overid_is_blind_under_gaussianity(make_sample):
    # Section 5.5: with Gaussian sources nonlinear functions carry no extra moments
    s = make_sample("s3_gaussian", n=20_000, lam=1.0)
    contaminated = s.z_oracle + s.H
    assert not D.overid_functions(s.y, s.x, s.W, contaminated).reject(0.01)


def test_c_test_flags_invalid_instrument(make_sample):
    s = make_sample("s1_latent_nongaussian", n=10_000, lam=1.0, dist_H="exponential")
    bad = s.H + 0.5 * np.random.default_rng(0).standard_normal(len(s.y))
    assert D.c_test(s.y, s.x, s.W, s.z_oracle, bad).reject()
    good = s.z_oracle ** 2 - 1
    assert D.c_test(s.y, s.x, s.W, s.z_oracle, good).pvalue > 0.01


def test_gmm_j_just_identified_has_no_pvalue(make_sample):
    s = make_sample("s1_latent_nongaussian", n=1000)
    j = D.gmm_j(s.y, s.x, s.W, s.z_oracle)
    assert j.df == 0 and np.isnan(j.pvalue) and j.statistic == pytest.approx(0, abs=1e-6)


def test_hausman(make_sample):
    s = make_sample("s1_latent_nongaussian", n=5000)
    assert D.hausman_ols_iv(s.y, s.x, s.W, s.z_oracle).reject()
    p = make_sample("s5_placebo", n=5000)
    assert not D.hausman_ols_iv(p.y, p.x, p.W, p.z_oracle).reject(0.01)


def test_residual_independence(make_sample):
    s = make_sample("s1_latent_nongaussian", n=3000, lam=1.0)
    assert D.residual_independence(s.y, s.x, s.W, s.z_oracle, n_perm=99, rng=0).pvalue > 0.01
    contaminated = s.H + s.z_oracle
    assert D.residual_independence(s.y, s.x, s.W, contaminated, n_perm=99, rng=0).reject()


def test_estimator_agreement(make_sample):
    s = make_sample("s1_latent_nongaussian", n=3000)
    ests = {"OLS": lambda y, x, W: E.ols(y, x, W).beta,
            "Lewbel": lambda y, x, W: E.lewbel(y, x, W).beta}
    out = D.estimator_agreement(ests, s.y, s.x, s.W, n_boot=49, rng=0)
    assert np.isfinite(out.statistic) and "diff" in out.extra


def test_estimator_agreement_detects_disagreement(make_sample):
    s = make_sample("s2_nonlinear_confounding", n=5000)
    ests = {"OLS": lambda y, x, W: E.ols(y, x, W).beta,
            "SP": lambda y, x, W: E.sp_projection(y, x, W).beta}
    assert D.estimator_agreement(ests, s.y, s.x, s.W, n_boot=29, rng=0).reject()


def test_negative_control(make_sample):
    s = make_sample("s1_latent_nongaussian", n=5000)
    placebo = s.W[:, 0] + np.random.default_rng(0).standard_normal(5000)
    assert D.negative_control(placebo, s.x, s.W, s.z_oracle).pvalue > 0.01
    # a placebo outcome that loads on the confounder is caught by an invalid instrument
    loaded = s.H + np.random.default_rng(1).standard_normal(5000)
    assert D.negative_control(loaded, s.x, s.W, s.H + s.z_oracle).reject()


def test_split_independence():
    rng = np.random.default_rng(0)
    a = rng.laplace(size=500)
    assert D.split_independence(a, rng.laplace(size=500), n_perm=99, rng=0).pvalue > 0.01
    assert D.split_independence(a, np.abs(a), n_perm=99, rng=0).reject()


def test_conley_union_widens_with_delta(make_sample):
    s = make_sample("s1_latent_nongaussian", n=3000)
    narrow = D.conley_union_ci(s.y, s.x, s.W, s.z_oracle, delta=0.0)
    wide = D.conley_union_ci(s.y, s.x, s.W, s.z_oracle, delta=0.2)
    assert wide["lo"] < narrow["lo"] and wide["hi"] > narrow["hi"]
    assert narrow["beta_range"][0] == pytest.approx(narrow["beta_range"][1])


def test_multiple_testing_corrections():
    p = [0.01, 0.04, 0.03, 0.5]
    np.testing.assert_allclose(D.holm(p), [0.04, 0.09, 0.09, 0.5])
    np.testing.assert_allclose(D.bonferroni(p), [0.04, 0.16, 0.12, 1.0])


def test_pre_estimation_battery(make_sample):
    s = make_sample("s1_latent_nongaussian", n=2000)
    out = D.pre_estimation_battery(s.y, s.x, s.W, n_boot=29, rng=0)
    assert set(out) == {"nongaussian_x", "nongaussian_ols_resid", "identification_D", "breusch_pagan"}
    assert out["nongaussian_x"].reject()
