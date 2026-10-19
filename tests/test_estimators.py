import numpy as np
import pytest

from ivsplit import estimators as E


def test_ols_and_2sls_recover_beta_without_endogeneity(make_sample):
    s = make_sample("s5_placebo", n=20_000)
    assert E.ols(s.y, s.x, s.W).beta == pytest.approx(1.0, abs=0.03)
    assert E.oracle_tsls(s.y, s.x, s.W, s.z_oracle).beta == pytest.approx(1.0, abs=0.05)


def test_ols_biased_oracle_consistent(make_sample):
    s = make_sample("s1_latent_nongaussian", n=20_000)
    ols = E.ols(s.y, s.x, s.W)
    orc = E.oracle_tsls(s.y, s.x, s.W, s.z_oracle)
    # plim OLS = beta + lam Var(H) / (Var(H) + Var(e_X)) = 1.25
    assert ols.beta == pytest.approx(1.25, abs=0.03)
    assert orc.ci[0] < 1.0 < orc.ci[1] or abs(orc.beta - 1.0) < 0.03


def test_tsls_matches_ratio_formula():
    rng = np.random.default_rng(0)
    n = 1000
    z = rng.standard_normal(n)
    x = z + rng.standard_normal(n)
    y = 2 * x + rng.standard_normal(n)
    b = E.tsls(y, x, None, z).beta
    zc, xc, yc = z - z.mean(), x - x.mean(), y - y.mean()
    assert b == pytest.approx(zc @ yc / (zc @ xc))


def test_robust_se_close_to_simulation_sd():
    draws, ses = [], []
    for seed in range(200):
        rng = np.random.default_rng(seed)
        z = rng.standard_normal(500)
        h = rng.standard_normal(500)
        x = z + h
        y = x + h + rng.standard_normal(500) * (1 + np.abs(z))
        e = E.tsls(y, x, None, z)
        draws.append(e.beta)
        ses.append(e.se)
    assert np.mean(ses) == pytest.approx(np.std(draws), rel=0.2)


def test_partial_out_methods(make_sample):
    s = make_sample("s1_latent_nonlinear_g", n=4000)
    for method in ("linear", "poly", "gbm"):
        r = E.partial_out(s.x, s.W, method, rng=0)
        assert r.shape == s.x.shape
    # the flexible residual removes the nonlinear g(w): closer to H + e_X
    target = s.H + s.z_oracle
    lin = E.partial_out(s.x, s.W, "linear")
    poly = E.partial_out(s.x, s.W, "poly")
    assert np.var(poly - target) < np.var(lin - target)
    with pytest.raises(ValueError):
        E.partial_out(s.x, s.W, "nope")


def test_lewbel_under_heteroskedasticity(make_sample):
    s = make_sample("aux_heteroskedastic", n=20_000)
    assert E.lewbel(s.y, s.x, s.W).beta == pytest.approx(1.0, abs=0.08)


def test_sp_wins_in_its_home_scenario(make_sample):
    s = make_sample("s2_nonlinear_confounding", n=10_000)
    ols = E.ols(s.y, s.x, s.W)
    sp = E.sp_projection(s.y, s.x, s.W)
    assert abs(ols.beta - 1.0) > 0.3
    assert abs(sp.beta - 1.0) < 0.2
    assert 2 in sp.extra["selected"]


def test_sp_cannot_remove_linear_confounding(make_sample):
    s = make_sample("s1_latent_nongaussian", n=10_000)
    assert abs(E.sp_projection(s.y, s.x, s.W).beta - 1.0) > 0.15


def test_copula_runs_with_bootstrap(make_sample):
    s = make_sample("s1_latent_nongaussian", n=1000)
    e = E.copula_cf(s.y, s.x, s.W, n_boot=20, rng=0)
    assert np.isfinite(e.beta) and e.se > 0


def test_cumulants_vanish_for_gaussian_data():
    rng = np.random.default_rng(0)
    a = rng.standard_normal(400_000)
    b = 0.5 * a + rng.standard_normal(400_000)
    assert np.max(np.abs(E.joint_cumulants4(a, b))) < 0.05
    assert np.max(np.abs(E.joint_cumulants3(a, b))) < 0.03


def test_joint_cumulants_multilinear():
    rng = np.random.default_rng(1)
    h = rng.laplace(size=400_000) / np.sqrt(2)  # excess kurtosis 3
    a, b = h, 2 * h
    c = E.joint_cumulants4(a, b)
    np.testing.assert_allclose(c, 3 * np.array([1, 2, 4, 8, 16]), rtol=0.1)


def test_cumulant_roots_recover_both_slopes(make_sample):
    s = make_sample("s1_latent_nongaussian", n=200_000, lam=1.0)
    xr = E.residualize(s.x, s.W)
    yr = E.residualize(s.y, s.W)
    roots, det, disc = E.cumulant_roots(xr, yr)
    np.testing.assert_allclose(roots, [1.0, 2.0], atol=0.12)
    assert det < 0 and disc > 0


def test_cumulant_ica_sign_labelling(make_sample):
    s = make_sample("s1_latent_nongaussian", n=50_000, lam=1.0)
    e = E.cumulant_ica(s.y, s.x, s.W, n_boot=49, rng=0)
    assert e.beta == pytest.approx(1.0, abs=0.15)
    assert e.identified
    flipped = E.cumulant_ica(s.y, s.x, s.W, lam_sign=-1, n_boot=0)
    assert flipped.beta == pytest.approx(2.0, abs=0.2)


def test_cumulant_ica_flags_gaussian_sources(make_sample):
    flags = []
    for seed in range(5):
        s = make_sample("s3_gaussian", n=5000, seed=seed)
        flags.append(E.cumulant_ica(s.y, s.x, s.W, n_boot=99, rng=seed).identified)
    assert sum(flags) <= 1


def test_aux_labelling_breaks_symmetry(make_sample):
    s = make_sample("aux_heteroskedastic", n=50_000, lam=1.0)
    xr, yr = E.residualize(s.x, s.W), E.residualize(s.y, s.W)
    assert E.heteroskedasticity_slope(xr, yr, s.W[:, 0]) == pytest.approx(1.0, abs=0.15)
    e = E.cumulant_ica(s.y, s.x, s.W, label="aux", aux=s.W[:, 0], n_boot=0)
    assert e.beta == pytest.approx(1.0, abs=0.2)


def test_label_roots_rejects_unknown_rule():
    with pytest.raises(ValueError):
        E.label_roots(np.array([0.0, 1.0]), "magic")
