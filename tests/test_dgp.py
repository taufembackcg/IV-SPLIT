import numpy as np
import pytest
from scipy import stats

from ivsplit.dgp import DISTRIBUTIONS, SCENARIOS, DGPConfig, draw_source, g_function, scenario, simulate


@pytest.mark.parametrize("dist", DISTRIBUTIONS)
def test_sources_are_standardised(dist):
    v = draw_source(np.random.default_rng(0), dist, 200_000)
    assert abs(v.mean()) < 0.02
    assert abs(v.var() - 1.0) < 0.03


@pytest.mark.parametrize("dist, sign", [("laplace", 1), ("exponential", 1), ("bimodal", -1),
                                        ("uniform", -1), ("gaussian", 0)])
def test_source_kurtosis_sign(dist, sign):
    k = stats.kurtosis(draw_source(np.random.default_rng(1), dist, 200_000))
    if sign == 0:
        assert abs(k) < 0.05
    else:
        assert np.sign(k) == sign and abs(k) > 0.5


def test_exponential_is_skewed():
    v = draw_source(np.random.default_rng(2), "exponential", 100_000)
    assert stats.skew(v) > 1.5


def test_unknown_distribution():
    with pytest.raises(ValueError):
        draw_source(np.random.default_rng(0), "cauchy", 10)


def test_base_structure_holds_exactly():
    cfg = DGPConfig(n=1000, beta=0.7, lam=0.4)
    s = simulate(cfg, 3)
    gw = g_function(s.W, cfg.g)
    np.testing.assert_allclose(s.x, gw + s.H + s.z_oracle)
    np.testing.assert_allclose(s.y, cfg.beta * s.x + s.W @ np.full(cfg.n_w, 0.5) + s.u)
    # u = lam H + e_Y with e_Y independent of (H, e_X)
    e_y = s.u - cfg.lam * s.H
    assert abs(np.corrcoef(e_y, s.z_oracle)[0, 1]) < 0.1


def test_endogeneity_and_oracle_exogeneity(make_sample):
    s = make_sample("s1_latent_nongaussian", n=50_000)
    assert np.corrcoef(s.x, s.u)[0, 1] > 0.1
    assert abs(np.corrcoef(s.z_oracle, s.u)[0, 1]) < 0.02


def test_placebo_has_exogenous_x(make_sample):
    s = make_sample("s5_placebo", n=50_000)
    assert abs(np.corrcoef(s.x, s.u)[0, 1]) < 0.02


def test_violations(make_sample):
    s = make_sample("s4_correlated_sources", n=50_000)
    assert np.corrcoef(s.H, s.z_oracle)[0, 1] == pytest.approx(0.3, abs=0.02)
    s = make_sample("s4_exclusion_violation", n=50_000)
    assert np.corrcoef(s.z_oracle, s.u)[0, 1] > 0.1


def test_nonlinear_confounding_has_no_H(make_sample):
    s = make_sample("s2_nonlinear_confounding", n=50_000)
    v = s.z_oracle
    # u depends on x only through v^2; the linear correlation comes from the skewness of v
    assert np.corrcoef(s.u, v ** 2)[0, 1] > 0.2


def test_heteroskedastic_aux_scales_variance(make_sample):
    s = make_sample("aux_heteroskedastic", n=100_000)
    hi = s.z_oracle[s.W[:, 0] > 1].var()
    lo = s.z_oracle[s.W[:, 0] < -1].var()
    assert hi > 3 * lo
    assert s.z_oracle.var() == pytest.approx(1.0, rel=0.15)


def test_all_scenarios_simulate():
    for name in SCENARIOS:
        s = simulate(scenario(name, n=200), 0)
        assert s.y.shape == s.x.shape == (200,)
        assert np.all(np.isfinite(s.y))


def test_unknown_scenario():
    with pytest.raises(KeyError):
        scenario("nope")


def test_reproducible():
    a = simulate(scenario("s1_latent_nongaussian", n=100), 5)
    b = simulate(scenario("s1_latent_nongaussian", n=100), 5)
    np.testing.assert_array_equal(a.y, b.y)


def test_subset(make_sample):
    s = make_sample("s1_latent_nongaussian", n=100)
    t = s.subset(np.arange(10))
    assert len(t.y) == 10 and t.W.shape == (10, s.W.shape[1])
