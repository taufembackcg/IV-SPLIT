import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ivsplit.estimators import ols  # noqa: E402
from ivsplit.gan_iv import GanIVConfig, _tests_torch, filter_candidates, gan_iv, groups_of  # noqa: E402


def test_differentiable_tests_match_closed_form():
    rng = np.random.default_rng(0)
    n = 2000
    Z = rng.standard_normal((n, 3))
    x = Z @ [0.5, 0.2, 0.0] + rng.standard_normal(n)
    y = x + rng.standard_normal(n) + 0.3 * Z[:, 2]
    x, y, Z = x - x.mean(), y - y.mean(), Z - Z.mean(0)
    F, S, _, beta = _tests_torch(*(torch.tensor(a, dtype=torch.float64) for a in (Z, x, y)), ridge=0.0)
    P = Z @ np.linalg.solve(Z.T @ Z, Z.T)
    r2 = x @ P @ x / (x @ x)
    assert float(F) == pytest.approx(r2 / 3 / ((1 - r2) / (n - 4)), rel=1e-6)
    xh = P @ x
    b = xh @ y / (xh @ x)
    e = y - b * x
    assert float(beta) == pytest.approx(b, rel=1e-6)
    assert float(S) == pytest.approx(n * e @ P @ e / (e @ e), rel=1e-6)


@pytest.mark.parametrize("method", ["hausman", "ctest"])
def test_filter_drops_an_invalid_candidate(make_sample, method):
    s = make_sample("s1_latent_nongaussian", n=10_000, lam=1.0, dist_H="exponential")
    rng = np.random.default_rng(0)
    Z = np.column_stack([s.z_oracle, s.z_oracle + rng.standard_normal(len(s.y)),
                         s.H + 0.5 * rng.standard_normal(len(s.y))])
    keep = filter_candidates(s.y, s.x, s.W, Z, [0, 0, 0], method)
    assert 2 not in keep and 0 in keep


def test_filter_keeps_one_instrument_per_endogenous_regressor(make_sample):
    s = make_sample("s1_latent_nongaussian", n=5000, lam=1.0)
    Z = np.column_stack([s.H, s.H + 0.1 * np.random.default_rng(0).standard_normal(len(s.y))])
    assert len(filter_candidates(s.y, s.x, s.W, Z, [0, 0], "hausman")) >= 1
    assert groups_of(3).tolist() == [0, 0, 1, 1, 2, 2]


def test_filter_none_keeps_everything(make_sample):
    s = make_sample("s1_latent_nongaussian", n=500)
    Z = np.column_stack([s.z_oracle, s.H])
    assert filter_candidates(s.y, s.x, s.W, Z, [0, 0], "none") == [0, 1]


def test_gan_instruments_pass_the_tests_but_reproduce_ols(make_sample):
    """The point of Section 2: passing F and Sargan says nothing about exogeneity."""
    s = make_sample("s1_latent_nongaussian", n=2000, lam=1.0)
    e = gan_iv(s.y, s.x, s.W, GanIVConfig(epochs=150))
    assert e.identified                       # F > 10 and Sargan not rejected
    assert e.extra["F"] > 10
    assert e.extra["n_candidates"] == 2
    assert len(e.extra["per_variable"]) == 1
    assert abs(e.beta - ols(s.y, s.x, s.W).beta) < 0.05
    assert abs(e.beta - s.beta) > 0.3
    u = s.u[e.extra["eval_idx"]]
    assert np.corrcoef(e.extra["xhat"], u)[0, 1] > 0.3
