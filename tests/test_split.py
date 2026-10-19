import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ivsplit.estimators import partial_out  # noqa: E402
from ivsplit.split import NeuralSplit, SplitConfig, split_iv  # noqa: E402

FAST = dict(epochs=15, hidden=16, batch_size=128)


def test_split_adds_up(make_sample):
    s = make_sample("s1_latent_nongaussian", n=1000)
    m = NeuralSplit(SplitConfig(**FAST)).fit(s.y, s.x, s.W)
    x_exo, r = m.transform(s.y, s.x, s.W)
    np.testing.assert_allclose(x_exo + r, partial_out(s.x, s.W), atol=1e-5)
    assert np.isfinite(m.beta_train)
    assert m.history and {"epoch", "loss", "beta", "rel"} <= set(m.history[0])


def test_relevance_band_avoids_degenerate_solutions(make_sample):
    s = make_sample("s1_latent_nongaussian", n=2000)
    cfg = SplitConfig(epochs=40, relevance_floor=0.2, relevance_cap=0.8, seed=1)
    m = NeuralSplit(cfg).fit(s.y, s.x, s.W)
    x_exo, r = m.transform(s.y, s.x, s.W)
    rho2 = np.corrcoef(x_exo, x_exo + r)[0, 1] ** 2
    assert np.std(x_exo) > 1e-3          # not constant
    assert 0.1 < rho2 < 0.9              # not x itself


def test_split_iv_uses_disjoint_halves(make_sample):
    s = make_sample("s1_latent_nongaussian", n=1000)
    res = split_iv(s.y, s.x, s.W, SplitConfig(**FAST), n_boot=19)
    assert len(res.test_idx) == 500 and len(np.unique(res.test_idx)) == 500
    assert res.x_exo.shape == (500,)
    for est in (res.iv, res.cf):
        assert np.isfinite(est.beta)
        assert {"F", "D", "identified", "beta_train"} <= set(est.extra)


def test_split_iv_battery(make_sample):
    s = make_sample("s1_latent_nongaussian", n=1000)
    res = split_iv(s.y, s.x, s.W, SplitConfig(**FAST), run_tests=True, n_boot=19)
    assert {"first_stage_F", "identification_D", "residual_hsic", "residual_dcov",
            "overid_powers", "hausman", "split_hsic"} == set(res.diagnostics)
    for t in res.diagnostics.values():
        assert np.isnan(t.pvalue) or 0 <= t.pvalue <= 1


def test_gaussian_sources_flagged_as_not_identified(make_sample):
    s = make_sample("s3_gaussian", n=2000)
    res = split_iv(s.y, s.x, s.W, SplitConfig(**FAST), n_boot=49)
    assert not res.iv.identified


def test_optional_inputs(make_sample):
    s = make_sample("s1_latent_nonlinear_g", n=600)
    for kw in (dict(use_w=False), dict(use_outcome=True), dict(partial="poly")):
        m = NeuralSplit(SplitConfig(**FAST, **kw)).fit(s.y, s.x, s.W)
        x_exo, _ = m.transform(s.y, s.x, s.W)
        assert np.all(np.isfinite(x_exo))


def test_reproducible_given_seed(make_sample):
    s = make_sample("s1_latent_nongaussian", n=600)
    a = split_iv(s.y, s.x, s.W, SplitConfig(**FAST, seed=3), n_boot=9)
    b = split_iv(s.y, s.x, s.W, SplitConfig(**FAST, seed=3), n_boot=9)
    assert a.iv.beta == pytest.approx(b.iv.beta)
