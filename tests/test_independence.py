import numpy as np
import pytest

from ivsplit.independence import dcov2, hsic, median_bandwidth, permutation_test


@pytest.mark.parametrize("method", ["hsic", "dcov"])
def test_detects_nonlinear_dependence(method):
    rng = np.random.default_rng(0)
    a = rng.standard_normal(400)
    b = a ** 2 + 0.3 * rng.standard_normal(400)
    assert abs(np.corrcoef(a, b)[0, 1]) < 0.2  # invisible to correlation
    assert permutation_test(a, b, method, n_perm=200, rng=1).pvalue < 0.01


@pytest.mark.parametrize("method", ["hsic", "dcov"])
def test_size_under_independence(method):
    rng = np.random.default_rng(2)
    pvals = [permutation_test(rng.standard_normal(150), rng.laplace(size=150), method,
                              n_perm=99, rng=k).pvalue for k in range(40)]
    rej = np.mean(np.array(pvals) < 0.05)
    assert rej <= 0.15


def test_statistics_nonnegative_and_scale():
    rng = np.random.default_rng(3)
    a, b = rng.standard_normal(200), rng.standard_normal(200)
    assert hsic(a, b) >= 0
    assert dcov2(a, b) >= -1e-12
    assert hsic(a, a) > hsic(a, b)
    assert median_bandwidth(10 * a) == pytest.approx(10 * median_bandwidth(a))


def test_subsampling_bounds_size():
    rng = np.random.default_rng(4)
    a = rng.standard_normal(5000)
    t = permutation_test(a, a + rng.standard_normal(5000), n_perm=20, max_n=300, rng=0)
    assert 0 < t.pvalue <= 1
