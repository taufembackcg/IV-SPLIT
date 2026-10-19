import numpy as np
import pytest

from ivsplit.dgp import scenario, simulate


@pytest.fixture
def make_sample():
    """Factory: make_sample(name, n=..., seed=..., **overrides) -> Sample."""
    def _make(name, n=5000, seed=0, **overrides):
        return simulate(scenario(name, n=n, **overrides), np.random.default_rng(seed))
    return _make
