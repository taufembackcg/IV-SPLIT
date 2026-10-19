"""Synthetic instruments by adversarial splitting with identification through non-Gaussianity."""

from .dgp import DGPConfig, Sample, scenario, simulate, SCENARIOS
from .estimators import Estimate

__all__ = ["DGPConfig", "Sample", "scenario", "simulate", "SCENARIOS", "Estimate"]
