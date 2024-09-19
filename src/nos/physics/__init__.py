"""Acoustic physics."""

from nos.physics.helmholtz_residual import HelmholtzDomainMSE, HelmholtzDomainResidual
from nos.physics.laplace import Laplace
from nos.physics.weight_scheduler_lin import WeightSchedulerLinear

__all__ = ["Laplace", "WeightSchedulerLinear", "HelmholtzDomainResidual", "HelmholtzDomainMSE"]
