"""
Flame elongation trajectory module.

Implements flame-coupled piston boundary condition using the
Clavin-Tofaili formulation:

    u_p = (sigma(t) - 1) * (rho_u / rho_b) * S_L

where:
    - sigma(t) is an imposed elongation function
    - rho_u/rho_b is the density ratio from flame data
    - S_L is the laminar flame speed from flame data

The challenge is that S_L and densities depend on local (P, T) at the
piston face, but (P, T) depend on u_p through shock relations. This
creates a nonlinear coupling requiring iteration at each timestep.

Components:
    - FlamePropertyInterpolator: 2D P-T interpolation of flame properties
    - ElongationBase, PowerLawElongation, etc.: Elongation functions sigma(t)
    - FlameElongationTrajectory: Combines elongation with flame lookup
    - FlameCoupledPistonBC: Iterative boundary condition implementation

Reference:
    Clavin, P., & Tofaili, H. (2021). Flame elongation model.
"""

from flame_property_interpolator import FlamePropertyInterpolator, FlameProperties
from flame_elongation_trajectory import (
    FlameElongationTrajectory,
    ElongationBase,
    PowerLawElongation,
    ExponentialElongation,
    LinearElongation,
    ConstantElongation,
)
from flame_coupled_piston_bc import FlameCoupledPistonBC, IterationResult

__all__ = [
    "FlamePropertyInterpolator",
    "FlameProperties",
    "FlameElongationTrajectory",
    "ElongationBase",
    "PowerLawElongation",
    "ExponentialElongation",
    "LinearElongation",
    "ConstantElongation",
    "FlameCoupledPistonBC",
    "IterationResult",
]
