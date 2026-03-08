"""Numerics module containing Riemann solvers, time integration, and artificial viscosity/heat conduction."""

from lagrangian_solver.numerics.riemann import (
    RiemannState,
    RiemannSolution,
    RiemannSolverBase,
    ExactRiemannSolver,
    HLLCRiemannSolver,
)
from lagrangian_solver.numerics.time_integration import (
    CompatibleHeunIntegrator,
    CompatibleForwardEulerIntegrator,
    CompatibleSSPRK3Integrator,
)
from lagrangian_solver.numerics.artificial_viscosity import (
    ArtificialViscosity,
    ArtificialViscosityConfig,
)
from lagrangian_solver.numerics.artificial_heat_conduction import (
    ArtificialHeatConduction,
    ArtificialHeatConductionConfig,
)
from lagrangian_solver.numerics.boundary_riemann import (
    BoundaryRiemannSolver,
    BoundaryState,
)

__all__ = [
    # Riemann solvers (for analytical solutions and boundary conditions)
    "RiemannState",
    "RiemannSolution",
    "RiemannSolverBase",
    "ExactRiemannSolver",
    "HLLCRiemannSolver",
    "BoundaryRiemannSolver",
    "BoundaryState",
    # Time integration (compatible discretization)
    "CompatibleHeunIntegrator",
    "CompatibleForwardEulerIntegrator",
    "CompatibleSSPRK3Integrator",
    # Artificial viscosity and heat conduction
    "ArtificialViscosity",
    "ArtificialViscosityConfig",
    "ArtificialHeatConduction",
    "ArtificialHeatConductionConfig",
]
