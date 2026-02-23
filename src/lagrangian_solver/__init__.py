"""
1D Compressible Lagrangian Solver

A robust and accurate Lagrangian fluid dynamics solver for evaluating
compressible flows in one dimension.

References:
    [Toro2009] Toro, E.F. "Riemann Solvers and Numerical Methods for Fluid
               Dynamics" (3rd ed., Springer, 2009)
    [Despres2017] Després, B. "Numerical Methods for Eulerian and Lagrangian
                  Conservation Laws" (Birkhäuser, 2017)
"""

__version__ = "0.1.0"

# Import order matters to avoid circular imports
# EOS first (no internal dependencies)
from lagrangian_solver.equations.eos import IdealGasEOS, CanteraEOS

# Grid next (no internal dependencies)
from lagrangian_solver.core.grid import LagrangianGrid, GridConfig

# State depends on EOS
from lagrangian_solver.core.state import FlowState, ConservedVariables

# Solver and other high-level components can be imported directly when needed
# from lagrangian_solver.core.solver import LagrangianSolver, SolverConfig

__all__ = [
    "FlowState",
    "ConservedVariables",
    "LagrangianGrid",
    "GridConfig",
    "IdealGasEOS",
    "CanteraEOS",
]
