"""Core module containing grid, state, and solver classes."""

from lagrangian_solver.core.grid import LagrangianGrid, GridConfig
from lagrangian_solver.core.state import FlowState, ConservedVariables

# Solver imported lazily to avoid circular imports
# Use: from lagrangian_solver.core.solver import LagrangianSolver, SolverConfig

__all__ = [
    "FlowState",
    "ConservedVariables",
    "LagrangianGrid",
    "GridConfig",
]
