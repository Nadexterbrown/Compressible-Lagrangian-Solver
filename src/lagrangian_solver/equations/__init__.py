"""Equations module containing EOS and conservation law implementations."""

from lagrangian_solver.equations.eos import EOSBase, IdealGasEOS, CanteraEOS

# LagrangianConservation imported lazily to avoid circular imports
# Use: from lagrangian_solver.equations.conservation import LagrangianConservation

__all__ = [
    "EOSBase",
    "IdealGasEOS",
    "CanteraEOS",
]
