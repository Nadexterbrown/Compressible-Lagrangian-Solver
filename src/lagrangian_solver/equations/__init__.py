"""Equations module containing EOS and conservation law implementations."""

from lagrangian_solver.equations.eos import EOSBase, IdealGasEOS, CanteraEOS

# CompatibleConservation is the main conservation class for compatible energy discretization
# Use: from lagrangian_solver.equations.conservation import CompatibleConservation

__all__ = [
    "EOSBase",
    "IdealGasEOS",
    "CanteraEOS",
]
