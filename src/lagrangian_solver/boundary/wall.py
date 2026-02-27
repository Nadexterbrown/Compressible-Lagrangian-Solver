"""
Solid wall boundary conditions for compatible energy discretization.

Implements stationary solid wall boundaries with adiabatic or isothermal
thermal conditions.

For compatible discretization:
    - apply_velocity(): Sets u = 0 at wall
    - apply_momentum(): Sets d_u = 0 at wall (velocity stays at zero)

References:
    [Toro2009] Section 6.3 - Reflective boundary conditions
    [Poinsot1992] Section 4 - Wall boundary conditions
    [Caramana1998] Section 4 - Boundary conditions for compatible hydrodynamics
"""

from typing import Optional
import numpy as np

from lagrangian_solver.boundary.base import (
    BoundaryCondition,
    BoundarySide,
    ThermalBCType,
    BoundaryFlux,
)
from lagrangian_solver.core.state import FlowState
from lagrangian_solver.core.grid import LagrangianGrid
from lagrangian_solver.equations.eos import EOSBase


class SolidWallBC(BoundaryCondition):
    """
    Stationary solid wall boundary condition.

    At a solid wall:
    - Velocity: u = 0 (no penetration)
    - Thermal: Either adiabatic (∂T/∂n = 0) or isothermal (T = T_wall)

    For compatible discretization:
    - apply_velocity(): Sets u[boundary] = 0
    - apply_momentum(): Sets d_u[boundary] = 0

    Reference: [Toro2009] Section 6.3, [Caramana1998] Section 4
    """

    def __init__(
        self,
        side: BoundarySide,
        eos: EOSBase,
        thermal_bc: ThermalBCType = ThermalBCType.ADIABATIC,
        wall_temperature: Optional[float] = None,
    ):
        """
        Initialize solid wall boundary condition.

        Args:
            side: Which side of the domain (LEFT or RIGHT)
            eos: Equation of state
            thermal_bc: Type of thermal boundary condition
            wall_temperature: Wall temperature [K] (required for isothermal)
        """
        super().__init__(side, eos)
        self._thermal_bc = thermal_bc
        self._wall_temperature = wall_temperature

        if thermal_bc == ThermalBCType.ISOTHERMAL and wall_temperature is None:
            raise ValueError("wall_temperature required for isothermal BC")

    @property
    def thermal_bc(self) -> ThermalBCType:
        """Type of thermal boundary condition."""
        return self._thermal_bc

    @property
    def wall_temperature(self) -> Optional[float]:
        """Wall temperature for isothermal BC [K]."""
        return self._wall_temperature

    def get_boundary_velocity(self, t: float) -> float:
        """Solid wall velocity is always zero."""
        return 0.0

    def apply_velocity(
        self,
        state: FlowState,
        grid: LagrangianGrid,
        t: float,
    ) -> None:
        """
        Set u = 0 at the wall.

        Reference: [Caramana1998] Section 4
        """
        state.u[self.face_index] = 0.0

    def apply_momentum(
        self,
        state: FlowState,
        grid: LagrangianGrid,
        d_u: np.ndarray,
        sigma: np.ndarray,
        t: float,
    ) -> None:
        """
        Set d_u = 0 at stationary wall.

        At a stationary wall, the velocity must remain zero,
        so the momentum rate is zero.
        """
        d_u[self.face_index] = 0.0

    # Legacy interface
    def apply(
        self,
        state: FlowState,
        grid: LagrangianGrid,
        t: float,
    ) -> None:
        """LEGACY: Use apply_velocity()."""
        self.apply_velocity(state, grid, t)

    def compute_flux(
        self,
        state: FlowState,
        grid: LagrangianGrid,
        t: float,
    ) -> BoundaryFlux:
        """
        LEGACY: Compute numerical flux at solid wall.

        At a stationary solid wall:
        - Pressure flux from adjacent cell
        - Velocity = 0, so energy flux = 0
        """
        p_wall = state.p[self.cell_index]
        return BoundaryFlux(
            p_flux=p_wall,
            pu_flux=0.0,
            u_flux=0.0,
        )

    def get_wall_pressure(
        self, state: FlowState, grid: LagrangianGrid
    ) -> float:
        """
        Get the pressure at the wall (for diagnostics).

        Args:
            state: Current flow state
            grid: Lagrangian grid

        Returns:
            Wall pressure [Pa]
        """
        return state.p[self.cell_index]

    def get_wall_heat_flux(
        self, state: FlowState, grid: LagrangianGrid
    ) -> float:
        """
        Get heat flux at the wall.

        For inviscid Euler equations, this is always zero.

        Args:
            state: Current flow state
            grid: Lagrangian grid

        Returns:
            Heat flux [W/m²] (always 0 for inviscid)
        """
        return 0.0


class SymmetryBC(SolidWallBC):
    """
    Symmetry boundary condition.

    Identical to adiabatic solid wall for 1D simulations.
    Provided as a semantic alias for clarity in symmetric problems.
    """

    def __init__(self, side: BoundarySide, eos: EOSBase):
        """
        Initialize symmetry boundary condition.

        Args:
            side: Which side of the domain (LEFT or RIGHT)
            eos: Equation of state
        """
        super().__init__(side, eos, ThermalBCType.ADIABATIC)
