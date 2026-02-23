"""
Solid wall boundary conditions.

Implements stationary solid wall boundaries with adiabatic or isothermal
thermal conditions.

References:
    [Toro2009] Section 6.3 - Reflective boundary conditions
    [Poinsot1992] Section 4 - Wall boundary conditions
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

    For inviscid Euler equations, the adiabatic and isothermal conditions
    are equivalent since there is no thermal diffusion. The distinction
    becomes important for viscous simulations or when coupling with heat
    transfer.

    Reference: [Toro2009] Section 6.3, [Poinsot1992] Section 4.2
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

    def apply(
        self,
        state: FlowState,
        grid: LagrangianGrid,
        t: float,
    ) -> None:
        """
        Apply solid wall boundary condition.

        Sets:
        - u = 0 at the wall face
        - For isothermal: adjusts temperature (affects EOS calculations)

        Reference: [Poinsot1992] Equations (4.5)-(4.7)
        """
        # Enforce no-penetration condition
        state.u[self.face_index] = 0.0

        # For isothermal BC, we would need to modify the temperature
        # at the adjacent cell to enforce the wall temperature.
        # This is more relevant for viscous flows with heat diffusion.
        # For inviscid flows, we just note the wall temperature for output.

    def compute_flux(
        self,
        state: FlowState,
        grid: LagrangianGrid,
        t: float,
    ) -> BoundaryFlux:
        """
        Compute numerical flux at solid wall.

        At a stationary solid wall:
        - The wall pressure is computed from the adjacent cell
          using acoustic impedance matching
        - Velocity is zero, so energy flux is zero

        For Lagrangian formulation with wall at rest:
            p_wall = p_adjacent (for inviscid flow)

        Reference: [Despres2017] Section 4.3
        """
        # Get adjacent cell state
        idx = self.cell_index
        p_adj = state.p[idx]
        rho_adj = state.rho[idx]
        c_adj = state.c[idx]

        # For a stationary wall, use acoustic relation to find wall pressure
        # In characteristic form: p - p_adj = ±ρc(u - u_adj)
        # With u_wall = 0:
        if self._side == BoundarySide.LEFT:
            # Wave going into domain from left wall
            # Use p = p_adj + ρc * u_adj (right-running wave reflected)
            u_adj = 0.5 * (state.u[0] + state.u[1])
            p_wall = p_adj + rho_adj * c_adj * u_adj
        else:
            # Wave going into domain from right wall
            # Use p = p_adj - ρc * u_adj (left-running wave reflected)
            u_adj = 0.5 * (state.u[-2] + state.u[-1])
            p_wall = p_adj - rho_adj * c_adj * u_adj

        # Ensure positive pressure
        p_wall = max(p_wall, 1e-10)

        # At stationary wall: u = 0, so energy flux = p * u = 0
        return BoundaryFlux(
            p_flux=p_wall,
            pu_flux=0.0,
            u_flux=0.0,
        )

    def get_boundary_velocity(self, t: float) -> float:
        """Solid wall velocity is always zero."""
        return 0.0

    def get_wall_pressure(
        self, state: FlowState, grid: LagrangianGrid
    ) -> float:
        """
        Get the pressure at the wall.

        Useful for diagnostics and force calculations.

        Args:
            state: Current flow state
            grid: Lagrangian grid

        Returns:
            Wall pressure [Pa]
        """
        flux = self.compute_flux(state, grid, 0.0)
        return flux.p_flux

    def get_wall_heat_flux(
        self, state: FlowState, grid: LagrangianGrid
    ) -> float:
        """
        Get heat flux at the wall.

        For inviscid Euler equations, this is always zero.
        Included for interface consistency with viscous solvers.

        Args:
            state: Current flow state
            grid: Lagrangian grid

        Returns:
            Heat flux [W/m²] (always 0 for inviscid)
        """
        if self._thermal_bc == ThermalBCType.ADIABATIC:
            return 0.0
        else:
            # For viscous flows, would compute from temperature gradient
            # q = -k * dT/dn
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
