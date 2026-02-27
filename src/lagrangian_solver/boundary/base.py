"""
Base classes for boundary conditions (compatible energy discretization).

This module defines the abstract interface and enumerations for
boundary conditions in the compatible Lagrangian solver.

For compatible energy discretization, boundary conditions must implement:
    - apply_velocity(): Set boundary face velocity
    - apply_momentum(): Set momentum rate d_u at boundary faces

The momentum rate uses cell-centered stress σ = p + Q, which is
consistent with the interior discretization.

References:
    [Poinsot1992] Poinsot & Lele, "Boundary conditions for direct simulations
                  of compressible viscous flows" JCP 1992
    [Toro2009] Chapter 6 - Boundary conditions
    [Caramana1998] Section 4 - Boundary conditions for compatible hydrodynamics
"""

from abc import ABC, abstractmethod
from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional, Callable, Tuple
import numpy as np

from lagrangian_solver.core.state import FlowState
from lagrangian_solver.core.grid import LagrangianGrid
from lagrangian_solver.equations.eos import EOSBase


class BoundarySide(Enum):
    """Which side of the domain the boundary is on."""

    LEFT = auto()
    RIGHT = auto()


class ThermalBCType(Enum):
    """Type of thermal boundary condition."""

    ADIABATIC = auto()  # Zero heat flux: ∂T/∂n = 0
    ISOTHERMAL = auto()  # Fixed temperature: T = T_wall


class BoundaryType(Enum):
    """Type of boundary condition."""

    SOLID_WALL = auto()  # Stationary solid wall
    MOVING_PISTON = auto()  # Moving solid boundary
    OPEN = auto()  # Open boundary (subsonic/supersonic)
    PERIODIC = auto()  # Periodic boundary


@dataclass
class BoundaryFlux:
    """
    Boundary fluxes for a single boundary (legacy interface).

    Attributes:
        p_flux: Pressure flux (for momentum equation) [Pa]
        pu_flux: Energy flux [W/m²]
        u_flux: Velocity at boundary (for position update) [m/s]
    """

    p_flux: float
    pu_flux: float
    u_flux: float


class BoundaryCondition(ABC):
    """
    Abstract base class for all boundary conditions.

    For compatible energy discretization, each boundary condition must implement:
    1. apply_velocity(): Set the velocity at the boundary face
    2. apply_momentum(): Set the momentum rate d_u at boundary faces

    The legacy interface (apply, compute_flux) is retained for backward
    compatibility but should not be used with CompatibleConservation.

    Reference: [Caramana1998] Section 4
    """

    def __init__(self, side: BoundarySide, eos: EOSBase):
        """
        Initialize boundary condition.

        Args:
            side: Which side of the domain (LEFT or RIGHT)
            eos: Equation of state
        """
        self._side = side
        self._eos = eos

    @property
    def side(self) -> BoundarySide:
        """Side of the domain."""
        return self._side

    @property
    def eos(self) -> EOSBase:
        """Equation of state."""
        return self._eos

    @property
    def face_index(self) -> int:
        """
        Face index for this boundary.

        Returns:
            0 for LEFT, -1 (last face) for RIGHT
        """
        return 0 if self._side == BoundarySide.LEFT else -1

    @property
    def cell_index(self) -> int:
        """
        Adjacent cell index for this boundary.

        Returns:
            0 for LEFT, -1 (last cell) for RIGHT
        """
        return 0 if self._side == BoundarySide.LEFT else -1

    # ==================== Compatible Interface ====================
    # These methods are used by CompatibleConservation

    @abstractmethod
    def apply_velocity(
        self,
        state: FlowState,
        grid: LagrangianGrid,
        t: float,
    ) -> None:
        """
        Set the velocity at the boundary face.

        This is called before computing the RHS to ensure boundary
        velocities are correctly set for computing d_tau.

        Args:
            state: Flow state to modify (u at boundary face)
            grid: Lagrangian grid
            t: Current time [s]
        """
        pass

    @abstractmethod
    def apply_momentum(
        self,
        state: FlowState,
        grid: LagrangianGrid,
        d_u: np.ndarray,
        sigma: np.ndarray,
        t: float,
    ) -> None:
        """
        Set the momentum rate d_u at the boundary face.

        For compatible discretization, this uses the same cell-centered
        stress σ = p + Q as the interior, ensuring energy conservation.

        Args:
            state: Current flow state
            grid: Lagrangian grid
            d_u: Momentum rate array (modified in-place at boundary)
            sigma: Cell-centered total stress σ = p + Q [Pa]
            t: Current time [s]
        """
        pass

    @abstractmethod
    def get_boundary_velocity(self, t: float) -> float:
        """
        Get the boundary velocity at time t.

        For stationary walls this returns 0.
        For moving boundaries this returns the prescribed velocity.

        Args:
            t: Current time [s]

        Returns:
            Boundary velocity [m/s]
        """
        pass

    # ==================== Legacy Interface ====================
    # These methods are kept for backward compatibility

    def apply(
        self,
        state: FlowState,
        grid: LagrangianGrid,
        t: float,
    ) -> None:
        """
        LEGACY: Apply boundary condition to the state.

        For compatible discretization, use apply_velocity() instead.
        """
        # Default: delegate to apply_velocity
        self.apply_velocity(state, grid, t)

    def compute_flux(
        self,
        state: FlowState,
        grid: LagrangianGrid,
        t: float,
    ) -> BoundaryFlux:
        """
        LEGACY: Compute the numerical flux at the boundary.

        For compatible discretization, use apply_momentum() instead.
        This method is retained for backward compatibility with the
        old Riemann-based solver.
        """
        # Default implementation using cell pressure
        p_wall = state.p[self.cell_index]
        u_boundary = state.u[self.face_index]
        return BoundaryFlux(
            p_flux=p_wall,
            pu_flux=p_wall * u_boundary,
            u_flux=u_boundary,
        )

    def update_position(
        self, grid: LagrangianGrid, dt: float, t: float
    ) -> float:
        """
        Compute the new boundary position after time step.

        Args:
            grid: Lagrangian grid
            dt: Time step [s]
            t: Current time [s]

        Returns:
            New position of the boundary [m]
        """
        current_x = grid.x[self.face_index]
        velocity = self.get_boundary_velocity(t)
        return current_x + dt * velocity


class ReflectiveBC(BoundaryCondition):
    """
    Simple reflective (solid wall) boundary condition.

    Reflects velocity while maintaining pressure and density.
    The wall is stationary with u = 0.

    For compatible discretization:
        - Velocity: u[boundary] = 0
        - Momentum: d_u[boundary] computed from wall pressure
    """

    def __init__(self, side: BoundarySide, eos: EOSBase):
        super().__init__(side, eos)

    def apply_velocity(
        self,
        state: FlowState,
        grid: LagrangianGrid,
        t: float,
    ) -> None:
        """Set u = 0 at wall."""
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
        Set momentum rate at stationary wall.

        At a stationary wall, d_u = 0 (velocity stays at zero).
        """
        d_u[self.face_index] = 0.0

    def get_boundary_velocity(self, t: float) -> float:
        return 0.0

    def compute_flux(
        self,
        state: FlowState,
        grid: LagrangianGrid,
        t: float,
    ) -> BoundaryFlux:
        """
        Compute flux for reflective boundary.

        At a reflective wall:
        - Pressure flux = wall pressure (from adjacent cell)
        - Energy flux = 0 (since u = 0)
        - Velocity = 0
        """
        p_wall = state.p[self.cell_index]
        return BoundaryFlux(p_flux=p_wall, pu_flux=0.0, u_flux=0.0)
