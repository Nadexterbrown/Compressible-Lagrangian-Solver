"""
Base classes for boundary conditions.

This module defines the abstract interface and enumerations for
boundary conditions in the Lagrangian solver.

References:
    [Poinsot1992] Poinsot & Lele, "Boundary conditions for direct simulations
                  of compressible viscous flows" JCP 1992
    [Toro2009] Chapter 6 - Boundary conditions
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
    Boundary fluxes for a single boundary.

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

    Each boundary condition must implement:
    1. apply(): Set ghost cell / boundary face values
    2. compute_flux(): Return the numerical flux at the boundary

    Reference: [Poinsot1992] for NSCBC-based boundary conditions
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

    @abstractmethod
    def apply(
        self,
        state: FlowState,
        grid: LagrangianGrid,
        t: float,
    ) -> None:
        """
        Apply boundary condition to the state.

        This method modifies the state in-place to enforce the
        boundary condition at the current time.

        Args:
            state: Flow state to modify
            grid: Lagrangian grid
            t: Current time [s]
        """
        pass

    @abstractmethod
    def compute_flux(
        self,
        state: FlowState,
        grid: LagrangianGrid,
        t: float,
    ) -> BoundaryFlux:
        """
        Compute the numerical flux at the boundary.

        Args:
            state: Current flow state
            grid: Lagrangian grid
            t: Current time [s]

        Returns:
            BoundaryFlux with pressure, energy, and velocity fluxes
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
    Simple reflective boundary condition.

    Reflects velocity while maintaining pressure and density.
    Useful for testing and simple configurations.
    """

    def __init__(self, side: BoundarySide, eos: EOSBase):
        super().__init__(side, eos)

    def apply(
        self,
        state: FlowState,
        grid: LagrangianGrid,
        t: float,
    ) -> None:
        """Apply reflective BC: u = 0 at wall."""
        state.u[self.face_index] = 0.0

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

    def get_boundary_velocity(self, t: float) -> float:
        return 0.0
