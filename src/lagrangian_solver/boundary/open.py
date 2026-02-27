"""
Open boundary conditions for compatible energy discretization.

Implements non-reflecting (characteristic-based) open boundary conditions
for subsonic and supersonic flows.

For compatible discretization:
    - apply_velocity(): Sets boundary velocity based on flow regime
    - apply_momentum(): Computes d_u from pressure difference

References:
    [Poinsot1992] Poinsot & Lele, "Boundary conditions for direct simulations
                  of compressible viscous flows" JCP 1992
    [Toro2009] Chapter 6 - Non-reflecting boundary conditions
    [Caramana1998] Section 4 - Boundary conditions
"""

from enum import Enum, auto
from typing import Optional, Tuple
import numpy as np

from lagrangian_solver.boundary.base import (
    BoundaryCondition,
    BoundarySide,
    BoundaryFlux,
)
from lagrangian_solver.core.state import FlowState
from lagrangian_solver.core.grid import LagrangianGrid
from lagrangian_solver.equations.eos import EOSBase


class FlowRegime(Enum):
    """Flow regime at the boundary."""

    SUBSONIC_INFLOW = auto()  # 2 incoming characteristics (specify u, T or ρ)
    SUBSONIC_OUTFLOW = auto()  # 1 incoming characteristic (specify p)
    SUPERSONIC_INFLOW = auto()  # 3 incoming (specify all)
    SUPERSONIC_OUTFLOW = auto()  # 0 incoming (extrapolate all)


class OpenBC(BoundaryCondition):
    """
    Open (non-reflecting) boundary condition.

    Uses characteristic analysis to determine which variables should be
    specified (incoming characteristics) and which should be extrapolated
    (outgoing characteristics).

    For compatible discretization:
    - apply_velocity(): Sets boundary velocity based on regime
    - apply_momentum(): Computes d_u from pressure difference

    Reference: [Poinsot1992] Section 3, [Toro2009] Section 6.4
    """

    def __init__(
        self,
        side: BoundarySide,
        eos: EOSBase,
        p_external: Optional[float] = None,
        T_external: Optional[float] = None,
        rho_external: Optional[float] = None,
        u_external: Optional[float] = None,
    ):
        """
        Initialize open boundary condition.

        Args:
            side: Which side of the domain (LEFT or RIGHT)
            eos: Equation of state
            p_external: External pressure [Pa]
            T_external: External temperature [K]
            rho_external: External density [kg/m³]
            u_external: External velocity [m/s]
        """
        super().__init__(side, eos)
        self._p_external = p_external
        self._T_external = T_external
        self._rho_external = rho_external
        self._u_external = u_external

    @property
    def p_external(self) -> Optional[float]:
        """External pressure [Pa]."""
        return self._p_external

    @property
    def T_external(self) -> Optional[float]:
        """External temperature [K]."""
        return self._T_external

    @property
    def rho_external(self) -> Optional[float]:
        """External density [kg/m³]."""
        return self._rho_external

    @property
    def u_external(self) -> Optional[float]:
        """External velocity [m/s]."""
        return self._u_external

    def get_boundary_velocity(self, t: float) -> float:
        """Get boundary velocity (external velocity if specified)."""
        return self._u_external if self._u_external is not None else 0.0

    def determine_regime(self, state: FlowState) -> FlowRegime:
        """
        Determine the flow regime at the boundary.

        Args:
            state: Current flow state

        Returns:
            FlowRegime indicating inflow/outflow and subsonic/supersonic
        """
        face_idx = self.face_index
        u = state.u[face_idx]

        cell_idx = self.cell_index
        c = state.c[cell_idx]

        mach = abs(u) / c if c > 0 else 0.0

        if self._side == BoundarySide.LEFT:
            is_outflow = u < 0
        else:
            is_outflow = u > 0

        if mach >= 1.0:
            if is_outflow:
                return FlowRegime.SUPERSONIC_OUTFLOW
            else:
                return FlowRegime.SUPERSONIC_INFLOW
        else:
            if is_outflow:
                return FlowRegime.SUBSONIC_OUTFLOW
            else:
                return FlowRegime.SUBSONIC_INFLOW

    def apply_velocity(
        self,
        state: FlowState,
        grid: LagrangianGrid,
        t: float,
    ) -> None:
        """
        Set boundary velocity based on flow regime.

        Reference: [Poinsot1992] Section 3.2
        """
        regime = self.determine_regime(state)
        face_idx = self.face_index

        if regime == FlowRegime.SUPERSONIC_OUTFLOW:
            # Extrapolate from interior
            if self._side == BoundarySide.LEFT:
                state.u[face_idx] = 2 * state.u[1] - state.u[2]
            else:
                state.u[face_idx] = 2 * state.u[-2] - state.u[-3]

        elif regime == FlowRegime.SUPERSONIC_INFLOW:
            # Use specified external velocity
            if self._u_external is not None:
                state.u[face_idx] = self._u_external

        elif regime == FlowRegime.SUBSONIC_OUTFLOW:
            # Extrapolate velocity
            if self._side == BoundarySide.LEFT:
                state.u[face_idx] = state.u[1]
            else:
                state.u[face_idx] = state.u[-2]

        else:  # SUBSONIC_INFLOW
            if self._u_external is not None:
                state.u[face_idx] = self._u_external
            else:
                if self._side == BoundarySide.LEFT:
                    state.u[face_idx] = state.u[1]
                else:
                    state.u[face_idx] = state.u[-2]

    def apply_momentum(
        self,
        state: FlowState,
        grid: LagrangianGrid,
        d_u: np.ndarray,
        sigma: np.ndarray,
        t: float,
    ) -> None:
        """
        Compute d_u at open boundary.

        Uses the cell-centered stress σ for consistency with interior.
        The boundary pressure may be specified externally (for inflow/outflow).

        Args:
            state: Current flow state
            grid: Lagrangian grid
            d_u: Momentum rate array (modified at boundary)
            sigma: Cell-centered total stress [Pa]
            t: Current time [s]
        """
        dm = grid.dm
        face_idx = self.face_index
        cell_idx = self.cell_index

        regime = self.determine_regime(state)

        if regime == FlowRegime.SUPERSONIC_OUTFLOW:
            # All information from interior - extrapolate d_u
            # Use the interior stress difference pattern
            if self._side == BoundarySide.LEFT:
                # d_u[0] based on extrapolated stress
                dm_face = 0.5 * dm[0]
                # Use interior stress, assume ghost cell has same stress
                d_u[face_idx] = -(sigma[0] - sigma[0]) / dm_face  # = 0
            else:
                dm_face = 0.5 * dm[-1]
                d_u[face_idx] = -(sigma[-1] - sigma[-1]) / dm_face  # = 0

        elif regime == FlowRegime.SUPERSONIC_INFLOW:
            # All information from exterior
            # d_u is determined by external state
            if self._u_external is not None:
                # Velocity is prescribed, d_u = 0 for constant inflow
                d_u[face_idx] = 0.0
            else:
                d_u[face_idx] = 0.0

        elif regime == FlowRegime.SUBSONIC_OUTFLOW:
            # Pressure from exterior, velocity extrapolated
            if self._p_external is not None:
                p_ext = self._p_external
            else:
                p_ext = state.p[cell_idx]

            # Compute d_u using external pressure as "ghost" stress
            if self._side == BoundarySide.LEFT:
                dm_face = 0.5 * dm[0]
                # Ghost stress = external pressure (no AV outside domain)
                d_u[face_idx] = -(sigma[0] - p_ext) / dm_face
            else:
                dm_face = 0.5 * dm[-1]
                d_u[face_idx] = -(p_ext - sigma[-1]) / dm_face

        else:  # SUBSONIC_INFLOW
            # Velocity specified, pressure from characteristic
            if self._p_external is not None:
                p_ext = self._p_external
            else:
                # Use acoustic approximation
                rho = state.rho[cell_idx]
                c = state.c[cell_idx]
                u_int = state.u[face_idx]
                u_ext = self._u_external if self._u_external else u_int

                if self._side == BoundarySide.LEFT:
                    p_ext = state.p[cell_idx] + rho * c * (u_ext - u_int)
                else:
                    p_ext = state.p[cell_idx] - rho * c * (u_ext - u_int)

            # Compute d_u
            if self._side == BoundarySide.LEFT:
                dm_face = 0.5 * dm[0]
                d_u[face_idx] = -(sigma[0] - p_ext) / dm_face
            else:
                dm_face = 0.5 * dm[-1]
                d_u[face_idx] = -(p_ext - sigma[-1]) / dm_face

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
        """LEGACY: Compute flux for open boundary."""
        regime = self.determine_regime(state)
        cell_idx = self.cell_index
        p_int = state.p[cell_idx]
        u_face = state.u[self.face_index]

        if regime in [FlowRegime.SUPERSONIC_OUTFLOW, FlowRegime.SUBSONIC_OUTFLOW]:
            p_bdry = self._p_external if self._p_external else p_int
        else:
            p_bdry = self._p_external if self._p_external else p_int

        return BoundaryFlux(
            p_flux=p_bdry,
            pu_flux=p_bdry * u_face,
            u_flux=u_face,
        )


class OutflowBC(OpenBC):
    """
    Simplified outflow boundary condition.

    Assumes subsonic outflow with specified back pressure.
    """

    def __init__(
        self,
        side: BoundarySide,
        eos: EOSBase,
        back_pressure: float,
    ):
        """
        Initialize outflow boundary condition.

        Args:
            side: Which side of the domain (LEFT or RIGHT)
            eos: Equation of state
            back_pressure: Back pressure [Pa]
        """
        super().__init__(side, eos, p_external=back_pressure)

    @property
    def back_pressure(self) -> float:
        """Back pressure [Pa]."""
        return self._p_external


class InflowBC(OpenBC):
    """
    Simplified inflow boundary condition.

    Assumes subsonic inflow with specified conditions.
    """

    def __init__(
        self,
        side: BoundarySide,
        eos: EOSBase,
        velocity: float,
        temperature: float,
        pressure: float,
    ):
        """
        Initialize inflow boundary condition.

        Args:
            side: Which side of the domain (LEFT or RIGHT)
            eos: Equation of state
            velocity: Inflow velocity [m/s]
            temperature: Inflow temperature [K]
            pressure: Inflow pressure [Pa]
        """
        super().__init__(
            side,
            eos,
            p_external=pressure,
            T_external=temperature,
            u_external=velocity,
        )

    @property
    def inflow_velocity(self) -> float:
        """Inflow velocity [m/s]."""
        return self._u_external

    @property
    def inflow_temperature(self) -> float:
        """Inflow temperature [K]."""
        return self._T_external

    @property
    def inflow_pressure(self) -> float:
        """Inflow pressure [Pa]."""
        return self._p_external
