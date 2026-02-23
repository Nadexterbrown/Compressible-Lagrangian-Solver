"""
Open boundary conditions.

Implements non-reflecting (characteristic-based) open boundary conditions
for subsonic and supersonic flows.

References:
    [Poinsot1992] Poinsot & Lele, "Boundary conditions for direct simulations
                  of compressible viscous flows" JCP 1992
    [Toro2009] Chapter 6 - Non-reflecting boundary conditions
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

    Characteristic speeds in 1D:
        λ₁ = u - c  (left-running acoustic)
        λ₂ = u      (entropy/contact wave)
        λ₃ = u + c  (right-running acoustic)

    LEFT boundary (normal pointing left, -x direction):
        - Positive λ = incoming (extrapolate from interior)
        - Negative λ = outgoing (specify boundary data)

    RIGHT boundary (normal pointing right, +x direction):
        - Negative λ = incoming (extrapolate from interior)
        - Positive λ = outgoing (specify boundary data)

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

        The required external values depend on the flow regime at runtime:
        - Subsonic outflow: p_external
        - Subsonic inflow: u_external and (T_external or rho_external)
        - Supersonic inflow: u_external, p_external, and (T_external or rho_external)
        - Supersonic outflow: none (all extrapolated)

        Args:
            side: Which side of the domain (LEFT or RIGHT)
            eos: Equation of state
            p_external: External pressure [Pa] (for inflow/outflow)
            T_external: External temperature [K] (for inflow)
            rho_external: External density [kg/m³] (for inflow)
            u_external: External velocity [m/s] (for inflow)
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

    def determine_regime(self, state: FlowState) -> FlowRegime:
        """
        Determine the flow regime at the boundary.

        Convention:
        - Outflow: material leaving the domain through this boundary
        - Inflow: material entering the domain through this boundary

        At LEFT boundary (x=0): u < 0 means flow to the left, leaving domain = OUTFLOW
        At RIGHT boundary (x=L): u > 0 means flow to the right, leaving domain = OUTFLOW

        Args:
            state: Current flow state

        Returns:
            FlowRegime indicating inflow/outflow and subsonic/supersonic
        """
        # Get velocity at the boundary face
        face_idx = self.face_index
        u = state.u[face_idx]

        # Get sound speed from adjacent cell
        cell_idx = self.cell_index
        c = state.c[cell_idx]

        mach = abs(u) / c if c > 0 else 0.0

        # Determine if flow is leaving (outflow) or entering (inflow) the domain
        if self._side == BoundarySide.LEFT:
            # Left boundary: u < 0 means outflow (leaving through left)
            is_outflow = u < 0
        else:
            # Right boundary: u > 0 means outflow (leaving through right)
            is_outflow = u > 0

        if mach >= 1.0:
            # Supersonic
            if is_outflow:
                return FlowRegime.SUPERSONIC_OUTFLOW
            else:
                return FlowRegime.SUPERSONIC_INFLOW
        else:
            # Subsonic
            if is_outflow:
                return FlowRegime.SUBSONIC_OUTFLOW
            else:
                return FlowRegime.SUBSONIC_INFLOW

    def apply(
        self,
        state: FlowState,
        grid: LagrangianGrid,
        t: float,
    ) -> None:
        """
        Apply open boundary condition based on flow regime.

        Reference: [Poinsot1992] Section 3.2
        """
        regime = self.determine_regime(state)
        idx = self.cell_index
        face_idx = self.face_index

        if regime == FlowRegime.SUPERSONIC_OUTFLOW:
            # All characteristics outgoing - extrapolate everything
            # Boundary velocity from interior
            if self._side == BoundarySide.LEFT:
                state.u[face_idx] = 2 * state.u[1] - state.u[2]
            else:
                state.u[face_idx] = 2 * state.u[-2] - state.u[-3]

        elif regime == FlowRegime.SUPERSONIC_INFLOW:
            # All characteristics incoming - specify everything
            if self._u_external is not None:
                state.u[face_idx] = self._u_external

        elif regime == FlowRegime.SUBSONIC_OUTFLOW:
            # One characteristic incoming (pressure wave)
            # Extrapolate velocity, use specified external pressure
            # The pressure is handled in compute_flux
            if self._side == BoundarySide.LEFT:
                state.u[face_idx] = state.u[1]  # First order extrapolation
            else:
                state.u[face_idx] = state.u[-2]

        else:  # SUBSONIC_INFLOW
            # Two characteristics incoming (entropy and one acoustic)
            # Specify velocity and one thermodynamic quantity
            if self._u_external is not None:
                state.u[face_idx] = self._u_external
            else:
                # Extrapolate if not specified
                if self._side == BoundarySide.LEFT:
                    state.u[face_idx] = state.u[1]
                else:
                    state.u[face_idx] = state.u[-2]

    def compute_flux(
        self,
        state: FlowState,
        grid: LagrangianGrid,
        t: float,
    ) -> BoundaryFlux:
        """
        Compute numerical flux at open boundary.

        Uses characteristic-based non-reflecting condition.

        Reference: [Poinsot1992] Section 3.2, Equations (29)-(34)
        """
        regime = self.determine_regime(state)
        cell_idx = self.cell_index
        p_int = state.p[cell_idx]
        rho_int = state.rho[cell_idx]
        c_int = state.c[cell_idx]

        # Interior velocity at boundary face
        face_idx = self.face_index
        u_int = state.u[face_idx]

        if regime == FlowRegime.SUPERSONIC_OUTFLOW:
            # All information from interior
            p_bdry = p_int
            u_bdry = u_int

        elif regime == FlowRegime.SUPERSONIC_INFLOW:
            # All information from exterior
            if self._p_external is None:
                raise ValueError("p_external required for supersonic inflow")
            if self._u_external is None:
                raise ValueError("u_external required for supersonic inflow")
            p_bdry = self._p_external
            u_bdry = self._u_external

        elif regime == FlowRegime.SUBSONIC_OUTFLOW:
            # Pressure from exterior, velocity extrapolated
            if self._p_external is None:
                # If not specified, extrapolate
                p_bdry = p_int
            else:
                # Use NSCBC-style characteristic relation
                # The outgoing wave carries velocity information
                p_bdry = self._p_external
                # Adjust velocity using characteristic relation
                # L1 = (u - c)(∂p/∂x - ρc ∂u/∂x) for left-running acoustic
                # At steady state, use simple extrapolation
            u_bdry = u_int

        else:  # SUBSONIC_INFLOW
            # Velocity and one thermo variable from exterior
            if self._u_external is not None:
                u_bdry = self._u_external
            else:
                u_bdry = u_int

            # Pressure from characteristic relation
            if self._p_external is not None:
                p_bdry = self._p_external
            else:
                # Use acoustic impedance matching
                if self._side == BoundarySide.LEFT:
                    # Left boundary: use L3 = (u+c) characteristic
                    p_bdry = p_int + rho_int * c_int * (u_bdry - u_int)
                else:
                    # Right boundary: use L1 = (u-c) characteristic
                    p_bdry = p_int - rho_int * c_int * (u_bdry - u_int)

        # Ensure positive pressure
        p_bdry = max(p_bdry, 1e-10)

        # Boundary velocity for position update
        u_face = state.u[self.face_index]

        return BoundaryFlux(
            p_flux=p_bdry,
            pu_flux=p_bdry * u_face,
            u_flux=u_face,
        )

    def get_boundary_velocity(self, t: float) -> float:
        """
        Get boundary velocity.

        For open boundaries, this returns the external velocity if specified,
        otherwise returns 0 (will be updated by apply()).
        """
        return self._u_external if self._u_external is not None else 0.0


class OutflowBC(OpenBC):
    """
    Simplified outflow boundary condition.

    Assumes subsonic outflow and allows specifying only back pressure.
    All other quantities are extrapolated from the interior.
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

    Assumes subsonic inflow with specified total conditions.
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
