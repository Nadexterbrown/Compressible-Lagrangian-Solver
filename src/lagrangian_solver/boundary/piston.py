"""
Moving piston boundary conditions.

Implements moving solid boundaries with prescribed velocity, including
adiabatic, isothermal, and porous conditions.

References:
    [Toro2009] Section 6.3.2 - Moving boundaries
    [BeaversJoseph1967] Beavers & Joseph, "Boundary conditions at a naturally
                        permeable wall" JFM 1967 - Porous boundary conditions
"""

from typing import Optional, Callable, Union
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


# Type alias for velocity profile function
VelocityProfile = Callable[[float], float]


class MovingPistonBC(BoundaryCondition):
    """
    Moving piston boundary condition.

    The piston moves with a prescribed velocity v_piston(t).
    The boundary can be:
    - Adiabatic: No heat transfer through piston
    - Isothermal: Piston maintains fixed temperature
    - Porous: Allows some mass flux through (Darcy flow with slip)

    Reference: [Toro2009] Section 6.3.2, [Despres2017] Section 4.3.2
    """

    def __init__(
        self,
        side: BoundarySide,
        eos: EOSBase,
        velocity: Union[float, VelocityProfile],
        thermal_bc: ThermalBCType = ThermalBCType.ADIABATIC,
        piston_temperature: Optional[float] = None,
        porous: bool = False,
        permeability: float = 0.0,
        slip_coefficient: float = 0.0,
    ):
        """
        Initialize moving piston boundary condition.

        Args:
            side: Which side of the domain (LEFT or RIGHT)
            eos: Equation of state
            velocity: Piston velocity [m/s] or function v(t)
                      Positive = moving right, Negative = moving left
            thermal_bc: Type of thermal boundary condition
            piston_temperature: Piston temperature [K] (for isothermal)
            porous: Whether piston is porous
            permeability: Darcy permeability K [m²] (for porous)
            slip_coefficient: Beavers-Joseph slip coefficient α_BJ [-]

        Reference: [BeaversJoseph1967] for porous BC parameters
        """
        super().__init__(side, eos)

        # Velocity profile
        if callable(velocity):
            self._velocity_func = velocity
        else:
            self._velocity_func = lambda t: velocity

        self._thermal_bc = thermal_bc
        self._piston_temperature = piston_temperature

        # Porous parameters
        self._porous = porous
        self._permeability = permeability  # K [m²]
        self._slip_coefficient = slip_coefficient  # α_BJ [-]

        if thermal_bc == ThermalBCType.ISOTHERMAL and piston_temperature is None:
            raise ValueError("piston_temperature required for isothermal BC")

        if porous and (permeability <= 0 or slip_coefficient <= 0):
            raise ValueError(
                "Positive permeability and slip_coefficient required for porous BC"
            )

    @property
    def thermal_bc(self) -> ThermalBCType:
        """Type of thermal boundary condition."""
        return self._thermal_bc

    @property
    def piston_temperature(self) -> Optional[float]:
        """Piston temperature for isothermal BC [K]."""
        return self._piston_temperature

    @property
    def is_porous(self) -> bool:
        """Whether piston is porous."""
        return self._porous

    @property
    def permeability(self) -> float:
        """Darcy permeability [m²]."""
        return self._permeability

    @property
    def slip_coefficient(self) -> float:
        """Beavers-Joseph slip coefficient [-]."""
        return self._slip_coefficient

    def get_boundary_velocity(self, t: float) -> float:
        """Get piston velocity at time t."""
        return self._velocity_func(t)

    def apply(
        self,
        state: FlowState,
        grid: LagrangianGrid,
        t: float,
    ) -> None:
        """
        Apply moving piston boundary condition.

        Sets boundary face velocity to match piston velocity,
        with possible correction for porous flow.

        Reference: [BeaversJoseph1967] Equation (1) for porous slip
        """
        v_piston = self.get_boundary_velocity(t)

        if self._porous:
            # Beavers-Joseph slip condition
            # u_slip - u_piston = (sqrt(K)/α_BJ) * du/dy
            # For 1D, approximate du/dy from adjacent cell
            idx = self.cell_index
            u_adj = 0.5 * (state.u[abs(idx)] + state.u[abs(idx) + 1])

            # Slip velocity correction
            sqrt_K = np.sqrt(self._permeability)
            slip_factor = sqrt_K / self._slip_coefficient

            # Estimate velocity gradient (simplified for 1D)
            dx = state.dx[idx]
            du_dn = (u_adj - v_piston) / dx

            u_boundary = v_piston + slip_factor * du_dn
        else:
            u_boundary = v_piston

        state.u[self.face_index] = u_boundary

    def compute_flux(
        self,
        state: FlowState,
        grid: LagrangianGrid,
        t: float,
    ) -> BoundaryFlux:
        """
        Compute numerical flux at moving piston.

        Uses acoustic impedance matching to determine wall pressure:
            p_wall = p_adj ± ρc(u_wall - u_adj)

        Reference: [Toro2009] Section 6.3.2, Equation (6.17)
        """
        v_piston = self.get_boundary_velocity(t)

        # Get adjacent cell state
        idx = self.cell_index
        p_adj = state.p[idx]
        rho_adj = state.rho[idx]
        c_adj = state.c[idx]

        # Cell-averaged velocity
        if self._side == BoundarySide.LEFT:
            u_adj = 0.5 * (state.u[0] + state.u[1])
            # Left boundary: right-running wave reflected from piston
            p_wall = p_adj + rho_adj * c_adj * (v_piston - u_adj)
        else:
            u_adj = 0.5 * (state.u[-2] + state.u[-1])
            # Right boundary: left-running wave reflected from piston
            p_wall = p_adj - rho_adj * c_adj * (v_piston - u_adj)

        # For porous piston, add Darcy pressure correction
        if self._porous:
            # Darcy law: ṁ = -(ρK/μ) ∂p/∂n
            # This adds a pressure drop across the porous interface
            # Simplified model: assume dynamic viscosity μ from EOS
            # For ideal gas, μ ~ 1.8e-5 Pa·s for air
            mu = 1.8e-5  # Simplified, could use Sutherland's law

            # Mass flux through porous interface
            dp_dn = (p_adj - p_wall) / state.dx[idx]  # Approximate gradient
            mass_flux = -rho_adj * self._permeability / mu * dp_dn

            # Adjust wall pressure based on mass flux
            # This is a simplified coupling
            p_wall = p_wall - 0.5 * mass_flux * c_adj

        # Ensure positive pressure
        p_wall = max(p_wall, 1e-10)

        # Get actual boundary velocity (may differ from piston if porous)
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
        Compute new piston position.

        For non-porous piston, position updates with piston velocity.
        For porous piston, there may be slight deviation due to slip.
        """
        current_x = grid.x[self.face_index]
        return current_x + dt * self.get_boundary_velocity(t)

    def get_piston_position(self, x0: float, t: float) -> float:
        """
        Get analytical piston position at time t.

        Integrates the velocity profile from 0 to t.
        For constant velocity, this is x0 + v*t.

        Args:
            x0: Initial piston position [m]
            t: Current time [s]

        Returns:
            Piston position [m]
        """
        # Numerical integration for general velocity profiles
        from scipy import integrate

        result, _ = integrate.quad(self._velocity_func, 0, t)
        return x0 + result


def sinusoidal_piston(
    amplitude: float, frequency: float, phase: float = 0.0
) -> VelocityProfile:
    """
    Create a sinusoidal piston velocity profile.

    v(t) = amplitude * sin(2π * frequency * t + phase)

    Args:
        amplitude: Velocity amplitude [m/s]
        frequency: Oscillation frequency [Hz]
        phase: Phase offset [rad]

    Returns:
        Velocity profile function
    """

    def velocity(t: float) -> float:
        return amplitude * np.sin(2 * np.pi * frequency * t + phase)

    return velocity


def ramp_piston(
    v_start: float, v_end: float, t_ramp: float
) -> VelocityProfile:
    """
    Create a linear ramp piston velocity profile.

    v(t) = v_start + (v_end - v_start) * t / t_ramp  for t < t_ramp
    v(t) = v_end                                      for t >= t_ramp

    Args:
        v_start: Initial velocity [m/s]
        v_end: Final velocity [m/s]
        t_ramp: Ramp duration [s]

    Returns:
        Velocity profile function
    """

    def velocity(t: float) -> float:
        if t < t_ramp:
            return v_start + (v_end - v_start) * t / t_ramp
        else:
            return v_end

    return velocity


def step_piston(v_before: float, v_after: float, t_step: float) -> VelocityProfile:
    """
    Create a step change piston velocity profile.

    v(t) = v_before  for t < t_step
    v(t) = v_after   for t >= t_step

    Args:
        v_before: Velocity before step [m/s]
        v_after: Velocity after step [m/s]
        t_step: Time of step change [s]

    Returns:
        Velocity profile function
    """

    def velocity(t: float) -> float:
        return v_before if t < t_step else v_after

    return velocity
