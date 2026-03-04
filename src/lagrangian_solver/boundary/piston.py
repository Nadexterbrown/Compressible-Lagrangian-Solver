"""
Moving piston boundary conditions with Rankine-Hugoniot shock relations.

Implements moving solid boundaries with prescribed velocity for
compatible energy discretization. Uses exact shock relations for
strong shocks via Newton iteration.

Key Design:
    For strong shocks (compression), the wall pressure is computed
    from Rankine-Hugoniot jump conditions using Newton iteration.
    This gives the correct post-shock state for piston problems.

    For weak perturbations or expansion, uses acoustic approximation
    or isentropic relations.

References:
    [Toro2009] Chapter 3 - Exact Riemann solver and shock relations
    [Caramana1998] Section 4 - Boundary conditions
    [GDTk] L1D solver - https://github.com/gdtk-uq/gdtk
"""

from typing import Optional, Callable, Union, Protocol, runtime_checkable
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


@runtime_checkable
class TrajectoryInterpolator(Protocol):
    """
    Protocol for trajectory interpolators used with MovingDataDrivenPistonBC.

    Any object with position(t) and velocity(t) methods satisfies this protocol.
    Compatible with PeleTrajectoryInterpolator and SyntheticTrajectoryInterpolator.
    """

    def position(self, t: float) -> float:
        """Return position at time t [m]."""
        ...

    def velocity(self, t: float) -> float:
        """Return velocity at time t [m/s]."""
        ...


class CompatiblePistonBC(BoundaryCondition):
    """
    Moving piston BC using Rankine-Hugoniot shock relations.

    For strong shocks, solves the exact shock jump conditions using
    Newton iteration. For weak perturbations, uses acoustic approximation.

    This BC is designed for compatible energy discretization:
        - apply_velocity(): Sets face velocity to piston velocity
        - apply_momentum(): Computes d_u from shock-based wall pressure

    The wall pressure comes from solving the Rankine-Hugoniot conditions,
    NOT from a Riemann solver. This ensures consistency with the
    cell-centered stress discretization.

    Automatic Velocity Ramping:
        By default, the piston velocity ramps up over 30 microseconds to
        prevent excessive artificial viscosity heating at the boundary
        during shock formation. This can be disabled by setting ramp_time=0.

    Reference: Toro (2009) Chapter 3, GDTk L1D solver
    """

    # Default ramp time for smooth shock formation (30 microseconds)
    DEFAULT_RAMP_TIME = 30e-6
    # Default startup time (delay before ramp begins)
    DEFAULT_STARTUP_TIME = 0.0

    def __init__(
        self,
        side: BoundarySide,
        eos: EOSBase,
        velocity: Union[float, VelocityProfile],
        thermal_bc: ThermalBCType = ThermalBCType.ADIABATIC,
        piston_temperature: Optional[float] = None,
        ramp_time: Optional[float] = None,
        startup_time: Optional[float] = None,
    ):
        """
        Initialize compatible piston boundary condition.

        Args:
            side: Which side of the domain (LEFT or RIGHT)
            eos: Equation of state
            velocity: Piston velocity [m/s] or function v(t)
                      Positive = moving right, Negative = moving left
            thermal_bc: Type of thermal boundary condition
            piston_temperature: Piston temperature [K] (for isothermal)
            ramp_time: Time to ramp velocity from 0 to target [s].
                      Default is 30e-6 (30 microseconds) to prevent
                      excessive AV heating during shock formation.
                      Set to 0 to disable ramping.
            startup_time: Delay before ramp begins [s]. Velocity stays
                         at zero until t = startup_time, then erf ramp starts.
                         Default is 0 (no delay).
        """
        super().__init__(side, eos)

        # Set ramp time (default 30 us for smooth shock formation)
        self._ramp_time = ramp_time if ramp_time is not None else self.DEFAULT_RAMP_TIME
        # Set startup delay (default 0 - no delay)
        self._startup_time = startup_time if startup_time is not None else self.DEFAULT_STARTUP_TIME

        # Velocity profile with optional ramping
        if callable(velocity):
            self._base_velocity_func = velocity
        else:
            self._base_velocity_func = lambda t: velocity

        # Apply ramping if ramp_time > 0
        if self._ramp_time > 0:
            self._velocity_func = self._create_ramped_velocity()
        else:
            self._velocity_func = self._base_velocity_func

        self._thermal_bc = thermal_bc
        self._piston_temperature = piston_temperature

        if thermal_bc == ThermalBCType.ISOTHERMAL and piston_temperature is None:
            raise ValueError("piston_temperature required for isothermal BC")

        # Store computed wall pressure for diagnostics
        self._p_wall = None

    def _create_ramped_velocity(self) -> VelocityProfile:
        """Create velocity profile with smooth erf ramp-up.

        Uses error function for smooth S-curve acceleration profile:
        v(t) = v_target * (1 + erf(6*(t-t_mid)/t_ramp)) / 2

        where t_mid = t_start + t_ramp/2 is the midpoint (inflection point).

        This maps [t_start, t_start+t_ramp] to [-3, +3] in erf space:
        - At t=t_start: erf(-3) ≈ -0.99998, v ≈ 0 (flat start, zero acceleration)
        - At t=t_mid: erf(0) = 0, v = v_target/2 (inflection point, max acceleration)
        - At t=t_start+t_ramp: erf(3) ≈ 0.99998, v ≈ v_target (flat end, zero acceleration)

        This gives the characteristic S-curve with smooth transitions at both ends.
        """
        from scipy.special import erf
        t_ramp = self._ramp_time
        t_start = self._startup_time
        base_func = self._base_velocity_func

        def ramped_velocity(t: float) -> float:
            v_target = base_func(t)
            if t <= t_start:
                return 0.0
            if t >= t_start + t_ramp:
                return v_target
            # Map [t_start, t_start+t_ramp] to [-3, +3] for smooth S-curve
            # Using 6/t_ramp as the scale factor (maps to [-3, 3])
            t_rel = t - t_start
            x = 6.0 * (t_rel / t_ramp) - 3.0
            return v_target * (1.0 + erf(x)) / 2.0

        return ramped_velocity

    @property
    def ramp_time(self) -> float:
        """Velocity ramp-up time [s]."""
        return self._ramp_time

    @property
    def startup_time(self) -> float:
        """Startup delay before ramp begins [s]."""
        return self._startup_time

    @property
    def thermal_bc(self) -> ThermalBCType:
        """Type of thermal boundary condition."""
        return self._thermal_bc

    @property
    def piston_temperature(self) -> Optional[float]:
        """Piston temperature for isothermal BC [K]."""
        return self._piston_temperature

    @property
    def wall_pressure(self) -> Optional[float]:
        """Most recently computed wall pressure [Pa]."""
        return self._p_wall

    def get_boundary_velocity(self, t: float) -> float:
        """Get piston velocity at time t."""
        return self._velocity_func(t)

    def apply_velocity(
        self,
        state: FlowState,
        grid: LagrangianGrid,
        t: float,
    ) -> None:
        """
        Set face velocity to piston velocity.

        This is called before computing the RHS.
        """
        v_piston = self.get_boundary_velocity(t)
        state.u[self.face_index] = v_piston

    def apply_momentum(
        self,
        state: FlowState,
        grid: LagrangianGrid,
        d_u: np.ndarray,
        sigma: np.ndarray,
        t: float,
    ) -> None:
        """
        Compute momentum rate at piston using shock-based wall pressure.

        For a moving piston:
            d_u[boundary] is set to maintain piston velocity

        The key insight: at the piston face, d_u should be such that
        the velocity stays equal to the piston velocity. For a constant
        velocity piston, d_u = 0. For an accelerating piston, d_u = dv_p/dt.

        For now, we assume constant or slowly varying piston velocity,
        so d_u[boundary] = 0.

        Args:
            state: Current flow state
            grid: Lagrangian grid
            d_u: Momentum rate array (modified at boundary)
            sigma: Cell-centered total stress [Pa]
            t: Current time [s]
        """
        # The piston velocity is prescribed, so d_u at the piston face
        # is determined by the piston motion, not the fluid dynamics.
        # For constant velocity piston: d_u = 0
        # For accelerating piston: d_u = dv_piston/dt

        # Simple finite difference for piston acceleration
        dt_check = 1e-8
        v_now = self._velocity_func(t)
        v_later = self._velocity_func(t + dt_check)
        dv_dt = (v_later - v_now) / dt_check

        d_u[self.face_index] = dv_dt

        # Also compute and store wall pressure for diagnostics
        self._compute_wall_pressure(state, grid, t)

    def _compute_wall_pressure(
        self,
        state: FlowState,
        grid: LagrangianGrid,
        t: float,
    ) -> float:
        """
        Compute wall pressure using Rankine-Hugoniot relations.

        For compression (piston moving into gas), solve shock jump conditions.
        For expansion (piston moving away), use isentropic relations.

        Reference: Toro (2009) Section 3.1
        """
        v_piston = self.get_boundary_velocity(t)

        # Get adjacent cell state
        idx = self.cell_index
        rho = state.rho[idx]
        p = state.p[idx]
        c = state.c[idx]
        gamma = state.gamma[idx]

        # Interior velocity (at face next to boundary)
        if self._side == BoundarySide.LEFT:
            # Left piston: interior face is u[1]
            u_int = state.u[1]
            # Compression: piston moving right into gas (v_piston > u_int)
            is_compression = v_piston > u_int
        else:
            # Right piston: interior face is u[-2]
            u_int = state.u[-2]
            # Compression: piston moving left into gas (v_piston < u_int)
            is_compression = v_piston < u_int

        if is_compression:
            # Shock wave - use Rankine-Hugoniot
            self._p_wall = self._shock_pressure(
                rho, p, c, gamma, u_int, v_piston
            )
        else:
            # Expansion wave - use isentropic relations
            self._p_wall = self._rarefaction_pressure(
                rho, p, c, gamma, u_int, v_piston
            )

        return self._p_wall

    def _shock_pressure(
        self,
        rho: float,
        p: float,
        c: float,
        gamma: float,
        u_int: float,
        v_piston: float,
    ) -> float:
        """
        Compute wall pressure from shock using Newton iteration.

        Solves the Rankine-Hugoniot jump conditions for the shock
        pressure given the piston velocity.

        Reference: Toro (2009) Section 3.1, Equations (3.6)-(3.12)

        Args:
            rho: Pre-shock density [kg/m³]
            p: Pre-shock pressure [Pa]
            c: Pre-shock sound speed [m/s]
            gamma: Ratio of specific heats
            u_int: Interior velocity [m/s]
            v_piston: Piston velocity [m/s]

        Returns:
            Post-shock (wall) pressure [Pa]
        """
        gp1 = gamma + 1.0
        gm1 = gamma - 1.0

        # Velocity jump
        if self._side == BoundarySide.LEFT:
            du = v_piston - u_int
        else:
            du = u_int - v_piston

        # Initial guess from acoustic approximation
        p_star = p + rho * c * abs(du)

        # Newton iteration for exact shock pressure
        # The shock relation: u* - u = (p* - p) / (ρ * S)
        # where S = c * sqrt[(γ+1)/(2γ) * (p*/p - 1) + 1] is shock speed

        for iteration in range(20):
            # Pressure ratio
            pr = p_star / p

            # Shock speed factor: A = sqrt[2/(γρ) * (γ+1)/2 * p* + (γ-1)/2 * p]
            A_sq = (2.0 / (rho * gp1)) * (p_star + gm1 / gp1 * p)
            A = np.sqrt(max(A_sq, 1e-30))

            # Function: f(p*) = du - (p* - p) * A
            f = abs(du) - (p_star - p) * A

            # Derivative
            dA_dp = 1.0 / (rho * gp1 * A)
            df_dp = -A - (p_star - p) * dA_dp

            # Newton update
            dp = -f / df_dp

            p_star_new = p_star + dp

            # Convergence check
            if abs(dp / max(p_star, 1e-10)) < 1e-8:
                break

            # Ensure positive pressure
            p_star = max(p_star_new, p * 0.01)

        return max(p_star, 1e-10)

    def _rarefaction_pressure(
        self,
        rho: float,
        p: float,
        c: float,
        gamma: float,
        u_int: float,
        v_piston: float,
    ) -> float:
        """
        Compute wall pressure from isentropic expansion.

        For expansion waves, use the isentropic relation:
            p* / p = [1 + (γ-1)/2 * (u - u*) / c]^(2γ/(γ-1))

        Reference: Toro (2009) Section 3.2

        Args:
            rho: Density [kg/m³]
            p: Interior pressure [Pa]
            c: Sound speed [m/s]
            gamma: Ratio of specific heats
            u_int: Interior velocity [m/s]
            v_piston: Piston velocity [m/s]

        Returns:
            Wall pressure [Pa]
        """
        gm1 = gamma - 1.0

        # Velocity difference
        if self._side == BoundarySide.LEFT:
            du = v_piston - u_int  # Negative for expansion
        else:
            du = u_int - v_piston  # Negative for expansion

        # Isentropic relation: p*/p = (1 + (γ-1)/2 * du/c)^(2γ/(γ-1))
        # For du < 0 (expansion), this gives p* < p
        term = 1.0 + gm1 / 2.0 * du / c

        if term <= 0:
            # Vacuum state - gas cannot expand faster than 2c/(γ-1)
            return 1e-10

        p_star = p * term ** (2.0 * gamma / gm1)

        return max(p_star, 1e-10)

    # Legacy interface for backward compatibility
    def apply(
        self,
        state: FlowState,
        grid: LagrangianGrid,
        t: float,
    ) -> None:
        """LEGACY: Use apply_velocity() instead."""
        self.apply_velocity(state, grid, t)

    def compute_flux(
        self,
        state: FlowState,
        grid: LagrangianGrid,
        t: float,
    ) -> BoundaryFlux:
        """
        LEGACY: Compute numerical flux at moving piston.

        For compatible discretization, use apply_momentum() instead.
        """
        v_piston = self.get_boundary_velocity(t)
        p_wall = self._compute_wall_pressure(state, grid, t)

        return BoundaryFlux(
            p_flux=p_wall,
            pu_flux=p_wall * v_piston,
            u_flux=v_piston,
        )

    def update_position(
        self, grid: LagrangianGrid, dt: float, t: float
    ) -> float:
        """Compute new piston position."""
        current_x = grid.x[self.face_index]
        return current_x + dt * self.get_boundary_velocity(t)

    def get_piston_position(self, x0: float, t: float) -> float:
        """
        Get analytical piston position at time t.

        Integrates the velocity profile from 0 to t.

        Args:
            x0: Initial piston position [m]
            t: Current time [s]

        Returns:
            Piston position [m]
        """
        from scipy import integrate

        result, _ = integrate.quad(self._velocity_func, 0, t)
        return x0 + result


# Keep MovingPistonBC as an alias for backward compatibility
class MovingPistonBC(CompatiblePistonBC):
    """
    Alias for CompatiblePistonBC for backward compatibility.

    NOTE: This now uses the compatible discretization interface.
    If you need the old Riemann-based BC, see the git history.

    Features automatic velocity ramping (default 30us) to prevent
    excessive AV heating at boundaries during shock formation.
    """

    pass


# Utility functions for creating piston velocity profiles

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


class RiemannGhostPistonBC(BoundaryCondition):
    """
    Moving piston BC using Riemann-based ghost cell.

    GENERAL: Works for any wave type (shock, rarefaction, acoustic).

    At each time step:
    1. Solve boundary Riemann problem with interior state and u_piston
    2. Get interface state (p*, rho*, etc.) valid for any wave pattern
    3. Use p* as ghost stress in momentum equation

    This is physically correct because the Riemann solver finds
    the exact wave structure connecting interior to boundary.

    Key difference from CompatiblePistonBC:
        - Solves full Riemann problem at boundary
        - Uses Riemann solution pressure for momentum equation
        - Handles shocks, rarefactions, and acoustic waves correctly

    Features automatic velocity ramping (default 30us) to prevent
    excessive AV heating at boundaries during shock formation.

    Reference: [Toro2009] Section 6.3
    """

    # Default ramp time for smooth shock formation (30 microseconds)
    DEFAULT_RAMP_TIME = 30e-6
    # Default startup time (delay before ramp begins)
    DEFAULT_STARTUP_TIME = 0.0

    def __init__(
        self,
        side: BoundarySide,
        eos: EOSBase,
        velocity: Union[float, VelocityProfile],
        tol: float = 1e-8,
        thermal_bc: ThermalBCType = ThermalBCType.ADIABATIC,
        piston_temperature: Optional[float] = None,
        ramp_time: Optional[float] = None,
        startup_time: Optional[float] = None,
    ):
        """
        Initialize Riemann ghost cell piston BC.

        Args:
            side: Which side of the domain (LEFT or RIGHT)
            eos: Equation of state
            velocity: Piston velocity [m/s] or function v(t)
            tol: Tolerance for Riemann solver iteration
            thermal_bc: Type of thermal boundary condition
            piston_temperature: Piston temperature [K] (for isothermal)
            ramp_time: Time to ramp velocity from 0 to target [s].
                      Default is 30e-6 (30 microseconds).
                      Set to 0 to disable ramping.
            startup_time: Delay before ramp begins [s]. Velocity stays
                         at zero until t = startup_time, then erf ramp starts.
                         Default is 0 (no delay).
        """
        super().__init__(side, eos)

        # Set ramp time (default 30 us for smooth shock formation)
        self._ramp_time = ramp_time if ramp_time is not None else self.DEFAULT_RAMP_TIME
        # Set startup delay (default 0 - no delay)
        self._startup_time = startup_time if startup_time is not None else self.DEFAULT_STARTUP_TIME

        # Base velocity function
        if callable(velocity):
            self._base_velocity_func = velocity
        else:
            self._base_velocity_func = lambda t: velocity

        # Apply ramping if ramp_time > 0
        if self._ramp_time > 0:
            self._velocity_func = self._create_ramped_velocity()
        else:
            self._velocity_func = self._base_velocity_func

        self._tol = tol
        self._thermal_bc = thermal_bc
        self._piston_temperature = piston_temperature

        # Import here to avoid circular imports
        from lagrangian_solver.numerics.boundary_riemann import BoundaryRiemannSolver
        self._boundary_solver = BoundaryRiemannSolver(eos, tol=tol)

        # Cache interface state for use in momentum equation
        self._interface_state = None

    def _create_ramped_velocity(self) -> VelocityProfile:
        """Create velocity profile with smooth erf ramp-up.

        Uses error function for smooth S-curve acceleration profile:
        v(t) = v_target * (1 + erf(6*(t-t_mid)/t_ramp)) / 2

        where t_mid = t_start + t_ramp/2 is the midpoint (inflection point).

        This maps [t_start, t_start+t_ramp] to [-3, +3] in erf space:
        - At t=t_start: erf(-3) ≈ -0.99998, v ≈ 0 (flat start, zero acceleration)
        - At t=t_mid: erf(0) = 0, v = v_target/2 (inflection point, max acceleration)
        - At t=t_start+t_ramp: erf(3) ≈ 0.99998, v ≈ v_target (flat end, zero acceleration)
        """
        from scipy.special import erf
        t_ramp = self._ramp_time
        t_start = self._startup_time
        base_func = self._base_velocity_func

        def ramped_velocity(t: float) -> float:
            v_target = base_func(t)
            if t <= t_start:
                return 0.0
            if t >= t_start + t_ramp:
                return v_target
            # Map [t_start, t_start+t_ramp] to [-3, +3] for smooth S-curve
            t_rel = t - t_start
            x = 6.0 * (t_rel / t_ramp) - 3.0
            return v_target * (1.0 + erf(x)) / 2.0

        return ramped_velocity

    @property
    def ramp_time(self) -> float:
        """Velocity ramp-up time [s]."""
        return self._ramp_time

    @property
    def startup_time(self) -> float:
        """Startup delay before ramp begins [s]."""
        return self._startup_time

    @property
    def thermal_bc(self) -> ThermalBCType:
        """Type of thermal boundary condition."""
        return self._thermal_bc

    @property
    def piston_temperature(self) -> Optional[float]:
        """Piston temperature for isothermal BC [K]."""
        return self._piston_temperature

    def get_boundary_velocity(self, t: float) -> float:
        """Get piston velocity at time t."""
        return self._velocity_func(t)

    def compute_interface_state(
        self,
        state: FlowState,
        grid: LagrangianGrid,
        t: float,
    ):
        """
        Compute interface state by solving boundary Riemann problem.

        This solves for the interface pressure p* such that the
        Riemann solution has velocity u* = u_piston. Works for
        any wave type (shock, rarefaction, acoustic).

        The interior velocity used is from the first interior face (u[1] for
        left boundary, u[-2] for right), NOT the boundary face itself.

        Args:
            state: Current flow state
            grid: Lagrangian grid
            t: Current time [s]

        Returns:
            BoundaryState at the interface
        """
        from lagrangian_solver.numerics.boundary_riemann import BoundaryState

        v_piston = self._velocity_func(t)
        idx = self.cell_index

        # Interior state from adjacent cell
        rho_int = state.rho[idx]
        p_int = state.p[idx]

        # Use the first interior face velocity, NOT the boundary face.
        # The boundary face velocity is set to piston velocity, so including
        # it would give wrong results.
        if self._side == BoundarySide.LEFT:
            # For left boundary: interior is cell 0, use face 1 velocity
            u_int = state.u[1]
            self._interface_state = self._boundary_solver.solve_left_boundary(
                rho_int, u_int, p_int, v_piston
            )
        else:
            # For right boundary: interior is cell -1, use face -2 velocity
            u_int = state.u[-2]
            self._interface_state = self._boundary_solver.solve_right_boundary(
                rho_int, u_int, p_int, v_piston
            )

        return self._interface_state

    def apply_velocity(
        self,
        state: FlowState,
        grid: LagrangianGrid,
        t: float,
    ) -> None:
        """
        Set face velocity to piston velocity.

        This is called before computing the RHS.
        """
        v_piston = self._velocity_func(t)
        state.u[self.face_index] = v_piston

    def apply_momentum(
        self,
        state: FlowState,
        grid: LagrangianGrid,
        d_u: np.ndarray,
        sigma: np.ndarray,
        t: float,
    ) -> None:
        """
        Apply momentum BC at piston face.

        For constant velocity piston: d_u = 0
        For accelerating piston: d_u = dv_piston/dt
        """
        # Finite difference for piston acceleration
        dt_check = 1e-8
        v_now = self._velocity_func(t)
        v_later = self._velocity_func(t + dt_check)
        dv_dt = (v_later - v_now) / dt_check

        d_u[self.face_index] = dv_dt

    def get_interface_stress(self) -> float:
        """
        Get interface stress for momentum equation.

        This returns the Riemann solution pressure p*, which is
        the correct interface pressure for any wave type.
        """
        if self._interface_state is None:
            raise RuntimeError(
                "Interface state not computed. Call compute_interface_state first."
            )
        return self._interface_state.sigma

    def has_ghost_cell(self) -> bool:
        """
        Indicate this BC provides ghost cell stress.

        Used by conservation equations to detect ghost cell BCs.
        """
        return True

    @property
    def interface_state(self):
        """Access the computed interface state (for diagnostics)."""
        return self._interface_state

    # Legacy interface for backward compatibility
    def apply(
        self,
        state: FlowState,
        grid: LagrangianGrid,
        t: float,
    ) -> None:
        """LEGACY: Use apply_velocity() instead."""
        self.apply_velocity(state, grid, t)

    def compute_flux(
        self,
        state: FlowState,
        grid: LagrangianGrid,
        t: float,
    ) -> BoundaryFlux:
        """
        LEGACY: Compute numerical flux at moving piston.

        For Riemann ghost BC, use compute_interface_state() instead.
        """
        v_piston = self._velocity_func(t)

        # Compute interface state if not already done
        if self._interface_state is None:
            self.compute_interface_state(state, grid, t)

        p_star = self._interface_state.p

        return BoundaryFlux(
            p_flux=p_star,
            pu_flux=p_star * v_piston,
            u_flux=v_piston,
        )


class MovingDataDrivenPistonBC(BoundaryCondition):
    """
    Moving piston BC using velocity from trajectory data.

    Uses a trajectory interpolator to get position/velocity from data files.
    This is a solid piston BC - gas velocity equals piston velocity.

    Features:
        - Linear interpolation between data points (via trajectory object)
        - Riemann-based ghost cell for interface state
        - velocity_scale: scales the trajectory velocity
        - velocity_offset: subtracts from the scaled velocity
        - velocity_min: minimum allowed velocity (clamping)
        - time_offset: for synchronization with data

    Velocity transformation order:
        v_final = max(velocity_min, v_data * velocity_scale - velocity_offset)

    Reference: [Toro2009] Section 6.3
    """

    def __init__(
        self,
        side: BoundarySide,
        eos: EOSBase,
        trajectory: TrajectoryInterpolator,
        velocity_scale: float = 1.0,
        velocity_offset: float = 0.0,
        velocity_min: Optional[float] = None,
        time_offset: float = 0.0,
        tol: float = 1e-8,
        thermal_bc: ThermalBCType = ThermalBCType.ADIABATIC,
        piston_temperature: Optional[float] = None,
    ):
        """
        Initialize data-driven piston boundary condition.

        Parameters
        ----------
        side : BoundarySide
            Which side of the domain (LEFT or RIGHT)
        eos : EOSBase
            Equation of state
        trajectory : TrajectoryInterpolator
            Interpolator for trajectory data. Must have position(t) and
            velocity(t) methods. Compatible with PeleTrajectoryInterpolator
            and SyntheticTrajectoryInterpolator.
        velocity_scale : float
            Scale factor for velocity (default 1.0)
        velocity_offset : float
            Value to subtract from scaled velocity [m/s] (default 0.0)
        velocity_min : float, optional
            Minimum allowed velocity [m/s] (default None = no clamping)
        time_offset : float
            Time offset [s] (simulation_time = data_time + time_offset)
        tol : float
            Tolerance for Riemann solver iteration
        thermal_bc : ThermalBCType
            Type of thermal boundary condition
        piston_temperature : float, optional
            Piston temperature [K] (for isothermal BC)
        """
        super().__init__(side, eos)

        self._trajectory = trajectory
        self._velocity_scale = velocity_scale
        self._velocity_offset = velocity_offset
        self._velocity_min = velocity_min
        self._time_offset = time_offset
        self._tol = tol
        self._thermal_bc = thermal_bc
        self._piston_temperature = piston_temperature

        # Import Riemann solver
        from lagrangian_solver.numerics.boundary_riemann import BoundaryRiemannSolver
        self._boundary_solver = BoundaryRiemannSolver(eos, tol=tol)

        # Cache interface state for momentum equation
        self._interface_state = None

    @property
    def trajectory(self) -> TrajectoryInterpolator:
        """Trajectory interpolator."""
        return self._trajectory

    @property
    def thermal_bc(self) -> ThermalBCType:
        """Type of thermal boundary condition."""
        return self._thermal_bc

    def _get_data_time(self, t: float) -> float:
        """Convert simulation time to data time."""
        return t - self._time_offset

    def get_piston_position(self, t: float) -> float:
        """
        Get piston position at time t.

        Returns position from trajectory data.
        """
        t_data = self._get_data_time(t)
        return self._trajectory.position(t_data)

    def get_piston_velocity(self, t: float) -> float:
        """
        Get piston velocity at time t.

        Applies transformations in order:
            v_final = max(velocity_min, v_data * velocity_scale - velocity_offset)
        """
        t_data = self._get_data_time(t)
        v = self._trajectory.velocity(t_data)

        # Apply scale then offset
        v_modified = v * self._velocity_scale - self._velocity_offset

        # Apply minimum velocity clamp if specified
        if self._velocity_min is not None:
            v_modified = max(self._velocity_min, v_modified)

        return v_modified

    def get_boundary_velocity(self, t: float) -> float:
        """Get boundary face velocity at time t."""
        return self.get_piston_velocity(t)

    def compute_interface_state(
        self,
        state: FlowState,
        grid: LagrangianGrid,
        t: float,
    ):
        """
        Compute interface state by solving boundary Riemann problem.

        Uses piston velocity to solve the Riemann problem.
        """
        from lagrangian_solver.numerics.boundary_riemann import BoundaryState

        v_piston = self.get_piston_velocity(t)
        idx = self.cell_index

        # Interior state from adjacent cell
        rho_int = state.rho[idx]
        p_int = state.p[idx]

        # Use first interior face velocity (not boundary face)
        if self._side == BoundarySide.LEFT:
            u_int = state.u[1]
            self._interface_state = self._boundary_solver.solve_left_boundary(
                rho_int, u_int, p_int, v_piston
            )
        else:
            u_int = state.u[-2]
            self._interface_state = self._boundary_solver.solve_right_boundary(
                rho_int, u_int, p_int, v_piston
            )

        return self._interface_state

    def apply_velocity(
        self,
        state: FlowState,
        grid: LagrangianGrid,
        t: float,
    ) -> None:
        """
        Set face velocity to piston velocity.

        For solid piston, gas velocity = piston velocity.
        """
        v_piston = self.get_piston_velocity(t)
        state.u[self.face_index] = v_piston

    def apply_momentum(
        self,
        state: FlowState,
        grid: LagrangianGrid,
        d_u: np.ndarray,
        sigma: np.ndarray,
        t: float,
    ) -> None:
        """
        Apply momentum BC at piston face.

        Computes d_u = dv_piston/dt using finite difference.
        """
        dt_check = 1e-8
        v_now = self.get_piston_velocity(t)
        v_later = self.get_piston_velocity(t + dt_check)
        dv_dt = (v_later - v_now) / dt_check

        d_u[self.face_index] = dv_dt

    def get_interface_stress(self) -> float:
        """
        Get interface stress for momentum equation.

        Returns Riemann solution pressure p*.
        """
        if self._interface_state is None:
            raise RuntimeError(
                "Interface state not computed. Call compute_interface_state first."
            )
        return self._interface_state.sigma

    def has_ghost_cell(self) -> bool:
        """Indicate this BC provides ghost cell stress."""
        return True

    @property
    def interface_state(self):
        """Access computed interface state (for diagnostics)."""
        return self._interface_state

    def compute_flux(
        self,
        state: FlowState,
        grid: LagrangianGrid,
        t: float,
    ) -> BoundaryFlux:
        """
        Compute numerical flux at moving piston.

        Legacy interface for backward compatibility.
        """
        v_piston = self.get_piston_velocity(t)

        if self._interface_state is None:
            self.compute_interface_state(state, grid, t)

        p_star = self._interface_state.p

        return BoundaryFlux(
            p_flux=p_star,
            pu_flux=p_star * v_piston,
            u_flux=v_piston,
        )
