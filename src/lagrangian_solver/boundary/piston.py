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


class PorousGhostPistonBC(BoundaryCondition):
    """
    Porous piston BC with ghost-cell Riemann pressure.

    This BC models a porous piston where the piston velocity u_p(t) differs
    from the gas velocity u_g(t) at the boundary, creating mass flux through
    the boundary while maintaining conservation.

    Physics:
        - Boundary node moves at u_p(t) (tracks piston position)
        - Gas at boundary has velocity u_g(t) (from experimental data)
        - When u_p != u_g: mass flux through boundary, cell 0 mass varies
        - Ghost cell provides correct boundary pressure (no spurious reflections)

    Key equation (acoustic impedance relation):
        p* = p_0 + Z_0 * (u_g - u_0)
        where Z_0 = rho_0 * c_0 is acoustic impedance, u_0 is cell 0 center velocity

    Features:
        - Two independent velocity inputs: piston position and gas velocity
        - Mass flux tracking for conservation verification
        - Optional merge-split for extreme mass ratio cells
        - CFL constraint based on leak rate

    Reference: [Toro2009] Section 6.3 - Boundary conditions via ghost cells
    """

    # Default ramp time for smooth velocity ramp-up
    DEFAULT_RAMP_TIME = 30e-6
    DEFAULT_STARTUP_TIME = 0.0

    # Merge-split thresholds
    DEFAULT_MERGE_LOW = 0.3   # merge when m[0] < 30% of m[1]
    DEFAULT_MERGE_HIGH = 3.0  # merge when m[0] > 300% of m[1]

    def __init__(
        self,
        side: BoundarySide,
        eos: EOSBase,
        piston_velocity: Union[float, VelocityProfile],
        gas_velocity: Union[float, VelocityProfile],
        ramp_time: Optional[float] = None,
        startup_time: Optional[float] = None,
        tol: float = 1e-8,
        thermal_bc: ThermalBCType = ThermalBCType.ADIABATIC,
        piston_temperature: Optional[float] = None,
        merge_low: Optional[float] = None,
        merge_high: Optional[float] = None,
    ):
        """
        Initialize porous ghost piston boundary condition.

        Args:
            side: Which side of the domain (LEFT or RIGHT)
            eos: Equation of state
            piston_velocity: Piston displacement velocity [m/s] or function u_p(t)
                            Controls boundary node position
            gas_velocity: Gas particle velocity at boundary [m/s] or function u_g(t)
                         From experimental data - controls mass flux
            ramp_time: Time to ramp velocity from 0 to target [s].
                      Default is 30e-6 (30 microseconds).
                      Set to 0 to disable ramping.
            startup_time: Delay before ramp begins [s].
            tol: Tolerance for Riemann solver iteration
            thermal_bc: Type of thermal boundary condition
            piston_temperature: Piston temperature [K] (for isothermal)
            merge_low: Merge threshold for small boundary cell (default 0.3)
            merge_high: Merge threshold for large boundary cell (default 3.0)
        """
        super().__init__(side, eos)

        # Set ramp time (default 30 us for smooth shock formation)
        self._ramp_time = ramp_time if ramp_time is not None else self.DEFAULT_RAMP_TIME
        self._startup_time = startup_time if startup_time is not None else self.DEFAULT_STARTUP_TIME

        # Piston velocity (controls boundary node position)
        if callable(piston_velocity):
            self._base_piston_velocity_func = piston_velocity
        else:
            self._base_piston_velocity_func = lambda t: piston_velocity

        # Gas velocity (controls mass flux)
        if callable(gas_velocity):
            self._base_gas_velocity_func = gas_velocity
        else:
            self._base_gas_velocity_func = lambda t: gas_velocity

        # Apply ramping if ramp_time > 0
        if self._ramp_time > 0:
            self._piston_velocity_func = self._create_ramped_velocity(
                self._base_piston_velocity_func
            )
            self._gas_velocity_func = self._create_ramped_velocity(
                self._base_gas_velocity_func
            )
        else:
            self._piston_velocity_func = self._base_piston_velocity_func
            self._gas_velocity_func = self._base_gas_velocity_func

        self._tol = tol
        self._thermal_bc = thermal_bc
        self._piston_temperature = piston_temperature

        # Merge-split thresholds
        self._merge_low = merge_low if merge_low is not None else self.DEFAULT_MERGE_LOW
        self._merge_high = merge_high if merge_high is not None else self.DEFAULT_MERGE_HIGH

        # Import boundary Riemann solver
        from lagrangian_solver.numerics.boundary_riemann import BoundaryRiemannSolver
        self._boundary_solver = BoundaryRiemannSolver(eos, tol=tol)

        # Interface state cache
        self._interface_state = None

        # Current time (updated in apply_velocity)
        self._current_time = 0.0

        # Mass tracking for diagnostics
        self._mass_leaked = 0.0
        self._initial_boundary_mass = None

    def _create_ramped_velocity(
        self, base_func: VelocityProfile
    ) -> VelocityProfile:
        """
        Create velocity profile with smooth erf ramp-up.

        Uses error function for smooth S-curve acceleration profile.
        """
        import math
        t_ramp = self._ramp_time
        t_start = self._startup_time

        def ramped_velocity(t: float) -> float:
            v_target = base_func(t)
            if t <= t_start:
                return 0.0
            if t >= t_start + t_ramp:
                return v_target
            # Map [t_start, t_start+t_ramp] to [-3, +3] for smooth S-curve
            t_rel = t - t_start
            x = 6.0 * (t_rel / t_ramp) - 3.0
            return v_target * (1.0 + math.erf(x)) / 2.0

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
    def mass_leaked(self) -> float:
        """
        Cumulative mass that has leaked through the boundary [kg].

        Positive = mass left the domain (piston catching up to gas).
        Negative = mass entered the domain (gas faster than piston).
        """
        return self._mass_leaked

    def get_piston_velocity(self, t: float) -> float:
        """Get piston velocity at time t (controls boundary position)."""
        return self._piston_velocity_func(t)

    def get_gas_velocity(self, t: float) -> float:
        """Get gas velocity at boundary at time t (controls mass flux)."""
        return self._gas_velocity_func(t)

    def get_boundary_velocity(self, t: float) -> float:
        """
        Get boundary face velocity at time t.

        For porous piston, the boundary node moves at piston velocity.
        """
        return self.get_piston_velocity(t)

    def compute_interface_state(
        self,
        state: FlowState,
        grid: LagrangianGrid,
        t: float,
    ):
        """
        Compute interface state by solving boundary Riemann problem.

        Uses the full Riemann solver with the GAS velocity (u_g) to find
        the interface pressure p*. This is the key difference from solid
        piston BCs which use the piston velocity.

        The piston velocity only controls boundary node position; the gas
        velocity controls the wave dynamics and interface state.

        Args:
            state: Current flow state
            grid: Lagrangian grid
            t: Current time [s]

        Returns:
            BoundaryState at the interface
        """
        self._current_time = t

        # Get gas velocity at boundary (this is what the gas is doing)
        u_g = self.get_gas_velocity(t)
        idx = self.cell_index

        # Interior state from adjacent cell
        rho_int = state.rho[idx]
        p_int = state.p[idx]

        # Use first interior face velocity, NOT the boundary face.
        # The boundary face velocity is set to piston velocity, which would
        # give wrong results for the Riemann problem.
        if self._side == BoundarySide.LEFT:
            u_int = state.u[1]
            self._interface_state = self._boundary_solver.solve_left_boundary(
                rho_int, u_int, p_int, u_g
            )
        else:
            u_int = state.u[-2]
            self._interface_state = self._boundary_solver.solve_right_boundary(
                rho_int, u_int, p_int, u_g
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

        For porous piston, the boundary node moves at piston velocity (u_p),
        tracking the physical piston/flame position. The gas velocity (u_g)
        is used separately for the Riemann solver and d_tau calculation.

        Args:
            state: Flow state to modify
            grid: Lagrangian grid
            t: Current time [s]
        """
        self._current_time = t
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

        For porous piston, d_u = dv_piston/dt (piston acceleration).

        Args:
            state: Current flow state
            grid: Lagrangian grid
            d_u: Momentum rate array (modified at boundary)
            sigma: Cell-centered total stress [Pa]
            t: Current time [s]
        """
        # Finite difference for piston acceleration
        dt_check = 1e-8
        v_now = self.get_piston_velocity(t)
        v_later = self.get_piston_velocity(t + dt_check)
        dv_dt = (v_later - v_now) / dt_check

        d_u[self.face_index] = dv_dt

    def get_interface_stress(self) -> float:
        """
        Get interface stress for momentum equation.

        Returns the ghost cell pressure p*, which provides the correct
        wave dynamics for the gas velocity condition.
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

    def update_boundary_mass(
        self,
        grid: LagrangianGrid,
        state: FlowState,
        dt: float,
    ) -> float:
        """
        Update boundary cell mass due to porosity flux.

        Mass flux through boundary:
            dm/dt = -rho_boundary * (u_p - u_g) * A

        where:
            u_p = piston velocity (boundary moves at this rate)
            u_g = gas velocity at boundary
            rho_boundary = density at boundary (from adjacent cell)
            A = cross-sectional area (1.0 for planar)

        When u_p > u_g: piston catches up, mass leaves domain (dm < 0)
        When u_p < u_g: gas escapes, mass enters domain (dm > 0)

        The mass change is limited to prevent the cell from going negative.
        Maximum drain is 50% of current cell mass per step.

        Args:
            grid: Lagrangian grid
            state: Current flow state
            dt: Time step [s]

        Returns:
            Mass change (positive = cell gained mass)
        """
        u_p = self.get_piston_velocity(self._current_time)
        u_g = self.get_gas_velocity(self._current_time)

        rho_boundary = state.rho[self.cell_index]
        A_boundary = 1.0  # Planar geometry; adjust for cylindrical/spherical

        # Mass flux: negative when piston catches up to gas
        dm = -rho_boundary * (u_p - u_g) * A_boundary * dt

        # Store initial mass for diagnostics
        if self._initial_boundary_mass is None:
            self._initial_boundary_mass = grid.dm[self.cell_index]

        # Limit mass drain to prevent negative mass
        # Maximum drain is 50% of current cell mass
        current_mass = grid.dm[self.cell_index]
        max_drain = -0.5 * current_mass  # Negative = mass leaving
        if dm < max_drain:
            dm = max_drain

        # Update grid mass
        grid.add_boundary_mass(self._side, dm)

        # Track cumulative mass leak
        self._mass_leaked -= dm

        return dm

    def get_max_dt_constraint(self, grid: LagrangianGrid) -> float:
        """
        Additional CFL constraint: don't drain >10% of boundary cell per step.

        This prevents the boundary cell from becoming too small in a single
        time step, which could cause numerical instability.

        Args:
            grid: Lagrangian grid

        Returns:
            Maximum allowed time step [s]
        """
        u_p = self.get_piston_velocity(self._current_time)
        u_g = self.get_gas_velocity(self._current_time)
        leak_speed = abs(u_p - u_g)

        if leak_speed < 1e-15:
            return float('inf')

        # Use cell width as proxy for mass drainage rate
        # More conservative: 10% instead of 20%
        dx_boundary = grid.dx[self.cell_index]
        return 0.1 * dx_boundary / leak_speed

    def get_effective_velocity_difference(self, grid: LagrangianGrid) -> float:
        """
        Get effective velocity difference, limited when cell is too small.

        When the boundary cell becomes very small, we need to limit
        the velocity difference to prevent grid inversion.

        Args:
            grid: Lagrangian grid

        Returns:
            Effective velocity difference [m/s]
        """
        u_p = self.get_piston_velocity(self._current_time)
        u_g = self.get_gas_velocity(self._current_time)
        velocity_diff = u_p - u_g

        # Get boundary cell size relative to average
        idx = self.cell_index
        dx_boundary = grid.dx[idx]
        dx_avg = grid.x[-1] - grid.x[0]  # Total domain length
        dx_avg /= len(grid.dx)  # Average cell size

        # When boundary cell is less than 10% of average, start limiting
        if dx_boundary < 0.1 * dx_avg:
            # Scale down the effective difference
            ratio = dx_boundary / (0.1 * dx_avg)
            velocity_diff *= ratio

        return velocity_diff

    def check_merge_split(
        self,
        grid: LagrangianGrid,
        state: FlowState,
    ) -> bool:
        """
        Check if boundary cell needs merge-split with neighbor.

        If the boundary cell mass becomes too small or too large compared
        to its neighbor, merge them and split equally to maintain grid quality.

        Args:
            grid: Lagrangian grid
            state: Current flow state

        Returns:
            True if merge-split was performed
        """
        idx = 0 if self._side == BoundarySide.LEFT else -1
        neighbor_idx = 1 if self._side == BoundarySide.LEFT else -2

        ratio = grid.dm[idx] / grid.dm[neighbor_idx]

        if ratio < self._merge_low or ratio > self._merge_high:
            self._conservative_merge_split(grid, state, idx, neighbor_idx)
            return True
        return False

    def _conservative_merge_split(
        self,
        grid: LagrangianGrid,
        state: FlowState,
        idx: int,
        neighbor_idx: int,
    ) -> None:
        """
        Merge cells idx and neighbor_idx, then split equally.

        Conserves mass, momentum, and total energy exactly.

        Args:
            grid: Lagrangian grid (modified in place)
            state: Current flow state (modified in place)
            idx: Boundary cell index
            neighbor_idx: Neighbor cell index
        """
        # Gather conserved quantities
        m0 = grid.dm[idx]
        m1 = grid.dm[neighbor_idx]
        m_total = m0 + m1

        # Cell center velocities
        if idx == 0:
            u_c0 = 0.5 * (state.u[0] + state.u[1])
            u_c1 = 0.5 * (state.u[1] + state.u[2])
        else:
            u_c0 = 0.5 * (state.u[-2] + state.u[-1])
            u_c1 = 0.5 * (state.u[-3] + state.u[-2])

        # Conserved quantities
        mom_total = m0 * u_c0 + m1 * u_c1
        E_total = m0 * state.e[idx] + m1 * state.e[neighbor_idx]

        # Redistribute mass equally
        m_half = m_total / 2.0

        # Update grid masses directly
        grid._dm[idx] = m_half
        grid._dm[neighbor_idx] = m_half

        # Update cumulative mass
        if idx == 0:
            # Left boundary: recalculate cumulative mass
            grid._m[1] = m_half
            grid._m[2] = m_half + m_half
            # Rest of cumulative mass shifts
            for j in range(3, grid.n_faces):
                grid._m[j] = grid._m[j-1] + grid._dm[j-1]
        else:
            # Right boundary: cumulative mass from left is unchanged
            pass

        # Average internal energy for both cells (conserves total internal energy)
        e_avg = E_total / m_total
        state.e[idx] = e_avg
        state.e[neighbor_idx] = e_avg

        # Update density: rho = dm / dx
        # Cell widths don't change, so density scales with mass
        if idx == 0:
            dx0 = state.x[1] - state.x[0]
            dx1 = state.x[2] - state.x[1]
        else:
            dx0 = state.x[-1] - state.x[-2]
            dx1 = state.x[-2] - state.x[-3]
        state.rho[idx] = m_half / dx0
        state.rho[neighbor_idx] = m_half / dx1

        # Update specific volume (tau = 1/rho)
        state.tau[idx] = 1.0 / state.rho[idx]
        state.tau[neighbor_idx] = 1.0 / state.rho[neighbor_idx]

        # Update interface velocity to conserve momentum
        # The shared face velocity is adjusted to conserve cell-centered momentum
        if idx == 0:
            # Shared face is u[1]
            # After merge-split: u_c0 = (u[0] + u[1])/2, u_c1 = (u[1] + u[2])/2
            # Momentum: m_half * u_c0 + m_half * u_c1 = mom_total
            # Simplify: m_half * (u[0] + 2*u[1] + u[2])/2 = mom_total
            # Solve for u[1]: u[1] = mom_total/m_half - (u[0] + u[2])/2
            state.u[1] = mom_total / m_half - 0.5 * (state.u[0] + state.u[2])
        else:
            # Shared face is u[-2]
            state.u[-2] = mom_total / m_half - 0.5 * (state.u[-3] + state.u[-1])

        # CRITICAL: Recalculate pressure and other thermodynamic properties
        # through the EOS to maintain consistency
        eos = self._eos
        for cell_idx in [idx, neighbor_idx]:
            rho_cell = state.rho[cell_idx]
            e_cell = state.e[cell_idx]
            # Use EOS to get consistent thermodynamic state
            p_new = eos.pressure(rho_cell, e_cell)
            T_new = eos.temperature(rho_cell, e_cell)
            c_new = eos.sound_speed(rho_cell, p_new)
            s_new = eos.entropy(rho_cell, p_new)
            # gamma may be a property (IdealGas) or method (Cantera)
            gamma_new = eos.get_gamma(rho_cell, p_new) if hasattr(eos, 'get_gamma') else eos.gamma

            state.p[cell_idx] = p_new
            state.T[cell_idx] = T_new
            state.c[cell_idx] = c_new
            state.s[cell_idx] = s_new
            state.gamma[cell_idx] = gamma_new

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

        For porous ghost BC, use compute_interface_state() instead.
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


class MovingPorousPistonBC(BoundaryCondition):
    """
    Porous piston BC with trajectory-based velocity and gas velocity offset.

    This BC models a porous piston where the gas velocity differs from the
    piston velocity by a configurable offset. The piston position is tracked
    from trajectory data, while the gas dynamics use a modified velocity.

    Key behavior:
        - Boundary node position moves at piston velocity (from trajectory)
        - Gas dynamics (Riemann solver, d_tau) use gas velocity
        - Gas velocity = piston velocity + offset, clamped to minimum
        - Mass flux occurs when piston velocity != gas velocity

    Use case: Flame-driven compression where flame position (piston) differs
    from unburned gas velocity at the flame front.

    Features:
        - Trajectory-based velocity input (compatible with Pele data)
        - gas_velocity_offset: additive offset to piston velocity
        - gas_velocity_min: minimum clamp for gas velocity
        - Mass flux tracking for conservation verification

    Reference: [Toro2009] Section 6.3 - Boundary conditions via ghost cells
    """

    def __init__(
        self,
        side: BoundarySide,
        eos: EOSBase,
        trajectory: TrajectoryInterpolator,
        gas_velocity_offset: float = 0.0,
        gas_velocity_min: Optional[float] = None,
        time_offset: float = 0.0,
        tol: float = 1e-8,
        thermal_bc: ThermalBCType = ThermalBCType.ADIABATIC,
        piston_temperature: Optional[float] = None,
    ):
        """
        Initialize porous piston boundary condition with trajectory.

        Parameters
        ----------
        side : BoundarySide
            Which side of the domain (LEFT or RIGHT)
        eos : EOSBase
            Equation of state
        trajectory : TrajectoryInterpolator
            Interpolator for trajectory data. Must have position(t) and
            velocity(t) methods.
        gas_velocity_offset : float
            Value added to piston velocity to get gas velocity [m/s].
            Negative = gas slower than piston.
        gas_velocity_min : float, optional
            Minimum allowed gas velocity [m/s] (default None = no clamping)
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
        self._gas_velocity_offset = gas_velocity_offset
        self._gas_velocity_min = gas_velocity_min
        self._time_offset = time_offset
        self._tol = tol
        self._thermal_bc = thermal_bc
        self._piston_temperature = piston_temperature

        # Import Riemann solver
        from lagrangian_solver.numerics.boundary_riemann import BoundaryRiemannSolver
        self._boundary_solver = BoundaryRiemannSolver(eos, tol=tol)

        # Cache interface state for momentum equation
        self._interface_state = None

        # Current time tracking
        self._current_time = 0.0

        # Mass tracking for diagnostics
        self._mass_leaked = 0.0
        self._initial_boundary_mass = None

        # Merge-split thresholds (same as PorousGhostPistonBC)
        self._merge_low = 0.5   # merge when ratio < 50% (more aggressive)
        self._merge_high = 2.0  # merge when ratio > 200% (more aggressive)

        # Minimum absolute cell size - forces merge-split regardless of ratio
        # Set based on initial grid spacing to prevent CFL collapse
        self._min_cell_size = None  # Will be set on first call to check_merge_split
        self._min_cell_size_factor = 0.1  # Trigger when dx < 10% of initial average dx

    @property
    def trajectory(self) -> TrajectoryInterpolator:
        """Trajectory interpolator."""
        return self._trajectory

    @property
    def thermal_bc(self) -> ThermalBCType:
        """Type of thermal boundary condition."""
        return self._thermal_bc

    @property
    def mass_leaked(self) -> float:
        """
        Cumulative mass that has leaked through the boundary [kg].

        Positive = mass left the domain (piston catching up to gas).
        Negative = mass entered the domain (gas faster than piston).
        """
        return self._mass_leaked

    def _get_data_time(self, t: float) -> float:
        """Convert simulation time to data time."""
        return t - self._time_offset

    def get_piston_velocity(self, t: float) -> float:
        """
        Get piston velocity at time t.

        Returns velocity directly from trajectory data.
        """
        t_data = self._get_data_time(t)
        return self._trajectory.velocity(t_data)

    def get_gas_velocity(self, t: float) -> float:
        """
        Get gas velocity at boundary at time t.

        Applies transformations:
            v_gas = v_piston + offset
            v_gas = max(v_gas, v_min) if v_min specified
        """
        v_piston = self.get_piston_velocity(t)
        v_gas = v_piston + self._gas_velocity_offset

        # Apply minimum velocity clamp if specified
        if self._gas_velocity_min is not None:
            v_gas = max(self._gas_velocity_min, v_gas)

        return v_gas

    def get_boundary_velocity(self, t: float) -> float:
        """
        Get boundary face velocity at time t.

        For porous piston, returns piston velocity (controls node position).
        """
        return self.get_piston_velocity(t)

    def get_piston_position(self, t: float) -> float:
        """Get piston position at time t from trajectory."""
        t_data = self._get_data_time(t)
        return self._trajectory.position(t_data)

    def get_mass_flux(
        self,
        state: FlowState,
        grid: LagrangianGrid,
        t: float,
    ) -> float:
        """
        Get instantaneous mass flux through boundary.

        Mass flux = rho * (v_piston - v_gas) [kg/(m²·s)]

        Positive = mass leaving domain (piston catching up to gas).

        Args:
            state: Current flow state
            grid: Lagrangian grid
            t: Current time [s]

        Returns:
            Mass flux [kg/(m²·s)]
        """
        v_piston = self.get_piston_velocity(t)
        v_gas = self.get_gas_velocity(t)
        rho_boundary = state.rho[self.cell_index]

        return rho_boundary * (v_piston - v_gas)

    def compute_interface_state(
        self,
        state: FlowState,
        grid: LagrangianGrid,
        t: float,
    ):
        """
        Compute interface state by solving boundary Riemann problem.

        Uses the GAS velocity (not piston velocity) to solve the Riemann
        problem. The piston velocity only controls boundary node position.

        Args:
            state: Current flow state
            grid: Lagrangian grid
            t: Current time [s]

        Returns:
            BoundaryState at the interface
        """
        self._current_time = t

        # Use GAS velocity for Riemann problem (controls wave dynamics)
        u_g = self.get_gas_velocity(t)
        idx = self.cell_index

        # Interior state from adjacent cell
        rho_int = state.rho[idx]
        p_int = state.p[idx]

        # Use first interior face velocity (not boundary face)
        if self._side == BoundarySide.LEFT:
            u_int = state.u[1]
            self._interface_state = self._boundary_solver.solve_left_boundary(
                rho_int, u_int, p_int, u_g
            )
        else:
            u_int = state.u[-2]
            self._interface_state = self._boundary_solver.solve_right_boundary(
                rho_int, u_int, p_int, u_g
            )

        return self._interface_state

    def apply_velocity(
        self,
        state: FlowState,
        grid: LagrangianGrid,
        t: float,
    ) -> None:
        """
        Set face velocity to GAS velocity.

        For porous piston, the face velocity for Riemann solver and d_tau
        calculation uses gas velocity. The boundary position is controlled
        separately via apply_position_rate.

        Args:
            state: Flow state to modify
            grid: Lagrangian grid
            t: Current time [s]
        """
        self._current_time = t
        v_gas = self.get_gas_velocity(t)
        state.u[self.face_index] = v_gas

    def apply_position_rate(
        self,
        d_x: np.ndarray,
        state: FlowState,
        grid: LagrangianGrid,
        t: float,
    ) -> None:
        """
        Set boundary position rate to piston velocity.

        This overrides the default d_x = u to track the piston position
        instead of following the gas velocity.

        Args:
            d_x: Position rate array (modified at boundary)
            state: Current flow state
            grid: Lagrangian grid
            t: Current time [s]
        """
        v_piston = self.get_piston_velocity(t)
        d_x[self.face_index] = v_piston

    def apply_momentum(
        self,
        state: FlowState,
        grid: LagrangianGrid,
        d_u: np.ndarray,
        sigma: np.ndarray,
        t: float,
    ) -> None:
        """
        Apply momentum BC at boundary face.

        For porous piston, d_u at boundary follows gas acceleration
        (not piston acceleration) since gas velocity controls the dynamics.

        Args:
            state: Current flow state
            grid: Lagrangian grid
            d_u: Momentum rate array (modified at boundary)
            sigma: Cell-centered total stress [Pa]
            t: Current time [s]
        """
        # Use gas velocity acceleration (controls wave dynamics)
        dt_check = 1e-8
        v_now = self.get_gas_velocity(t)
        v_later = self.get_gas_velocity(t + dt_check)
        dv_dt = (v_later - v_now) / dt_check

        d_u[self.face_index] = dv_dt

    def get_interface_stress(self) -> float:
        """
        Get interface stress for momentum equation.

        Returns the Riemann solution pressure p*, computed using gas velocity.
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

    def update_boundary_mass(
        self,
        grid: LagrangianGrid,
        state: FlowState,
        dt: float,
    ) -> float:
        """
        Update boundary cell mass due to porosity flux.

        Mass flux through boundary:
            dm/dt = -rho_boundary * (u_p - u_g) * A

        When u_p > u_g: piston catches up, mass leaves domain (dm < 0)
        When u_p < u_g: gas escapes, mass enters domain (dm > 0)

        Args:
            grid: Lagrangian grid
            state: Current flow state
            dt: Time step [s]

        Returns:
            Mass change (positive = cell gained mass)
        """
        u_p = self.get_piston_velocity(self._current_time)
        u_g = self.get_gas_velocity(self._current_time)

        rho_boundary = state.rho[self.cell_index]
        A_boundary = 1.0  # Planar geometry

        # Mass flux: negative when piston catches up to gas
        dm = -rho_boundary * (u_p - u_g) * A_boundary * dt

        # Store initial mass for diagnostics
        if self._initial_boundary_mass is None:
            self._initial_boundary_mass = grid.dm[self.cell_index]

        # Limit mass drain to prevent negative mass (50% max per step)
        current_mass = grid.dm[self.cell_index]
        max_drain = -0.5 * current_mass
        if dm < max_drain:
            dm = max_drain

        # Update grid mass
        grid.add_boundary_mass(self._side, dm)

        # Track cumulative mass leak
        self._mass_leaked -= dm

        return dm

    def check_merge_split(
        self,
        grid: LagrangianGrid,
        state: FlowState,
    ) -> bool:
        """
        Check if boundary cells need recursive merge-split.

        Triggers merge-split based on THREE criteria:
        1. Mass ratio: if boundary cell mass is too small/large vs neighbor
        2. Volume ratio: if boundary cell volume (dx) is too small vs neighbor
        3. Absolute minimum: if cell volume drops below absolute threshold

        The absolute minimum check prevents CFL collapse even when ratios
        are acceptable (e.g., when ALL cells are shrinking uniformly).

        Uses RECURSIVE merge-split: after merging cells 0+1, also check if
        cell 1 needs to merge with cell 2, and so on.

        Args:
            grid: Lagrangian grid
            state: Current flow state

        Returns:
            True if any merge-split was performed
        """
        # Initialize minimum cell size on first call
        if self._min_cell_size is None:
            # Use average initial cell size as reference
            domain_length = state.x[-1] - state.x[0]
            avg_dx = domain_length / grid.n_cells
            self._min_cell_size = self._min_cell_size_factor * avg_dx

        any_merged = False

        if self._side == BoundarySide.LEFT:
            # Start from boundary and work inward
            # Allow redistribution up to half the grid
            idx = 0
            max_idx = min(grid.n_cells - 2, grid.n_cells // 2)
            while idx < max_idx:
                neighbor_idx = idx + 1
                if grid.dm[neighbor_idx] < 1e-15:
                    break  # Avoid division by zero

                # Check mass ratio
                mass_ratio = grid.dm[idx] / grid.dm[neighbor_idx]
                needs_merge_mass = mass_ratio < self._merge_low or mass_ratio > self._merge_high

                # Check volume ratio (CRITICAL for CFL stability)
                dx_idx = state.x[idx + 1] - state.x[idx]
                dx_neighbor = state.x[neighbor_idx + 1] - state.x[neighbor_idx]
                if dx_neighbor > 1e-15:
                    vol_ratio = dx_idx / dx_neighbor
                    needs_merge_vol = vol_ratio < self._merge_low or vol_ratio > self._merge_high
                else:
                    needs_merge_vol = True

                # Check absolute minimum cell size
                needs_merge_abs = dx_idx < self._min_cell_size

                if needs_merge_mass or needs_merge_vol or needs_merge_abs:
                    self._conservative_merge_split(grid, state, idx, neighbor_idx)
                    any_merged = True
                    idx += 1  # Check next pair
                else:
                    break  # No more merging needed
        else:
            # Right boundary: work from right to left
            # Allow redistribution up to half the grid
            idx = grid.n_cells - 1
            min_idx = max(1, grid.n_cells // 2)
            while idx > min_idx:
                neighbor_idx = idx - 1
                if grid.dm[neighbor_idx] < 1e-15:
                    break

                # Check mass ratio
                mass_ratio = grid.dm[idx] / grid.dm[neighbor_idx]
                needs_merge_mass = mass_ratio < self._merge_low or mass_ratio > self._merge_high

                # Check volume ratio (CRITICAL for CFL stability)
                dx_idx = state.x[idx + 1] - state.x[idx]
                dx_neighbor = state.x[neighbor_idx + 1] - state.x[neighbor_idx]
                if dx_neighbor > 1e-15:
                    vol_ratio = dx_idx / dx_neighbor
                    needs_merge_vol = vol_ratio < self._merge_low or vol_ratio > self._merge_high
                else:
                    needs_merge_vol = True

                # Check absolute minimum cell size
                needs_merge_abs = dx_idx < self._min_cell_size

                if needs_merge_mass or needs_merge_vol or needs_merge_abs:
                    self._conservative_merge_split(grid, state, idx, neighbor_idx)
                    any_merged = True
                    idx -= 1
                else:
                    break

        return any_merged

    def _conservative_merge_split(
        self,
        grid: LagrangianGrid,
        state: FlowState,
        idx: int,
        neighbor_idx: int,
    ) -> None:
        """
        Merge cells idx and neighbor_idx, then split equally.

        Conserves mass and internal energy exactly.
        Works for any adjacent cell pair, not just boundary cells.

        CRITICAL: This method also moves the cell interface so that both
        cells end up with the SAME DENSITY (the average density of the
        merged region). This prevents artificial pressure discontinuities.

        The algorithm:
        1. Compute total mass and volume of merged region
        2. Average density = total mass / total volume
        3. Split mass equally: m_half = m_total / 2
        4. Move interface so each cell has volume = m_half / rho_avg = V_total / 2
        5. Both cells now have same density → no pressure discontinuity

        Args:
            grid: Lagrangian grid (modified in place)
            state: Current flow state (modified in place)
            idx: First cell index
            neighbor_idx: Second cell index (must be adjacent)
        """
        # Determine which cell is left vs right
        left_idx = min(idx, neighbor_idx)
        right_idx = max(idx, neighbor_idx)

        # Gather conserved quantities
        m0 = grid.dm[left_idx]
        m1 = grid.dm[right_idx]
        m_total = m0 + m1

        # Cell volumes
        dx0 = state.x[left_idx + 1] - state.x[left_idx]
        dx1 = state.x[right_idx + 1] - state.x[right_idx]
        V_total = dx0 + dx1

        # Average density of merged region
        rho_avg = m_total / V_total

        # Internal energy (conserved)
        E_total = m0 * state.e[left_idx] + m1 * state.e[right_idx]

        # Redistribute mass equally
        m_half = m_total / 2.0

        # Update grid masses
        grid._dm[left_idx] = m_half
        grid._dm[right_idx] = m_half

        # Recalculate cumulative mass from scratch
        grid._m[0] = 0.0
        for j in range(grid.n_cells):
            grid._m[j + 1] = grid._m[j] + grid._dm[j]

        # CRITICAL: Move the interface so both cells have equal volume
        # This maintains constant density across the merged region
        V_half = V_total / 2.0
        x_left = state.x[left_idx]
        x_right = state.x[right_idx + 1]
        x_interface_new = x_left + V_half

        # Update interface position in state and grid
        state.x[left_idx + 1] = x_interface_new
        grid._x[left_idx + 1] = x_interface_new

        # Average internal energy for both cells (conserves total internal energy)
        e_avg = E_total / m_total
        state.e[left_idx] = e_avg
        state.e[right_idx] = e_avg

        # Both cells now have the same density (rho_avg)
        state.rho[left_idx] = rho_avg
        state.rho[right_idx] = rho_avg
        state.tau[left_idx] = 1.0 / rho_avg
        state.tau[right_idx] = 1.0 / rho_avg

        # CRITICAL: Recalculate pressure and other thermodynamic properties
        # through the EOS to maintain consistency
        # Since both cells have same rho and e, they'll have the same pressure
        eos = self._eos
        p_new = eos.pressure(rho_avg, e_avg)
        T_new = eos.temperature(rho_avg, e_avg)
        c_new = eos.sound_speed(rho_avg, p_new)
        s_new = eos.entropy(rho_avg, p_new)
        gamma_new = eos.get_gamma(rho_avg, p_new) if hasattr(eos, 'get_gamma') else eos.gamma

        for cell_idx in [left_idx, right_idx]:
            state.p[cell_idx] = p_new
            state.T[cell_idx] = T_new
            state.c[cell_idx] = c_new
            state.s[cell_idx] = s_new
            state.gamma[cell_idx] = gamma_new

    # Legacy interface
    def compute_flux(
        self,
        state: FlowState,
        grid: LagrangianGrid,
        t: float,
    ) -> BoundaryFlux:
        """Compute numerical flux at boundary (legacy interface)."""
        if self._interface_state is None:
            self.compute_interface_state(state, grid, t)

        p_star = self._interface_state.p
        v_gas = self.get_gas_velocity(t)

        return BoundaryFlux(
            p_flux=p_star,
            pu_flux=p_star * v_gas,
            u_flux=v_gas,
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
        v_final = max(velocity_min, v_data * velocity_scale + velocity_offset)

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
            Value to add to scaled velocity [m/s] (default 0.0, use negative to subtract)
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
            v_final = max(velocity_min, v_data * velocity_scale + velocity_offset)
        """
        t_data = self._get_data_time(t)
        v = self._trajectory.velocity(t_data)

        # Apply scale then offset
        v_modified = v * self._velocity_scale + self._velocity_offset

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
