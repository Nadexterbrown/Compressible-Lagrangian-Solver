"""
Flame-coupled piston boundary condition.

Implements iterative coupling between piston velocity and flame properties
using the Clavin-Tofaili formulation:

    u_p = (sigma(t) - 1) * (rho_u / rho_b) * S_L

where S_L and densities depend on the local (P, T) at the piston face,
which in turn depends on u_p through shock relations. This creates a
nonlinear coupling that requires iteration at each timestep.

Reference:
    Clavin, P., & Tofaili, H. (2021). Flame elongation model.
    Toro, E.F. (2009). Riemann Solvers and Numerical Methods for Fluid Dynamics,
        Section 6.3 - Boundary conditions.
"""

from dataclasses import dataclass
from typing import Optional, Union, Callable
import numpy as np
from scipy.special import erf

# Import solver components
import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).resolve().parents[2] / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from lagrangian_solver.boundary.base import (
    BoundaryCondition,
    BoundarySide,
    ThermalBCType,
    BoundaryFlux,
)
from lagrangian_solver.core.state import FlowState
from lagrangian_solver.core.grid import LagrangianGrid
from lagrangian_solver.equations.eos import EOSBase
from lagrangian_solver.numerics.boundary_riemann import BoundaryRiemannSolver, BoundaryState

# Import local modules
from flame_elongation_trajectory import FlameElongationTrajectory


@dataclass
class IterationResult:
    """
    Result of the iterative velocity solve.

    Attributes:
        velocity: Converged piston velocity [m/s]
        P_face: Converged face pressure [Pa]
        T_face: Converged face temperature [K]
        iterations: Number of iterations taken
        converged: Whether iteration converged
        relative_change: Final relative change in velocity
    """

    velocity: float
    P_face: float
    T_face: float
    iterations: int
    converged: bool
    relative_change: float


class FlameCoupledPistonBC(BoundaryCondition):
    """
    Piston BC with iterative flame-velocity coupling.

    At each timestep, solves the nonlinear coupling:
    1. Guess piston velocity u_p
    2. Solve boundary Riemann problem to get (P_face, T_face)
    3. Look up flame properties at (P_face, T_face)
    4. Compute new u_p from Clavin-Tofaili formula
    5. Repeat until converged

    This BC is physically correct for flame-driven piston motion where
    the flame properties depend on the local thermodynamic state.

    Features:
    - Automatic velocity ramping to prevent shock startup issues
    - Caching to avoid redundant solves in Heun's method
    - Under-relaxation for iteration stability
    - Convergence diagnostics

    Reference: Clavin & Tofaili (2021), Toro (2009) Section 6.3
    """

    # Default parameters
    DEFAULT_TOL = 1e-6
    DEFAULT_MAX_ITER = 20
    DEFAULT_RAMP_TIME = 30e-6
    DEFAULT_RELAXATION = 0.7

    def __init__(
        self,
        side: BoundarySide,
        eos: EOSBase,
        flame_trajectory: FlameElongationTrajectory,
        tol: float = DEFAULT_TOL,
        max_iter: int = DEFAULT_MAX_ITER,
        ramp_time: float = DEFAULT_RAMP_TIME,
        startup_time: float = 0.0,
        relaxation: float = DEFAULT_RELAXATION,
        thermal_bc: ThermalBCType = ThermalBCType.ADIABATIC,
        piston_temperature: Optional[float] = None,
    ):
        """
        Initialize flame-coupled piston BC.

        Args:
            side: Which side of the domain (LEFT or RIGHT)
            eos: Equation of state
            flame_trajectory: FlameElongationTrajectory with elongation function
                             and flame property interpolator
            tol: Convergence tolerance for velocity iteration
            max_iter: Maximum iterations per timestep
            ramp_time: Time to ramp velocity from 0 to target [s]
            startup_time: Delay before ramp begins [s]
            relaxation: Under-relaxation factor (0 < w <= 1)
            thermal_bc: Type of thermal boundary condition
            piston_temperature: Piston temperature [K] (for isothermal)
        """
        super().__init__(side, eos)

        self._trajectory = flame_trajectory
        self._tol = tol
        self._max_iter = max_iter
        self._ramp_time = ramp_time
        self._startup_time = startup_time
        self._relaxation = relaxation
        self._thermal_bc = thermal_bc
        self._piston_temperature = piston_temperature

        # Boundary Riemann solver for computing interface state
        self._boundary_solver = BoundaryRiemannSolver(eos, tol=1e-8)

        # Cache for avoiding redundant solves (Heun's method calls twice per step)
        self._cached_time: Optional[float] = None
        self._cached_velocity: Optional[float] = None
        self._cached_result: Optional[IterationResult] = None

        # Interface state (for diagnostics and momentum equation)
        self._interface_state: Optional[BoundaryState] = None

        # Iteration history for diagnostics
        self._iteration_history: list[IterationResult] = []

    @property
    def trajectory(self) -> FlameElongationTrajectory:
        """The flame elongation trajectory."""
        return self._trajectory

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
    def interface_state(self) -> Optional[BoundaryState]:
        """Most recent interface state from boundary Riemann solve."""
        return self._interface_state

    @property
    def iteration_history(self) -> list[IterationResult]:
        """History of iteration results for diagnostics."""
        return self._iteration_history

    def _apply_ramp(self, velocity: float, t: float) -> float:
        """
        Apply smooth erf ramp to velocity.

        Uses error function for smooth S-curve acceleration profile:
        v(t) = v_target * (1 + erf(6*(t-t_mid)/t_ramp)) / 2

        Args:
            velocity: Target velocity [m/s]
            t: Current time [s]

        Returns:
            Ramped velocity [m/s]
        """
        if self._ramp_time <= 0:
            return velocity

        t_start = self._startup_time

        if t <= t_start:
            return 0.0
        if t >= t_start + self._ramp_time:
            return velocity

        # Map [t_start, t_start+t_ramp] to [-3, +3] for smooth S-curve
        t_rel = t - t_start
        x = 6.0 * (t_rel / self._ramp_time) - 3.0
        return velocity * (1.0 + erf(x)) / 2.0

    def _solve_coupled_velocity(
        self,
        state: FlowState,
        grid: LagrangianGrid,
        t: float,
    ) -> IterationResult:
        """
        Iteratively solve for converged piston velocity.

        Algorithm:
        1. Initial guess from cached value or uncoupled velocity
        2. Solve boundary Riemann problem for (P_face, T_face)
        3. Compute new velocity from Clavin-Tofaili formula
        4. Check convergence; if not converged, apply under-relaxation
        5. Repeat until converged or max_iter reached

        Args:
            state: Current flow state
            grid: Lagrangian grid
            t: Current time [s]

        Returns:
            IterationResult with converged velocity and diagnostics
        """
        # Check cache first (avoids redundant solves in Heun's method)
        if self._cached_time is not None and np.isclose(t, self._cached_time, rtol=1e-12):
            return self._cached_result

        # Initial guess: use cached velocity or uncoupled
        if self._cached_velocity is not None:
            u_p = self._cached_velocity
        else:
            u_p = self._trajectory.velocity_uncoupled(t)

        # Apply ramp to initial guess
        u_p = self._apply_ramp(u_p, t)

        # Get interior state
        idx = self.cell_index
        rho_int = state.rho[idx]
        p_int = state.p[idx]

        # Use first interior face velocity (not boundary face)
        if self._side == BoundarySide.LEFT:
            u_int = state.u[1]
        else:
            u_int = state.u[-2]

        # Iteration loop
        converged = False
        iterations = 0
        rel_change = 1.0
        P_face = p_int
        T_face = state.T[idx] if hasattr(state, 'T') else self._estimate_temperature(p_int, rho_int)

        for iteration in range(self._max_iter):
            iterations = iteration + 1

            # 1. Solve boundary Riemann problem
            if self._side == BoundarySide.LEFT:
                bstate = self._boundary_solver.solve_left_boundary(
                    rho_int, u_int, p_int, u_p
                )
            else:
                bstate = self._boundary_solver.solve_right_boundary(
                    rho_int, u_int, p_int, u_p
                )

            P_face = bstate.p
            # Estimate T_face from interface state
            T_face = self._temperature_from_state(bstate)

            # 2. Compute new velocity from Clavin-Tofaili formula
            sigma = self._trajectory.sigma(t)
            props = self._trajectory.interpolator.get_properties(P_face, T_face)
            u_p_new = (sigma - 1.0) * props.density_ratio * props.S_L

            # Apply ramp
            u_p_new = self._apply_ramp(u_p_new, t)

            # 3. Check convergence
            if abs(u_p) > 1e-10:
                rel_change = abs(u_p_new - u_p) / abs(u_p)
            else:
                rel_change = abs(u_p_new - u_p)

            if rel_change < self._tol:
                converged = True
                u_p = u_p_new
                break

            # 4. Under-relaxation for stability
            u_p = self._relaxation * u_p_new + (1.0 - self._relaxation) * u_p

        # Store interface state
        self._interface_state = bstate

        # Create result
        result = IterationResult(
            velocity=u_p,
            P_face=P_face,
            T_face=T_face,
            iterations=iterations,
            converged=converged,
            relative_change=rel_change,
        )

        # Update cache
        self._cached_time = t
        self._cached_velocity = u_p
        self._cached_result = result

        # Store in history
        self._iteration_history.append(result)

        return result

    def _estimate_temperature(self, p: float, rho: float) -> float:
        """
        Estimate temperature from pressure and density using ideal gas.

        T = p / (rho * R_specific)

        For air: R_specific ≈ 287 J/(kg·K)
        """
        R_specific = 287.0  # Approximate for air
        return p / (rho * R_specific)

    def _temperature_from_state(self, bstate: BoundaryState) -> float:
        """
        Compute temperature from boundary state.

        Uses ideal gas law: T = p / (rho * R_specific)
        """
        R_specific = 287.0  # Approximate for air
        return bstate.p / (bstate.rho * R_specific)

    def get_boundary_velocity(self, t: float) -> float:
        """
        Get piston velocity at time t.

        NOTE: This returns the cached velocity from the last solve.
        For accurate results, call apply_velocity() first.
        """
        if self._cached_velocity is not None and self._cached_time is not None:
            if np.isclose(t, self._cached_time, rtol=1e-12):
                return self._cached_velocity

        # Fall back to uncoupled velocity with ramp
        v_uncoupled = self._trajectory.velocity_uncoupled(t)
        return self._apply_ramp(v_uncoupled, t)

    def apply_velocity(
        self,
        state: FlowState,
        grid: LagrangianGrid,
        t: float,
    ) -> None:
        """
        Set face velocity to converged flame-coupled value.

        This solves the iterative coupling and sets the boundary
        face velocity to the converged piston velocity.
        """
        result = self._solve_coupled_velocity(state, grid, t)
        state.u[self.face_index] = result.velocity

    def apply_momentum(
        self,
        state: FlowState,
        grid: LagrangianGrid,
        d_u: np.ndarray,
        sigma: np.ndarray,
        t: float,
    ) -> None:
        """
        Compute momentum rate at piston face.

        For time-varying piston velocity, d_u = du_p/dt computed
        via finite difference.
        """
        # Finite difference for piston acceleration
        dt_check = 1e-8

        # Get current velocity (should be cached from apply_velocity)
        v_now = self.get_boundary_velocity(t)

        # Estimate velocity at t + dt (uncoupled + ramp)
        v_later_uncoupled = self._trajectory.velocity_uncoupled(t + dt_check)
        v_later = self._apply_ramp(v_later_uncoupled, t + dt_check)

        dv_dt = (v_later - v_now) / dt_check
        d_u[self.face_index] = dv_dt

    def compute_interface_state(
        self,
        state: FlowState,
        grid: LagrangianGrid,
        t: float,
    ) -> BoundaryState:
        """
        Compute interface state by solving boundary Riemann problem.

        This is called automatically during apply_velocity(), but can
        be called explicitly for diagnostics.

        Returns:
            BoundaryState at the interface
        """
        # Ensure coupling has been solved
        if self._interface_state is None or not np.isclose(t, self._cached_time, rtol=1e-12):
            self._solve_coupled_velocity(state, grid, t)

        return self._interface_state

    def get_interface_stress(self) -> float:
        """
        Get interface stress for momentum equation.

        Returns the Riemann solution pressure p*, which is the
        correct interface pressure for any wave type.
        """
        if self._interface_state is None:
            raise RuntimeError(
                "Interface state not computed. Call apply_velocity or "
                "compute_interface_state first."
            )
        return self._interface_state.sigma

    def has_ghost_cell(self) -> bool:
        """
        Indicate this BC provides ghost cell stress.

        Used by conservation equations to detect ghost cell BCs.
        """
        return True

    def clear_cache(self) -> None:
        """Clear cached velocity (call at start of new timestep if needed)."""
        self._cached_time = None
        self._cached_velocity = None
        self._cached_result = None

    def clear_history(self) -> None:
        """Clear iteration history."""
        self._iteration_history.clear()

    def get_convergence_stats(self) -> dict:
        """
        Get statistics on iteration convergence.

        Returns:
            Dictionary with convergence statistics
        """
        if not self._iteration_history:
            return {}

        iterations = [r.iterations for r in self._iteration_history]
        converged = [r.converged for r in self._iteration_history]

        return {
            "n_timesteps": len(self._iteration_history),
            "avg_iterations": np.mean(iterations),
            "max_iterations": max(iterations),
            "min_iterations": min(iterations),
            "convergence_rate": sum(converged) / len(converged),
            "failures": sum(1 for c in converged if not c),
        }

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
        LEGACY: Compute numerical flux at piston.

        For flame-coupled BC, use apply_velocity() + get_interface_stress().
        """
        # Ensure interface state is computed
        if self._interface_state is None:
            self.compute_interface_state(state, grid, t)

        v_piston = self.get_boundary_velocity(t)
        p_star = self._interface_state.p

        return BoundaryFlux(
            p_flux=p_star,
            pu_flux=p_star * v_piston,
            u_flux=v_piston,
        )

    def __repr__(self) -> str:
        return (
            f"FlameCoupledPistonBC(\n"
            f"  side={self._side.name},\n"
            f"  trajectory={self._trajectory.elongation.name},\n"
            f"  tol={self._tol}, max_iter={self._max_iter},\n"
            f"  ramp_time={self._ramp_time*1e6:.1f} us\n"
            f")"
        )
