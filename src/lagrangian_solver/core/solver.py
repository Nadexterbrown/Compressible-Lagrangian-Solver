"""
Compatible Lagrangian solver with exact energy conservation.

This module provides the CompatibleLagrangianSolver class that uses
compatible energy discretization to guarantee exact total energy
conservation to machine precision.

Key Design:
    - Internal energy e is PRIMARY evolved variable (not total E)
    - Uses cell-centered stress σ = p + Q for BOTH momentum AND energy
    - State reconstruction uses from_internal_energy() - NO subtraction
    - Guarantees |ΔE/E| < 10^-12 over any simulation

References:
    [Caramana1998] Caramana et al. (1998) JCP 146:227-262
    [Burton1992] Burton (1992) UCRL-JC-105926
    [Despres2017] Chapter 5 - Complete Lagrangian algorithm
    [Toro2009] Chapter 6 - Godunov-type methods
"""

from dataclasses import dataclass, field
from typing import Optional, Callable, List, Tuple
import numpy as np
import time as timer

from lagrangian_solver.core.state import (
    FlowState,
    ConservedVariables,
    create_riemann_state,
    create_uniform_state,
)
from lagrangian_solver.core.grid import LagrangianGrid, GridConfig
from lagrangian_solver.equations.eos import EOSBase, IdealGasEOS, CanteraEOS
from lagrangian_solver.equations.conservation import (
    CompatibleConservation,
    compute_mass_error,
    compute_total_energy,
    compute_momentum,
    compute_internal_energy,
    compute_kinetic_energy,
)
from lagrangian_solver.numerics.time_integration import (
    TimeIntegratorBase,
    CompatibleHeunIntegrator,
    TimeStepInfo,
)
from lagrangian_solver.numerics.artificial_viscosity import (
    ArtificialViscosity,
    ArtificialViscosityConfig,
)
from lagrangian_solver.boundary.base import BoundaryCondition


@dataclass
class SolverConfig:
    """
    Solver configuration parameters.

    Attributes:
        cfl: CFL number for time step control
        t_end: Final simulation time [s]
        dt_output: Time interval between output writes [s]
        dt_max: Maximum time step [s] (None for no limit)
        dt_min: Minimum time step floor [s] (None for no floor)
        verbose: Print progress messages
        av_linear: Linear AV coefficient (Landshoff, default 0.1)
        av_quad: Quadratic AV coefficient (VNR, default 0.5)
        av_enabled: Whether artificial viscosity is enabled (default True)

    Note:
        Default AV coefficients (0.1, 0.5) are lower than traditional values
        (0.3, 2.0) to prevent excessive heating at piston boundaries during
        shock formation. Higher values can cause 15%+ density errors at
        boundary-adjacent cells due to over-dissipation.
    """

    cfl: float = 0.5
    t_end: float = 1.0
    dt_output: float = 0.1
    dt_max: Optional[float] = None
    dt_min: Optional[float] = None
    verbose: bool = True
    av_linear: float = 0.1
    av_quad: float = 0.5
    av_enabled: bool = True

    # Legacy field for backward compatibility
    artificial_viscosity: Optional[ArtificialViscosityConfig] = None

    def __post_init__(self):
        """Initialize AV config from legacy field or new fields."""
        if self.artificial_viscosity is not None:
            # Use legacy config
            self.av_linear = self.artificial_viscosity.c_linear
            self.av_quad = self.artificial_viscosity.c_quad
            self.av_enabled = self.artificial_viscosity.enabled


@dataclass
class SolverStatistics:
    """
    Statistics from a solver run.

    Attributes:
        n_steps: Total number of time steps
        wall_time: Total wall-clock time [s]
        final_time: Final simulation time [s]
        min_dt: Minimum time step used [s]
        max_dt: Maximum time step used [s]
        avg_dt: Average time step [s]
        mass_error: Final relative mass conservation error
        energy_change: Relative change in total energy
        initial_energy: Initial total energy [J]
        final_energy: Final total energy [J]
    """

    n_steps: int = 0
    wall_time: float = 0.0
    final_time: float = 0.0
    min_dt: float = float("inf")
    max_dt: float = 0.0
    avg_dt: float = 0.0
    mass_error: float = 0.0
    energy_change: float = 0.0
    initial_energy: float = 0.0
    final_energy: float = 0.0


class CompatibleLagrangianSolver:
    """
    1D Lagrangian solver with compatible energy discretization.

    Key features:
    - Internal energy e is PRIMARY evolved variable (not E)
    - Uses cell-centered stress σ = p + Q for BOTH momentum and energy
    - Guarantees exact total energy conservation to machine precision
    - Artificial viscosity for shock capturing

    The key insight: by using the SAME stress in both momentum and
    energy equations, the discrete work done accelerating the fluid
    exactly equals the discrete energy change. This gives energy
    conservation to machine precision (O(10^-15) relative error).

    Reference: Caramana et al. (1998), Burton (1992)
    """

    def __init__(
        self,
        grid: LagrangianGrid,
        eos: EOSBase,
        bc_left: BoundaryCondition,
        bc_right: BoundaryCondition,
        config: Optional[SolverConfig] = None,
    ):
        """
        Initialize the compatible Lagrangian solver.

        Args:
            grid: Lagrangian grid
            eos: Equation of state
            bc_left: Left boundary condition
            bc_right: Right boundary condition
            config: Solver configuration
        """
        self._grid = grid
        self._eos = eos
        self._bc_left = bc_left
        self._bc_right = bc_right
        self._config = config or SolverConfig()

        # Create artificial viscosity
        if self._config.av_enabled:
            av_config = ArtificialViscosityConfig(
                c_linear=self._config.av_linear,
                c_quad=self._config.av_quad,
                enabled=True,
            )
            self._av = ArtificialViscosity(av_config)
        else:
            self._av = None

        # Create conservation law handler (compatible discretization)
        self._conservation = CompatibleConservation(eos, self._av)

        # Create time integrator (compatible - evolves internal energy)
        self._integrator = CompatibleHeunIntegrator(eos, cfl=self._config.cfl)

        # State
        self._state: Optional[FlowState] = None
        self._time = 0.0
        self._step = 0

        # Statistics
        self._stats = SolverStatistics()

        # Callbacks
        self._step_callbacks: List[Callable[[FlowState, float, int], None]] = []

    @property
    def grid(self) -> LagrangianGrid:
        """Lagrangian grid."""
        return self._grid

    @property
    def eos(self) -> EOSBase:
        """Equation of state."""
        return self._eos

    @property
    def state(self) -> Optional[FlowState]:
        """Current flow state."""
        return self._state

    @property
    def time(self) -> float:
        """Current simulation time."""
        return self._time

    @property
    def step(self) -> int:
        """Current time step number."""
        return self._step

    @property
    def config(self) -> SolverConfig:
        """Solver configuration."""
        return self._config

    @property
    def statistics(self) -> SolverStatistics:
        """Solver statistics."""
        return self._stats

    def set_initial_condition(self, state: FlowState) -> None:
        """
        Set the initial condition.

        Args:
            state: Initial flow state
        """
        self._state = state.copy()
        self._grid.initialize_mass(state.rho)
        self._time = 0.0
        self._step = 0

    def set_riemann_ic(
        self,
        x_disc: float,
        rho_L: float,
        u_L: float,
        p_L: float,
        rho_R: float,
        u_R: float,
        p_R: float,
    ) -> None:
        """
        Set a Riemann problem initial condition.

        Args:
            x_disc: Position of discontinuity [m]
            rho_L, u_L, p_L: Left state
            rho_R, u_R, p_R: Right state
        """
        state = create_riemann_state(
            n_cells=self._grid.n_cells,
            x_left=self._grid.x[0],
            x_right=self._grid.x[-1],
            x_discontinuity=x_disc,
            rho_L=rho_L,
            u_L=u_L,
            p_L=p_L,
            rho_R=rho_R,
            u_R=u_R,
            p_R=p_R,
            eos=self._eos,
        )
        self.set_initial_condition(state)

    def add_step_callback(
        self, callback: Callable[[FlowState, float, int], None]
    ) -> None:
        """
        Add a callback function called after each time step.

        Args:
            callback: Function(state, time, step) called after each step
        """
        self._step_callbacks.append(callback)

    def _compute_rhs(
        self, state: FlowState, grid: LagrangianGrid
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the right-hand side of the ODE system.

        Returns (d_tau, d_u, d_e, d_x) using compatible discretization.
        """
        # Apply boundary velocities first
        self._bc_left.apply_velocity(state, grid, self._time)
        self._bc_right.apply_velocity(state, grid, self._time)

        # Compute residual using compatible discretization
        return self._conservation.compute_residual(
            state, grid, self._bc_left, self._bc_right, self._time
        )

    def compute_total_energy(self) -> float:
        """
        Compute total energy (for diagnostics).

        Uses the staggered grid formula with face-centered kinetic energy.

        Returns:
            Total energy [J]
        """
        return compute_total_energy(self._state, self._grid)

    def step_forward(self, dt: Optional[float] = None) -> TimeStepInfo:
        """
        Advance the solution by one time step.

        Args:
            dt: Time step size (computed from CFL if None)

        Returns:
            TimeStepInfo with step diagnostics
        """
        if self._state is None:
            raise RuntimeError("Initial condition not set")

        # Apply boundary velocities before computing time step
        self._bc_left.apply_velocity(self._state, self._grid, self._time)
        self._bc_right.apply_velocity(self._state, self._grid, self._time)

        # Compute time step if not provided
        if dt is None:
            ts_info = self._integrator.compute_timestep(self._state, self._grid)
            dt = ts_info.dt
        else:
            ts_info = TimeStepInfo(
                dt=dt,
                cfl_used=self._config.cfl,
                limiting_cell=0,
                max_wave_speed=0.0,
            )

        # Apply time step limits
        if self._config.dt_max is not None:
            dt = min(dt, self._config.dt_max)
            ts_info.dt = dt

        if self._config.dt_min is not None:
            dt = max(dt, self._config.dt_min)
            ts_info.dt = dt

        # Don't exceed final time
        if self._time + dt > self._config.t_end:
            dt = self._config.t_end - self._time
            ts_info.dt = dt

        # Advance solution using compatible integrator
        self._state = self._integrator.step(
            self._state,
            self._grid,
            dt,
            self._compute_rhs,
        )

        # Update time and step counter
        self._time += dt
        self._step += 1

        # Update statistics
        self._stats.min_dt = min(self._stats.min_dt, dt)
        self._stats.max_dt = max(self._stats.max_dt, dt)

        # Call callbacks
        for callback in self._step_callbacks:
            callback(self._state, self._time, self._step)

        return ts_info

    def run(self, writer=None) -> SolverStatistics:
        """
        Run the simulation to completion.

        Args:
            writer: Output writer (optional)

        Returns:
            SolverStatistics with run diagnostics
        """
        if self._state is None:
            raise RuntimeError("Initial condition not set")

        # Record initial state
        initial_energy = self.compute_total_energy()
        self._stats.initial_energy = initial_energy
        t_next_output = 0.0

        # Start timing
        wall_start = timer.perf_counter()

        # Write initial condition
        if writer is not None:
            from lagrangian_solver.io.output import OutputFrame
            frame = OutputFrame.from_state(
                self._state, self._grid, self._time, self._step
            )
            writer.write_frame(frame)
            t_next_output = self._config.dt_output

        # Main time loop
        while self._time < self._config.t_end - 1e-12:
            # Take time step
            ts_info = self.step_forward()

            # Progress output
            if self._config.verbose and self._step % 100 == 0:
                current_energy = self.compute_total_energy()
                energy_error = abs(current_energy - initial_energy) / abs(initial_energy)
                print(
                    f"Step {self._step:6d}: t = {self._time:.6e}, "
                    f"dt = {ts_info.dt:.6e}, "
                    f"ΔE/E = {energy_error:.6e}"
                )

            # Write output at specified intervals
            if writer is not None and self._time >= t_next_output - 1e-12:
                from lagrangian_solver.io.output import OutputFrame
                frame = OutputFrame.from_state(
                    self._state, self._grid, self._time, self._step
                )
                writer.write_frame(frame)
                t_next_output += self._config.dt_output

        # Write final state
        if writer is not None:
            from lagrangian_solver.io.output import OutputFrame
            frame = OutputFrame.from_state(
                self._state, self._grid, self._time, self._step
            )
            writer.write_frame(frame)

        # Compute final statistics
        wall_end = timer.perf_counter()

        self._stats.n_steps = self._step
        self._stats.wall_time = wall_end - wall_start
        self._stats.final_time = self._time
        self._stats.avg_dt = self._time / max(self._step, 1)
        self._stats.mass_error = compute_mass_error(self._state, self._grid)

        final_energy = self.compute_total_energy()
        self._stats.final_energy = final_energy
        if abs(initial_energy) > 1e-15:
            self._stats.energy_change = (
                abs(final_energy - initial_energy) / initial_energy
            )

        if self._config.verbose:
            self._print_statistics()

        return self._stats

    def _print_statistics(self) -> None:
        """Print solver statistics."""
        print("\n" + "=" * 60)
        print("COMPATIBLE LAGRANGIAN SOLVER STATISTICS")
        print("=" * 60)
        print(f"Total steps:      {self._stats.n_steps}")
        print(f"Wall time:        {self._stats.wall_time:.3f} s")
        print(f"Final time:       {self._stats.final_time:.6e} s")
        print(f"Min dt:           {self._stats.min_dt:.6e} s")
        print(f"Max dt:           {self._stats.max_dt:.6e} s")
        print(f"Avg dt:           {self._stats.avg_dt:.6e} s")
        print(f"Mass error:       {self._stats.mass_error:.6e}")
        print(f"Energy change:    {self._stats.energy_change:.6e}")
        print(f"Initial energy:   {self._stats.initial_energy:.6e} J")
        print(f"Final energy:     {self._stats.final_energy:.6e} J")
        print("=" * 60)


# Keep LagrangianSolver as an alias for backward compatibility
class LagrangianSolver(CompatibleLagrangianSolver):
    """
    Alias for CompatibleLagrangianSolver.

    NOTE: This now uses compatible energy discretization.
    For the old Riemann-based solver, see git history.
    """

    def __init__(
        self,
        grid: LagrangianGrid,
        eos: EOSBase,
        riemann_solver=None,
        time_integrator=None,
        bc_left: Optional[BoundaryCondition] = None,
        bc_right: Optional[BoundaryCondition] = None,
        config: Optional[SolverConfig] = None,
    ):
        """
        Initialize Lagrangian solver (backward compatible interface).

        NOTE: riemann_solver and time_integrator parameters are ignored.
        The compatible discretization doesn't use Riemann solvers for
        interior faces.
        """
        if riemann_solver is not None:
            import warnings
            warnings.warn(
                "riemann_solver parameter is ignored. Compatible discretization "
                "uses cell-centered stress, not Riemann fluxes.",
                DeprecationWarning,
                stacklevel=2,
            )

        if time_integrator is not None:
            import warnings
            warnings.warn(
                "time_integrator parameter is ignored. Compatible discretization "
                "uses CompatibleHeunIntegrator.",
                DeprecationWarning,
                stacklevel=2,
            )

        # Create default reflective BCs if not provided
        if bc_left is None:
            from lagrangian_solver.boundary.base import ReflectiveBC, BoundarySide
            bc_left = ReflectiveBC(BoundarySide.LEFT, eos)

        if bc_right is None:
            from lagrangian_solver.boundary.base import ReflectiveBC, BoundarySide
            bc_right = ReflectiveBC(BoundarySide.RIGHT, eos)

        super().__init__(
            grid=grid,
            eos=eos,
            bc_left=bc_left,
            bc_right=bc_right,
            config=config,
        )
