"""
Main Lagrangian solver orchestrator.

This module provides the high-level LagrangianSolver class that coordinates
all components for running a complete simulation.

References:
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
    ConservedVariablesCompatible,
    create_riemann_state,
)
from lagrangian_solver.core.grid import LagrangianGrid, GridConfig
from lagrangian_solver.equations.eos import EOSBase, IdealGasEOS, CanteraEOS
from lagrangian_solver.equations.conservation import (
    LagrangianConservation,
    LagrangianFlux,
    compute_mass_error,
    compute_total_energy,
    compute_momentum,
    compute_total_energy_compatible,
    compute_energy_conservation_error,
)
from lagrangian_solver.numerics.riemann import (
    ExactRiemannSolver,
    HLLCRiemannSolver,
    RiemannSolverBase,
)
from lagrangian_solver.numerics.time_integration import (
    TimeIntegratorBase,
    HeunIntegrator,
    ForwardEulerIntegrator,
    SSPRK3Integrator,
    HeunIntegratorCompatible,
    ForwardEulerIntegratorCompatible,
    TimeStepInfo,
)
from lagrangian_solver.numerics.artificial_viscosity import (
    ArtificialViscosity,
    ArtificialViscosityConfig,
)
from lagrangian_solver.boundary.base import BoundaryCondition, BoundaryFlux
from lagrangian_solver.io.output import OutputWriter, OutputFrame, create_writer


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
                When set, clips time step to this minimum value.
                WARNING: May cause stability issues if set too large.
        verbose: Print progress messages
        artificial_viscosity: Configuration for Von Neumann-Richtmyer
                              artificial viscosity (None to disable)
        use_compatible_energy: If True, use compatible energy discretization
                               that solves for internal energy directly while
                               ensuring exact total energy conservation.
                               Reference: [Burton1992] UCRL-JC-105926
    """

    cfl: float = 0.5
    t_end: float = 1.0
    dt_output: float = 0.1
    dt_max: Optional[float] = None
    dt_min: Optional[float] = None
    verbose: bool = True
    artificial_viscosity: Optional[ArtificialViscosityConfig] = None
    use_compatible_energy: bool = False


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
        compatible_energy_error: Energy conservation error for compatible
                                 discretization (should be O(machine epsilon))
    """

    n_steps: int = 0
    wall_time: float = 0.0
    final_time: float = 0.0
    min_dt: float = float("inf")
    max_dt: float = 0.0
    avg_dt: float = 0.0
    mass_error: float = 0.0
    energy_change: float = 0.0
    compatible_energy_error: float = 0.0


class LagrangianSolver:
    """
    Main orchestrator for 1D compressible Lagrangian simulations.

    Coordinates:
    - Grid management and position updates
    - EOS calculations
    - Riemann solver for interface fluxes
    - Time integration
    - Boundary condition application
    - Output writing

    Reference: [Despres2017] Chapter 5, Algorithm 5.1
    """

    def __init__(
        self,
        grid: LagrangianGrid,
        eos: EOSBase,
        riemann_solver: Optional[RiemannSolverBase] = None,
        time_integrator: Optional[TimeIntegratorBase] = None,
        bc_left: Optional[BoundaryCondition] = None,
        bc_right: Optional[BoundaryCondition] = None,
        config: Optional[SolverConfig] = None,
    ):
        """
        Initialize the Lagrangian solver.

        Args:
            grid: Lagrangian grid
            eos: Equation of state
            riemann_solver: Riemann solver (default: ExactRiemannSolver)
            time_integrator: Time integrator (default: HeunIntegrator)
            bc_left: Left boundary condition (default: reflective)
            bc_right: Right boundary condition (default: reflective)
            config: Solver configuration
        """
        self._grid = grid
        self._eos = eos
        self._config = config or SolverConfig()

        # Set up Riemann solver
        if riemann_solver is None:
            self._riemann_solver = ExactRiemannSolver(eos)
        else:
            self._riemann_solver = riemann_solver

        # Set up time integrator
        # Use compatible energy integrator if configured
        if time_integrator is None:
            if self._config.use_compatible_energy:
                self._integrator = HeunIntegratorCompatible(eos, cfl=self._config.cfl)
            else:
                self._integrator = HeunIntegrator(eos, cfl=self._config.cfl)
        else:
            self._integrator = time_integrator
            self._integrator.cfl = self._config.cfl

        # Track compatible energy mode
        self._use_compatible_energy = self._config.use_compatible_energy

        # Set up artificial viscosity if configured
        if self._config.artificial_viscosity is not None:
            self._artificial_viscosity = ArtificialViscosity(
                self._config.artificial_viscosity
            )
        else:
            self._artificial_viscosity = None

        # Set up conservation law handler
        self._conservation = LagrangianConservation(
            eos, self._riemann_solver, self._artificial_viscosity
        )

        # Boundary conditions
        self._bc_left = bc_left
        self._bc_right = bc_right

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

        This is passed to the time integrator.

        Returns:
            Tuple of (d_tau, d_u, d_E, d_x) for standard method
            Tuple of (d_tau, d_u, d_e, d_x) for compatible energy method
        """
        # Compute interior fluxes
        fluxes = self._conservation.compute_fluxes(state, grid)

        # Apply boundary conditions
        if self._bc_left is not None:
            self._bc_left.apply(state, grid, self._time)
            bc_flux_left = self._bc_left.compute_flux(state, grid, self._time)
            fluxes.p_flux[0] = bc_flux_left.p_flux
            fluxes.pu_flux[0] = bc_flux_left.pu_flux
            fluxes.u_flux[0] = bc_flux_left.u_flux

        if self._bc_right is not None:
            self._bc_right.apply(state, grid, self._time)
            bc_flux_right = self._bc_right.compute_flux(state, grid, self._time)
            fluxes.p_flux[-1] = bc_flux_right.p_flux
            fluxes.pu_flux[-1] = bc_flux_right.pu_flux
            fluxes.u_flux[-1] = bc_flux_right.u_flux

        # Compute residual using appropriate method
        if self._use_compatible_energy:
            # Compatible discretization: returns (d_tau, d_u, d_e, d_x)
            # where d_e is internal energy rate derived from kinetic energy constraint
            return self._conservation.compute_compatible_residual(state, grid, fluxes)
        else:
            # Standard method: returns (d_tau, d_u, d_E, d_x)
            return self._conservation.compute_residual(state, grid, fluxes)

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
        # This ensures CFL accounts for moving boundaries (e.g., supersonic piston)
        if self._bc_left is not None:
            self._bc_left.apply(self._state, self._grid, self._time)
        if self._bc_right is not None:
            self._bc_right.apply(self._state, self._grid, self._time)

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

        # Apply maximum time step limit
        if self._config.dt_max is not None:
            dt = min(dt, self._config.dt_max)
            ts_info.dt = dt

        # Apply minimum time step floor (if set)
        # WARNING: This may cause stability issues if dt_min is too large
        if self._config.dt_min is not None:
            dt = max(dt, self._config.dt_min)
            ts_info.dt = dt

        # Don't exceed final time
        if self._time + dt > self._config.t_end:
            dt = self._config.t_end - self._time
            ts_info.dt = dt

        # Advance solution
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

    def run(
        self,
        writer: Optional[OutputWriter] = None,
    ) -> SolverStatistics:
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
        initial_energy = compute_total_energy(self._state)
        # For compatible energy, also track using the mass-based formulation
        if self._use_compatible_energy:
            initial_energy_compatible = compute_total_energy_compatible(
                self._state, self._grid
            )
        else:
            initial_energy_compatible = None
        t_next_output = 0.0

        # Start timing
        wall_start = timer.perf_counter()

        # Write initial condition
        if writer is not None:
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
                print(
                    f"Step {self._step:6d}: t = {self._time:.6e}, "
                    f"dt = {ts_info.dt:.6e}"
                )

            # Write output at specified intervals
            if writer is not None and self._time >= t_next_output - 1e-12:
                frame = OutputFrame.from_state(
                    self._state, self._grid, self._time, self._step
                )
                writer.write_frame(frame)
                t_next_output += self._config.dt_output

        # Write final state if not already written
        if writer is not None:
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

        final_energy = compute_total_energy(self._state)
        if abs(initial_energy) > 1e-15:
            self._stats.energy_change = (
                abs(final_energy - initial_energy) / initial_energy
            )

        # Compute compatible energy conservation error if using compatible method
        if self._use_compatible_energy and initial_energy_compatible is not None:
            self._stats.compatible_energy_error = compute_energy_conservation_error(
                self._state, self._grid, initial_energy_compatible
            )

        if self._config.verbose:
            self._print_statistics()

        return self._stats

    def _print_statistics(self) -> None:
        """Print solver statistics."""
        print("\n" + "=" * 60)
        print("SOLVER STATISTICS")
        print("=" * 60)
        print(f"Total steps:      {self._stats.n_steps}")
        print(f"Wall time:        {self._stats.wall_time:.3f} s")
        print(f"Final time:       {self._stats.final_time:.6e} s")
        print(f"Min dt:           {self._stats.min_dt:.6e} s")
        print(f"Max dt:           {self._stats.max_dt:.6e} s")
        print(f"Avg dt:           {self._stats.avg_dt:.6e} s")
        print(f"Mass error:       {self._stats.mass_error:.6e}")
        print(f"Energy change:    {self._stats.energy_change:.6e}")
        if self._use_compatible_energy:
            print(f"Compatible E err: {self._stats.compatible_energy_error:.6e}")
        print("=" * 60)


def create_solver_from_config(config: dict) -> LagrangianSolver:
    """
    Create a solver from a configuration dictionary.

    Args:
        config: Configuration dictionary

    Returns:
        Configured LagrangianSolver
    """
    from lagrangian_solver.io.input import SimulationConfig

    sim_config = SimulationConfig.from_dict(config)
    sim_config.validate()

    # Create grid
    grid_config = GridConfig(
        n_cells=sim_config.grid.n_cells,
        x_min=sim_config.grid.x_min,
        x_max=sim_config.grid.x_max,
        stretch_factor=sim_config.grid.stretch_factor,
    )
    grid = LagrangianGrid(grid_config)

    # Create EOS
    if sim_config.eos.use_cantera:
        eos = CanteraEOS(sim_config.eos.mechanism_file)
        if sim_config.eos.fuel and sim_config.eos.oxidizer:
            eos.set_mixture(
                sim_config.eos.fuel,
                sim_config.eos.oxidizer,
                sim_config.eos.phi or 1.0,
            )
    else:
        eos = IdealGasEOS(
            gamma=sim_config.eos.gamma,
            R=sim_config.eos.R,
        )

    # Create solver config
    solver_config = SolverConfig(
        cfl=sim_config.time.cfl,
        t_end=sim_config.time.t_end,
        dt_output=sim_config.time.dt_output,
        dt_max=sim_config.time.dt_max,
    )

    # Create solver (boundary conditions handled separately)
    solver = LagrangianSolver(
        grid=grid,
        eos=eos,
        config=solver_config,
    )

    # Set initial condition
    if sim_config.initial.is_riemann:
        solver.set_riemann_ic(
            x_disc=sim_config.initial.x_discontinuity,
            rho_L=sim_config.initial.rho_L,
            u_L=sim_config.initial.u_L,
            p_L=sim_config.initial.p_L,
            rho_R=sim_config.initial.rho_R,
            u_R=sim_config.initial.u_R,
            p_R=sim_config.initial.p_R,
        )
    else:
        from lagrangian_solver.core.state import create_uniform_state

        state = create_uniform_state(
            n_cells=sim_config.grid.n_cells,
            x_left=sim_config.grid.x_min,
            x_right=sim_config.grid.x_max,
            rho=sim_config.initial.rho,
            u=sim_config.initial.u,
            p=sim_config.initial.p,
            eos=eos,
        )
        solver.set_initial_condition(state)

    return solver
