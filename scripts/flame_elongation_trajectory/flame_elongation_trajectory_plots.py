"""
Piston Simulation with Configurable Velocity Trajectories

Supports various piston velocity profiles:
- Linear ramp: u_p = u_final * min(t/ramp_time, 1)
- Power law: u_p = (k * t)^n
- Exponential: u_p = A * exp(B * t)
- Flame-coupled: u_p = (sigma(t) - 1) * (rho_u/rho_b) * S_L
  (with iterative P-T coupling from Clavin-Tofaili formulation)

Uses Cantera for all equation of state calculations.

Reference: [Toro2009] Section 4.2, 6.3.2
Reference: Clavin & Tofaili (2021) - Flame elongation model
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import argparse
import time as time_module
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from lagrangian_solver.equations.eos import CanteraEOS
from lagrangian_solver.core.grid import LagrangianGrid, GridConfig
from lagrangian_solver.core.state import create_uniform_state
from lagrangian_solver.core.solver import LagrangianSolver, SolverConfig
from lagrangian_solver.numerics.riemann import ExactRiemannSolver
from lagrangian_solver.numerics.artificial_viscosity import ArtificialViscosityConfig
from lagrangian_solver.boundary.piston import RiemannGhostPistonBC
from lagrangian_solver.boundary.open import OpenBC
from lagrangian_solver.boundary.base import BoundarySide

# Import flame-coupled components
from flame_property_interpolator import FlamePropertyInterpolator
from flame_elongation_trajectory import (
    FlameElongationTrajectory,
    PowerLawElongation,
    ExponentialElongation,
    LinearElongation,
    ConstantElongation,
)
from flame_coupled_piston_bc import FlameCoupledPistonBC

# Constants
T_STP = 298.15  # K
P_STP = 101325  # Pa


# =============================================================================
# Piston Trajectory Classes
# =============================================================================

class PistonTrajectory(ABC):
    """Abstract base class for piston velocity trajectories."""

    @abstractmethod
    def velocity(self, t: float) -> float:
        """Return piston velocity at time t [m/s]."""
        pass

    @abstractmethod
    def position(self, t: float) -> float:
        """Return piston position at time t [m]."""
        pass

    @abstractmethod
    def name(self) -> str:
        """Return trajectory name for labeling."""
        pass

    @abstractmethod
    def params_str(self) -> str:
        """Return parameter string for file naming."""
        pass


@dataclass
class LinearRampTrajectory(PistonTrajectory):
    """
    Linear ramp to final velocity.

    u_p(t) = u_final * min(t / ramp_time, 1)
    """
    u_final: float  # Final velocity [m/s]
    ramp_time: float  # Ramp duration [s]

    def velocity(self, t: float) -> float:
        if t <= 0:
            return 0.0
        elif t < self.ramp_time:
            return self.u_final * t / self.ramp_time
        else:
            return self.u_final

    def position(self, t: float) -> float:
        if t <= 0:
            return 0.0
        elif t < self.ramp_time:
            return 0.5 * self.u_final * t**2 / self.ramp_time
        else:
            # Position at end of ramp + constant velocity portion
            x_ramp = 0.5 * self.u_final * self.ramp_time
            return x_ramp + self.u_final * (t - self.ramp_time)

    def name(self) -> str:
        return f"Linear Ramp (u_f={self.u_final:.0f} m/s, tau={self.ramp_time*1e6:.0f} us)"

    def params_str(self) -> str:
        return f"linear_uf{self.u_final:.0f}_tau{self.ramp_time*1e6:.0f}us"


@dataclass
class PowerLawTrajectory(PistonTrajectory):
    """
    Power law velocity profile (LGDCS power_law_alt equivalent).

    u_p(t) = (k * t)^n

    This is equivalent to LGDCS power_law_alt with sigma_0=1:
        sigma(t) = sigma_0 * (1 + (k*t)^n)
        u_piston ∝ sigma - 1 = (k*t)^n  [when sigma_0=1]

    Parameters:
        k: Rate coefficient [1/s]. Controls how fast velocity grows.
        n: Power exponent (dimensionless). Shape of the velocity curve.
           n=1: linear (constant acceleration)
           n=2: quadratic (increasing acceleration)
           n=3: cubic (rapidly increasing acceleration)
        u_max: Optional velocity cap [m/s]
    """
    k: float  # Rate coefficient [1/s]
    n: float  # Power exponent (dimensionless)
    u_max: Optional[float] = None  # Optional velocity cap [m/s]

    def velocity(self, t: float) -> float:
        if t <= 0:
            return 0.0
        u = (self.k * t) ** self.n
        if self.u_max is not None:
            u = min(u, self.u_max)
        return u

    def position(self, t: float) -> float:
        if t <= 0:
            return 0.0
        # Integrate (k*t)^n dt = k^n * t^(n+1) / (n+1)
        if self.u_max is None:
            return (self.k ** self.n) * (t ** (self.n + 1)) / (self.n + 1)
        else:
            # Find time when u_max is reached
            t_max = (self.u_max ** (1/self.n)) / self.k
            if t <= t_max:
                return (self.k ** self.n) * (t ** (self.n + 1)) / (self.n + 1)
            else:
                # Position at t_max + constant velocity portion
                x_max = (self.k ** self.n) * (t_max ** (self.n + 1)) / (self.n + 1)
                return x_max + self.u_max * (t - t_max)

    def name(self) -> str:
        cap_str = f", cap={self.u_max:.0f} m/s" if self.u_max else ""
        return f"Power Law (k={self.k:.1f}, n={self.n:.2f}{cap_str})"

    def params_str(self) -> str:
        cap_str = f"_cap{self.u_max:.0f}" if self.u_max else ""
        return f"power_k{self.k:.1f}_n{self.n:.2f}{cap_str}"


@dataclass
class ExponentialTrajectory(PistonTrajectory):
    """
    Exponential velocity profile.

    u_p(t) = A * (exp(B * t) - 1)

    Note: Using exp(B*t) - 1 to ensure u_p(0) = 0.

    Parameters:
        A: Amplitude coefficient [m/s]
        B: Growth rate [1/s]
    """
    A: float  # Amplitude [m/s]
    B: float  # Growth rate [1/s]
    u_max: Optional[float] = None  # Optional velocity cap [m/s]

    def velocity(self, t: float) -> float:
        if t <= 0:
            return 0.0
        u = self.A * (np.exp(self.B * t) - 1)
        if self.u_max is not None:
            u = min(u, self.u_max)
        return u

    def position(self, t: float) -> float:
        if t <= 0:
            return 0.0
        # Integrate A*(exp(B*t) - 1) dt = A * (exp(B*t)/B - t) - A/B
        # With x(0) = 0: x(t) = A * (exp(B*t) - 1) / B - A*t
        if self.u_max is None:
            return self.A * (np.exp(self.B * t) - 1) / self.B - self.A * t
        else:
            # Find time when u_max is reached
            t_max = np.log(self.u_max / self.A + 1) / self.B
            if t <= t_max:
                return self.A * (np.exp(self.B * t) - 1) / self.B - self.A * t
            else:
                x_max = self.A * (np.exp(self.B * t_max) - 1) / self.B - self.A * t_max
                return x_max + self.u_max * (t - t_max)

    def name(self) -> str:
        cap_str = f", cap={self.u_max:.0f} m/s" if self.u_max else ""
        return f"Exponential (A={self.A:.1f}, B={self.B:.1f}{cap_str})"

    def params_str(self) -> str:
        cap_str = f"_cap{self.u_max:.0f}" if self.u_max else ""
        return f"exp_A{self.A:.1f}_B{self.B:.1f}{cap_str}"


@dataclass
class ExponentialAltTrajectory(PistonTrajectory):
    """
    Exponential velocity profile (LGDCS format).

    u_p(t) = A * exp(B * t + C)

    This matches the LGDCS exponential mode.

    Parameters:
        A: Amplitude coefficient [m/s]
        B: Growth rate [1/s]
        C: Phase shift (dimensionless, typically negative)
    """
    A: float  # Amplitude [m/s]
    B: float  # Growth rate [1/s]
    C: float  # Phase shift (dimensionless)
    u_max: Optional[float] = None  # Optional velocity cap [m/s]

    def velocity(self, t: float) -> float:
        if t <= 0:
            return self.A * np.exp(self.C)  # Initial velocity at t=0
        u = self.A * np.exp(self.B * t + self.C)
        if self.u_max is not None:
            u = min(u, self.u_max)
        return u

    def position(self, t: float) -> float:
        if t <= 0:
            return 0.0
        # Integrate A*exp(B*t+C) dt = A*exp(C) * exp(B*t) / B
        # With x(0) = 0: x(t) = A*exp(C) * (exp(B*t) - 1) / B
        if self.u_max is None:
            return self.A * np.exp(self.C) * (np.exp(self.B * t) - 1) / self.B
        else:
            # Find time when u_max is reached: A*exp(B*t+C) = u_max
            # t_max = (ln(u_max/A) - C) / B
            t_max = (np.log(self.u_max / self.A) - self.C) / self.B
            if t_max < 0:
                # Already above u_max at t=0
                return self.u_max * t
            elif t <= t_max:
                return self.A * np.exp(self.C) * (np.exp(self.B * t) - 1) / self.B
            else:
                x_max = self.A * np.exp(self.C) * (np.exp(self.B * t_max) - 1) / self.B
                return x_max + self.u_max * (t - t_max)

    def name(self) -> str:
        cap_str = f", cap={self.u_max:.0f} m/s" if self.u_max else ""
        return f"Exponential (A={self.A:.1f}, B={self.B:.0f}, C={self.C:.0f}{cap_str})"

    def params_str(self) -> str:
        cap_str = f"_cap{self.u_max:.0f}" if self.u_max else ""
        c_str = f"Cm{abs(int(self.C))}" if self.C < 0 else f"C{int(self.C)}"
        return f"exponential_B{int(self.B)}_{c_str}{cap_str}"


class FlameCoupledTrajectory(PistonTrajectory):
    """
    Flame-coupled velocity profile using Clavin-Tofaili formulation.

    u_p(t) = (sigma(t) - 1) * (rho_u / rho_b) * S_L

    This trajectory uses flame property interpolation from pre-computed
    Cantera data. The actual velocity depends on local (P, T) conditions
    at the piston face through iterative coupling.

    NOTE: This trajectory provides UNCOUPLED velocity for initial guess
    and plotting purposes. The actual FlameCoupledPistonBC handles the
    iterative coupling during simulation.

    Parameters:
        flame_trajectory: FlameElongationTrajectory object
        elongation_type: Type of elongation function ("power", "exponential", "linear")
    """

    def __init__(
        self,
        flame_trajectory: FlameElongationTrajectory,
        elongation_type: str = "power",
    ):
        self._flame_trajectory = flame_trajectory
        self._elongation_type = elongation_type
        # Cache for position integration
        self._position_cache: Dict[float, float] = {}

    @property
    def flame_trajectory(self) -> FlameElongationTrajectory:
        return self._flame_trajectory

    def velocity(self, t: float) -> float:
        """Return uncoupled piston velocity (at reference P, T)."""
        return self._flame_trajectory.velocity_uncoupled(t)

    def position(self, t: float) -> float:
        """Approximate position by numerical integration."""
        if t <= 0:
            return 0.0
        # Use simple trapezoidal integration
        from scipy import integrate
        result, _ = integrate.quad(self.velocity, 0, t)
        return result

    def name(self) -> str:
        elong = self._flame_trajectory.elongation
        props = self._flame_trajectory.properties_ref
        return (f"Flame-Coupled ({elong.name})\n"
                f"S_L={props.S_L:.1f} m/s, rho_u/rho_b={props.density_ratio:.2f}")

    def params_str(self) -> str:
        elong = self._flame_trajectory.elongation
        if hasattr(elong, 'k') and hasattr(elong, 'n'):
            return f"flame_power_k{elong.k:.0f}_n{elong.n:.1f}"
        elif hasattr(elong, 'B'):
            return f"flame_exp_B{elong.B:.0f}"
        elif hasattr(elong, 'k'):
            return f"flame_linear_k{elong.k:.0f}"
        else:
            return "flame_constant"


# =============================================================================
# Callable Wrapper for BC
# =============================================================================

class TrajectoryVelocityFunction:
    """Wrapper to make trajectory velocity callable for BC."""

    def __init__(self, trajectory: PistonTrajectory):
        self.trajectory = trajectory

    def __call__(self, t: float) -> float:
        return self.trajectory.velocity(t)


# =============================================================================
# Simulation Functions
# =============================================================================

def run_trajectory_simulation(
    trajectory: PistonTrajectory,
    eos: CanteraEOS,
    n_cells: int = 200,
    domain_length: float = 1.0,
    t_end: float = 1e-3,
    cfl: float = 0.5,
    dt_min: float = 1e-9,
    av_config: Optional[ArtificialViscosityConfig] = None,
    hc_enabled: bool = False,
    hc_linear: float = 0.2,
    hc_quad: float = 1.0,
    save_interval: int = 10,
    verbose: bool = False,
    u_terminate: Optional[float] = None,
    flame_coupled: bool = False,
    flame_tol: float = 1e-6,
    flame_max_iter: int = 20,
    flame_ramp_time: float = 30e-6,
    flame_relaxation: float = 0.7,
) -> Dict[str, Any]:
    """
    Run piston simulation with given velocity trajectory.

    Args:
        trajectory: PistonTrajectory object defining velocity profile
        eos: Cantera equation of state
        n_cells: Number of grid cells
        domain_length: Domain length [m]
        t_end: Simulation end time [s]
        cfl: CFL number
        dt_min: Minimum time step [s]
        av_config: Artificial viscosity configuration
        hc_enabled: Enable heat conduction
        hc_linear: HC linear coefficient
        hc_quad: HC quadratic coefficient
        save_interval: Save solution every N steps
        verbose: Print progress
        u_terminate: Terminate simulation when piston velocity exceeds this value [m/s]
                    (default: None = no termination)
        flame_coupled: Use FlameCoupledPistonBC (requires FlameCoupledTrajectory)
        flame_tol: Convergence tolerance for flame coupling iteration
        flame_max_iter: Maximum iterations for flame coupling
        flame_ramp_time: Ramp time for flame-coupled BC [s]
        flame_relaxation: Under-relaxation factor for flame coupling

    Returns:
        Dictionary with simulation results and saved data
    """
    print(f"Running: {trajectory.name()}")
    print(f"  Domain: {domain_length:.2f} m, {n_cells} cells")
    print(f"  t_end: {t_end*1e3:.2f} ms")

    # Get initial state from EOS
    eos.set_state_TP(T_STP, P_STP)
    rho_init = eos.gas.density
    c_init = eos.gas.sound_speed

    print(f"  Initial: rho={rho_init:.4f} kg/m3, c={c_init:.1f} m/s")

    # Create grid
    grid_config = GridConfig(n_cells=n_cells, x_min=0.0, x_max=domain_length)
    grid = LagrangianGrid(grid_config)

    # Left BC: Choose based on trajectory type
    if flame_coupled and isinstance(trajectory, FlameCoupledTrajectory):
        # Use FlameCoupledPistonBC for iterative P-T coupling
        bc_left = FlameCoupledPistonBC(
            side=BoundarySide.LEFT,
            eos=eos,
            flame_trajectory=trajectory.flame_trajectory,
            tol=flame_tol,
            max_iter=flame_max_iter,
            ramp_time=flame_ramp_time,
            relaxation=flame_relaxation,
        )
        print(f"  BC: FlameCoupledPistonBC (tol={flame_tol}, max_iter={flame_max_iter})")
    else:
        # Use standard RiemannGhostPistonBC with velocity function
        velocity_func = TrajectoryVelocityFunction(trajectory)
        bc_left = RiemannGhostPistonBC(
            side=BoundarySide.LEFT,
            eos=eos,
            velocity=velocity_func,  # Pass callable
            ramp_time=0,  # Ramp handled by trajectory
            startup_time=0,
        )
        print(f"  BC: RiemannGhostPistonBC")

    # Right BC: Open
    bc_right = OpenBC(
        side=BoundarySide.RIGHT,
        eos=eos,
        p_external=P_STP,
        u_external=0.0,
        rho_external=rho_init,
    )

    # Solver configuration
    solver_config = SolverConfig(
        cfl=cfl,
        t_end=t_end,
        dt_output=t_end,
        dt_min=dt_min,
        verbose=verbose,
        artificial_viscosity=av_config,
        hc_enabled=hc_enabled,
        hc_linear=hc_linear,
        hc_quad=hc_quad,
    )

    # Create solver
    riemann_solver = ExactRiemannSolver(eos, tol=1e-8, max_iter=100)
    solver = LagrangianSolver(
        grid=grid,
        eos=eos,
        riemann_solver=riemann_solver,
        bc_left=bc_left,
        bc_right=bc_right,
        config=solver_config,
    )

    # Set uniform initial condition
    initial_state = create_uniform_state(
        n_cells=n_cells,
        x_left=0.0,
        x_right=domain_length,
        rho=rho_init,
        u=0.0,
        p=P_STP,
        eos=eos,
    )
    solver.set_initial_condition(initial_state)

    # Storage for time-series data
    saved_times = []
    saved_x = []
    saved_rho = []
    saved_u = []
    saved_p = []
    saved_e = []
    saved_T = []
    saved_s = []
    saved_u_piston = []

    # Helper to get current piston velocity (handles both BC types)
    def get_piston_velocity(t: float) -> float:
        if flame_coupled and isinstance(bc_left, FlameCoupledPistonBC):
            return bc_left.get_boundary_velocity(t)
        else:
            return trajectory.velocity(t)

    # Save initial state
    saved_times.append(solver.time)
    saved_x.append(solver.state.x.copy())
    saved_rho.append(solver.state.rho.copy())
    saved_u.append(solver.state.u.copy())
    saved_p.append(solver.state.p.copy())
    saved_e.append(solver.state.e.copy())
    saved_T.append(solver.state.T.copy())
    saved_s.append(solver.state.s.copy())
    saved_u_piston.append(get_piston_velocity(0.0))

    # For flame-coupled, also track iteration stats
    saved_iterations = []
    saved_P_face = []
    saved_T_face = []
    if flame_coupled and isinstance(bc_left, FlameCoupledPistonBC):
        saved_iterations.append(0)
        saved_P_face.append(P_STP)
        saved_T_face.append(T_STP)

    # Run simulation
    wall_start = time_module.perf_counter()
    step = 0
    failed = False
    error_msg = None
    terminated_early = False
    termination_reason = None

    try:
        while solver.time < t_end:
            solver.step_forward()
            step += 1

            # Check piston velocity termination condition
            current_u_piston = get_piston_velocity(solver.time)
            if u_terminate is not None and abs(current_u_piston) >= u_terminate:
                terminated_early = True
                termination_reason = f"Piston velocity {current_u_piston:.1f} m/s >= {u_terminate:.1f} m/s"
                print(f"  TERMINATING: {termination_reason}")
                # Save state before breaking
                saved_times.append(solver.time)
                saved_x.append(solver.state.x.copy())
                saved_rho.append(solver.state.rho.copy())
                saved_u.append(solver.state.u.copy())
                saved_p.append(solver.state.p.copy())
                saved_e.append(solver.state.e.copy())
                saved_T.append(solver.state.T.copy())
                saved_s.append(solver.state.s.copy())
                saved_u_piston.append(current_u_piston)
                # Save flame-coupled data on termination
                if flame_coupled and isinstance(bc_left, FlameCoupledPistonBC):
                    result = bc_left._cached_result
                    if result is not None:
                        saved_iterations.append(result.iterations)
                        saved_P_face.append(result.P_face)
                        saved_T_face.append(result.T_face)
                    else:
                        saved_iterations.append(0)
                        saved_P_face.append(solver.state.p[0])
                        saved_T_face.append(solver.state.T[0])
                break

            if step % save_interval == 0:
                saved_times.append(solver.time)
                saved_x.append(solver.state.x.copy())
                saved_rho.append(solver.state.rho.copy())
                saved_u.append(solver.state.u.copy())
                saved_p.append(solver.state.p.copy())
                saved_e.append(solver.state.e.copy())
                saved_T.append(solver.state.T.copy())
                saved_s.append(solver.state.s.copy())
                saved_u_piston.append(current_u_piston)

                # Save flame-coupled iteration data
                if flame_coupled and isinstance(bc_left, FlameCoupledPistonBC):
                    result = bc_left._cached_result
                    if result is not None:
                        saved_iterations.append(result.iterations)
                        saved_P_face.append(result.P_face)
                        saved_T_face.append(result.T_face)
                    else:
                        saved_iterations.append(0)
                        saved_P_face.append(solver.state.p[0])
                        saved_T_face.append(solver.state.T[0])

    except Exception as e:
        failed = True
        error_msg = str(e)
        import traceback
        traceback.print_exc()

    wall_time = time_module.perf_counter() - wall_start

    # Save final state
    if len(saved_times) == 0 or saved_times[-1] != solver.time:
        saved_times.append(solver.time)
        saved_x.append(solver.state.x.copy())
        saved_rho.append(solver.state.rho.copy())
        saved_u.append(solver.state.u.copy())
        saved_p.append(solver.state.p.copy())
        saved_e.append(solver.state.e.copy())
        saved_T.append(solver.state.T.copy())
        saved_s.append(solver.state.s.copy())
        saved_u_piston.append(get_piston_velocity(solver.time))
        # Save flame-coupled data for final state
        if flame_coupled and isinstance(bc_left, FlameCoupledPistonBC):
            result = bc_left._cached_result
            if result is not None:
                saved_iterations.append(result.iterations)
                saved_P_face.append(result.P_face)
                saved_T_face.append(result.T_face)
            else:
                saved_iterations.append(0)
                saved_P_face.append(solver.state.p[0])
                saved_T_face.append(solver.state.T[0])

    print(f"  Completed: {step} steps, {wall_time:.2f}s")
    if terminated_early:
        print(f"  Terminated early: {termination_reason}")
    if failed:
        print(f"  FAILED: {error_msg}")

    # Print flame-coupled convergence statistics
    if flame_coupled and isinstance(bc_left, FlameCoupledPistonBC):
        stats = bc_left.get_convergence_stats()
        print(f"  Flame coupling: {stats['n_timesteps']} solves, "
              f"avg {stats['avg_iterations']:.1f} iters, "
              f"{stats['convergence_rate']*100:.1f}% converged")

    # Build results
    saved_data = {
        't': np.array(saved_times),
        'x': saved_x,
        'rho': saved_rho,
        'u': saved_u,
        'p': saved_p,
        'e': saved_e,
        'T': saved_T,
        's': saved_s,
        'u_piston': np.array(saved_u_piston),
    }

    # Add flame-coupled specific data
    if flame_coupled and isinstance(bc_left, FlameCoupledPistonBC):
        saved_data['iterations'] = np.array(saved_iterations)
        saved_data['P_face'] = np.array(saved_P_face)
        saved_data['T_face'] = np.array(saved_T_face)

    results = {
        'state': solver.state,
        'grid': solver.grid,
        'stats': solver.statistics,
        'failed': failed,
        'error_msg': error_msg,
        'terminated_early': terminated_early,
        'termination_reason': termination_reason,
        'saved_data': saved_data,
        'trajectory': trajectory,
        'flame_coupled': flame_coupled,
        'bc_left': bc_left,
        'config': {
            'n_cells': n_cells,
            'domain_length': domain_length,
            't_end': t_end,
            'cfl': cfl,
            'rho_init': rho_init,
            'c_init': c_init,
            'T_init': T_STP,
            'p_init': P_STP,
            'u_terminate': u_terminate,
            'flame_coupled': flame_coupled,
        }
    }

    return results


def save_snapshots(
    results: Dict[str, Any],
    output_dir: str,
    snapshot_interval: int = 1,
):
    """
    Save simulation snapshots in format compatible with interactive_plotter.py.

    Parameters:
        results: Simulation results dictionary
        output_dir: Base output directory
        snapshot_interval: Save every N saved frames (default: 1 = all frames)
    """
    from pathlib import Path
    output_path = Path(output_dir)
    snapshots_path = output_path / 'snapshots'
    snapshots_path.mkdir(parents=True, exist_ok=True)

    saved_data = results['saved_data']
    config = results['config']

    n_frames = len(saved_data['t'])
    rho_init = config['rho_init']
    L_init = config['domain_length']
    n_cells = config['n_cells']

    # Compute mass coordinate (Lagrangian invariant)
    dm = rho_init * L_init / n_cells
    m_centers = (np.arange(n_cells) + 0.5) * dm

    saved_count = 0
    for i in range(0, n_frames, snapshot_interval):
        t_snap = saved_data['t'][i]
        x_snap = saved_data['x'][i]
        x_centers = 0.5 * (x_snap[:-1] + x_snap[1:])

        # Node velocity to cell velocity
        u_snap = saved_data['u'][i]
        u_centers = 0.5 * (u_snap[:-1] + u_snap[1:])

        save_dict = {
            't': t_snap,
            'step': i,
            'x_centers': x_centers,
            'x_interfaces': x_snap,
            'm_centers': m_centers,
            'rho': saved_data['rho'][i],
            'u': u_centers,
            'u_nodes': u_snap,
            'p': saved_data['p'][i],
            'e': saved_data['e'][i],
            'T': saved_data['T'][i],
            's': saved_data['s'][i],
            'piston_velocity': saved_data['u_piston'][i],
        }

        np.savez(
            snapshots_path / f'snapshot_{saved_count:06d}.npz',
            **save_dict
        )
        saved_count += 1

    # Also save a history file for compatibility
    np.savez(
        output_path / 'piston_history.npz',
        times=saved_data['t'],
        piston_velocity=saved_data['u_piston'],
    )

    print(f"  Saved snapshots: {saved_count} files in {snapshots_path}")


def _create_animation_core(
    results: Dict[str, Any],
    output_file: str,
    use_mass_coord: bool = False,
    fps: int = 30,
):
    """Create MP4 animation from simulation results.

    Parameters:
        use_mass_coord: If True, plot vs mass coordinate m. If False, plot vs x.
    """
    import cv2

    saved_data = results['saved_data']
    trajectory = results['trajectory']
    config = results['config']

    n_frames = len(saved_data['t'])
    t_end = config['t_end']
    rho_init = config['rho_init']
    p_init = config['p_init']
    T_init = config['T_init']
    c_init = config['c_init']
    s_init = saved_data['s'][0][0]  # Initial entropy

    # Compute mass coordinate (Lagrangian invariant)
    # m_cell[i] = (i + 0.5) * dm where dm = rho_init * L_init / n_cells
    n_cells = len(saved_data['rho'][0])
    L_init = config['domain_length']
    dm = rho_init * L_init / n_cells  # mass per cell
    m_cell = (np.arange(n_cells) + 0.5) * dm  # cell-centered mass coordinate
    m_total = rho_init * L_init

    # Create figure with 2x3 subplots + piston velocity panel
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.6])

    axes = [
        fig.add_subplot(gs[0, 0]),  # rho
        fig.add_subplot(gs[0, 1]),  # u
        fig.add_subplot(gs[0, 2]),  # p
        fig.add_subplot(gs[1, 0]),  # T
        fig.add_subplot(gs[1, 1]),  # e
        fig.add_subplot(gs[1, 2]),  # ds
    ]
    ax_piston = fig.add_subplot(gs[2, :])  # piston velocity spanning all columns

    # Initialize lines
    num_lines = []
    variables = ['rho', 'u', 'p', 'T', 'e', 'ds']
    labels = ['Density [kg/m³]', 'Velocity [m/s]', 'Pressure [Pa]',
              'Temperature [K]', 'Int. Energy [J/kg]', 'Entropy Change [J/(kg·K)]']

    x_label = 'm [kg/m²]' if use_mass_coord else 'x [m]'
    x_max = m_total if use_mass_coord else config['domain_length']

    for idx, (var, label) in enumerate(zip(variables, labels)):
        ax = axes[idx]
        line, = ax.plot([], [], 'bo', markersize=2)
        num_lines.append(line)
        ax.set_xlabel(x_label)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, x_max)

    # Pre-compute axis limits
    all_rho = np.concatenate([np.array(r) for r in saved_data['rho']])
    all_u = np.concatenate([np.array(u) for u in saved_data['u']])
    all_p = np.concatenate([np.array(p) for p in saved_data['p']])
    all_T = np.concatenate([np.array(T) for T in saved_data['T']])
    all_e = np.concatenate([np.array(e) for e in saved_data['e']])
    all_s = np.concatenate([np.array(s) - s_init for s in saved_data['s']])

    y_data = [all_rho, all_u, all_p, all_T, all_e, all_s]
    for ax, y in zip(axes, y_data):
        ymin, ymax = y.min(), y.max()
        margin = 0.1 * (ymax - ymin) if ymax > ymin else 0.1 * abs(ymax) + 0.1
        ax.set_ylim(ymin - margin, ymax + margin)

    # Piston velocity panel
    t_all = saved_data['t']
    u_piston_all = saved_data['u_piston']

    ax_piston.plot(t_all * 1e6, u_piston_all, 'g-', linewidth=1, alpha=0.3, label='Full trajectory')
    piston_line, = ax_piston.plot([], [], 'g-', linewidth=2, label='Piston velocity')
    piston_marker, = ax_piston.plot([], [], 'go', markersize=8)

    # Sound speed reference
    ax_piston.axhline(c_init, color='purple', linestyle='--', alpha=0.7,
                      label=f'c_1 = {c_init:.0f} m/s')

    ax_piston.set_xlim(0, t_all[-1] * 1e6)  # Use actual final time, not t_end
    ax_piston.set_ylim(-0.05 * max(u_piston_all), max(u_piston_all) * 1.15)
    ax_piston.set_xlabel('Time [us]')
    ax_piston.set_ylabel('Piston Velocity [m/s]')
    ax_piston.legend(loc='lower right')
    ax_piston.grid(True, alpha=0.3)

    coord_label = "Mass Coordinate" if use_mass_coord else "Physical Space"
    title = fig.suptitle('', fontsize=14, fontweight='bold')

    def update(frame):
        t = saved_data['t'][frame]
        x = saved_data['x'][frame]
        x_cell = 0.5 * (x[:-1] + x[1:])

        # Choose coordinate
        coord = m_cell if use_mass_coord else x_cell

        # Update numerical lines
        rho = saved_data['rho'][frame]
        u = saved_data['u'][frame]
        p = saved_data['p'][frame]
        T = saved_data['T'][frame]
        e = saved_data['e'][frame]
        ds = saved_data['s'][frame] - s_init

        u_cell = 0.5 * (u[:-1] + u[1:])

        data_list = [rho, u_cell, p, T, e, ds]
        for line, data in zip(num_lines, data_list):
            line.set_data(coord, data)

        # Update piston velocity
        piston_line.set_data(t_all[:frame+1] * 1e6, u_piston_all[:frame+1])
        piston_marker.set_data([t * 1e6], [u_piston_all[frame]])

        title.set_text(f'{trajectory.name()} ({coord_label})\nt = {t*1e6:.1f} us')
        return num_lines + [piston_line, piston_marker, title]

    # Generate MP4 using OpenCV
    fig.tight_layout()
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    for frame in range(n_frames):
        update(frame)
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        video_writer.write(img_bgr)

    video_writer.release()
    plt.close(fig)
    print(f"  Saved animation: {output_file}")


def create_trajectory_animation(
    results: Dict[str, Any],
    output_file: str,
    fps: int = 30,
):
    """Create MP4 animation in physical space (x coordinate)."""
    _create_animation_core(results, output_file, use_mass_coord=False, fps=fps)


def create_trajectory_animation_mass(
    results: Dict[str, Any],
    output_file: str,
    fps: int = 30,
):
    """Create MP4 animation in mass/Lagrangian coordinate (m)."""
    _create_animation_core(results, output_file, use_mass_coord=True, fps=fps)


def _plot_final_state_core(
    results: Dict[str, Any],
    output_file: str,
    use_mass_coord: bool = False,
):
    """Plot final state comparison.

    Parameters:
        use_mass_coord: If True, plot vs mass coordinate m. If False, plot vs x.
    """
    saved_data = results['saved_data']
    trajectory = results['trajectory']
    config = results['config']

    # Get final state
    x_final = saved_data['x'][-1]
    x_cell = 0.5 * (x_final[:-1] + x_final[1:])

    rho = saved_data['rho'][-1]
    u = saved_data['u'][-1]
    u_cell = 0.5 * (u[:-1] + u[1:])
    p = saved_data['p'][-1]
    T = saved_data['T'][-1]
    e = saved_data['e'][-1]
    s = saved_data['s'][-1]
    s_init = saved_data['s'][0][0]
    ds = s - s_init

    # Compute mass coordinate (Lagrangian invariant)
    n_cells = len(rho)
    rho_init = config['rho_init']
    L_init = config['domain_length']
    dm = rho_init * L_init / n_cells
    m_cell = (np.arange(n_cells) + 0.5) * dm

    # Choose coordinate
    coord = m_cell if use_mass_coord else x_cell
    x_label = 'm [kg/m²]' if use_mass_coord else 'x [m]'
    coord_label = "Mass Coordinate" if use_mass_coord else "Physical Space"

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    variables = [
        ('Density', rho, 'kg/m³'),
        ('Velocity', u_cell, 'm/s'),
        ('Pressure', p, 'Pa'),
        ('Temperature', T, 'K'),
        ('Internal Energy', e, 'J/kg'),
        ('Entropy Change', ds, 'J/(kg·K)'),
    ]

    for idx, (name, data, unit) in enumerate(variables):
        ax = axes[idx // 3, idx % 3]
        ax.plot(coord, data, 'bo', markersize=3)
        ax.set_xlabel(x_label)
        ax.set_ylabel(f'{name} [{unit}]')
        ax.set_title(name)
        ax.grid(True, alpha=0.3)

    t_final = saved_data['t'][-1]
    fig.suptitle(f'{trajectory.name()} ({coord_label})\nt = {t_final*1e6:.1f} us', fontsize=14, fontweight='bold')
    plt.tight_layout()

    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved plot: {output_file}")


def plot_final_state(
    results: Dict[str, Any],
    output_file: str,
):
    """Plot final state in physical space (x coordinate)."""
    _plot_final_state_core(results, output_file, use_mass_coord=False)


def plot_final_state_mass(
    results: Dict[str, Any],
    output_file: str,
):
    """Plot final state in mass/Lagrangian coordinate (m)."""
    _plot_final_state_core(results, output_file, use_mass_coord=True)


def plot_flame_coupling_diagnostics(
    results: Dict[str, Any],
    output_file: str,
):
    """
    Plot flame-coupling specific diagnostics.

    Shows P_face, T_face, and iteration count vs time.
    Only applicable for flame-coupled simulations.
    """
    if not results.get('flame_coupled', False):
        print("  Skipping flame diagnostics (not a flame-coupled simulation)")
        return

    saved_data = results['saved_data']
    trajectory = results['trajectory']

    if 'P_face' not in saved_data or 'T_face' not in saved_data:
        print("  Skipping flame diagnostics (no P_face/T_face data)")
        return

    t = saved_data['t'] * 1e6  # Convert to microseconds
    P_face = saved_data['P_face'] / 1e5  # Convert to bar
    T_face = saved_data['T_face']
    iterations = saved_data['iterations']
    u_piston = saved_data['u_piston']

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # P_face vs time
    ax = axes[0, 0]
    ax.plot(t, P_face, 'b-', linewidth=1.5)
    ax.set_xlabel('Time [us]')
    ax.set_ylabel('P_face [bar]')
    ax.set_title('Piston Face Pressure')
    ax.grid(True, alpha=0.3)

    # T_face vs time
    ax = axes[0, 1]
    ax.plot(t, T_face, 'r-', linewidth=1.5)
    ax.set_xlabel('Time [us]')
    ax.set_ylabel('T_face [K]')
    ax.set_title('Piston Face Temperature')
    ax.grid(True, alpha=0.3)

    # Iterations vs time
    ax = axes[1, 0]
    ax.plot(t, iterations, 'g-', linewidth=1.5)
    ax.set_xlabel('Time [us]')
    ax.set_ylabel('Iterations')
    ax.set_title('Coupling Iterations per Timestep')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max(iterations) * 1.2 if max(iterations) > 0 else 10)

    # u_piston vs time (for reference)
    ax = axes[1, 1]
    ax.plot(t, u_piston, 'purple', linewidth=1.5)
    ax.set_xlabel('Time [us]')
    ax.set_ylabel('Piston Velocity [m/s]')
    ax.set_title('Piston Velocity')
    ax.grid(True, alpha=0.3)

    fig.suptitle(f'Flame Coupling Diagnostics\n{trajectory.name()}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved flame diagnostics: {output_file}")


def plot_xt_diagrams(
    results: Dict[str, Any],
    output_dir: str,
    use_mass_coord: bool = False,
):
    """
    Create X-T (space-time) diagrams for all variables.

    Parameters:
        results: Simulation results dictionary
        output_dir: Directory to save the figures
        use_mass_coord: If True, use mass coordinate. If False, use physical x.
    """
    from pathlib import Path
    output_path = Path(output_dir)

    saved_data = results['saved_data']
    trajectory = results['trajectory']
    config = results['config']

    # Get time array
    times = saved_data['t']
    n_times = len(times)
    n_cells = len(saved_data['rho'][0])

    # Compute coordinates
    rho_init = config['rho_init']
    L_init = config['domain_length']
    dm = rho_init * L_init / n_cells
    m_cell = (np.arange(n_cells) + 0.5) * dm

    # Build 2D arrays for each variable
    rho_xt = np.array([saved_data['rho'][i] for i in range(n_times)])
    p_xt = np.array([saved_data['p'][i] for i in range(n_times)])
    T_xt = np.array([saved_data['T'][i] for i in range(n_times)])
    e_xt = np.array([saved_data['e'][i] for i in range(n_times)])

    # Velocity needs cell-centering
    u_xt = np.array([0.5 * (saved_data['u'][i][:-1] + saved_data['u'][i][1:])
                     for i in range(n_times)])

    # Entropy change
    s_init = saved_data['s'][0][0]
    ds_xt = np.array([saved_data['s'][i] - s_init for i in range(n_times)])

    # Physical x coordinates (time-varying)
    x_xt = np.array([0.5 * (saved_data['x'][i][:-1] + saved_data['x'][i][1:])
                     for i in range(n_times)])

    # Choose coordinate system
    if use_mass_coord:
        # Mass coordinate is constant for all times
        coord_2d = np.tile(m_cell, (n_times, 1))
        x_label = 'm [kg/m²]'
        coord_name = 'mass'
        suffix = '_mt'
    else:
        coord_2d = x_xt
        x_label = 'x [m]'
        coord_name = 'physical'
        suffix = '_xt'

    # Time in microseconds for plotting
    t_us = times * 1e6

    variables = [
        ('Density', rho_xt, 'kg/m³', 'rho'),
        ('Velocity', u_xt, 'm/s', 'u'),
        ('Pressure', p_xt, 'Pa', 'p'),
        ('Temperature', T_xt, 'K', 'T'),
        ('Internal Energy', e_xt, 'J/kg', 'e'),
        ('Entropy Change', ds_xt, 'J/(kg·K)', 'ds'),
    ]

    for name, data, unit, var_name in variables:
        fig, ax = plt.subplots(figsize=(10, 8))

        # Create pcolormesh plot
        # Need to create mesh grids for time and coordinate
        T_mesh, X_mesh = np.meshgrid(t_us, np.arange(n_cells), indexing='ij')

        # For physical coordinates, use actual x values
        if use_mass_coord:
            X_plot = np.tile(m_cell, (n_times, 1))
        else:
            X_plot = coord_2d

        # Use pcolormesh with actual coordinates
        pcm = ax.pcolormesh(X_plot, T_mesh, data, shading='auto', cmap='viridis')
        cbar = fig.colorbar(pcm, ax=ax, label=f'{name} [{unit}]')

        ax.set_xlabel(x_label)
        ax.set_ylabel('Time [us]')
        ax.set_title(f'{name} - {coord_name.title()} Space\n{trajectory.name()}')

        plt.tight_layout()
        filename = output_path / f'{var_name}{suffix}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close(fig)

    print(f"  Saved X-T diagrams ({coord_name}): {output_path}")


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run piston simulation with configurable velocity trajectories"
    )

    # Trajectory type
    parser.add_argument(
        "--trajectory", type=str, default="linear",
        choices=["linear", "power", "exponential", "flame"],
        help="Trajectory type (default: linear). 'flame' uses Clavin-Tofaili formulation."
    )

    # Linear ramp parameters
    parser.add_argument(
        "--u-final", type=float, default=500.0,
        help="Final piston velocity [m/s] (default: 500)"
    )
    parser.add_argument(
        "--ramp-time", type=float, default=100e-6,
        help="Ramp time [s] for linear trajectory (default: 100e-6)"
    )

    # Power law parameters
    parser.add_argument(
        "--k", type=float, default=1e6,
        help="Power law rate coefficient k (default: 1e6)"
    )
    parser.add_argument(
        "--power-n", type=float, default=1.0,
        help="Power law exponent n (default: 1.0)"
    )

    # Exponential parameters
    parser.add_argument(
        "--A", type=float, default=10.0,
        help="Exponential amplitude A [m/s] (default: 10)"
    )
    parser.add_argument(
        "--B", type=float, default=5000.0,
        help="Exponential growth rate B [1/s] (default: 5000)"
    )

    # Velocity cap (applies to power and exponential)
    parser.add_argument(
        "--u-max", type=float, default=None,
        help="Maximum velocity cap [m/s] (default: None)"
    )

    # Domain and simulation parameters
    parser.add_argument(
        "--n-cells", type=int, default=200,
        help="Number of grid cells (default: 200)"
    )
    parser.add_argument(
        "--domain-length", type=float, default=1.0,
        help="Domain length [m] (default: 1.0)"
    )
    parser.add_argument(
        "--t-end", type=float, default=1e-3,
        help="Simulation end time [s] (default: 1e-3)"
    )
    parser.add_argument(
        "--cfl", type=float, default=0.5,
        help="CFL number (default: 0.5)"
    )
    parser.add_argument(
        "--dt-min", type=float, default=1e-9,
        help="Minimum time step [s] (default: 1e-9)"
    )

    # Artificial viscosity
    parser.add_argument("--av", action="store_true", help="Enable artificial viscosity")
    parser.add_argument("--av-linear", type=float, default=0.3, help="AV linear coefficient")
    parser.add_argument("--av-quad", type=float, default=2.0, help="AV quadratic coefficient")

    # Heat conduction
    parser.add_argument("--hc", action="store_true", help="Enable heat conduction")
    parser.add_argument("--hc-linear", type=float, default=0.2, help="HC linear coefficient")
    parser.add_argument("--hc-quad", type=float, default=1.0, help="HC quadratic coefficient")

    # Output
    parser.add_argument(
        "--save-interval", type=int, default=10,
        help="Save solution every N steps (default: 10)"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: auto-generated)"
    )

    # Termination condition
    parser.add_argument(
        "--u-terminate", type=float, default=None,
        help="Terminate when piston velocity exceeds this value [m/s] (default: None = no termination)"
    )

    # Flame-coupled parameters
    parser.add_argument(
        "--flame-data", type=str, default=None,
        help="Path to flame_properties.csv (default: cantera_data/output/flame_properties.csv)"
    )
    parser.add_argument(
        "--flame-elongation", type=str, default="power",
        choices=["power", "exponential", "linear", "constant"],
        help="Elongation function type for flame trajectory (default: power)"
    )
    parser.add_argument(
        "--flame-sigma0", type=float, default=1.0,
        help="Initial elongation sigma_0 (default: 1.0)"
    )
    parser.add_argument(
        "--flame-k", type=float, default=1000.0,
        help="Elongation rate k [1/s] for power/linear (default: 1000)"
    )
    parser.add_argument(
        "--flame-n", type=float, default=2.0,
        help="Power law exponent n for power elongation (default: 2.0)"
    )
    parser.add_argument(
        "--flame-B", type=float, default=1000.0,
        help="Exponential growth rate B [1/s] (default: 1000)"
    )
    parser.add_argument(
        "--flame-P-ref", type=float, default=1e6,
        help="Reference pressure for uncoupled velocity [Pa] (default: 1e6 = 10 bar)"
    )
    parser.add_argument(
        "--flame-T-ref", type=float, default=500.0,
        help="Reference temperature for uncoupled velocity [K] (default: 500)"
    )
    parser.add_argument(
        "--flame-tol", type=float, default=1e-6,
        help="Convergence tolerance for flame coupling iteration (default: 1e-6)"
    )
    parser.add_argument(
        "--flame-max-iter", type=int, default=20,
        help="Maximum iterations for flame coupling (default: 20)"
    )
    parser.add_argument(
        "--flame-ramp-time", type=float, default=30e-6,
        help="Ramp time for flame-coupled BC [s] (default: 30e-6)"
    )
    parser.add_argument(
        "--flame-relaxation", type=float, default=0.7,
        help="Under-relaxation factor for flame coupling (default: 0.7)"
    )
    parser.add_argument(
        "--flame-uncoupled", action="store_true",
        help="Use uncoupled velocity (no P-T iteration) for flame trajectory"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Flag for flame-coupled simulation
    use_flame_coupled = False

    # Create trajectory based on type
    if args.trajectory == "linear":
        trajectory = LinearRampTrajectory(
            u_final=args.u_final,
            ramp_time=args.ramp_time,
        )
    elif args.trajectory == "power":
        trajectory = PowerLawTrajectory(
            k=args.k,
            n=args.power_n,
            u_max=args.u_max,
        )
    elif args.trajectory == "exponential":
        trajectory = ExponentialTrajectory(
            A=args.A,
            B=args.B,
            u_max=args.u_max,
        )
    elif args.trajectory == "flame":
        # Load flame property interpolator
        if args.flame_data:
            flame_csv = Path(args.flame_data)
        else:
            flame_csv = Path(__file__).parent / "cantera_data" / "output" / "flame_properties.csv"

        if not flame_csv.exists():
            print(f"ERROR: Flame properties file not found: {flame_csv}")
            print("Run the Cantera flame calculations first, or specify --flame-data path.")
            sys.exit(1)

        print(f"Loading flame data: {flame_csv}")
        flame_interp = FlamePropertyInterpolator(str(flame_csv))
        print(f"  {flame_interp}")

        # Create elongation function
        if args.flame_elongation == "power":
            elongation = PowerLawElongation(
                sigma_0=args.flame_sigma0,
                k=args.flame_k,
                n=args.flame_n,
            )
        elif args.flame_elongation == "exponential":
            elongation = ExponentialElongation(
                sigma_0=args.flame_sigma0,
                B=args.flame_B,
            )
        elif args.flame_elongation == "linear":
            elongation = LinearElongation(
                sigma_0=args.flame_sigma0,
                k=args.flame_k,
            )
        elif args.flame_elongation == "constant":
            elongation = ConstantElongation(
                sigma_0=args.flame_sigma0,
            )
        else:
            raise ValueError(f"Unknown elongation type: {args.flame_elongation}")

        print(f"  Elongation: {elongation.name}")

        # Create flame trajectory
        flame_traj = FlameElongationTrajectory(
            elongation=elongation,
            flame_interpolator=flame_interp,
            P_ref=args.flame_P_ref,
            T_ref=args.flame_T_ref,
        )

        print(f"  Reference: P={args.flame_P_ref/1e5:.1f} bar, T={args.flame_T_ref:.0f} K")
        print(f"  S_L_ref = {flame_traj.properties_ref.S_L:.2f} m/s")
        print(f"  (rho_u/rho_b)_ref = {flame_traj.properties_ref.density_ratio:.2f}")

        # Wrap in PistonTrajectory interface
        trajectory = FlameCoupledTrajectory(
            flame_trajectory=flame_traj,
            elongation_type=args.flame_elongation,
        )

        # Use flame-coupled BC unless --flame-uncoupled is specified
        use_flame_coupled = not args.flame_uncoupled
        if use_flame_coupled:
            print(f"  Mode: COUPLED (iterative P-T)")
        else:
            print(f"  Mode: UNCOUPLED (reference P, T)")
    else:
        raise ValueError(f"Unknown trajectory type: {args.trajectory}")

    # Create EOS
    eos = CanteraEOS(mechanism_file="gri30.yaml")
    eos.gas.X = {"N2": 0.79, "O2": 0.21}
    eos.set_state_TP(T_STP, P_STP)

    # AV config
    av_config = None
    if args.av:
        av_config = ArtificialViscosityConfig(
            c_linear=args.av_linear,
            c_quad=args.av_quad,
        )

    # Create output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = time_module.strftime("%Y%m%d_%H%M%S")
        output_dir = Path(__file__).parent / "output" / "piston_trajectory" / f"{timestamp}_{trajectory.params_str()}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PISTON TRAJECTORY SIMULATION")
    print("=" * 70)
    print(f"Trajectory: {trajectory.name()}")
    print(f"Domain: {args.domain_length:.2f} m, {args.n_cells} cells")
    print(f"t_end: {args.t_end*1e3:.2f} ms")
    if args.av:
        print(f"AV: c_linear={args.av_linear}, c_quad={args.av_quad}")
    if args.hc:
        print(f"HC: kappa_linear={args.hc_linear}, kappa_quad={args.hc_quad}")
    if args.u_terminate:
        print(f"Terminate at: u_piston >= {args.u_terminate:.0f} m/s")
    if use_flame_coupled:
        print(f"Flame coupling: tol={args.flame_tol}, max_iter={args.flame_max_iter}, "
              f"relaxation={args.flame_relaxation}")
    print(f"Output: {output_dir}")
    print("=" * 70)

    # Run simulation
    results = run_trajectory_simulation(
        trajectory=trajectory,
        eos=eos,
        n_cells=args.n_cells,
        domain_length=args.domain_length,
        t_end=args.t_end,
        cfl=args.cfl,
        dt_min=args.dt_min,
        av_config=av_config,
        hc_enabled=args.hc,
        hc_linear=args.hc_linear,
        hc_quad=args.hc_quad,
        save_interval=args.save_interval,
        u_terminate=args.u_terminate,
        flame_coupled=use_flame_coupled,
        flame_tol=args.flame_tol,
        flame_max_iter=args.flame_max_iter,
        flame_ramp_time=args.flame_ramp_time,
        flame_relaxation=args.flame_relaxation,
    )

    # Save data
    np.savez(
        output_dir / "timeseries.npz",
        **results['saved_data'],
        config=results['config'],
    )
    print(f"  Saved data: {output_dir / 'timeseries.npz'}")

    # Create plots
    plot_final_state(results, str(output_dir / "final_state.png"))

    # Create flame coupling diagnostics if applicable
    if use_flame_coupled:
        plot_flame_coupling_diagnostics(results, str(output_dir / "flame_diagnostics.png"))

    # Create animation
    create_trajectory_animation(results, str(output_dir / "animation.mp4"))

    print()
    print("=" * 70)
    print("SIMULATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
