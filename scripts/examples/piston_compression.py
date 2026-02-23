"""
Piston Compression Simulation

Demonstrates a gas compression driven by a moving piston.
The piston moves into the gas domain, compressing the gas and
generating a shock wave.

Reference: Classic piston problem for testing moving boundary conditions

Usage:
    python scripts/examples/piston_compression.py
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from lagrangian_solver.equations.eos import IdealGasEOS
from lagrangian_solver.core.grid import LagrangianGrid, GridConfig
from lagrangian_solver.core.state import create_uniform_state
from lagrangian_solver.core.solver import LagrangianSolver, SolverConfig
from lagrangian_solver.numerics.riemann import ExactRiemannSolver
from lagrangian_solver.boundary.wall import SolidWallBC
from lagrangian_solver.boundary.piston import MovingPistonBC, ramp_piston
from lagrangian_solver.boundary.base import BoundarySide, ThermalBCType
from lagrangian_solver.io.output import CSVWriter


def run_piston_compression(
    n_cells=100,
    cfl=0.4,
    t_end=0.1,
    piston_velocity=100.0,
    output_dir=None,
):
    """
    Run piston compression simulation.

    Args:
        n_cells: Number of grid cells
        cfl: CFL number
        t_end: Final time
        piston_velocity: Final piston velocity [m/s]
        output_dir: Directory for output (optional)

    Returns:
        Final FlowState, LagrangianGrid, and history of states
    """
    # Create EOS (ideal gas with gamma=1.4)
    gamma = 1.4
    R = 287.05
    eos = IdealGasEOS(gamma=gamma, R=R)

    # Initial conditions (standard atmosphere)
    rho_0 = 1.225  # kg/m³
    p_0 = 101325.0  # Pa
    T_0 = p_0 / (rho_0 * R)  # K
    u_0 = 0.0  # m/s

    # Domain length
    L = 1.0  # m

    # Create grid
    grid_config = GridConfig(n_cells=n_cells, x_min=0.0, x_max=L)
    grid = LagrangianGrid(grid_config)

    # Create Riemann solver
    riemann_solver = ExactRiemannSolver(eos, tol=1e-8, max_iter=100)

    # Create boundary conditions
    # Left: moving piston with ramp velocity profile
    t_ramp = t_end * 0.1  # Ramp over first 10% of simulation
    piston_profile = ramp_piston(v_start=0.0, v_end=piston_velocity, t_ramp=t_ramp)
    bc_left = MovingPistonBC(
        BoundarySide.LEFT,
        eos,
        velocity=piston_profile,
        thermal_bc=ThermalBCType.ADIABATIC,
    )

    # Right: solid wall
    bc_right = SolidWallBC(BoundarySide.RIGHT, eos, ThermalBCType.ADIABATIC)

    # Solver configuration
    solver_config = SolverConfig(
        cfl=cfl,
        t_end=t_end,
        dt_output=t_end / 10,
        verbose=True,
    )

    # Create solver
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
        x_right=L,
        rho=rho_0,
        u=u_0,
        p=p_0,
        eos=eos,
    )
    solver.set_initial_condition(initial_state)

    # Store history
    history = []

    def record_state(state, time, step):
        history.append({
            "time": time,
            "step": step,
            "x": grid.x.copy(),
            "x_cell": grid.x_cell.copy(),
            "rho": state.rho.copy(),
            "u": state.u.copy(),
            "p": state.p.copy(),
            "T": state.T.copy(),
        })

    # Record initial state
    record_state(initial_state, 0.0, 0)
    solver.add_step_callback(lambda s, t, n: record_state(s, t, n) if n % 50 == 0 else None)

    # Run simulation
    print("\n" + "=" * 60)
    print("PISTON COMPRESSION SIMULATION")
    print("=" * 60)
    print(f"Grid cells: {n_cells}")
    print(f"CFL number: {cfl}")
    print(f"Final time: {t_end}")
    print(f"Piston velocity: {piston_velocity} m/s")
    print(f"Initial pressure: {p_0} Pa")
    print(f"Initial temperature: {T_0:.1f} K")
    print("=" * 60 + "\n")

    # Create output writer if directory specified
    writer = None
    if output_dir is not None:
        writer = CSVWriter(output_dir, "piston_compression")

    stats = solver.run(writer)

    if writer is not None:
        writer.finalize()

    # Record final state
    record_state(solver.state, solver.time, solver.step)

    return solver.state, grid, eos, history


def plot_results(state, grid, eos, history, output_dir=None):
    """
    Plot simulation results.
    """
    # Final state
    x_cell = grid.x_cell
    u_cell = 0.5 * (state.u[:-1] + state.u[1:])

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(
        "Piston Compression - Results",
        fontsize=14,
        fontweight="bold",
    )

    # Final state plots (top row)
    # Density
    ax = axes[0, 0]
    ax.plot(x_cell, state.rho, "b-", linewidth=2)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("Density ρ [kg/m³]")
    ax.set_title("Final Density")
    ax.grid(True, alpha=0.3)

    # Velocity
    ax = axes[0, 1]
    ax.plot(x_cell, u_cell, "r-", linewidth=2)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("Velocity u [m/s]")
    ax.set_title("Final Velocity")
    ax.grid(True, alpha=0.3)

    # Pressure
    ax = axes[0, 2]
    ax.plot(x_cell, state.p / 1e3, "g-", linewidth=2)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("Pressure p [kPa]")
    ax.set_title("Final Pressure")
    ax.grid(True, alpha=0.3)

    # Time evolution plots (bottom row)
    # Piston position vs time
    ax = axes[1, 0]
    times = [h["time"] for h in history]
    x_piston = [h["x"][0] for h in history]
    ax.plot(times, x_piston, "k-", linewidth=2)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Piston position [m]")
    ax.set_title("Piston Position")
    ax.grid(True, alpha=0.3)

    # Pressure at right wall vs time
    ax = axes[1, 1]
    p_wall = [h["p"][-1] / 1e3 for h in history]  # kPa
    ax.plot(times, p_wall, "g-", linewidth=2)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Wall pressure [kPa]")
    ax.set_title("Right Wall Pressure")
    ax.grid(True, alpha=0.3)

    # Compression ratio vs time
    ax = axes[1, 2]
    domain_length = [h["x"][-1] - h["x"][0] for h in history]
    compression_ratio = [domain_length[0] / L for L in domain_length]
    ax.plot(times, compression_ratio, "m-", linewidth=2)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Compression ratio")
    ax.set_title("Domain Compression")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path / "piston_compression.png", dpi=150, bbox_inches="tight")
        print(f"\nPlot saved to: {output_path / 'piston_compression.png'}")

    return fig


def plot_spacetime_diagram(history, output_dir=None):
    """
    Create x-t diagram showing wave propagation.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot cell boundaries at each recorded time
    for h in history:
        t = h["time"]
        x = h["x"]
        ax.plot(x, np.full_like(x, t), "b-", alpha=0.3, linewidth=0.5)

    # Plot piston trajectory
    times = [h["time"] for h in history]
    x_piston = [h["x"][0] for h in history]
    ax.plot(x_piston, times, "r-", linewidth=2, label="Piston")

    # Plot right wall (stationary)
    x_wall = [h["x"][-1] for h in history]
    ax.plot(x_wall, times, "k-", linewidth=2, label="Wall")

    ax.set_xlabel("x [m]")
    ax.set_ylabel("Time [s]")
    ax.set_title("Space-Time Diagram (Lagrangian Grid)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if output_dir is not None:
        output_path = Path(output_dir)
        fig.savefig(output_path / "piston_xt_diagram.png", dpi=150, bbox_inches="tight")
        print(f"Plot saved to: {output_path / 'piston_xt_diagram.png'}")

    return fig


def main():
    """Main entry point."""
    output_dir = Path(__file__).parent.parent.parent / "output" / "piston_compression"

    # Run simulation
    state, grid, eos, history = run_piston_compression(
        n_cells=100,
        cfl=0.4,
        t_end=0.005,  # Short time for demonstration
        piston_velocity=100.0,  # 100 m/s piston
        output_dir=output_dir,
    )

    # Plot results
    fig1 = plot_results(state, grid, eos, history, output_dir)
    fig2 = plot_spacetime_diagram(history, output_dir)

    plt.show()


if __name__ == "__main__":
    main()
