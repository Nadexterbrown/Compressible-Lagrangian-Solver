"""
Sod Shock Tube Simulation

Classic Riemann problem test case for 1D compressible flow solvers.
Demonstrates shock wave, contact discontinuity, and rarefaction fan.

Reference: [Toro2009] Section 4.3.3, Test 1 (modified Sod)

Initial conditions:
    Left state:  ρ = 1.0,   u = 0.0, p = 1.0
    Right state: ρ = 0.125, u = 0.0, p = 0.1

Usage:
    python scripts/examples/sod_shock_tube.py
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from lagrangian_solver.equations.eos import IdealGasEOS
from lagrangian_solver.core.grid import LagrangianGrid, GridConfig
from lagrangian_solver.core.state import create_riemann_state
from lagrangian_solver.core.solver import LagrangianSolver, SolverConfig
from lagrangian_solver.numerics.riemann import ExactRiemannSolver, RiemannState
from lagrangian_solver.boundary.wall import SolidWallBC
from lagrangian_solver.boundary.base import BoundarySide
from lagrangian_solver.io.output import CSVWriter, OutputFrame


def compute_exact_solution(eos, t_end, x, x_disc=0.5):
    """
    Compute exact solution for Sod shock tube at time t_end.

    Args:
        eos: Equation of state
        t_end: Time at which to sample solution
        x: Array of x positions
        x_disc: Discontinuity position

    Returns:
        rho_exact, u_exact, p_exact arrays
    """
    gamma = eos.gamma

    # Left and right states
    left = RiemannState(
        rho=1.0,
        u=0.0,
        p=1.0,
        c=np.sqrt(gamma * 1.0 / 1.0),
        e=1.0 / (1.0 * (gamma - 1)),
    )
    right = RiemannState(
        rho=0.125,
        u=0.0,
        p=0.1,
        c=np.sqrt(gamma * 0.1 / 0.125),
        e=0.1 / (0.125 * (gamma - 1)),
    )

    solver = ExactRiemannSolver(eos)

    n = len(x)
    rho = np.zeros(n)
    u = np.zeros(n)
    p = np.zeros(n)

    for i in range(n):
        if t_end > 0:
            S = (x[i] - x_disc) / t_end
        else:
            S = 0.0 if x[i] < x_disc else 1e10
        rho[i], u[i], p[i] = solver.sample(left, right, S)

    return rho, u, p


def run_sod_shock_tube(n_cells=200, cfl=0.5, t_end=0.25, output_dir=None):
    """
    Run Sod shock tube simulation.

    Args:
        n_cells: Number of grid cells
        cfl: CFL number
        t_end: Final time
        output_dir: Directory for output (optional)

    Returns:
        Final FlowState and LagrangianGrid
    """
    # Create EOS (ideal gas with gamma=1.4)
    gamma = 1.4
    eos = IdealGasEOS(gamma=gamma)

    # Create grid
    grid_config = GridConfig(n_cells=n_cells, x_min=0.0, x_max=1.0)
    grid = LagrangianGrid(grid_config)

    # Create Riemann solver
    riemann_solver = ExactRiemannSolver(eos, tol=1e-8, max_iter=100)

    # Create boundary conditions (reflective walls)
    bc_left = SolidWallBC(BoundarySide.LEFT, eos)
    bc_right = SolidWallBC(BoundarySide.RIGHT, eos)

    # Solver configuration
    solver_config = SolverConfig(
        cfl=cfl,
        t_end=t_end,
        dt_output=0.05,
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

    # Set initial condition: Sod shock tube
    solver.set_riemann_ic(
        x_disc=0.5,
        rho_L=1.0, u_L=0.0, p_L=1.0,
        rho_R=0.125, u_R=0.0, p_R=0.1,
    )

    # Run simulation
    print("\n" + "=" * 60)
    print("SOD SHOCK TUBE SIMULATION")
    print("=" * 60)
    print(f"Grid cells: {n_cells}")
    print(f"CFL number: {cfl}")
    print(f"Final time: {t_end}")
    print("=" * 60 + "\n")

    # Create output writer if directory specified
    writer = None
    if output_dir is not None:
        writer = CSVWriter(output_dir, "sod_shock")

    stats = solver.run(writer)

    if writer is not None:
        writer.finalize()

    return solver.state, grid, eos


def plot_results(state, grid, eos, t_end=0.25, output_dir=None):
    """
    Plot numerical results compared to exact solution.
    """
    # Cell centers
    x_cell = grid.x_cell

    # Cell-averaged velocity
    u_cell = 0.5 * (state.u[:-1] + state.u[1:])

    # Compute exact solution
    x_exact = np.linspace(0, 1, 1000)
    rho_exact, u_exact, p_exact = compute_exact_solution(eos, t_end, x_exact)

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(
        f"Sod Shock Tube at t = {t_end}\n"
        f"N = {grid.n_cells} cells, CFL = 0.5",
        fontsize=14,
        fontweight="bold",
    )

    # Density
    ax = axes[0, 0]
    ax.plot(x_exact, rho_exact, "b-", linewidth=2, label="Exact")
    ax.plot(x_cell, state.rho, "ro", markersize=3, label="Numerical")
    ax.set_xlabel("x")
    ax.set_ylabel("Density ρ")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)

    # Velocity
    ax = axes[0, 1]
    ax.plot(x_exact, u_exact, "b-", linewidth=2, label="Exact")
    ax.plot(x_cell, u_cell, "ro", markersize=3, label="Numerical")
    ax.set_xlabel("x")
    ax.set_ylabel("Velocity u")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)

    # Pressure
    ax = axes[1, 0]
    ax.plot(x_exact, p_exact, "b-", linewidth=2, label="Exact")
    ax.plot(x_cell, state.p, "ro", markersize=3, label="Numerical")
    ax.set_xlabel("x")
    ax.set_ylabel("Pressure p")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)

    # Internal energy
    ax = axes[1, 1]
    e_exact = p_exact / (rho_exact * (eos.gamma - 1))
    ax.plot(x_exact, e_exact, "b-", linewidth=2, label="Exact")
    ax.plot(x_cell, state.e, "ro", markersize=3, label="Numerical")
    ax.set_xlabel("x")
    ax.set_ylabel("Internal Energy e")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)

    plt.tight_layout()

    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path / "sod_shock_tube.png", dpi=150, bbox_inches="tight")
        print(f"\nPlot saved to: {output_path / 'sod_shock_tube.png'}")

    return fig


def main():
    """Main entry point."""
    output_dir = Path(__file__).parent.parent.parent / "output" / "sod_shock"

    # Run simulation
    state, grid, eos = run_sod_shock_tube(
        n_cells=200,
        cfl=0.5,
        t_end=0.25,
        output_dir=output_dir,
    )

    # Plot results
    fig = plot_results(state, grid, eos, t_end=0.25, output_dir=output_dir)

    plt.show()


if __name__ == "__main__":
    main()
