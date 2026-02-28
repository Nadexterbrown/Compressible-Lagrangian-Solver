"""
Toro Riemann Problems - Exact vs Numerical Comparison

Runs the full solver on all 5 Toro Riemann problems and compares
numerical results to exact analytical solutions.

Outputs are saved to timestamped subdirectories with:
- config.json: Test configuration for reproducibility
- results.json: Test results summary
- Individual plots for each test showing exact vs numerical

Reference: [Toro2009] Section 4.3.3, Table 4.1
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from lagrangian_solver.equations.eos import IdealGasEOS
from lagrangian_solver.core.grid import LagrangianGrid, GridConfig
from lagrangian_solver.core.state import create_riemann_state
from lagrangian_solver.core.solver import LagrangianSolver, SolverConfig
from lagrangian_solver.numerics.riemann import ExactRiemannSolver, RiemannState
from lagrangian_solver.numerics.artificial_viscosity import ArtificialViscosityConfig
from lagrangian_solver.boundary.wall import SolidWallBC
from lagrangian_solver.boundary.base import BoundarySide
from lagrangian_solver.boundary.open import OpenBC

from output_manager import create_output_manager, TestResult, OutputManager


# Toro Table 4.1 test data
TORO_TESTS = {
    1: {
        "name": "Sod shock tube",
        "left": {"rho": 1.0, "u": 0.0, "p": 1.0},
        "right": {"rho": 0.125, "u": 0.0, "p": 0.1},
        "t_end": 0.25,
        "x_disc": 0.5,
    },
    2: {
        "name": "Two rarefactions",
        "left": {"rho": 1.0, "u": -2.0, "p": 0.4},
        "right": {"rho": 1.0, "u": 2.0, "p": 0.4},
        "t_end": 0.15,
        "x_disc": 0.5,
    },
    3: {
        "name": "Left blast wave",
        "left": {"rho": 1.0, "u": 0.0, "p": 1000.0},
        "right": {"rho": 1.0, "u": 0.0, "p": 0.01},
        "t_end": 0.012,
        "x_disc": 0.5,
    },
    4: {
        "name": "Right blast wave",
        "left": {"rho": 1.0, "u": 0.0, "p": 0.01},
        "right": {"rho": 1.0, "u": 0.0, "p": 100.0},
        "t_end": 0.035,
        "x_disc": 0.5,
    },
    5: {
        "name": "Two shock collision",
        "left": {"rho": 5.99924, "u": 19.5975, "p": 460.894},
        "right": {"rho": 5.99242, "u": -6.19633, "p": 46.0950},
        "t_end": 0.035,
        "x_disc": 0.5,
    },
}

GAMMA = 1.4
N_CELLS = 400


def create_riemann_state_obj(data: dict, gamma: float) -> RiemannState:
    """Create RiemannState object from test data."""
    rho = data["rho"]
    p = data["p"]
    c = np.sqrt(gamma * p / rho)
    e = p / (rho * (gamma - 1))
    return RiemannState(rho=rho, u=data["u"], p=p, c=c, e=e)


def sample_exact_solution(solver, left, right, x_disc, t, n_points=1000):
    """Sample exact Riemann solution at multiple points."""
    x = np.linspace(0, 1, n_points)
    rho = np.zeros(n_points)
    u = np.zeros(n_points)
    p = np.zeros(n_points)

    for i, xi in enumerate(x):
        if t > 0:
            S = (xi - x_disc) / t
        else:
            S = 0.0 if xi < x_disc else 1e10
        rho[i], u[i], p[i] = solver.sample(left, right, S)

    # Internal energy from EOS
    e = p / (rho * (GAMMA - 1))

    return x, rho, u, p, e


def run_toro_test(
    test_num: int,
    eos: IdealGasEOS,
    n_cells: int = N_CELLS,
    cfl: float = 0.5,
    dt_min: float = None,
    av_config: ArtificialViscosityConfig = None,
    hc_enabled: bool = False,
    hc_linear: float = 0.1,
    hc_quad: float = 0.5,
    verbose: bool = False,
):
    """
    Run a single Toro test and return numerical solution.

    Args:
        test_num: Toro test number (1-5)
        eos: Equation of state
        n_cells: Number of cells
        cfl: CFL number
        dt_min: Minimum time step floor (None for no floor)
        av_config: Artificial viscosity configuration (None to disable)
        hc_enabled: Enable artificial heat conduction
        hc_linear: HC linear coefficient
        hc_quad: HC quadratic coefficient
        verbose: Print progress

    Returns:
        Tuple of (state, grid, statistics, failed, error_msg)
    """
    test_data = TORO_TESTS[test_num]

    # Create grid
    grid_config = GridConfig(n_cells=n_cells, x_min=0.0, x_max=1.0)
    grid = LagrangianGrid(grid_config)

    # Create Riemann solver
    riemann_solver = ExactRiemannSolver(eos, tol=1e-8, max_iter=100)

    # Create boundary conditions - use Open BC for Riemann problems
    # Provide external conditions from the initial state for cases with inflow
    left_data = test_data["left"]
    right_data = test_data["right"]

    bc_left = OpenBC(
        BoundarySide.LEFT,
        eos,
        p_external=left_data["p"],
        u_external=left_data["u"],
        rho_external=left_data["rho"],
    )
    bc_right = OpenBC(
        BoundarySide.RIGHT,
        eos,
        p_external=right_data["p"],
        u_external=right_data["u"],
        rho_external=right_data["rho"],
    )

    # Solver configuration with optional dt_min
    solver_config = SolverConfig(
        cfl=cfl,
        t_end=test_data["t_end"],
        dt_output=test_data["t_end"],
        dt_min=dt_min,
        verbose=verbose,
        artificial_viscosity=av_config,
        hc_enabled=hc_enabled,
        hc_linear=hc_linear,
        hc_quad=hc_quad,
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

    # Set Riemann problem initial condition
    solver.set_riemann_ic(
        x_disc=test_data["x_disc"],
        rho_L=test_data["left"]["rho"],
        u_L=test_data["left"]["u"],
        p_L=test_data["left"]["p"],
        rho_R=test_data["right"]["rho"],
        u_R=test_data["right"]["u"],
        p_R=test_data["right"]["p"],
    )

    # Run simulation - catch failures and return partial results
    try:
        stats = solver.run()
        failed = False
        error_msg = None
    except Exception as e:
        # Get partial stats from solver
        stats = solver.statistics
        failed = True
        error_msg = str(e)

    return solver.state, solver.grid, stats, failed, error_msg


def plot_toro_test_comparison(
    test_num: int,
    output_manager: OutputManager,
    eos: IdealGasEOS,
    n_cells: int = N_CELLS,
    cfl: float = 0.5,
    dt_min: float = None,
    av_config: ArtificialViscosityConfig = None,
    hc_enabled: bool = False,
    hc_linear: float = 0.1,
    hc_quad: float = 0.5,
):
    """
    Plot exact vs numerical comparison for a single Toro test.

    Args:
        test_num: Toro test number (1-5)
        output_manager: OutputManager for saving results
        eos: Equation of state
        n_cells: Number of cells
        cfl: CFL number
        dt_min: Optional minimum time step floor
        av_config: Artificial viscosity configuration
        hc_enabled: Enable artificial heat conduction
        hc_linear: HC linear coefficient
        hc_quad: HC quadratic coefficient
    """
    test_data = TORO_TESTS[test_num]

    dt_min_str = f" (dt_min={dt_min:.0e})" if dt_min else ""
    print(f"Running Toro Test {test_num}: {test_data['name']}{dt_min_str}...")

    # Get numerical solution (may return partial results on failure)
    state, grid, stats, failed, error_msg = run_toro_test(
        test_num, eos, n_cells=n_cells, cfl=cfl, dt_min=dt_min,
        av_config=av_config, hc_enabled=hc_enabled, hc_linear=hc_linear,
        hc_quad=hc_quad
    )

    if failed:
        print(f"  WARNING: Simulation failed at t={stats.final_time:.4e}: {error_msg}")
        print(f"  Generating plot with partial results...")

    # Get exact solution
    riemann_solver = ExactRiemannSolver(eos)
    left = create_riemann_state_obj(test_data["left"], GAMMA)
    right = create_riemann_state_obj(test_data["right"], GAMMA)

    x_exact, rho_exact, u_exact, p_exact, e_exact = sample_exact_solution(
        riemann_solver, left, right,
        test_data["x_disc"], test_data["t_end"]
    )

    # Numerical solution
    x_num = grid.x_cell
    rho_num = state.rho
    u_num = 0.5 * (state.u[:-1] + state.u[1:])
    p_num = state.p
    e_num = state.e
    T_num = state.T
    s_num = state.s

    # Exact entropy and temperature
    T_exact = p_exact / (rho_exact * (GAMMA - 1) / eos.cv)  # From e = cv*T
    T_exact = e_exact / eos.cv
    s_exact = eos.entropy(rho_exact, p_exact)

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Build title with stats
    status_str = "FAILED" if failed else ""
    title_lines = [f"Toro Test {test_num}: {test_data['name']} {status_str}"]
    t_reached = stats.final_time
    title_lines.append(f"t = {t_reached:.4e} / {test_data['t_end']}, N = {n_cells} cells, steps = {stats.n_steps}")
    if dt_min:
        min_dt_str = f"{stats.min_dt:.2e}" if stats.min_dt < float('inf') else "N/A"
        title_lines.append(f"dt_min = {dt_min:.0e} (min_dt used: {min_dt_str})")
    if failed:
        title_lines.append(f"Error: {error_msg[:60]}..." if len(error_msg) > 60 else f"Error: {error_msg}")

    fig.suptitle("\n".join(title_lines), fontsize=14, fontweight="bold")

    # Plot variables
    variables = [
        ("Density ρ", rho_exact, rho_num, "[kg/m³]"),
        ("Velocity u", u_exact, u_num, "[m/s]"),
        ("Pressure p", p_exact, p_num, "[Pa]"),
        ("Internal Energy e", e_exact, e_num, "[J/kg]"),
        ("Temperature T", T_exact, T_num, "[K]"),
        ("Entropy s", s_exact, s_num, "[J/(kg·K)]"),
    ]

    for idx, (name, exact, num, unit) in enumerate(variables):
        ax = axes[idx // 3, idx % 3]
        ax.plot(x_exact, exact, "b-", linewidth=2, label="Exact")
        ax.plot(x_num, num, "ro", markersize=3, label="Numerical")
        ax.set_xlabel("x")
        ax.set_ylabel(f"{name} {unit}")
        ax.set_title(name)
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)

    plt.tight_layout()

    # Save figure using output manager
    filename = f"toro_test_{test_num}.png"
    output_manager.save_figure(fig, filename)
    print(f"    Steps: {stats.n_steps}, Wall time: {stats.wall_time:.2f}s, "
          f"Min dt: {stats.min_dt:.2e}")

    plt.close(fig)

    # Record result
    result = TestResult(
        test_id=test_num,
        steps=stats.n_steps,
        wall_time=stats.wall_time,
        final_time=stats.final_time,
        failed=failed,
        error_message=error_msg if error_msg else "",
    )
    output_manager.add_result(result)

    return x_num, rho_num, u_num, p_num, stats, failed


def create_all_tests_comparison(output_dir: Path):
    """Create a summary figure comparing all 5 tests."""
    eos = IdealGasEOS(gamma=GAMMA)
    riemann_solver = ExactRiemannSolver(eos)

    fig, axes = plt.subplots(5, 4, figsize=(16, 20))
    fig.suptitle(
        "Toro's Five Riemann Problems - Exact vs Numerical\n"
        f"N = {N_CELLS} cells, Reference: [Toro2009] Section 4.3.3",
        fontsize=14,
        fontweight="bold",
    )

    for test_num in range(1, 6):
        test_data = TORO_TESTS[test_num]

        # Get numerical solution
        state, grid, stats, failed, error_msg = run_toro_test(test_num, eos)

        # Get exact solution
        left = create_riemann_state_obj(test_data["left"], GAMMA)
        right = create_riemann_state_obj(test_data["right"], GAMMA)
        x_exact, rho_exact, u_exact, p_exact, e_exact = sample_exact_solution(
            riemann_solver, left, right,
            test_data["x_disc"], test_data["t_end"]
        )

        # Numerical
        x_num = grid.x_cell
        rho_num = state.rho
        u_num = 0.5 * (state.u[:-1] + state.u[1:])
        p_num = state.p
        e_num = state.e

        row = test_num - 1

        # Density
        axes[row, 0].plot(x_exact, rho_exact, "b-", linewidth=1.5, label="Exact")
        axes[row, 0].plot(x_num, rho_num, "r.", markersize=2, label="Numerical")
        axes[row, 0].set_ylabel(f"Test {test_num}\nρ")
        axes[row, 0].grid(True, alpha=0.3)
        if row == 0:
            axes[row, 0].set_title("Density")
            axes[row, 0].legend(loc="best", fontsize=8)

        # Velocity
        axes[row, 1].plot(x_exact, u_exact, "b-", linewidth=1.5)
        axes[row, 1].plot(x_num, u_num, "r.", markersize=2)
        axes[row, 1].grid(True, alpha=0.3)
        if row == 0:
            axes[row, 1].set_title("Velocity")

        # Pressure
        axes[row, 2].plot(x_exact, p_exact, "b-", linewidth=1.5)
        axes[row, 2].plot(x_num, p_num, "r.", markersize=2)
        axes[row, 2].grid(True, alpha=0.3)
        if row == 0:
            axes[row, 2].set_title("Pressure")

        # Internal Energy
        axes[row, 3].plot(x_exact, e_exact, "b-", linewidth=1.5)
        axes[row, 3].plot(x_num, e_num, "r.", markersize=2)
        axes[row, 3].grid(True, alpha=0.3)
        if row == 0:
            axes[row, 3].set_title("Internal Energy")

        # Add test name
        axes[row, 3].text(
            1.02, 0.5, test_data["name"],
            transform=axes[row, 3].transAxes,
            rotation=-90,
            va="center",
            fontsize=9,
        )

    # Set x-labels on bottom row
    for col in range(4):
        axes[4, col].set_xlabel("x")

    plt.tight_layout()

    fig.savefig(output_dir / "toro_all_tests_comparison.png",
                dpi=150, bbox_inches="tight")
    print(f"Saved: toro_all_tests_comparison.png")

    plt.close(fig)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Toro Riemann problem verification tests"
    )
    parser.add_argument(
        "--n-cells", type=int, default=N_CELLS,
        help=f"Number of grid cells (default: {N_CELLS})"
    )
    parser.add_argument(
        "--cfl", type=float, default=0.3,
        help="CFL number (default: 0.3)"
    )
    parser.add_argument(
        "--dt-min", type=float, default=1e-9,
        help="Minimum time step floor (default: 1e-9)"
    )
    parser.add_argument(
        "--av", action="store_true",
        help="Enable artificial viscosity"
    )
    parser.add_argument(
        "--av-linear", type=float, default=0.3,
        help="AV linear coefficient (default: 0.3)"
    )
    parser.add_argument(
        "--av-quad", type=float, default=2.0,
        help="AV quadratic coefficient (default: 2.0)"
    )
    parser.add_argument(
        "--description", type=str, default="",
        help="Description for this test run"
    )
    parser.add_argument(
        "--tests", type=str, default="1,2,3,4,5",
        help="Comma-separated list of tests to run (default: 1,2,3,4,5)"
    )
    parser.add_argument(
        "--hc", action="store_true",
        help="Enable artificial heat conduction for contact discontinuity spreading"
    )
    parser.add_argument(
        "--hc-linear", type=float, default=0.1,
        help="HC linear coefficient (default: 0.1)"
    )
    parser.add_argument(
        "--hc-quad", type=float, default=0.5,
        help="HC quadratic coefficient (default: 0.5)"
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Parse test numbers
    test_nums = [int(t.strip()) for t in args.tests.split(",")]

    # Build description if not provided
    if not args.description:
        parts = ["first_order"]
        if args.av:
            parts = [f"av_clin{args.av_linear}_cq{args.av_quad}"]
        if args.hc:
            parts.append(f"hc_kl{args.hc_linear}_kq{args.hc_quad}")
        args.description = "_".join(parts)

    # Create artificial viscosity config if enabled
    av_config = None
    if args.av:
        av_config = ArtificialViscosityConfig(
            c_linear=args.av_linear,
            c_quad=args.av_quad,
        )

    # Create output manager
    output_manager = create_output_manager(
        test_type="toro",
        n_cells=args.n_cells,
        cfl=args.cfl,
        gamma=GAMMA,
        description=args.description,
        dt_min=args.dt_min,
        av_enabled=args.av,
        av_c_linear=args.av_linear if args.av else 0.0,
        av_c_quad=args.av_quad if args.av else 0.0,
        hc_enabled=args.hc,
        hc_linear=args.hc_linear if args.hc else 0.0,
        hc_quad=args.hc_quad if args.hc else 0.0,
    )

    print("=" * 70)
    print("TORO RIEMANN PROBLEM VERIFICATION - EXACT VS NUMERICAL")
    print("Reference: [Toro2009] Section 4.3.3, Table 4.1")
    print("=" * 70)
    print(f"Configuration: {args.description}")
    print(f"  N_cells: {args.n_cells}, CFL: {args.cfl}, dt_min: {args.dt_min:.0e}")
    if args.av:
        print(f"  AV: c_linear={args.av_linear}, c_quad={args.av_quad}")
    if args.hc:
        print(f"  HC: kappa_linear={args.hc_linear}, kappa_quad={args.hc_quad}")
    print(f"Output: {output_manager.output_dir}")
    print("=" * 70 + "\n")

    eos = IdealGasEOS(gamma=GAMMA)

    # Run requested tests
    for test_num in test_nums:
        if test_num == 5:
            print("\n" + "-" * 70)
            print("TEST 5 (Two Shock Collision)")
            print("-" * 70)
            print("  NOTE: Test 5 may fail due to cell collapse from shock-shock")
            print("  interaction. This is a fundamental Lagrangian limitation.")
        else:
            print("-" * 70)
            print(f"TEST {test_num}")
            print("-" * 70)

        plot_toro_test_comparison(
            test_num,
            output_manager,
            eos,
            n_cells=args.n_cells,
            cfl=args.cfl,
            dt_min=args.dt_min,
            av_config=av_config,
            hc_enabled=args.hc,
            hc_linear=args.hc_linear,
            hc_quad=args.hc_quad,
        )
        print()

    # Print summary
    output_manager.print_summary()


if __name__ == "__main__":
    main()
