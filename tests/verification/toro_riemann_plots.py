"""
Toro Riemann Problems - Exact vs Numerical Comparison

Runs the full solver on all 5 Toro Riemann problems and compares
numerical results to exact analytical solutions.

Outputs:
- Individual plots for each test showing exact vs numerical
- Summary comparison figure
- Plots saved to tests/verification/output/

Reference: [Toro2009] Section 4.3.3, Table 4.1
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
from lagrangian_solver.numerics.adaptive import AdaptiveConfig
from lagrangian_solver.boundary.wall import SolidWallBC
from lagrangian_solver.boundary.base import BoundarySide
from lagrangian_solver.boundary.open import OpenBC


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
    adaptive_config: AdaptiveConfig = None,
    verbose: bool = False,
):
    """
    Run a single Toro test and return numerical solution.

    Args:
        test_num: Toro test number (1-5)
        eos: Equation of state
        n_cells: Number of cells
        adaptive_config: Optional adaptive mesh configuration
        verbose: Print progress

    Returns:
        Tuple of (state, grid, statistics)
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

    # Solver configuration
    solver_config = SolverConfig(
        cfl=0.5,
        t_end=test_data["t_end"],
        dt_output=test_data["t_end"],
        verbose=verbose,
    )

    # Create solver with optional adaptive mesh
    solver = LagrangianSolver(
        grid=grid,
        eos=eos,
        riemann_solver=riemann_solver,
        bc_left=bc_left,
        bc_right=bc_right,
        config=solver_config,
        adaptive_config=adaptive_config,
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

    # Run simulation
    stats = solver.run()

    return solver.state, solver.grid, stats


def plot_toro_test_comparison(
    test_num: int,
    output_dir: Path,
    adaptive_config: AdaptiveConfig = None,
):
    """Plot exact vs numerical comparison for a single Toro test."""
    test_data = TORO_TESTS[test_num]
    eos = IdealGasEOS(gamma=GAMMA)

    amr_label = " (with AMR)" if adaptive_config is not None else ""
    print(f"Running Toro Test {test_num}: {test_data['name']}{amr_label}...")

    # Get numerical solution
    state, grid, stats = run_toro_test(test_num, eos, adaptive_config=adaptive_config)

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

    # Build title with adaptive mesh info if enabled
    title_lines = [f"Toro Test {test_num}: {test_data['name']}"]
    if adaptive_config is not None:
        title_lines.append(
            f"t = {test_data['t_end']}, N = {stats.cells_initial}→{stats.cells_final} cells (AMR)"
        )
        title_lines.append(
            f"Splits: {stats.total_splits}, Merges: {stats.total_merges}, "
            f"Levels: [{stats.min_refine_level}, {stats.max_refine_level}]"
        )
    else:
        title_lines.append(f"t = {test_data['t_end']}, N = {N_CELLS} cells")

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

    # Save figure
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_amr" if adaptive_config is not None else ""
    filename = f"toro_test_{test_num}_comparison{suffix}.png"
    fig.savefig(output_dir / filename, dpi=150, bbox_inches="tight")
    print(f"  Saved: {filename}")

    # Print stats summary
    print(f"    Steps: {stats.n_steps}, Wall time: {stats.wall_time:.2f}s, "
          f"Min dt: {stats.min_dt:.2e}")
    if adaptive_config is not None:
        print(f"    Cells: {stats.cells_initial}→{stats.cells_final}, "
              f"Splits: {stats.total_splits}, Merges: {stats.total_merges}")

    plt.close(fig)

    return x_num, rho_num, u_num, p_num, stats


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
        state, grid, stats = run_toro_test(test_num, eos)

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


def get_adaptive_config_for_test(test_num: int) -> AdaptiveConfig:
    """
    Get adaptive mesh configuration tuned for each Toro test.

    Different tests have different challenges:
    - Tests 1-4: Standard AMR with level limits
    - Test 5: Aggressive coarsening with dt_floor to handle shock collision
    """
    if test_num == 5:
        # Two shock collision - challenging case for Lagrangian methods
        # Use aggressive coarsening, NO dt_floor (can cause cell collapse)
        # Accept that this test may take many steps with small dt
        return AdaptiveConfig(
            enabled=True,
            refine_threshold=3.0,    # Less aggressive refinement
            coarsen_threshold=0.5,   # Very aggressive coarsening
            min_cells=15,            # Allow very few cells
            max_cells=150,           # Limit max cells
            check_interval=1,        # Check every step
            max_refine_level=1,      # Minimal refinement
            max_coarsen_level=-4,    # Allow deep coarsening
            min_dt=1e-4,             # Early coarsening trigger
            min_dx=2e-3,             # Larger minimum cell size
            dt_floor=None,           # No dt_floor - let physics dictate
            dt_floor_enabled=False,  # Disabled
        )
    else:
        # Standard tests - moderate AMR settings
        return AdaptiveConfig(
            enabled=True,
            refine_threshold=2.0,
            coarsen_threshold=0.25,
            min_cells=50,
            max_cells=600,
            check_interval=10,
            max_refine_level=4,      # Allow 4 levels of refinement
            max_coarsen_level=-2,    # Allow 2 levels of coarsening
            min_dt=None,             # No emergency coarsening
            dt_floor=None,           # No dt floor
            dt_floor_enabled=False,
        )


def main():
    """Main entry point."""
    output_dir = Path(__file__).parent / "output"

    print("=" * 70)
    print("TORO RIEMANN PROBLEM VERIFICATION - EXACT VS NUMERICAL")
    print("Reference: [Toro2009] Section 4.3.3, Table 4.1")
    print("=" * 70 + "\n")

    # Run individual tests WITHOUT adaptive mesh (baseline)
    print("-" * 70)
    print("BASELINE TESTS (No AMR)")
    print("-" * 70)
    for test_num in range(1, 6):
        plot_toro_test_comparison(test_num, output_dir)

    # Run individual tests WITH adaptive mesh (skip Test 5 - known Lagrangian limitation)
    print("\n" + "-" * 70)
    print("ADAPTIVE MESH TESTS (With AMR and Level Controls)")
    print("-" * 70)
    for test_num in range(1, 6):
        if test_num == 5:
            print(f"\n  Skipping Test 5 (two shock collision) with AMR:")
            print(f"    This test exhibits fundamental Lagrangian CFL limitation")
            print(f"    when shocks collide, causing impractical time steps.")
            print(f"    The baseline test (without AMR) is available for reference.")
            continue

        adaptive_config = get_adaptive_config_for_test(test_num)
        print(f"\n  AMR Config for Test {test_num}:")
        print(f"    max_refine_level={adaptive_config.max_refine_level}, "
              f"max_coarsen_level={adaptive_config.max_coarsen_level}")
        if adaptive_config.dt_floor_enabled:
            print(f"    dt_floor={adaptive_config.dt_floor} (ENABLED)")
        plot_toro_test_comparison(test_num, output_dir, adaptive_config=adaptive_config)

    # Create summary comparison (baseline only)
    print("\n" + "-" * 70)
    print("Creating summary comparison figure (baseline)...")
    create_all_tests_comparison(output_dir)

    print(f"\nAll plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
