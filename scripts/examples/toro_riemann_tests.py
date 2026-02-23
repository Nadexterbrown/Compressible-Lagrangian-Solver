"""
Toro Riemann Problem Test Suite with Visualization

Runs all five Toro Riemann problems and plots the exact solutions
across the spatial domain.

Reference: [Toro2009] Section 4.3.3, Table 4.1

Usage:
    python scripts/examples/toro_riemann_tests.py
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from lagrangian_solver.equations.eos import IdealGasEOS
from lagrangian_solver.numerics.riemann import ExactRiemannSolver, RiemannState


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


def create_riemann_state(data: dict, gamma: float) -> RiemannState:
    """Create RiemannState from test data."""
    rho = data["rho"]
    p = data["p"]
    c = np.sqrt(gamma * p / rho)
    e = p / (rho * (gamma - 1))
    return RiemannState(rho=rho, u=data["u"], p=p, c=c, e=e)


def sample_exact_solution(solver, left, right, x_disc, t, n_points=500):
    """
    Sample the exact Riemann solution at multiple points.

    Args:
        solver: ExactRiemannSolver instance
        left: Left state
        right: Right state
        x_disc: Discontinuity position
        t: Time
        n_points: Number of sample points

    Returns:
        x, rho, u, p, e arrays
    """
    x = np.linspace(0, 1, n_points)
    rho = np.zeros(n_points)
    u = np.zeros(n_points)
    p = np.zeros(n_points)

    for i, xi in enumerate(x):
        # Similarity variable S = (x - x_disc) / t
        if t > 0:
            S = (xi - x_disc) / t
        else:
            S = 0.0 if xi < x_disc else 1e10

        rho[i], u[i], p[i] = solver.sample(left, right, S)

    # Internal energy from EOS
    e = p / (rho * (GAMMA - 1))

    return x, rho, u, p, e


def plot_toro_test(test_num, solver, output_dir=None):
    """
    Plot the exact solution for a single Toro test.

    Args:
        test_num: Test number (1-5)
        solver: ExactRiemannSolver instance
        output_dir: Directory for saving plots (optional)

    Returns:
        Figure object
    """
    test_data = TORO_TESTS[test_num]

    left = create_riemann_state(test_data["left"], GAMMA)
    right = create_riemann_state(test_data["right"], GAMMA)
    t_end = test_data["t_end"]
    x_disc = test_data["x_disc"]

    # Get exact solution
    x, rho, u, p, e = sample_exact_solution(solver, left, right, x_disc, t_end)

    # Also get star region values for annotation
    solution = solver.solve(left, right)

    # Create figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(
        f"Toro Test {test_num}: {test_data['name']}\n"
        f"t = {t_end:.4f}, γ = {GAMMA}",
        fontsize=14,
        fontweight="bold",
    )

    # Density
    ax = axes[0, 0]
    ax.plot(x, rho, "b-", linewidth=2)
    ax.set_xlabel("x")
    ax.set_ylabel("Density ρ [kg/m³]")
    ax.set_title("Density")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)

    # Velocity
    ax = axes[0, 1]
    ax.plot(x, u, "r-", linewidth=2)
    ax.axhline(y=solution.u_star, color="k", linestyle="--", alpha=0.5, label=f"u* = {solution.u_star:.4f}")
    ax.set_xlabel("x")
    ax.set_ylabel("Velocity u [m/s]")
    ax.set_title("Velocity")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    ax.set_xlim(0, 1)

    # Pressure
    ax = axes[1, 0]
    ax.plot(x, p, "g-", linewidth=2)
    ax.axhline(y=solution.p_star, color="k", linestyle="--", alpha=0.5, label=f"p* = {solution.p_star:.4f}")
    ax.set_xlabel("x")
    ax.set_ylabel("Pressure p [Pa]")
    ax.set_title("Pressure")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    ax.set_xlim(0, 1)

    # Internal energy
    ax = axes[1, 1]
    ax.plot(x, e, "m-", linewidth=2)
    ax.set_xlabel("x")
    ax.set_ylabel("Internal Energy e [J/kg]")
    ax.set_title("Internal Energy")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)

    plt.tight_layout()

    # Save if output directory specified
    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path / f"toro_test_{test_num}.png", dpi=150, bbox_inches="tight")
        print(f"Saved: toro_test_{test_num}.png")

    return fig


def plot_all_toro_tests(output_dir=None):
    """
    Plot all five Toro tests.

    Args:
        output_dir: Directory for saving plots (optional)
    """
    eos = IdealGasEOS(gamma=GAMMA)
    solver = ExactRiemannSolver(eos, tol=1e-8, max_iter=100)

    print("=" * 70)
    print("TORO RIEMANN PROBLEM TEST SUITE - VISUALIZATION")
    print("Reference: [Toro2009] Section 4.3.3, Table 4.1")
    print("=" * 70)

    figures = []
    for test_num in range(1, 6):
        test_data = TORO_TESTS[test_num]
        left = create_riemann_state(test_data["left"], GAMMA)
        right = create_riemann_state(test_data["right"], GAMMA)

        solution = solver.solve(left, right)

        print(f"\nTest {test_num}: {test_data['name']}")
        print(f"  p* = {solution.p_star:.6f}")
        print(f"  u* = {solution.u_star:.6f}")
        print(f"  Left wave: {solution.wave_L.value}")
        print(f"  Right wave: {solution.wave_R.value}")

        fig = plot_toro_test(test_num, solver, output_dir)
        figures.append(fig)

    print("\n" + "=" * 70)

    return figures


def create_comparison_figure(output_dir=None):
    """
    Create a single figure comparing all 5 Toro tests.
    """
    eos = IdealGasEOS(gamma=GAMMA)
    solver = ExactRiemannSolver(eos, tol=1e-8, max_iter=100)

    fig, axes = plt.subplots(5, 4, figsize=(16, 20))
    fig.suptitle(
        "Toro's Five Riemann Problems - Exact Solutions\n"
        "Reference: [Toro2009] Section 4.3.3",
        fontsize=14,
        fontweight="bold",
    )

    for test_num in range(1, 6):
        test_data = TORO_TESTS[test_num]
        left = create_riemann_state(test_data["left"], GAMMA)
        right = create_riemann_state(test_data["right"], GAMMA)
        t_end = test_data["t_end"]
        x_disc = test_data["x_disc"]

        x, rho, u, p, e = sample_exact_solution(solver, left, right, x_disc, t_end)

        row = test_num - 1

        # Density
        axes[row, 0].plot(x, rho, "b-", linewidth=1.5)
        axes[row, 0].set_ylabel(f"Test {test_num}\nρ")
        axes[row, 0].grid(True, alpha=0.3)
        if row == 0:
            axes[row, 0].set_title("Density")

        # Velocity
        axes[row, 1].plot(x, u, "r-", linewidth=1.5)
        axes[row, 1].grid(True, alpha=0.3)
        if row == 0:
            axes[row, 1].set_title("Velocity")

        # Pressure
        axes[row, 2].plot(x, p, "g-", linewidth=1.5)
        axes[row, 2].grid(True, alpha=0.3)
        if row == 0:
            axes[row, 2].set_title("Pressure")

        # Internal Energy
        axes[row, 3].plot(x, e, "m-", linewidth=1.5)
        axes[row, 3].grid(True, alpha=0.3)
        if row == 0:
            axes[row, 3].set_title("Internal Energy")

        # Add test name annotation
        axes[row, 3].text(
            1.02, 0.5, test_data["name"],
            transform=axes[row, 3].transAxes,
            rotation=-90,
            va="center",
            fontsize=10,
        )

    # Set x-labels only on bottom row
    for col in range(4):
        axes[4, col].set_xlabel("x")

    plt.tight_layout()

    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path / "toro_all_tests.png", dpi=150, bbox_inches="tight")
        print(f"Saved: toro_all_tests.png")

    return fig


def main():
    """Main entry point."""
    # Output directory
    output_dir = Path(__file__).parent.parent.parent / "output" / "toro_tests"

    # Plot all tests
    figures = plot_all_toro_tests(output_dir)

    # Create comparison figure
    comparison_fig = create_comparison_figure(output_dir)

    print(f"\nPlots saved to: {output_dir}")

    # Show plots
    plt.show()


if __name__ == "__main__":
    main()
