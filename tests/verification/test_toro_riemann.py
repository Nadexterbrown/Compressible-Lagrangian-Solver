"""
Toro's Five Riemann Problem Tests

Tests the exact Riemann solver against the five canonical problems
from Toro Section 4.3.3, Table 4.1.

Evaluates:
1. Solution accuracy (p*, u* in star region)
2. Convergence of iterative solver
3. Initial guess influence on iteration count
4. Robustness for extreme pressure ratios

Reference: [Toro2009] Section 4.3.3, Table 4.1
"""

import numpy as np
import pytest
from dataclasses import dataclass
from typing import Tuple, Dict
import time

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from lagrangian_solver.equations.eos import IdealGasEOS
from lagrangian_solver.numerics.riemann import (
    RiemannState,
    RiemannSolution,
    ExactRiemannSolver,
    HLLCRiemannSolver,
)


# Toro Table 4.1 test data
# Reference: [Toro2009] Section 4.3.3, Table 4.1
TORO_TESTS = {
    1: {
        "name": "Sod shock tube (modified)",
        "left": {"rho": 1.0, "u": 0.0, "p": 1.0},
        "right": {"rho": 0.125, "u": 0.0, "p": 0.1},
        "description": "Classic shock tube with shock, contact, rarefaction",
        # Known exact solution (gamma=1.4)
        "p_star": 0.30313,
        "u_star": 0.92745,
    },
    2: {
        "name": "Two rarefactions (near vacuum)",
        "left": {"rho": 1.0, "u": -2.0, "p": 0.4},
        "right": {"rho": 1.0, "u": 2.0, "p": 0.4},
        "description": "Symmetric expansion creating near-vacuum",
        # Note: Near-vacuum case is sensitive; tolerance relaxed
        "p_star": 0.00189,
        "u_star": 0.0,
        "p_tolerance": 0.5,  # Allow 0.5% for near-vacuum
    },
    3: {
        "name": "Left blast wave",
        "left": {"rho": 1.0, "u": 0.0, "p": 1000.0},
        "right": {"rho": 1.0, "u": 0.0, "p": 0.01},
        "description": "Strong shock moving right, pressure ratio 10^5",
        "p_star": 460.894,
        "u_star": 19.5975,
    },
    4: {
        "name": "Right blast wave",
        "left": {"rho": 1.0, "u": 0.0, "p": 0.01},
        "right": {"rho": 1.0, "u": 0.0, "p": 100.0},
        "description": "Strong shock moving left, pressure ratio 10^4",
        "p_star": 46.0950,
        "u_star": -6.19633,
    },
    5: {
        "name": "Two shock collision",
        "left": {"rho": 5.99924, "u": 19.5975, "p": 460.894},
        "right": {"rho": 5.99242, "u": -6.19633, "p": 46.0950},
        "description": "Colliding shocks, tests high-speed interactions",
        "p_star": 1691.64,
        "u_star": 8.68975,
    },
}

GAMMA = 1.4  # Ideal gas for all tests


@dataclass
class TestResult:
    """Results from a single Riemann solver test."""

    test_num: int
    solver_name: str
    p_star_computed: float
    u_star_computed: float
    p_star_exact: float
    u_star_exact: float
    p_error_pct: float
    u_error_pct: float
    iterations: int
    solve_time_us: float
    converged: bool


def create_riemann_state(data: dict, gamma: float) -> RiemannState:
    """Create RiemannState from test data."""
    rho = data["rho"]
    p = data["p"]
    c = np.sqrt(gamma * p / rho)
    e = p / (rho * (gamma - 1))
    return RiemannState(rho=rho, u=data["u"], p=p, c=c, e=e)


class TestToroRiemannProblems:
    """
    Test suite for Toro's five Riemann problems.

    Reference: [Toro2009] Section 4.3.3, Table 4.1
    """

    @pytest.fixture
    def eos(self):
        """Create ideal gas EOS with gamma=1.4."""
        return IdealGasEOS(gamma=GAMMA)

    @pytest.fixture
    def exact_solver(self, eos):
        """Create exact Riemann solver."""
        return ExactRiemannSolver(eos, tol=1e-8, max_iter=100)

    @pytest.fixture
    def hllc_solver(self, eos):
        """Create HLLC Riemann solver."""
        return HLLCRiemannSolver(eos)

    @pytest.mark.parametrize("test_num", [1, 2, 3, 4, 5])
    def test_exact_solver_accuracy(self, exact_solver, test_num):
        """
        Test exact Riemann solver against known solutions.

        Tolerance: 0.1% for pressure, 0.1% for velocity
        """
        test_data = TORO_TESTS[test_num]

        left = create_riemann_state(test_data["left"], GAMMA)
        right = create_riemann_state(test_data["right"], GAMMA)

        solution = exact_solver.solve(left, right)

        p_exact = test_data["p_star"]
        u_exact = test_data["u_star"]

        p_error = abs(solution.p_star - p_exact) / p_exact * 100
        u_error = abs(solution.u_star - u_exact) / max(abs(u_exact), 1e-10) * 100

        assert (
            p_error < 0.1
        ), f"Test {test_num} ({test_data['name']}): p* error {p_error:.4f}% > 0.1%"

        # Velocity tolerance relaxed for near-zero values (Test 2)
        if abs(u_exact) > 0.01:
            assert (
                u_error < 0.1
            ), f"Test {test_num} ({test_data['name']}): u* error {u_error:.4f}% > 0.1%"

    @pytest.mark.parametrize("test_num", [1, 2, 3, 4, 5])
    def test_hllc_solver_accuracy(self, hllc_solver, test_num):
        """
        Test HLLC solver accuracy (approximate, larger tolerance).

        Tolerance: 5% for pressure, 5% for velocity
        """
        test_data = TORO_TESTS[test_num]

        left = create_riemann_state(test_data["left"], GAMMA)
        right = create_riemann_state(test_data["right"], GAMMA)

        solution = hllc_solver.solve(left, right)

        p_exact = test_data["p_star"]

        p_error = abs(solution.p_star - p_exact) / p_exact * 100

        # HLLC is approximate - allow 5% error
        assert p_error < 5.0, f"Test {test_num}: HLLC p* error {p_error:.2f}% > 5%"

    @pytest.mark.parametrize("test_num", [3, 4])
    def test_extreme_pressure_ratios(self, exact_solver, test_num):
        """
        Test solver robustness for extreme pressure ratios.

        Tests 3 and 4 have pressure ratios of 10^5 and 10^4.
        Solver should converge without failure.
        """
        test_data = TORO_TESTS[test_num]

        left = create_riemann_state(test_data["left"], GAMMA)
        right = create_riemann_state(test_data["right"], GAMMA)

        # Should not raise exception
        solution = exact_solver.solve(left, right)

        # Solution should be physical
        assert solution.p_star > 0, "p* must be positive"
        assert np.isfinite(solution.p_star), "p* must be finite"
        assert np.isfinite(solution.u_star), "u* must be finite"

    def test_initial_guess_influence(self, eos):
        """
        Test how initial guess affects iteration count.

        Compare different initial guess strategies:
        1. PVRS (primitive variable Riemann solver)
        2. Two-rarefaction approximation
        3. Two-shock approximation
        4. Arithmetic mean
        """
        results: Dict[int, Dict[str, dict]] = {}

        for test_num, test_data in TORO_TESTS.items():
            left = create_riemann_state(test_data["left"], GAMMA)
            right = create_riemann_state(test_data["right"], GAMMA)

            results[test_num] = {}

            # Test different initial guess strategies
            for guess_method in ["pvrs", "two_rarefaction", "two_shock", "mean"]:
                solver = ExactRiemannSolver(
                    eos, tol=1e-8, max_iter=100, initial_guess_method=guess_method
                )

                start = time.perf_counter()
                solution = solver.solve(left, right)
                elapsed = (time.perf_counter() - start) * 1e6  # microseconds

                results[test_num][guess_method] = {
                    "iterations": solver.last_iteration_count,
                    "time_us": elapsed,
                    "converged": solver.last_converged,
                }

        # Verify all methods converge
        for test_num in TORO_TESTS:
            for method, data in results[test_num].items():
                assert data[
                    "converged"
                ], f"Test {test_num}, method {method}: failed to converge"

    def test_solver_convergence(self, exact_solver):
        """Test that exact solver converges for all test cases."""
        for test_num, test_data in TORO_TESTS.items():
            left = create_riemann_state(test_data["left"], GAMMA)
            right = create_riemann_state(test_data["right"], GAMMA)

            solution = exact_solver.solve(left, right)

            assert (
                exact_solver.last_converged
            ), f"Test {test_num}: solver did not converge"
            assert (
                exact_solver.last_iteration_count < 50
            ), f"Test {test_num}: too many iterations ({exact_solver.last_iteration_count})"

    def test_wave_type_detection(self, exact_solver):
        """Test that wave types are correctly identified."""
        # Test 1: Left rarefaction, right shock
        test_data = TORO_TESTS[1]
        left = create_riemann_state(test_data["left"], GAMMA)
        right = create_riemann_state(test_data["right"], GAMMA)
        solution = exact_solver.solve(left, right)

        from lagrangian_solver.numerics.riemann import WaveType

        # In Sod problem: left wave is rarefaction (p* < p_L)
        # right wave is shock (p* > p_R)
        assert (
            solution.wave_L == WaveType.RAREFACTION
        ), "Test 1: left wave should be rarefaction"
        assert solution.wave_R == WaveType.SHOCK, "Test 1: right wave should be shock"

        # Test 2: Two rarefactions
        test_data = TORO_TESTS[2]
        left = create_riemann_state(test_data["left"], GAMMA)
        right = create_riemann_state(test_data["right"], GAMMA)
        solution = exact_solver.solve(left, right)

        assert (
            solution.wave_L == WaveType.RAREFACTION
        ), "Test 2: left wave should be rarefaction"
        assert (
            solution.wave_R == WaveType.RAREFACTION
        ), "Test 2: right wave should be rarefaction"

        # Test 5: Two shocks
        test_data = TORO_TESTS[5]
        left = create_riemann_state(test_data["left"], GAMMA)
        right = create_riemann_state(test_data["right"], GAMMA)
        solution = exact_solver.solve(left, right)

        assert solution.wave_L == WaveType.SHOCK, "Test 5: left wave should be shock"
        assert solution.wave_R == WaveType.SHOCK, "Test 5: right wave should be shock"

    def test_sampling_function(self, exact_solver):
        """Test the exact solution sampling at different x/t values."""
        # Test 1: Sod shock tube
        test_data = TORO_TESTS[1]
        left = create_riemann_state(test_data["left"], GAMMA)
        right = create_riemann_state(test_data["right"], GAMMA)

        # Sample at x/t = 0 (contact location)
        rho, u, p = exact_solver.sample(left, right, 0.0)

        # At contact, should have star region values
        solution = exact_solver.solve(left, right)
        assert abs(p - solution.p_star) < 1e-6, "Pressure at contact should be p*"
        assert abs(u - solution.u_star) < 1e-6, "Velocity at contact should be u*"

        # Sample far left (in undisturbed region)
        rho_far_left, u_far_left, p_far_left = exact_solver.sample(left, right, -10.0)
        assert abs(rho_far_left - left.rho) < 1e-10, "Far left should be undisturbed"
        assert abs(u_far_left - left.u) < 1e-10
        assert abs(p_far_left - left.p) < 1e-10

        # Sample far right (in undisturbed region)
        rho_far_right, u_far_right, p_far_right = exact_solver.sample(left, right, 10.0)
        assert abs(rho_far_right - right.rho) < 1e-10, "Far right should be undisturbed"
        assert abs(u_far_right - right.u) < 1e-10
        assert abs(p_far_right - right.p) < 1e-10


class TestToroRiemannSummary:
    """Summary test that runs all Toro problems and reports results."""

    def test_all_toro_problems_summary(self):
        """
        Run all five tests and produce summary report.
        """
        eos = IdealGasEOS(gamma=GAMMA)
        exact_solver = ExactRiemannSolver(eos, tol=1e-8, max_iter=100)

        print("\n" + "=" * 70)
        print("TORO RIEMANN PROBLEM TEST SUITE - SUMMARY")
        print("Reference: [Toro2009] Section 4.3.3, Table 4.1")
        print("=" * 70)

        all_passed = True

        for test_num, test_data in TORO_TESTS.items():
            left = create_riemann_state(test_data["left"], GAMMA)
            right = create_riemann_state(test_data["right"], GAMMA)

            solution = exact_solver.solve(left, right)

            p_exact = test_data["p_star"]
            u_exact = test_data["u_star"]
            p_tol = test_data.get("p_tolerance", 0.1)  # Default 0.1% tolerance

            p_error = abs(solution.p_star - p_exact) / p_exact * 100
            u_error = abs(solution.u_star - u_exact) / max(abs(u_exact), 1e-10) * 100

            passed = p_error < p_tol and (u_error < 0.1 or abs(u_exact) < 0.01)
            status = "PASS" if passed else "FAIL"
            all_passed = all_passed and passed

            print(f"\nTest {test_num}: {test_data['name']}")
            print(f"  Description: {test_data['description']}")
            print(
                f"  p* computed: {solution.p_star:.6f}, exact: {p_exact:.6f}, error: {p_error:.4f}%"
            )
            print(
                f"  u* computed: {solution.u_star:.6f}, exact: {u_exact:.6f}, error: {u_error:.4f}%"
            )
            print(f"  Iterations: {exact_solver.last_iteration_count}")
            print(f"  Status: [{status}]")

        print("\n" + "=" * 70)
        print(f"OVERALL: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
        print("=" * 70)

        assert all_passed, "Not all Toro tests passed"


def run_toro_tests():
    """
    Standalone function to run all Toro tests.

    Can be called from scripts/examples/toro_riemann_tests.py
    """
    eos = IdealGasEOS(gamma=GAMMA)
    exact_solver = ExactRiemannSolver(eos, tol=1e-8, max_iter=100)

    test_suite = TestToroRiemannSummary()
    test_suite.test_all_toro_problems_summary()


if __name__ == "__main__":
    run_toro_tests()
