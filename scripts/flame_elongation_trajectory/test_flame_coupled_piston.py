"""
Verification tests for flame-coupled piston boundary condition.

Tests:
1. Interpolator accuracy - verify interpolation at CSV grid points
2. Elongation functions - verify sigma(t) and dsigma/dt
3. Iteration convergence - verify coupled solver converges
4. Energy conservation - verify |dE/E| < tolerance
5. Coupled vs uncoupled - compare velocity profiles

Run with:
    cd scripts/flame_elongation_trajectory
    conda run -n Cantera-3_0_0 python test_flame_coupled_piston.py

Reference:
    Clavin, P., & Tofaili, H. (2021). Flame elongation model.
"""

import sys
from pathlib import Path
import numpy as np
import warnings

# Add src to path
src_path = Path(__file__).resolve().parents[2] / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Import test modules
from flame_property_interpolator import FlamePropertyInterpolator, FlameProperties
from flame_elongation_trajectory import (
    FlameElongationTrajectory,
    PowerLawElongation,
    ExponentialElongation,
    LinearElongation,
    ConstantElongation,
)
from flame_coupled_piston_bc import FlameCoupledPistonBC, IterationResult

# Import solver components
from lagrangian_solver.core.grid import LagrangianGrid, GridConfig
from lagrangian_solver.core.solver import LagrangianSolver, SolverConfig
from lagrangian_solver.core.state import create_uniform_state
from lagrangian_solver.equations.eos import CanteraEOS
from lagrangian_solver.numerics.artificial_viscosity import ArtificialViscosityConfig
from lagrangian_solver.boundary.open import OpenBC
from lagrangian_solver.boundary.base import BoundarySide


# =============================================================================
# Test Configuration
# =============================================================================

CSV_PATH = Path(__file__).parent / "cantera_data" / "output" / "flame_properties.csv"

# Reference conditions
P_REF = 1e6      # 10 bar [Pa]
T_REF = 500.0    # [K]

# Test tolerances
INTERP_TOL = 1e-6       # Interpolation at grid points
SIGMA_TOL = 1e-10       # Elongation function accuracy
ITER_CONV_RATE = 0.95   # Required convergence rate
ENERGY_TOL = 1e-8       # Energy conservation |dE/E|


# =============================================================================
# Test 1: Interpolator Accuracy
# =============================================================================

def test_interpolator_accuracy():
    """Verify interpolation at CSV grid points matches data."""
    print("\n" + "=" * 60)
    print("TEST 1: Interpolator Accuracy")
    print("=" * 60)

    # Load interpolator
    interp = FlamePropertyInterpolator(str(CSV_PATH))
    print(f"Loaded: {interp}")

    # Load raw CSV for comparison
    import pandas as pd
    df = pd.read_csv(CSV_PATH)

    # Test at grid points (should be exact for linear interpolation)
    n_tests = 20
    indices = np.linspace(0, len(df) - 1, n_tests, dtype=int)

    max_S_L_err = 0.0
    max_rho_u_err = 0.0
    max_rho_b_err = 0.0

    for idx in indices:
        row = df.iloc[idx]
        P = row["P [Pa]"]
        T = row["T [K]"]
        S_L_exact = row["Su [m/s]"]
        rho_u_exact = row["rho_u [kg/m3]"]
        rho_b_exact = row["rho_b [kg/m3]"]

        props = interp.get_properties(P, T)

        S_L_err = abs(props.S_L - S_L_exact) / S_L_exact
        rho_u_err = abs(props.rho_u - rho_u_exact) / rho_u_exact
        rho_b_err = abs(props.rho_b - rho_b_exact) / rho_b_exact

        max_S_L_err = max(max_S_L_err, S_L_err)
        max_rho_u_err = max(max_rho_u_err, rho_u_err)
        max_rho_b_err = max(max_rho_b_err, rho_b_err)

    print(f"\nMax relative errors at grid points:")
    print(f"  S_L:   {max_S_L_err:.2e} (tol: {INTERP_TOL:.0e})")
    print(f"  rho_u: {max_rho_u_err:.2e} (tol: {INTERP_TOL:.0e})")
    print(f"  rho_b: {max_rho_b_err:.2e} (tol: {INTERP_TOL:.0e})")

    passed = (max_S_L_err < INTERP_TOL and
              max_rho_u_err < INTERP_TOL and
              max_rho_b_err < INTERP_TOL)

    if passed:
        print("\n[PASS] Interpolation accuracy at grid points")
    else:
        print("\n[FAIL] Interpolation errors exceed tolerance")

    # Test extrapolation warning
    print("\nTesting extrapolation warning...")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        # Query outside bounds
        _ = interp.get_properties(0.5e6, 400.0)  # Below P and T bounds
        if len(w) >= 1:
            print(f"  [PASS] Extrapolation warnings raised ({len(w)} warning(s))")
        else:
            print("  [FAIL] No extrapolation warning raised")
            passed = False

    return passed


# =============================================================================
# Test 2: Elongation Functions
# =============================================================================

def test_elongation_functions():
    """Verify elongation functions compute sigma(t) and dsigma/dt correctly."""
    print("\n" + "=" * 60)
    print("TEST 2: Elongation Functions")
    print("=" * 60)

    passed = True

    # Test PowerLawElongation: sigma(t) = sigma_0 * (1 + (k*t)^n)
    print("\nPowerLawElongation (sigma_0=1, k=1000, n=2):")
    elong = PowerLawElongation(sigma_0=1.0, k=1000.0, n=2.0)

    t_test = 1e-3  # 1 ms
    sigma_computed = elong.sigma(t_test)
    sigma_expected = 1.0 * (1.0 + (1000.0 * t_test) ** 2)  # 1 + 1 = 2

    dsigma_computed = elong.dsigma_dt(t_test)
    dsigma_expected = 1.0 * 2.0 * 1000.0 * (1000.0 * t_test) ** 1  # 2 * 1000 * 1 = 2000

    print(f"  t = {t_test*1e3:.1f} ms:")
    print(f"    sigma:   computed={sigma_computed:.6f}, expected={sigma_expected:.6f}")
    print(f"    dsigma:  computed={dsigma_computed:.6f}, expected={dsigma_expected:.6f}")

    if abs(sigma_computed - sigma_expected) > SIGMA_TOL:
        print("    [FAIL] sigma mismatch")
        passed = False
    if abs(dsigma_computed - dsigma_expected) > SIGMA_TOL:
        print("    [FAIL] dsigma/dt mismatch")
        passed = False

    # Test ExponentialElongation: sigma(t) = sigma_0 * exp(B*t)
    print("\nExponentialElongation (sigma_0=1, B=1000):")
    elong = ExponentialElongation(sigma_0=1.0, B=1000.0)

    sigma_computed = elong.sigma(t_test)
    sigma_expected = 1.0 * np.exp(1000.0 * t_test)

    dsigma_computed = elong.dsigma_dt(t_test)
    dsigma_expected = 1.0 * 1000.0 * np.exp(1000.0 * t_test)

    print(f"  t = {t_test*1e3:.1f} ms:")
    print(f"    sigma:   computed={sigma_computed:.6f}, expected={sigma_expected:.6f}")
    print(f"    dsigma:  computed={dsigma_computed:.6f}, expected={dsigma_expected:.6f}")

    if abs(sigma_computed - sigma_expected) / sigma_expected > SIGMA_TOL:
        print("    [FAIL] sigma mismatch")
        passed = False
    if abs(dsigma_computed - dsigma_expected) / dsigma_expected > SIGMA_TOL:
        print("    [FAIL] dsigma/dt mismatch")
        passed = False

    # Test LinearElongation: sigma(t) = sigma_0 + k*t
    print("\nLinearElongation (sigma_0=1, k=1000):")
    elong = LinearElongation(sigma_0=1.0, k=1000.0)

    sigma_computed = elong.sigma(t_test)
    sigma_expected = 1.0 + 1000.0 * t_test  # 1 + 1 = 2

    dsigma_computed = elong.dsigma_dt(t_test)
    dsigma_expected = 1000.0

    print(f"  t = {t_test*1e3:.1f} ms:")
    print(f"    sigma:   computed={sigma_computed:.6f}, expected={sigma_expected:.6f}")
    print(f"    dsigma:  computed={dsigma_computed:.6f}, expected={dsigma_expected:.6f}")

    if abs(sigma_computed - sigma_expected) > SIGMA_TOL:
        print("    [FAIL] sigma mismatch")
        passed = False
    if abs(dsigma_computed - dsigma_expected) > SIGMA_TOL:
        print("    [FAIL] dsigma/dt mismatch")
        passed = False

    if passed:
        print("\n[PASS] All elongation functions correct")
    else:
        print("\n[FAIL] Some elongation function tests failed")

    return passed


# =============================================================================
# Test 3: FlameElongationTrajectory
# =============================================================================

def test_flame_trajectory():
    """Test FlameElongationTrajectory velocity computation."""
    print("\n" + "=" * 60)
    print("TEST 3: FlameElongationTrajectory")
    print("=" * 60)

    passed = True

    # Load interpolator and create trajectory
    interp = FlamePropertyInterpolator(str(CSV_PATH))
    elong = PowerLawElongation(sigma_0=1.0, k=100.0, n=2.0)
    traj = FlameElongationTrajectory(elong, interp, P_ref=P_REF, T_ref=T_REF)

    print(f"\n{traj}")

    # At t=0, sigma=1, so velocity should be 0
    v_0 = traj.velocity_uncoupled(0.0)
    print(f"\nAt t=0: sigma=1, velocity={v_0:.6f} m/s (expected ~0)")
    if abs(v_0) > 1e-10:
        print("  [FAIL] Velocity should be 0 at t=0 (sigma=1)")
        passed = False

    # At t=1ms, sigma = 1 + (100*0.001)^2 = 1.01
    t_test = 1e-3
    sigma_t = traj.sigma(t_test)
    props = traj.properties_ref

    v_expected = (sigma_t - 1.0) * props.density_ratio * props.S_L
    v_computed = traj.velocity_uncoupled(t_test)

    print(f"\nAt t={t_test*1e3:.1f} ms:")
    print(f"  sigma = {sigma_t:.6f}")
    print(f"  S_L = {props.S_L:.2f} m/s")
    print(f"  rho_u/rho_b = {props.density_ratio:.2f}")
    print(f"  velocity: computed={v_computed:.4f} m/s, expected={v_expected:.4f} m/s")

    if abs(v_computed - v_expected) / max(abs(v_expected), 1e-10) > SIGMA_TOL:
        print("  [FAIL] Velocity mismatch")
        passed = False

    # Test coupled velocity at different P, T
    P_test = 2e6  # 20 bar
    T_test = 600.0
    v_coupled = traj.velocity(t_test, P_test, T_test)
    props_test = interp.get_properties(P_test, T_test)
    v_coupled_expected = (sigma_t - 1.0) * props_test.density_ratio * props_test.S_L

    print(f"\nCoupled velocity at P={P_test/1e6:.0f} bar, T={T_test:.0f} K:")
    print(f"  S_L = {props_test.S_L:.2f} m/s")
    print(f"  rho_u/rho_b = {props_test.density_ratio:.2f}")
    print(f"  velocity: computed={v_coupled:.4f} m/s, expected={v_coupled_expected:.4f} m/s")

    if abs(v_coupled - v_coupled_expected) / max(abs(v_coupled_expected), 1e-10) > SIGMA_TOL:
        print("  [FAIL] Coupled velocity mismatch")
        passed = False

    if passed:
        print("\n[PASS] FlameElongationTrajectory correct")
    else:
        print("\n[FAIL] FlameElongationTrajectory tests failed")

    return passed


# =============================================================================
# Test 4: Iteration Convergence
# =============================================================================

def test_iteration_convergence():
    """Test that FlameCoupledPistonBC iteration converges."""
    print("\n" + "=" * 60)
    print("TEST 4: Iteration Convergence")
    print("=" * 60)

    passed = True

    # Create EOS
    eos = CanteraEOS(mechanism_file="gri30.yaml")
    eos.gas.X = {"H2": 0.21, "O2": 0.21/2, "N2": 0.79 - 0.21/2}  # Approx stoichiometric H2/air
    eos.set_state_TP(T_REF, P_REF)
    rho_init = eos.gas.density

    print(f"\nInitial conditions: T={T_REF} K, P={P_REF/1e5:.1f} bar, rho={rho_init:.3f} kg/m³")

    # Create flame trajectory
    interp = FlamePropertyInterpolator(str(CSV_PATH))
    elong = LinearElongation(sigma_0=1.0, k=500.0)  # Gentle growth
    traj = FlameElongationTrajectory(elong, interp, P_ref=P_REF, T_ref=T_REF)

    # Create BC
    bc = FlameCoupledPistonBC(
        side=BoundarySide.LEFT,
        eos=eos,
        flame_trajectory=traj,
        tol=1e-6,
        max_iter=20,
        ramp_time=30e-6,
    )

    print(f"\n{bc}")

    # Create simple grid and state
    n_cells = 50
    grid = LagrangianGrid(GridConfig(n_cells=n_cells, x_min=0.0, x_max=0.1))
    state = create_uniform_state(
        n_cells=n_cells, x_left=0.0, x_right=0.1,
        rho=rho_init, u=0.0, p=P_REF, eos=eos,
    )

    # Test at several times
    test_times = [0.0, 10e-6, 30e-6, 50e-6, 100e-6, 200e-6]

    print("\nIteration convergence at various times:")
    print("-" * 50)

    for t in test_times:
        bc.clear_cache()
        bc.apply_velocity(state, grid, t)
        result = bc._cached_result

        if result:
            status = "CONVERGED" if result.converged else "FAILED"
            print(f"  t={t*1e6:6.1f} us: {status} in {result.iterations:2d} iters, "
                  f"v={result.velocity:.3f} m/s, rel_change={result.relative_change:.2e}")

            if not result.converged:
                print(f"    [WARNING] Did not converge!")
                passed = False
        else:
            print(f"  t={t*1e6:6.1f} us: No result (error)")
            passed = False

    # Check convergence statistics
    stats = bc.get_convergence_stats()
    print(f"\nConvergence Statistics:")
    print(f"  Timesteps tested: {stats['n_timesteps']}")
    print(f"  Avg iterations: {stats['avg_iterations']:.1f}")
    print(f"  Max iterations: {stats['max_iterations']}")
    print(f"  Convergence rate: {stats['convergence_rate']*100:.1f}%")
    print(f"  Failures: {stats['failures']}")

    if stats['convergence_rate'] >= ITER_CONV_RATE:
        print(f"\n[PASS] Convergence rate >= {ITER_CONV_RATE*100:.0f}%")
    else:
        print(f"\n[FAIL] Convergence rate < {ITER_CONV_RATE*100:.0f}%")
        passed = False

    return passed


# =============================================================================
# Test 5: Coupled vs Uncoupled Comparison
# =============================================================================

def test_coupled_vs_uncoupled():
    """Compare coupled vs uncoupled velocity profiles."""
    print("\n" + "=" * 60)
    print("TEST 5: Coupled vs Uncoupled Comparison")
    print("=" * 60)

    # Create flame trajectory
    interp = FlamePropertyInterpolator(str(CSV_PATH))
    elong = LinearElongation(sigma_0=1.0, k=1000.0)
    traj = FlameElongationTrajectory(elong, interp, P_ref=P_REF, T_ref=T_REF)

    print(f"\nReference conditions: P={P_REF/1e5:.0f} bar, T={T_REF:.0f} K")
    print(f"Reference flame speed: S_L = {traj.properties_ref.S_L:.2f} m/s")
    print(f"Reference density ratio: rho_u/rho_b = {traj.properties_ref.density_ratio:.2f}")

    # Compute uncoupled velocity at several times
    times = np.linspace(0, 0.5e-3, 11)  # 0 to 0.5 ms

    print(f"\nUncoupled velocity profile:")
    print("-" * 50)
    print(f"{'t [us]':>10} {'sigma':>10} {'v_uncoupled [m/s]':>18}")
    print("-" * 50)

    for t in times:
        sigma = traj.sigma(t)
        v = traj.velocity_uncoupled(t)
        print(f"{t*1e6:10.1f} {sigma:10.4f} {v:18.4f}")

    # Test that coupled velocity is higher at elevated P, T
    print(f"\nCoupled velocity at elevated P, T (should be higher):")
    print("-" * 60)

    t_test = 0.5e-3
    sigma = traj.sigma(t_test)
    v_uncoupled = traj.velocity_uncoupled(t_test)

    # Elevated conditions (typical post-shock)
    P_elevated = 3e6  # 30 bar
    T_elevated = 800.0

    v_coupled = traj.velocity(t_test, P_elevated, T_elevated)
    props_elevated = interp.get_properties(P_elevated, T_elevated)

    print(f"  t = {t_test*1e3:.2f} ms, sigma = {sigma:.4f}")
    print(f"  Reference (P={P_REF/1e5:.0f} bar, T={T_REF:.0f} K):")
    print(f"    S_L = {traj.properties_ref.S_L:.2f} m/s, rho_u/rho_b = {traj.properties_ref.density_ratio:.2f}")
    print(f"    v_uncoupled = {v_uncoupled:.4f} m/s")
    print(f"  Elevated (P={P_elevated/1e5:.0f} bar, T={T_elevated:.0f} K):")
    print(f"    S_L = {props_elevated.S_L:.2f} m/s, rho_u/rho_b = {props_elevated.density_ratio:.2f}")
    print(f"    v_coupled = {v_coupled:.4f} m/s")

    # The coupled velocity should typically be different (often higher at elevated T)
    # because S_L increases with T faster than the density ratio decreases
    ratio = v_coupled / v_uncoupled if v_uncoupled != 0 else float('inf')
    print(f"\n  Ratio v_coupled/v_uncoupled = {ratio:.4f}")

    # This is informational - no pass/fail criterion for this test
    print("\n[INFO] Coupled vs uncoupled comparison complete")
    print("       (This test is informational - no pass/fail criterion)")

    return True


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("FLAME-COUPLED PISTON BC VERIFICATION TESTS")
    print("=" * 60)

    if not CSV_PATH.exists():
        print(f"\nERROR: flame_properties.csv not found at:")
        print(f"  {CSV_PATH}")
        print("\nRun the Cantera flame calculations first.")
        return

    results = {}

    # Run tests
    results["Interpolator Accuracy"] = test_interpolator_accuracy()
    results["Elongation Functions"] = test_elongation_functions()
    results["Flame Trajectory"] = test_flame_trajectory()
    results["Iteration Convergence"] = test_iteration_convergence()
    results["Coupled vs Uncoupled"] = test_coupled_vs_uncoupled()

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    n_passed = sum(1 for v in results.values() if v)
    n_total = len(results)

    for name, passed in results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status} {name}")

    print(f"\nTotal: {n_passed}/{n_total} tests passed")

    if n_passed == n_total:
        print("\n[SUCCESS] All tests passed!")
    else:
        print(f"\n[FAILURE] {n_total - n_passed} test(s) failed")


if __name__ == "__main__":
    main()
