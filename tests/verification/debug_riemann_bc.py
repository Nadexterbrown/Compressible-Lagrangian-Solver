"""
Debug script for RiemannGhostPistonBC.

Diagnoses the interface state computation to identify bugs.
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from sdtoolbox.postshock import PostShock_fr
import cantera as ct

from lagrangian_solver.core.grid import LagrangianGrid, GridConfig
from lagrangian_solver.core.state import create_uniform_state
from lagrangian_solver.equations.eos import CanteraEOS
from lagrangian_solver.numerics.boundary_riemann import BoundaryRiemannSolver

T_STP = 298.15
P_STP = 101325


def compute_shock_state(M_s: float):
    """Compute exact post-shock state from SDToolbox."""
    air_composition = {"N2": 0.79, "O2": 0.21}
    mech = "gri30.yaml"
    gas1 = ct.Solution(mech)
    gas1.TPX = T_STP, P_STP, air_composition
    rho1, c1 = gas1.density, gas1.sound_speed
    D = M_s * c1
    gas2 = PostShock_fr(D, P_STP, T_STP, air_composition, mech)
    u_p = D * (1.0 - rho1 / gas2.density)
    return {
        "rho1": rho1, "rho2": gas2.density,
        "p1": P_STP, "p2": gas2.P,
        "D": D, "u_p": u_p, "c1": c1,
        "gamma": gas1.cp / gas1.cv,
    }


def test_boundary_riemann_solver():
    """Test the boundary Riemann solver directly."""
    print("=" * 70)
    print("DEBUGGING BOUNDARY RIEMANN SOLVER")
    print("=" * 70)

    # Create EOS
    eos = CanteraEOS(mechanism_file="gri30.yaml")
    eos.gas.X = {"N2": 0.79, "O2": 0.21}
    eos.set_state_TP(T_STP, P_STP)
    rho_init = eos.gas.density
    c_init = eos.gas.sound_speed
    gamma = eos.gas.cp / eos.gas.cv

    print(f"\nInitial state:")
    print(f"  rho = {rho_init:.4f} kg/m^3")
    print(f"  p   = {P_STP:.0f} Pa")
    print(f"  c   = {c_init:.1f} m/s")
    print(f"  gamma = {gamma:.4f}")

    boundary_solver = BoundaryRiemannSolver(eos, tol=1e-8)

    for M_s in [2, 3, 4]:
        shock = compute_shock_state(M_s)
        u_piston = shock['u_p']
        p_exact = shock['p2']
        rho_exact = shock['rho2']

        print(f"\n{'='*60}")
        print(f"MACH {M_s} SHOCK")
        print(f"{'='*60}")
        print(f"Piston velocity:   {u_piston:.1f} m/s")
        print(f"Expected pressure: {p_exact/1000:.1f} kPa")
        print(f"Expected density:  {rho_exact:.4f} kg/m^3")

        # Test 1: Interior gas at rest (correct initial state)
        print(f"\n--- Test 1: Interior gas at rest (u_int = 0) ---")
        u_int = 0.0  # Gas initially at rest
        result = boundary_solver.solve_left_boundary(
            rho_int=rho_init,
            u_int=u_int,
            p_int=P_STP,
            u_bc=u_piston,
        )
        p_error = 100 * (result.p - p_exact) / p_exact
        rho_error = 100 * (result.rho - rho_exact) / rho_exact
        print(f"  delta_u = u_int - u_bc = {u_int:.1f} - {u_piston:.1f} = {u_int - u_piston:.1f} m/s")
        print(f"  Computed p* = {result.p/1000:.1f} kPa ({p_error:+.1f}% error)")
        print(f"  Computed rho* = {result.rho:.4f} kg/m^3 ({rho_error:+.1f}% error)")

        # Test 2: Interior velocity = average of piston and face 1 (BUG)
        print(f"\n--- Test 2: Buggy u_int = 0.5*(u_piston + 0) ---")
        u_int_bug = 0.5 * (u_piston + 0.0)  # The bug: includes piston velocity
        result_bug = boundary_solver.solve_left_boundary(
            rho_int=rho_init,
            u_int=u_int_bug,
            p_int=P_STP,
            u_bc=u_piston,
        )
        p_error_bug = 100 * (result_bug.p - p_exact) / p_exact
        rho_error_bug = 100 * (result_bug.rho - rho_exact) / rho_exact
        print(f"  delta_u = u_int - u_bc = {u_int_bug:.1f} - {u_piston:.1f} = {u_int_bug - u_piston:.1f} m/s")
        print(f"  Computed p* = {result_bug.p/1000:.1f} kPa ({p_error_bug:+.1f}% error)")
        print(f"  Computed rho* = {result_bug.rho:.4f} kg/m^3 ({rho_error_bug:+.1f}% error)")

        # Test 3: Check wave function direction
        print(f"\n--- Test 3: Wave function analysis ---")

        # Correct delta_u for left boundary: u_int - u_bc
        # If u_int = 0 and u_bc = u_piston > 0: delta_u < 0
        # This is interpreted as expansion in the current code!

        delta_u_correct = 0.0 - u_piston
        print(f"  delta_u (correct) = {delta_u_correct:.1f} m/s")
        print(f"  Current code interprets delta_u < 0 as EXPANSION")
        print(f"  But piston moving into gas is COMPRESSION!")


def test_sign_convention():
    """Test the sign convention in detail."""
    print("\n" + "=" * 70)
    print("SIGN CONVENTION ANALYSIS")
    print("=" * 70)

    print("""
For a LEFT boundary piston problem:

    [PISTON] ---> |====== GAS ======|
    u = u_p       |  u = 0 (at rest) |

Wave structure:
    - Shock wave travels RIGHT into the gas
    - Contact (piston face) moves at u = u_p

Riemann problem viewpoint:
    - Left state: piston (boundary condition)
    - Right state: quiescent gas (interior)

For a RIGHT-traveling shock connecting interior (R) to contact (C):
    u_C = u_R + f_R(p*)   (velocity increases toward contact)

Since u_C = u_piston and u_R = 0:
    u_piston = 0 + f_R(p*)
    f_R(p*) = u_piston > 0

This means p* > p (compression), which is correct!

Current code uses:
    delta_u = u_int - u_bc = 0 - u_piston < 0

This treats it as expansion (wrong sign).

Fix: For LEFT boundary, use delta_u = u_bc - u_int
     (opposite of current implementation)
""")


if __name__ == "__main__":
    test_boundary_riemann_solver()
    test_sign_convention()
