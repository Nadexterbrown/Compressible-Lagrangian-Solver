"""
Debug script to verify interface state calculation for solid vs porous modes.
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from lagrangian_solver.equations.eos import CanteraEOS
from lagrangian_solver.boundary.base import BoundarySide, ThermalBCType
from lagrangian_solver.core.state import create_uniform_state
from lagrangian_solver.core.grid import LagrangianGrid, GridConfig

from pele_data_loader import PeleDataLoader, PeleTrajectoryInterpolator
from data_driven_piston_bc import DataDrivenPistonBC, PistonMode


def main():
    print("=" * 70)
    print("DEBUG: INTERFACE STATE CALCULATION")
    print("=" * 70)

    # Load trajectory data
    data_dir = Path(__file__).parent / "pele_data" / "truncated_raw_data"
    loader = PeleDataLoader(data_dir)
    data = loader.load()
    trajectory = PeleTrajectoryInterpolator(data, extrapolate=False)

    # Create EOS
    mech_path = Path(__file__).parent.parent / "flame_elongation_trajectory" / "cantera_data" / "Li-Dryer-H2-mechanism.yaml"
    eos = CanteraEOS(str(mech_path))
    eos.set_mixture('H2', 'O2:1', 1.0)
    eos.set_state_TP(503, 10e5)

    # Create grid and initial state
    n_cells = 100
    grid_config = GridConfig(n_cells=n_cells, x_min=0.0, x_max=2.0)
    grid = LagrangianGrid(grid_config)

    eos.set_state_TP(503, 10e5)
    rho_init = eos.gas.density
    c_init = eos.gas.sound_speed

    state = create_uniform_state(
        n_cells=n_cells,
        x_left=0.0,
        x_right=2.0,
        rho=rho_init,
        u=0.0,
        p=10e5,
        eos=eos,
    )

    print(f"\nInitial state:")
    print(f"  rho = {rho_init:.4f} kg/m³")
    print(f"  p = {10e5/1e6:.2f} MPa")
    print(f"  c = {c_init:.1f} m/s")

    # Create BCs for solid and porous modes
    bc_solid = DataDrivenPistonBC(
        side=BoundarySide.LEFT, eos=eos, trajectory=trajectory, mode=PistonMode.SOLID,
        velocity_offset=0.0, velocity_min=0.0, thermal_bc=ThermalBCType.ADIABATIC,
    )

    bc_porous = DataDrivenPistonBC(
        side=BoundarySide.LEFT, eos=eos, trajectory=trajectory, mode=PistonMode.POROUS,
        velocity_offset=-119.2, velocity_min=0.0, thermal_bc=ThermalBCType.ADIABATIC,
    )

    # Test at several times
    test_times = [0.0, 100e-6, 500e-6, 1000e-6]

    print("\n" + "=" * 70)
    print("COMPARING SOLID VS POROUS INTERFACE STATES")
    print("=" * 70)

    for t in test_times:
        print(f"\n--- Time: t = {t*1e6:.1f} µs ---")

        # Get velocities
        v_piston_solid = bc_solid.get_piston_velocity(t)
        v_gas_solid = bc_solid.get_gas_velocity(t)
        v_piston_porous = bc_porous.get_piston_velocity(t)
        v_gas_porous = bc_porous.get_gas_velocity(t)

        print(f"  Solid:  v_piston = {v_piston_solid:.1f} m/s, v_gas = {v_gas_solid:.1f} m/s")
        print(f"  Porous: v_piston = {v_piston_porous:.1f} m/s, v_gas = {v_gas_porous:.1f} m/s")

        # Compute interface states
        state_solid = bc_solid.compute_interface_state(state, grid, t)
        state_porous = bc_porous.compute_interface_state(state, grid, t)

        print(f"\n  Interface state (p*, sigma):")
        print(f"    Solid:  p* = {state_solid.p/1e6:.4f} MPa, sigma = {state_solid.sigma/1e6:.4f} MPa")
        print(f"    Porous: p* = {state_porous.p/1e6:.4f} MPa, sigma = {state_porous.sigma/1e6:.4f} MPa")
        print(f"    Difference: Δp* = {(state_solid.p - state_porous.p)/1e6:.4f} MPa")

        # Check u* values
        print(f"\n  Interface velocity (u*):")
        print(f"    Solid:  u* = {state_solid.u:.1f} m/s")
        print(f"    Porous: u* = {state_porous.u:.1f} m/s")

    # Simulate a few steps to see if states diverge
    print("\n" + "=" * 70)
    print("RUNNING SHORT SIMULATION TO CHECK DIVERGENCE")
    print("=" * 70)

    from lagrangian_solver.core.solver import LagrangianSolver, SolverConfig
    from lagrangian_solver.boundary.open import OpenBC

    # Create two solvers
    grid_solid = LagrangianGrid(grid_config)
    grid_porous = LagrangianGrid(grid_config)

    state_solid = create_uniform_state(
        n_cells=n_cells, x_left=0.0, x_right=2.0, rho=rho_init, u=0.0, p=10e5, eos=eos,
    )
    state_porous = create_uniform_state(
        n_cells=n_cells, x_left=0.0, x_right=2.0, rho=rho_init, u=0.0, p=10e5, eos=eos,
    )

    bc_right = OpenBC(side=BoundarySide.RIGHT, eos=eos, p_external=10e5)

    solver_config = SolverConfig(cfl=0.4, av_linear=0.3, av_quad=2.0, av_enabled=True)

    solver_solid = LagrangianSolver(
        grid=grid_solid, eos=eos, bc_left=bc_solid, bc_right=bc_right, config=solver_config
    )
    solver_porous = LagrangianSolver(
        grid=grid_porous, eos=eos, bc_left=bc_porous, bc_right=bc_right, config=solver_config
    )

    solver_solid.set_initial_condition(state_solid)
    solver_porous.set_initial_condition(state_porous)

    # Run for a few steps
    t = 0.0
    dt = 1e-7
    n_steps = 100

    for step in range(n_steps):
        solver_solid.step_forward(dt)
        solver_porous.step_forward(dt)
        t += dt

    print(f"\nAfter {n_steps} steps (t = {t*1e6:.1f} µs):")

    state_s = solver_solid.state
    state_p = solver_porous.state

    print(f"\n  Piston face (cell 0):")
    print(f"    Solid:  rho = {state_s.rho[0]:.4f} kg/m³, p = {state_s.p[0]/1e6:.4f} MPa")
    print(f"    Porous: rho = {state_p.rho[0]:.4f} kg/m³, p = {state_p.p[0]/1e6:.4f} MPa")
    print(f"    Difference: Δrho = {state_s.rho[0] - state_p.rho[0]:.6f} kg/m³, "
          f"Δp = {(state_s.p[0] - state_p.p[0])/1e6:.6f} MPa")

    print(f"\n  Cell 10:")
    print(f"    Solid:  rho = {state_s.rho[10]:.4f} kg/m³, p = {state_s.p[10]/1e6:.4f} MPa")
    print(f"    Porous: rho = {state_p.rho[10]:.4f} kg/m³, p = {state_p.p[10]/1e6:.4f} MPa")

    print(f"\n  Max pressure:")
    print(f"    Solid:  p_max = {state_s.p.max()/1e6:.4f} MPa")
    print(f"    Porous: p_max = {state_p.p.max()/1e6:.4f} MPa")

    print(f"\n  Grid positions:")
    print(f"    Solid:  x_piston = {grid_solid.x[0]:.6f} m, x_max = {grid_solid.x[-1]:.6f} m")
    print(f"    Porous: x_piston = {grid_porous.x[0]:.6f} m, x_max = {grid_porous.x[-1]:.6f} m")

    if abs(state_s.p[0] - state_p.p[0]) < 1e-6:
        print("\n  WARNING: Pressures are IDENTICAL - BC is not being applied correctly!")
    else:
        print("\n  Pressures differ - BC is working!")


if __name__ == "__main__":
    main()
