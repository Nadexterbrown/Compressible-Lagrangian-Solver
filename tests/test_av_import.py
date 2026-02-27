"""Quick test to verify artificial viscosity implementation."""
import numpy as np

from lagrangian_solver.core.solver import LagrangianSolver, SolverConfig
from lagrangian_solver.numerics.artificial_viscosity import (
    ArtificialViscosity,
    ArtificialViscosityConfig,
)
from lagrangian_solver.core.grid import LagrangianGrid, GridConfig
from lagrangian_solver.core.state import create_uniform_state
from lagrangian_solver.equations.eos import IdealGasEOS


def test_av_module_import():
    """Test that AV module imports correctly."""
    av = ArtificialViscosity(ArtificialViscosityConfig(c_quad=2.0))
    assert av.enabled
    assert av.config.c_quad == 2.0
    print("PASS: AV module imports correctly")


def test_av_computation():
    """Test that AV computes viscous stress correctly."""
    # Create minimal setup
    grid_config = GridConfig(n_cells=10, x_min=0.0, x_max=1.0)
    grid = LagrangianGrid(grid_config)
    eos = IdealGasEOS(gamma=1.4)

    # Create uniform state with zero velocity - no compression
    state = create_uniform_state(
        n_cells=10, x_left=0.0, x_right=1.0, rho=1.0, u=0.0, p=1.0, eos=eos
    )
    grid.initialize_mass(state.rho)

    av = ArtificialViscosity(ArtificialViscosityConfig(c_quad=2.0))
    Q = av.compute_viscous_stress(state, grid)

    # No compression, so Q should be zero
    assert np.allclose(Q, 0.0), f"Expected Q=0 for uniform flow, got Q={Q}"
    print("PASS: AV returns zero for non-compression")

    # Now test with compression (negative velocity gradient)
    # Set velocity to decrease from left to right
    state.u[:] = np.linspace(1.0, -1.0, state.n_faces)

    Q = av.compute_viscous_stress(state, grid)

    # All cells should have Q > 0 since du/dx < 0 everywhere
    assert np.all(Q > 0), f"Expected Q>0 for compression, got Q={Q}"
    print("PASS: AV returns positive Q for compression")


def test_solver_with_av():
    """Test solver creation with artificial viscosity."""
    from lagrangian_solver.boundary.base import ReflectiveBC, BoundarySide

    grid_config = GridConfig(n_cells=100, x_min=0.0, x_max=1.0)
    grid = LagrangianGrid(grid_config)
    eos = IdealGasEOS(gamma=1.4)

    # Create solver with artificial viscosity (new interface)
    solver_config = SolverConfig(
        cfl=0.5, t_end=0.1, av_enabled=True, av_quad=2.0
    )
    bc_left = ReflectiveBC(BoundarySide.LEFT, eos)
    bc_right = ReflectiveBC(BoundarySide.RIGHT, eos)
    solver = LagrangianSolver(
        grid=grid, eos=eos, bc_left=bc_left, bc_right=bc_right, config=solver_config
    )

    # Check AV is enabled through the conservation handler
    assert solver._av is not None
    assert solver._av.enabled
    assert solver._av.config.c_quad == 2.0
    print("PASS: Solver created with artificial viscosity")


def test_solver_without_av():
    """Test solver creation without artificial viscosity."""
    from lagrangian_solver.boundary.base import ReflectiveBC, BoundarySide

    grid_config = GridConfig(n_cells=100, x_min=0.0, x_max=1.0)
    grid = LagrangianGrid(grid_config)
    eos = IdealGasEOS(gamma=1.4)

    # Create solver without artificial viscosity
    solver_config = SolverConfig(cfl=0.5, t_end=0.1, av_enabled=False)
    bc_left = ReflectiveBC(BoundarySide.LEFT, eos)
    bc_right = ReflectiveBC(BoundarySide.RIGHT, eos)
    solver = LagrangianSolver(
        grid=grid, eos=eos, bc_left=bc_left, bc_right=bc_right, config=solver_config
    )

    assert solver._av is None
    print("PASS: Solver created without artificial viscosity")


if __name__ == "__main__":
    test_av_module_import()
    test_av_computation()
    test_solver_with_av()
    test_solver_without_av()
    print("\nAll tests passed!")
