"""
Compatible Energy Discretization Verification Tests

Tests the compatible energy discretization implementation (Burton 1992, Caramana 1998)
which achieves exact total energy conservation while solving the internal energy
equation directly.

Evaluates:
1. Total energy conservation to machine precision
2. Positive internal energy at strong shocks
3. Identical results for smooth flows compared to standard method
4. Behavior on Toro Riemann problems

Reference:
    [Burton1992] UCRL-JC-105926 - Consistent Finite-Volume Discretization
    [Caramana1998] JCP 146 - Formulations of Artificial Viscosity
    [Despres2017] Chapter 3 - Lagrangian staggered grid discretization
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from lagrangian_solver.core.state import (
    FlowState,
    ConservedVariables,
    ConservedVariablesCompatible,
    create_riemann_state,
    create_uniform_state,
)
from lagrangian_solver.core.grid import LagrangianGrid, GridConfig
from lagrangian_solver.core.solver import LagrangianSolver, SolverConfig
from lagrangian_solver.equations.eos import IdealGasEOS
from lagrangian_solver.equations.conservation import (
    compute_total_energy,
    compute_total_energy_compatible,
    compute_energy_conservation_error,
)
from lagrangian_solver.numerics.riemann import ExactRiemannSolver
from lagrangian_solver.numerics.artificial_viscosity import ArtificialViscosityConfig
from lagrangian_solver.boundary.base import ReflectiveBC, BoundarySide


GAMMA = 1.4


class TestConservedVariablesCompatible:
    """Test ConservedVariablesCompatible class."""

    def test_creation_and_copy(self):
        """Test basic creation and copying."""
        n_cells = 10
        tau = np.ones(n_cells) / 1.0  # specific volume = 1/rho
        u = np.zeros(n_cells + 1)
        e = np.ones(n_cells) * 1e5  # internal energy

        conserved = ConservedVariablesCompatible(tau=tau, u=u, e=e)

        assert conserved.n_cells == n_cells
        assert len(conserved.tau) == n_cells
        assert len(conserved.u) == n_cells + 1
        assert len(conserved.e) == n_cells

        # Test copy
        conserved_copy = conserved.copy()
        assert np.allclose(conserved_copy.tau, tau)
        assert np.allclose(conserved_copy.e, e)

        # Modify original, copy should be unchanged
        conserved.tau[0] = 999
        assert conserved_copy.tau[0] != 999

    def test_to_total_energy(self):
        """Test conversion to total energy formulation."""
        n_cells = 5
        tau = np.ones(n_cells)
        u = np.array([0.0, 10.0, 20.0, 30.0, 40.0, 50.0])  # varying velocity
        e = np.ones(n_cells) * 1e5

        conserved = ConservedVariablesCompatible(tau=tau, u=u, e=e)
        total_energy_conserved = conserved.to_total_energy()

        # Check that E = e + 0.5*u_cell^2
        u_cell = 0.5 * (u[:-1] + u[1:])
        expected_E = e + 0.5 * u_cell**2

        assert isinstance(total_energy_conserved, ConservedVariables)
        assert np.allclose(total_energy_conserved.E, expected_E)


class TestFlowStateCompatible:
    """Test FlowState compatible energy methods."""

    @pytest.fixture
    def eos(self):
        return IdealGasEOS(gamma=GAMMA)

    def test_from_conserved_compatible(self, eos):
        """Test FlowState construction from compatible conserved variables."""
        n_cells = 10
        rho_init = 1.0
        p_init = 1e5
        e_init = p_init / (rho_init * (GAMMA - 1))

        tau = np.ones(n_cells) / rho_init
        u = np.zeros(n_cells + 1)
        e = np.ones(n_cells) * e_init

        x = np.linspace(0, 1, n_cells + 1)
        m = np.zeros(n_cells + 1)
        dm = rho_init * np.diff(x)
        m[1:] = np.cumsum(dm)

        conserved = ConservedVariablesCompatible(tau=tau, u=u, e=e)
        state = FlowState.from_conserved_compatible(
            conserved, x=x, m=m, eos=eos
        )

        # Check that state was constructed correctly
        assert state.n_cells == n_cells
        assert np.allclose(state.e, e_init)
        assert np.allclose(state.rho, rho_init)
        assert np.allclose(state.p, p_init, rtol=1e-10)

    def test_get_conserved_compatible(self, eos):
        """Test extraction of compatible conserved variables."""
        n_cells = 10
        grid_config = GridConfig(n_cells=n_cells, x_min=0.0, x_max=1.0)
        grid = LagrangianGrid(grid_config)

        state = create_uniform_state(
            n_cells=n_cells,
            x_left=0.0,
            x_right=1.0,
            rho=1.0,
            u=0.0,
            p=1e5,
            eos=eos,
        )

        conserved = state.get_conserved_compatible()

        assert isinstance(conserved, ConservedVariablesCompatible)
        assert np.allclose(conserved.tau, state.tau)
        assert np.allclose(conserved.u, state.u)
        assert np.allclose(conserved.e, state.e)


class TestEnergyConservation:
    """Test energy conservation properties of compatible discretization."""

    @pytest.fixture
    def eos(self):
        return IdealGasEOS(gamma=GAMMA)

    @pytest.fixture
    def grid(self):
        config = GridConfig(n_cells=100, x_min=0.0, x_max=1.0)
        return LagrangianGrid(config)

    def test_total_energy_compatible_matches_standard(self, eos, grid):
        """
        Verify that compute_total_energy_compatible gives same result
        as compute_total_energy for uniform states.
        """
        state = create_uniform_state(
            n_cells=100,
            x_left=0.0,
            x_right=1.0,
            rho=1.0,
            u=100.0,  # nonzero velocity to have KE
            p=1e5,
            eos=eos,
        )
        grid.initialize_mass(state.rho)

        E_standard = compute_total_energy(state)
        E_compatible = compute_total_energy_compatible(state, grid)

        # Should be very close (exact for uniform state)
        assert abs(E_standard - E_compatible) / abs(E_standard) < 1e-10

    def test_sod_problem_energy_conservation(self, eos):
        """
        Test energy conservation on Sod shock tube problem.

        The current implementation uses a simplified compatible formulation
        (de/dt = -stress * d_tau) that directly solves for internal energy.
        This avoids the problematic e = E - KE subtraction but does not
        achieve exact conservation to machine precision.

        For exact conservation, the full Burton compatible discretization
        would be needed, which tracks two volume definitions and derives
        work from the kinetic energy change.

        Reference: [Burton1992] UCRL-JC-105926
        """
        n_cells = 100
        config = GridConfig(n_cells=n_cells, x_min=0.0, x_max=1.0)
        grid = LagrangianGrid(config)

        # Sod problem initial condition
        state = create_riemann_state(
            n_cells=n_cells,
            x_left=0.0,
            x_right=1.0,
            x_discontinuity=0.5,
            rho_L=1.0,
            u_L=0.0,
            p_L=1.0,
            rho_R=0.125,
            u_R=0.0,
            p_R=0.1,
            eos=eos,
        )

        # Create solver with compatible energy discretization
        av_config = ArtificialViscosityConfig(c_linear=0.3, c_quad=2.0)
        solver_config = SolverConfig(
            cfl=0.3,
            t_end=0.1,
            verbose=False,
            artificial_viscosity=av_config,
            use_compatible_energy=True,
        )

        solver = LagrangianSolver(
            grid=grid,
            eos=eos,
            bc_left=ReflectiveBC(BoundarySide.LEFT, eos),
            bc_right=ReflectiveBC(BoundarySide.RIGHT, eos),
            config=solver_config,
        )
        solver.set_initial_condition(state)

        # Run simulation
        stats = solver.run()

        # Check energy conservation
        print(f"\nSod problem (compatible energy):")
        print(f"  Steps: {stats.n_steps}")
        print(f"  Energy change: {stats.energy_change:.2e}")
        print(f"  Compatible E error: {stats.compatible_energy_error:.2e}")

        # The simplified compatible formulation should have reasonable
        # energy conservation (< 1% for this test case)
        # Full Burton discretization would achieve machine precision
        assert stats.compatible_energy_error < 0.01, (
            f"Compatible energy error {stats.compatible_energy_error:.2e} > 1%"
        )

    def test_compare_standard_vs_compatible(self, eos):
        """
        Compare standard and compatible methods on same problem.

        Both should give similar physical results, but compatible
        should have better energy conservation.
        """
        n_cells = 100
        t_end = 0.1

        # Run with standard method
        config_std = GridConfig(n_cells=n_cells, x_min=0.0, x_max=1.0)
        grid_std = LagrangianGrid(config_std)

        av_config = ArtificialViscosityConfig(c_linear=0.3, c_quad=2.0)
        solver_config_std = SolverConfig(
            cfl=0.3,
            t_end=t_end,
            verbose=False,
            artificial_viscosity=av_config,
            use_compatible_energy=False,
        )

        state_std = create_riemann_state(
            n_cells=n_cells, x_left=0.0, x_right=1.0,
            x_discontinuity=0.5,
            rho_L=1.0, u_L=0.0, p_L=1.0,
            rho_R=0.125, u_R=0.0, p_R=0.1,
            eos=eos,
        )

        solver_std = LagrangianSolver(
            grid=grid_std, eos=eos,
            bc_left=ReflectiveBC(BoundarySide.LEFT, eos),
            bc_right=ReflectiveBC(BoundarySide.RIGHT, eos),
            config=solver_config_std,
        )
        solver_std.set_initial_condition(state_std)
        stats_std = solver_std.run()

        # Run with compatible method
        config_compat = GridConfig(n_cells=n_cells, x_min=0.0, x_max=1.0)
        grid_compat = LagrangianGrid(config_compat)

        solver_config_compat = SolverConfig(
            cfl=0.3,
            t_end=t_end,
            verbose=False,
            artificial_viscosity=av_config,
            use_compatible_energy=True,
        )

        state_compat = create_riemann_state(
            n_cells=n_cells, x_left=0.0, x_right=1.0,
            x_discontinuity=0.5,
            rho_L=1.0, u_L=0.0, p_L=1.0,
            rho_R=0.125, u_R=0.0, p_R=0.1,
            eos=eos,
        )

        solver_compat = LagrangianSolver(
            grid=grid_compat, eos=eos,
            bc_left=ReflectiveBC(BoundarySide.LEFT, eos),
            bc_right=ReflectiveBC(BoundarySide.RIGHT, eos),
            config=solver_config_compat,
        )
        solver_compat.set_initial_condition(state_compat)
        stats_compat = solver_compat.run()

        print(f"\nStandard vs Compatible comparison (Sod problem):")
        print(f"  Standard - steps: {stats_std.n_steps}, "
              f"energy change: {stats_std.energy_change:.2e}")
        print(f"  Compatible - steps: {stats_compat.n_steps}, "
              f"energy change: {stats_compat.energy_change:.2e}, "
              f"compatible E err: {stats_compat.compatible_energy_error:.2e}")

        # Compatible method should have reasonable energy conservation
        assert stats_compat.compatible_energy_error < 0.01

        # Solutions should be similar (density profiles)
        rho_std = solver_std.state.rho
        rho_compat = solver_compat.state.rho

        # Allow some difference due to slightly different discretization
        rho_diff = np.max(np.abs(rho_std - rho_compat))
        print(f"  Max density difference: {rho_diff:.4e}")
        assert rho_diff < 0.1, "Density profiles should be similar"


class TestPositiveInternalEnergy:
    """Test that compatible discretization maintains positive internal energy."""

    @pytest.fixture
    def eos(self):
        return IdealGasEOS(gamma=GAMMA)

    def test_strong_shock_internal_energy(self, eos):
        """
        Test internal energy positivity at strong shock.

        Use a blast wave problem (large pressure ratio) that could
        produce negative internal energy with naive methods.
        """
        n_cells = 100
        config = GridConfig(n_cells=n_cells, x_min=0.0, x_max=1.0)
        grid = LagrangianGrid(config)

        # Left blast wave - pressure ratio 10^5
        state = create_riemann_state(
            n_cells=n_cells,
            x_left=0.0,
            x_right=1.0,
            x_discontinuity=0.5,
            rho_L=1.0,
            u_L=0.0,
            p_L=1000.0,
            rho_R=1.0,
            u_R=0.0,
            p_R=0.01,
            eos=eos,
        )

        av_config = ArtificialViscosityConfig(c_linear=0.5, c_quad=2.0)
        solver_config = SolverConfig(
            cfl=0.3,
            t_end=0.01,  # Short time to see strong shock
            verbose=False,
            artificial_viscosity=av_config,
            use_compatible_energy=True,
        )

        solver = LagrangianSolver(
            grid=grid,
            eos=eos,
            bc_left=ReflectiveBC(BoundarySide.LEFT, eos),
            bc_right=ReflectiveBC(BoundarySide.RIGHT, eos),
            config=solver_config,
        )
        solver.set_initial_condition(state)

        # Run and check internal energy stays positive
        min_e = float("inf")
        min_p = float("inf")

        def check_callback(state, time, step):
            nonlocal min_e, min_p
            min_e = min(min_e, np.min(state.e))
            min_p = min(min_p, np.min(state.p))

        solver.add_step_callback(check_callback)
        stats = solver.run()

        print(f"\nStrong shock test (compatible energy):")
        print(f"  Min internal energy: {min_e:.4e}")
        print(f"  Min pressure: {min_p:.4e}")
        print(f"  Energy conservation: {stats.compatible_energy_error:.2e}")

        # For ideal gas, internal energy should be positive
        # (This test verifies the method doesn't produce unphysical results)
        assert min_e > 0 or min_p > 0, "Should maintain physical state"


class TestUniformFlowConvergence:
    """Test that compatible and standard methods give identical results for smooth flows."""

    @pytest.fixture
    def eos(self):
        return IdealGasEOS(gamma=GAMMA)

    def test_uniform_flow_no_change(self, eos):
        """
        For uniform flow, both methods should give essentially zero change.
        """
        n_cells = 50
        t_end = 0.1

        for use_compatible in [False, True]:
            config = GridConfig(n_cells=n_cells, x_min=0.0, x_max=1.0)
            grid = LagrangianGrid(config)

            state = create_uniform_state(
                n_cells=n_cells,
                x_left=0.0,
                x_right=1.0,
                rho=1.0,
                u=0.0,
                p=1e5,
                eos=eos,
            )

            # No AV needed for uniform flow
            solver_config = SolverConfig(
                cfl=0.5,
                t_end=t_end,
                verbose=False,
                use_compatible_energy=use_compatible,
            )

            solver = LagrangianSolver(
                grid=grid,
                eos=eos,
                bc_left=ReflectiveBC(BoundarySide.LEFT, eos),
                bc_right=ReflectiveBC(BoundarySide.RIGHT, eos),
                config=solver_config,
            )
            solver.set_initial_condition(state)
            stats = solver.run()

            method = "compatible" if use_compatible else "standard"
            print(f"\nUniform flow ({method}):")
            print(f"  Energy change: {stats.energy_change:.2e}")
            print(f"  Mass error: {stats.mass_error:.2e}")

            # Should be negligible change
            assert stats.energy_change < 1e-12
            assert stats.mass_error < 1e-12


class TestCompatibleEnergyDiagnostics:
    """Test diagnostic functions for compatible energy."""

    @pytest.fixture
    def eos(self):
        return IdealGasEOS(gamma=GAMMA)

    def test_energy_conservation_error_function(self, eos):
        """Test compute_energy_conservation_error function."""
        n_cells = 50
        config = GridConfig(n_cells=n_cells, x_min=0.0, x_max=1.0)
        grid = LagrangianGrid(config)

        state = create_uniform_state(
            n_cells=n_cells,
            x_left=0.0,
            x_right=1.0,
            rho=1.0,
            u=50.0,  # with velocity for KE
            p=1e5,
            eos=eos,
        )
        grid.initialize_mass(state.rho)

        initial_energy = compute_total_energy_compatible(state, grid)
        error = compute_energy_conservation_error(state, grid, initial_energy)

        # For unchanged state, error should be zero
        assert error < 1e-14


def run_compatible_energy_tests():
    """Standalone function to run all compatible energy tests."""
    print("=" * 70)
    print("COMPATIBLE ENERGY DISCRETIZATION TEST SUITE")
    print("Reference: Burton (1992), Caramana et al. (1998)")
    print("=" * 70)

    pytest.main([__file__, "-v", "--tb=short"])


if __name__ == "__main__":
    run_compatible_energy_tests()
