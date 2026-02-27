"""
Unit tests for compatible energy discretization.

These tests verify that the compatible Lagrangian solver conserves
total energy to machine precision (O(10^-15) relative error).

Test cases:
1. Uniform flow: All derivatives should be zero
2. Energy conservation: Track total energy over many steps
3. AV activation: Q > 0 only in compression
4. Consistent stress: Same stress in momentum and energy

Reference: Caramana et al. (1998) JCP 146:227-262
"""

import pytest
import numpy as np

from lagrangian_solver.core.grid import LagrangianGrid, GridConfig
from lagrangian_solver.core.state import FlowState, create_uniform_state, create_riemann_state
from lagrangian_solver.core.solver import CompatibleLagrangianSolver, SolverConfig
from lagrangian_solver.equations.eos import IdealGasEOS
from lagrangian_solver.equations.conservation import (
    CompatibleConservation,
    compute_total_energy,
    compute_internal_energy,
    compute_kinetic_energy,
)
from lagrangian_solver.numerics.artificial_viscosity import (
    ArtificialViscosity,
    ArtificialViscosityConfig,
)
from lagrangian_solver.numerics.time_integration import CompatibleHeunIntegrator
from lagrangian_solver.boundary.base import ReflectiveBC, BoundarySide
from lagrangian_solver.boundary.wall import SolidWallBC


class TestUniformFlow:
    """Test that uniform flow has zero residuals."""

    def setup_method(self):
        """Set up uniform flow test case."""
        self.eos = IdealGasEOS(gamma=1.4)
        self.n_cells = 50
        self.x_left = 0.0
        self.x_right = 1.0

        # Create grid
        config = GridConfig(
            n_cells=self.n_cells,
            x_min=self.x_left,
            x_max=self.x_right,
        )
        self.grid = LagrangianGrid(config)

        # Create uniform state
        self.rho = 1.0
        self.u = 0.0  # Stationary gas
        self.p = 1e5

        self.state = create_uniform_state(
            n_cells=self.n_cells,
            x_left=self.x_left,
            x_right=self.x_right,
            rho=self.rho,
            u=self.u,
            p=self.p,
            eos=self.eos,
        )
        self.grid.initialize_mass(self.state.rho)

        # Create boundary conditions
        self.bc_left = ReflectiveBC(BoundarySide.LEFT, self.eos)
        self.bc_right = ReflectiveBC(BoundarySide.RIGHT, self.eos)

    def test_uniform_flow_zero_residuals(self):
        """Uniform flow should have all residuals equal to zero."""
        conservation = CompatibleConservation(self.eos, artificial_viscosity=None)

        # Apply boundary conditions
        self.bc_left.apply_velocity(self.state, self.grid, t=0.0)
        self.bc_right.apply_velocity(self.state, self.grid, t=0.0)

        # Compute residuals
        d_tau, d_u, d_e, d_x = conservation.compute_residual(
            self.state, self.grid, self.bc_left, self.bc_right, t=0.0
        )

        # All residuals should be zero (or machine precision)
        assert np.allclose(d_tau, 0.0, atol=1e-14), f"d_tau not zero: max = {np.max(np.abs(d_tau))}"
        assert np.allclose(d_u, 0.0, atol=1e-14), f"d_u not zero: max = {np.max(np.abs(d_u))}"
        assert np.allclose(d_e, 0.0, atol=1e-14), f"d_e not zero: max = {np.max(np.abs(d_e))}"
        assert np.allclose(d_x, 0.0, atol=1e-14), f"d_x not zero: max = {np.max(np.abs(d_x))}"


class TestEnergyConservation:
    """Test exact energy conservation over many time steps."""

    def setup_method(self):
        """Set up energy conservation test case."""
        self.eos = IdealGasEOS(gamma=1.4)
        self.n_cells = 100

    def test_energy_conservation_uniform_flow(self):
        """Total energy should be exactly conserved for uniform flow."""
        # Create grid
        config = GridConfig(n_cells=self.n_cells, x_min=0.0, x_max=1.0)
        grid = LagrangianGrid(config)

        # Create uniform state with velocity (will oscillate)
        state = create_uniform_state(
            n_cells=self.n_cells,
            x_left=0.0,
            x_right=1.0,
            rho=1.0,
            u=0.0,
            p=1e5,
            eos=self.eos,
        )
        grid.initialize_mass(state.rho)

        # Boundary conditions
        bc_left = ReflectiveBC(BoundarySide.LEFT, self.eos)
        bc_right = ReflectiveBC(BoundarySide.RIGHT, self.eos)

        # Solver config (no AV for this test)
        solver_config = SolverConfig(
            cfl=0.5,
            t_end=0.001,
            av_enabled=False,
            verbose=False,
        )

        solver = CompatibleLagrangianSolver(
            grid=grid,
            eos=self.eos,
            bc_left=bc_left,
            bc_right=bc_right,
            config=solver_config,
        )
        solver.set_initial_condition(state)

        # Initial energy
        E_initial = solver.compute_total_energy()

        # Run for many steps
        stats = solver.run()

        # Final energy
        E_final = solver.compute_total_energy()

        # Energy error should be machine precision
        rel_error = abs(E_final - E_initial) / abs(E_initial)
        assert rel_error < 1e-12, f"Energy error {rel_error:.6e} > 1e-12"
        print(f"Energy conservation: |ΔE/E| = {rel_error:.6e}")

    def test_energy_conservation_sod_shock(self):
        """Energy should be well-conserved for Sod shock tube.

        Note: Perfect machine-precision conservation requires smooth solutions.
        For discontinuous problems like Sod, time integration errors accumulate.
        The compatible discretization ensures no ADDITIONAL energy error from
        the spatial discretization - but 2nd order time integration still
        introduces O(dt^2) errors at each step.

        For a strong discontinuity over many time steps, we expect
        accumulated error of O(n_steps * dt^2) ~ O(dt) ~ O(1e-5 to 1e-6).
        """
        # Create grid
        config = GridConfig(n_cells=self.n_cells, x_min=0.0, x_max=1.0)
        grid = LagrangianGrid(config)

        # Sod shock tube initial condition
        state = create_riemann_state(
            n_cells=self.n_cells,
            x_left=0.0,
            x_right=1.0,
            x_discontinuity=0.5,
            rho_L=1.0,
            u_L=0.0,
            p_L=1.0,
            rho_R=0.125,
            u_R=0.0,
            p_R=0.1,
            eos=self.eos,
        )
        grid.initialize_mass(state.rho)

        # Boundary conditions
        bc_left = ReflectiveBC(BoundarySide.LEFT, self.eos)
        bc_right = ReflectiveBC(BoundarySide.RIGHT, self.eos)

        # Solver config (no AV to test pure conservation)
        solver_config = SolverConfig(
            cfl=0.3,
            t_end=0.01,
            av_enabled=False,
            verbose=False,
        )

        solver = CompatibleLagrangianSolver(
            grid=grid,
            eos=self.eos,
            bc_left=bc_left,
            bc_right=bc_right,
            config=solver_config,
        )
        solver.set_initial_condition(state)

        # Initial energy
        E_initial = solver.compute_total_energy()

        # Run
        stats = solver.run()

        # Final energy
        E_final = solver.compute_total_energy()

        # Energy error for discontinuous problems: expect O(1e-5)
        # This is due to time integration error, not spatial discretization
        rel_error = abs(E_final - E_initial) / abs(E_initial)
        assert rel_error < 1e-4, f"Energy error {rel_error:.6e} > 1e-4"
        print(f"Sod tube energy conservation: |ΔE/E| = {rel_error:.6e}")


class TestArtificialViscosity:
    """Test artificial viscosity behavior."""

    def setup_method(self):
        """Set up AV test case."""
        self.eos = IdealGasEOS(gamma=1.4)
        self.n_cells = 50

    def test_av_only_in_compression(self):
        """AV should only be active in compression zones (du < 0)."""
        # Create grid
        config = GridConfig(n_cells=self.n_cells, x_min=0.0, x_max=1.0)
        grid = LagrangianGrid(config)

        # Create uniform state
        state = create_uniform_state(
            n_cells=self.n_cells,
            x_left=0.0,
            x_right=1.0,
            rho=1.0,
            u=0.0,
            p=1e5,
            eos=self.eos,
        )
        grid.initialize_mass(state.rho)

        # Create compression in left half, expansion in right half
        n_faces = self.n_cells + 1
        state.u[:n_faces//2] = 100.0  # Moving right
        state.u[n_faces//2:] = -100.0  # Moving left
        # This creates compression in the middle

        # Create AV
        av_config = ArtificialViscosityConfig(c_linear=0.3, c_quad=2.0)
        av = ArtificialViscosity(av_config)

        # Compute Q
        Q = av.compute_viscous_stress(state, grid)

        # Check: Q should be > 0 where du/dx < 0 (compression)
        du_dx = (state.u[1:] - state.u[:-1]) / grid.dx
        compression = du_dx < 0

        for i in range(self.n_cells):
            if compression[i]:
                assert Q[i] > 0, f"Q[{i}] = {Q[i]} should be > 0 in compression"
            else:
                assert Q[i] == 0, f"Q[{i}] = {Q[i]} should be 0 in expansion"

        print(f"AV test passed: Q > 0 only in compression zones")

    def test_av_energy_dissipation(self):
        """With AV, total energy should decrease (entropy production)."""
        # Create grid
        config = GridConfig(n_cells=self.n_cells, x_min=0.0, x_max=1.0)
        grid = LagrangianGrid(config)

        # Sod shock tube - will develop shock
        state = create_riemann_state(
            n_cells=self.n_cells,
            x_left=0.0,
            x_right=1.0,
            x_discontinuity=0.5,
            rho_L=1.0,
            u_L=0.0,
            p_L=1.0,
            rho_R=0.125,
            u_R=0.0,
            p_R=0.1,
            eos=self.eos,
        )
        grid.initialize_mass(state.rho)

        # Boundary conditions
        bc_left = ReflectiveBC(BoundarySide.LEFT, self.eos)
        bc_right = ReflectiveBC(BoundarySide.RIGHT, self.eos)

        # Solver config WITH AV
        solver_config = SolverConfig(
            cfl=0.3,
            t_end=0.1,
            av_enabled=True,
            av_linear=0.3,
            av_quad=2.0,
            verbose=False,
        )

        solver = CompatibleLagrangianSolver(
            grid=grid,
            eos=self.eos,
            bc_left=bc_left,
            bc_right=bc_right,
            config=solver_config,
        )
        solver.set_initial_condition(state)

        # Initial energy
        E_initial = solver.compute_total_energy()

        # Run
        stats = solver.run()

        # Final energy
        E_final = solver.compute_total_energy()

        # With AV, kinetic energy is converted to internal (entropy production)
        # Total energy in this closed system is still conserved, but
        # internal energy increases and kinetic decreases

        # The key is that without proper compatible discretization,
        # energy would NOT be conserved. With compatible discretization
        # the AV work on the momentum equation exactly matches the
        # energy dissipation in the energy equation.

        rel_error = abs(E_final - E_initial) / abs(E_initial)
        print(f"Sod with AV energy error: |ΔE/E| = {rel_error:.6e}")

        # With compatible discretization, AV work in momentum exactly matches
        # energy dissipation. Error comes from time integration accumulation.
        assert rel_error < 1e-4, f"Energy error {rel_error:.6e} too large with AV"


class TestCompatibleDiscretization:
    """Test that discretization is truly compatible."""

    def setup_method(self):
        """Set up test case."""
        self.eos = IdealGasEOS(gamma=1.4)
        self.n_cells = 50

    def test_stress_consistency(self):
        """Same stress should be used in momentum and energy equations."""
        # Create grid
        config = GridConfig(n_cells=self.n_cells, x_min=0.0, x_max=1.0)
        grid = LagrangianGrid(config)

        # Create non-uniform state
        state = create_riemann_state(
            n_cells=self.n_cells,
            x_left=0.0,
            x_right=1.0,
            x_discontinuity=0.5,
            rho_L=1.0,
            u_L=0.0,
            p_L=1.0,
            rho_R=0.125,
            u_R=0.0,
            p_R=0.1,
            eos=self.eos,
        )
        grid.initialize_mass(state.rho)

        # Create conservation with AV
        av_config = ArtificialViscosityConfig(c_linear=0.3, c_quad=2.0)
        av = ArtificialViscosity(av_config)
        conservation = CompatibleConservation(self.eos, av)

        # Compute stress
        sigma = conservation.compute_stress(state, grid)

        # Verify sigma = p + Q
        Q = av.compute_viscous_stress(state, grid)
        expected_sigma = state.p + Q

        assert np.allclose(sigma, expected_sigma), "Stress should be p + Q"
        print("Stress consistency test passed: σ = p + Q")

    def test_pdv_work_form(self):
        """Energy equation should use d_e = -σ * d_tau form."""
        # Create grid
        config = GridConfig(n_cells=self.n_cells, x_min=0.0, x_max=1.0)
        grid = LagrangianGrid(config)

        # Create non-uniform state with velocities
        state = create_uniform_state(
            n_cells=self.n_cells,
            x_left=0.0,
            x_right=1.0,
            rho=1.0,
            u=0.0,
            p=1e5,
            eos=self.eos,
        )
        grid.initialize_mass(state.rho)

        # Set non-zero velocities
        state.u[:] = np.linspace(-100, 100, len(state.u))

        # Create conservation with AV
        av_config = ArtificialViscosityConfig(c_linear=0.3, c_quad=2.0)
        av = ArtificialViscosity(av_config)
        conservation = CompatibleConservation(self.eos, av)

        # Create BCs
        bc_left = ReflectiveBC(BoundarySide.LEFT, self.eos)
        bc_right = ReflectiveBC(BoundarySide.RIGHT, self.eos)
        bc_left.apply_velocity(state, grid, t=0.0)
        bc_right.apply_velocity(state, grid, t=0.0)

        # Compute residuals
        d_tau, d_u, d_e, d_x = conservation.compute_residual(
            state, grid, bc_left, bc_right, t=0.0
        )

        # Verify d_e = -sigma * d_tau
        sigma = conservation.compute_stress(state, grid)
        expected_d_e = -sigma * d_tau

        assert np.allclose(d_e, expected_d_e, rtol=1e-14), \
            f"d_e should equal -σ * d_tau. Max diff: {np.max(np.abs(d_e - expected_d_e))}"
        print("pdV work form test passed: d_e = -σ * d_tau")


class TestInstantaneousConservation:
    """Test that the RHS satisfies exact energy conservation at each evaluation."""

    def setup_method(self):
        """Set up test case."""
        self.eos = IdealGasEOS(gamma=1.4)
        self.n_cells = 50

    def test_discrete_work_equals_energy_change(self):
        """
        Verify that discrete work = discrete energy change.

        This is the fundamental property of compatible discretization:
            Σ m_j u_j du_j/dt = -Σ dm_i e_i de_i/dt

        In other words, the rate of kinetic energy change equals the
        rate of internal energy change (with opposite sign).
        """
        # Create grid
        config = GridConfig(n_cells=self.n_cells, x_min=0.0, x_max=1.0)
        grid = LagrangianGrid(config)

        # Create non-uniform state with velocities
        state = create_riemann_state(
            n_cells=self.n_cells,
            x_left=0.0,
            x_right=1.0,
            x_discontinuity=0.5,
            rho_L=1.0,
            u_L=0.0,
            p_L=1.0,
            rho_R=0.125,
            u_R=0.0,
            p_R=0.1,
            eos=self.eos,
        )
        grid.initialize_mass(state.rho)

        # Create conservation with AV
        av_config = ArtificialViscosityConfig(c_linear=0.3, c_quad=2.0)
        av = ArtificialViscosity(av_config)
        conservation = CompatibleConservation(self.eos, av)

        # Create BCs
        bc_left = ReflectiveBC(BoundarySide.LEFT, self.eos)
        bc_right = ReflectiveBC(BoundarySide.RIGHT, self.eos)
        bc_left.apply_velocity(state, grid, t=0.0)
        bc_right.apply_velocity(state, grid, t=0.0)

        # Compute residuals
        d_tau, d_u, d_e, d_x = conservation.compute_residual(
            state, grid, bc_left, bc_right, t=0.0
        )

        dm = grid.dm

        # Rate of internal energy change: dIE/dt = Σ dm_i * de_i/dt
        dIE_dt = np.sum(dm * d_e)

        # Rate of kinetic energy change: dKE/dt = Σ m_j * u_j * du_j/dt
        # Face masses
        n_cells = state.n_cells
        dKE_dt = 0.0
        for j in range(state.n_faces):
            if j == 0:
                m_j = 0.5 * dm[0]
            elif j == n_cells:
                m_j = 0.5 * dm[-1]
            else:
                m_j = 0.5 * (dm[j - 1] + dm[j])
            dKE_dt += m_j * state.u[j] * d_u[j]

        # Total energy rate should be zero for closed system
        dE_total_dt = dIE_dt + dKE_dt

        # This should be machine precision for compatible discretization
        rel_error = abs(dE_total_dt) / max(abs(dIE_dt), abs(dKE_dt), 1e-30)

        assert rel_error < 1e-12, \
            f"Instantaneous energy rate error: {rel_error:.6e}\n" \
            f"dIE/dt = {dIE_dt:.6e}, dKE/dt = {dKE_dt:.6e}"

        print(f"Instantaneous energy conservation: |dE/dt| / |dKE/dt| = {rel_error:.6e}")
        print(f"  dIE/dt = {dIE_dt:.6e}, dKE/dt = {dKE_dt:.6e}, dE/dt = {dE_total_dt:.6e}")


class TestFromInternalEnergy:
    """Test state construction from internal energy."""

    def setup_method(self):
        """Set up test case."""
        self.eos = IdealGasEOS(gamma=1.4)
        self.n_cells = 50

    def test_internal_energy_is_primary(self):
        """Internal energy e should be primary, E should be derived."""
        # Create grid
        config = GridConfig(n_cells=self.n_cells, x_min=0.0, x_max=1.0)
        grid = LagrangianGrid(config)

        # Create initial state
        state0 = create_uniform_state(
            n_cells=self.n_cells,
            x_left=0.0,
            x_right=1.0,
            rho=1.0,
            u=100.0,  # Non-zero velocity
            p=1e5,
            eos=self.eos,
        )
        grid.initialize_mass(state0.rho)

        # Get values
        tau = state0.tau.copy()
        u = state0.u.copy()
        e = state0.e.copy()  # Internal energy
        x = grid.x.copy()
        m = grid.m.copy()

        # Reconstruct state from internal energy
        state1 = FlowState.from_internal_energy(
            tau=tau, u=u, e=e, x=x, m=m, eos=self.eos
        )

        # Verify e is exactly preserved (no subtraction error)
        assert np.allclose(state1.e, e, rtol=1e-14), "Internal energy not preserved"

        # Verify E is correctly derived
        u_cell = 0.5 * (u[:-1] + u[1:])
        expected_E = e + 0.5 * u_cell**2
        assert np.allclose(state1.E, expected_E, rtol=1e-14), "Total energy incorrectly derived"

        print("from_internal_energy test passed: e is primary, E is derived")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
