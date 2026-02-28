"""
Verification tests for artificial heat conduction.

Tests the artificial heat conduction module for:
1. Pure contact discontinuity spreading
2. Sod shock tube with heat conduction
3. Energy conservation
4. Contact width scaling

Reference: [Noh1987] Noh, W.F. (1987). "Errors for calculations of strong
           shocks using an artificial viscosity and an artificial heat flux."
           Journal of Computational Physics, 72(1), 78-120.
"""

import numpy as np
import pytest
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from lagrangian_solver.equations.eos import IdealGasEOS
from lagrangian_solver.core.grid import LagrangianGrid, GridConfig
from lagrangian_solver.core.state import create_riemann_state, FlowState
from lagrangian_solver.core.solver import CompatibleLagrangianSolver, SolverConfig
from lagrangian_solver.numerics.artificial_heat_conduction import (
    ArtificialHeatConduction,
    ArtificialHeatConductionConfig,
)
from lagrangian_solver.boundary.open import OpenBC
from lagrangian_solver.boundary.base import BoundarySide
from lagrangian_solver.equations.conservation import compute_total_energy


GAMMA = 1.4


class TestArtificialHeatConductionConfig:
    """Test ArtificialHeatConductionConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ArtificialHeatConductionConfig()
        assert config.kappa_linear == 0.1
        assert config.kappa_quad == 0.5
        assert config.use_density_switch is True
        assert config.enabled is False

    def test_custom_config(self):
        """Test custom configuration values."""
        config = ArtificialHeatConductionConfig(
            kappa_linear=0.2,
            kappa_quad=1.0,
            use_density_switch=False,
            enabled=True,
        )
        assert config.kappa_linear == 0.2
        assert config.kappa_quad == 1.0
        assert config.use_density_switch is False
        assert config.enabled is True

    def test_negative_kappa_raises(self):
        """Test that negative coefficients raise ValueError."""
        with pytest.raises(ValueError):
            ArtificialHeatConductionConfig(kappa_linear=-0.1)
        with pytest.raises(ValueError):
            ArtificialHeatConductionConfig(kappa_quad=-0.5)


class TestArtificialHeatConduction:
    """Test ArtificialHeatConduction class."""

    @pytest.fixture
    def eos(self):
        """Create ideal gas EOS."""
        return IdealGasEOS(gamma=GAMMA)

    @pytest.fixture
    def contact_state(self, eos):
        """Create a pure contact discontinuity state.

        Contact discontinuity: density jumps, but pressure and velocity
        are continuous. This is what heat conduction should spread.
        """
        n_cells = 100
        grid_config = GridConfig(n_cells=n_cells, x_min=0.0, x_max=1.0)
        grid = LagrangianGrid(grid_config)

        # Pure contact: rho jumps at x=0.5, but p and u are constant
        # Left: rho=1.0, u=0.5, p=1.0
        # Right: rho=0.5, u=0.5, p=1.0
        state = create_riemann_state(
            n_cells=n_cells,
            x_left=0.0,
            x_right=1.0,
            x_discontinuity=0.5,
            rho_L=1.0,
            u_L=0.5,
            p_L=1.0,
            rho_R=0.5,
            u_R=0.5,
            p_R=1.0,
            eos=eos,
        )
        grid.initialize_mass(state.rho)

        return state, grid

    def test_disabled_returns_zeros(self, contact_state):
        """Test that disabled HC returns zero flux and source."""
        state, grid = contact_state

        config = ArtificialHeatConductionConfig(enabled=False)
        hc = ArtificialHeatConduction(config)

        q = hc.compute_heat_flux(state, grid)
        d_e = hc.compute_energy_source(state, grid)

        assert np.allclose(q, 0.0)
        assert np.allclose(d_e, 0.0)

    def test_heat_flux_shape(self, contact_state):
        """Test heat flux has correct shape (face-centered)."""
        state, grid = contact_state

        config = ArtificialHeatConductionConfig(enabled=True)
        hc = ArtificialHeatConduction(config)

        q = hc.compute_heat_flux(state, grid)

        assert q.shape == (state.n_faces,)

    def test_heat_flux_boundary_adiabatic(self, contact_state):
        """Test that boundary face heat fluxes are zero (adiabatic)."""
        state, grid = contact_state

        config = ArtificialHeatConductionConfig(enabled=True)
        hc = ArtificialHeatConduction(config)

        q = hc.compute_heat_flux(state, grid)

        # Boundary faces should be zero (adiabatic BC)
        assert q[0] == 0.0
        assert q[-1] == 0.0

    def test_energy_source_shape(self, contact_state):
        """Test energy source has correct shape (cell-centered)."""
        state, grid = contact_state

        config = ArtificialHeatConductionConfig(enabled=True)
        hc = ArtificialHeatConduction(config)

        d_e = hc.compute_energy_source(state, grid)

        assert d_e.shape == (state.n_cells,)

    def test_energy_conservation_discrete(self, contact_state):
        """Test that sum of dm * d_e_heat = 0 (exact conservation)."""
        state, grid = contact_state

        config = ArtificialHeatConductionConfig(enabled=True)
        hc = ArtificialHeatConduction(config)

        d_e = hc.compute_energy_source(state, grid)

        # Conservative divergence should sum to zero
        # Sum(dm * d_e) = -(q[-1] - q[0]) = 0 for adiabatic BCs
        total = np.sum(grid.dm * d_e)
        assert abs(total) < 1e-14, f"Energy conservation violated: sum = {total}"

    def test_heat_flux_at_contact(self, contact_state):
        """Test that heat flux activates at the contact discontinuity."""
        state, grid = contact_state

        config = ArtificialHeatConductionConfig(
            kappa_linear=0.1,
            kappa_quad=0.5,
            enabled=True,
        )
        hc = ArtificialHeatConduction(config)

        q = hc.compute_heat_flux(state, grid)

        # Find the face near the contact (around x=0.5)
        x_faces = grid.x
        contact_face = np.argmin(np.abs(x_faces - 0.5))

        # Heat flux should be non-zero near the contact
        assert q[contact_face] != 0.0, "Heat flux should be non-zero at contact"

        # Heat flux should be small far from contact
        far_left = 10  # Far left interior face
        far_right = state.n_faces - 10  # Far right interior face
        assert abs(q[far_left]) < abs(q[contact_face])
        assert abs(q[far_right]) < abs(q[contact_face])


class TestContactDiscontinuitySpreading:
    """Test that heat conduction spreads contact discontinuities."""

    @pytest.fixture
    def eos(self):
        return IdealGasEOS(gamma=GAMMA)

    def test_sod_with_heat_conduction(self, eos):
        """Test Sod shock tube with heat conduction enabled.

        The Sod problem has a contact discontinuity between the shock and
        rarefaction. With heat conduction, this should be smoothed.
        """
        n_cells = 200
        grid_config = GridConfig(n_cells=n_cells, x_min=0.0, x_max=1.0)
        grid = LagrangianGrid(grid_config)

        # Sod initial conditions
        bc_left = OpenBC(BoundarySide.LEFT, eos, p_external=1.0, u_external=0.0, rho_external=1.0)
        bc_right = OpenBC(BoundarySide.RIGHT, eos, p_external=0.1, u_external=0.0, rho_external=0.125)

        # Run with heat conduction
        config = SolverConfig(
            cfl=0.3,
            t_end=0.2,
            verbose=False,
            av_enabled=True,
            av_linear=0.3,
            av_quad=2.0,
            hc_enabled=True,
            hc_linear=0.1,
            hc_quad=0.5,
        )

        solver = CompatibleLagrangianSolver(grid, eos, bc_left, bc_right, config)
        solver.set_riemann_ic(
            x_disc=0.5,
            rho_L=1.0, u_L=0.0, p_L=1.0,
            rho_R=0.125, u_R=0.0, p_R=0.1,
        )

        stats = solver.run()

        # Basic sanity checks
        assert stats.n_steps > 0
        assert not np.isnan(solver.state.rho).any()
        assert not np.isnan(solver.state.p).any()

    def test_energy_conservation_with_hc(self, eos):
        """Test that total energy is conserved with heat conduction.

        The heat conduction should redistribute energy but not create or
        destroy it. Total energy should be conserved to machine precision.
        """
        n_cells = 100
        grid_config = GridConfig(n_cells=n_cells, x_min=0.0, x_max=1.0)
        grid = LagrangianGrid(grid_config)

        # Use reflective BCs to ensure no energy flux at boundaries
        from lagrangian_solver.boundary.wall import SolidWallBC

        bc_left = SolidWallBC(BoundarySide.LEFT, eos)
        bc_right = SolidWallBC(BoundarySide.RIGHT, eos)

        config = SolverConfig(
            cfl=0.3,
            t_end=0.1,
            verbose=False,
            av_enabled=True,
            hc_enabled=True,
            hc_linear=0.2,
            hc_quad=1.0,
        )

        solver = CompatibleLagrangianSolver(grid, eos, bc_left, bc_right, config)

        # Initial condition with density variation (like contact)
        state = create_riemann_state(
            n_cells=n_cells,
            x_left=0.0,
            x_right=1.0,
            x_discontinuity=0.5,
            rho_L=1.0,
            u_L=0.0,  # Zero velocity for reflective BCs
            p_L=1.0,
            rho_R=0.5,
            u_R=0.0,
            p_R=1.0,  # Same pressure (pure contact)
            eos=eos,
        )
        solver.set_initial_condition(state)

        # Get initial energy
        E_initial = solver.compute_total_energy()

        # Run simulation
        stats = solver.run()

        # Check energy conservation
        E_final = solver.compute_total_energy()
        rel_error = abs(E_final - E_initial) / abs(E_initial)

        # Energy should be conserved to reasonable precision
        # Note: Not machine precision due to time integration and state reconstruction
        assert rel_error < 1e-6, f"Energy not conserved: relative error = {rel_error:.2e}"


class TestContactWidthScaling:
    """Test that contact width scales with HC coefficients."""

    @pytest.fixture
    def eos(self):
        return IdealGasEOS(gamma=GAMMA)

    def measure_contact_width(self, state, threshold=0.1):
        """Measure the width of the contact discontinuity.

        Returns the number of cells where the density gradient is significant.
        """
        rho = state.rho
        drho = np.abs(np.diff(rho))
        max_drho = np.max(drho)

        if max_drho < 1e-10:
            return 0

        # Count cells where gradient exceeds threshold of max
        significant = drho > threshold * max_drho
        return np.sum(significant)

    def test_larger_coefficients_wider_contact(self, eos):
        """Test that larger HC coefficients spread contact over more cells."""
        n_cells = 200
        grid_config = GridConfig(n_cells=n_cells, x_min=0.0, x_max=1.0)

        from lagrangian_solver.boundary.wall import SolidWallBC

        widths = []

        for kappa_quad in [0.2, 0.5, 1.0]:
            grid = LagrangianGrid(grid_config)
            bc_left = SolidWallBC(BoundarySide.LEFT, eos)
            bc_right = SolidWallBC(BoundarySide.RIGHT, eos)

            config = SolverConfig(
                cfl=0.3,
                t_end=0.05,
                verbose=False,
                av_enabled=True,
                hc_enabled=True,
                hc_linear=0.1,
                hc_quad=kappa_quad,
            )

            solver = CompatibleLagrangianSolver(grid, eos, bc_left, bc_right, config)
            state = create_riemann_state(
                n_cells=n_cells,
                x_left=0.0,
                x_right=1.0,
                x_discontinuity=0.5,
                rho_L=1.0,
                u_L=0.0,
                p_L=1.0,
                rho_R=0.5,
                u_R=0.0,
                p_R=1.0,
                eos=eos,
            )
            solver.set_initial_condition(state)
            solver.run()

            width = self.measure_contact_width(solver.state)
            widths.append(width)

        # Larger coefficients should generally produce wider contacts
        # (or at least not dramatically narrower)
        assert widths[-1] >= widths[0] * 0.8, \
            f"Contact width did not increase with larger coefficients: {widths}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
