"""
Unit tests for Lagrangian grid.
"""

import numpy as np
import pytest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from lagrangian_solver.core.grid import LagrangianGrid, GridConfig


class TestGridConfig:
    """Tests for GridConfig validation."""

    def test_valid_config(self):
        """Test valid configuration."""
        config = GridConfig(n_cells=100, x_min=0.0, x_max=1.0)
        assert config.n_cells == 100
        assert config.x_min == 0.0
        assert config.x_max == 1.0

    def test_invalid_n_cells(self):
        """Test that invalid n_cells raises error."""
        with pytest.raises(ValueError):
            GridConfig(n_cells=0, x_min=0.0, x_max=1.0)

    def test_invalid_domain(self):
        """Test that x_max <= x_min raises error."""
        with pytest.raises(ValueError):
            GridConfig(n_cells=10, x_min=1.0, x_max=0.0)
        with pytest.raises(ValueError):
            GridConfig(n_cells=10, x_min=0.5, x_max=0.5)

    def test_invalid_stretch(self):
        """Test that invalid stretch factor raises error."""
        with pytest.raises(ValueError):
            GridConfig(n_cells=10, x_min=0.0, x_max=1.0, stretch_factor=-1.0)


class TestLagrangianGrid:
    """Tests for LagrangianGrid class."""

    @pytest.fixture
    def grid(self):
        """Create a standard test grid."""
        config = GridConfig(n_cells=10, x_min=0.0, x_max=1.0)
        return LagrangianGrid(config)

    def test_initialization(self, grid):
        """Test grid initialization."""
        assert grid.n_cells == 10
        assert grid.n_faces == 11
        assert grid.x_min == 0.0
        assert grid.x_max == 1.0

    def test_uniform_spacing(self, grid):
        """Test that uniform grid has equal spacing."""
        dx = grid.dx
        expected_dx = 0.1  # 1.0 / 10

        np.testing.assert_allclose(dx, np.full(10, expected_dx), rtol=1e-10)

    def test_face_positions(self, grid):
        """Test face positions are correct."""
        expected = np.linspace(0.0, 1.0, 11)
        np.testing.assert_allclose(grid.x, expected, rtol=1e-10)

    def test_cell_centers(self, grid):
        """Test cell center positions."""
        x_cell = grid.x_cell
        expected = np.linspace(0.05, 0.95, 10)
        np.testing.assert_allclose(x_cell, expected, rtol=1e-10)

    def test_mass_initialization(self, grid):
        """Test mass coordinate initialization."""
        rho = np.ones(10)  # Uniform density

        grid.initialize_mass(rho)

        # Total mass should be 1.0 (rho=1, L=1)
        assert abs(grid.total_mass - 1.0) < 1e-10

        # Cell masses should be uniform
        expected_dm = 0.1
        np.testing.assert_allclose(grid.dm, np.full(10, expected_dm), rtol=1e-10)

    def test_mass_initialization_nonuniform(self, grid):
        """Test mass initialization with non-uniform density."""
        rho = np.linspace(1.0, 2.0, 10)

        grid.initialize_mass(rho)

        # Cell masses = rho * dx
        expected_dm = rho * 0.1
        np.testing.assert_allclose(grid.dm, expected_dm, rtol=1e-10)

    def test_cumulative_mass(self, grid):
        """Test cumulative mass at faces."""
        rho = np.ones(10)
        grid.initialize_mass(rho)

        # m should start at 0 and increase
        assert grid.m[0] == 0.0
        assert abs(grid.m[-1] - 1.0) < 1e-10

        # Should be monotonically increasing
        assert np.all(np.diff(grid.m) > 0)

    def test_set_positions(self, grid):
        """Test setting face positions."""
        new_x = np.linspace(0.0, 2.0, 11)
        grid.set_positions(new_x)

        np.testing.assert_allclose(grid.x, new_x, rtol=1e-10)
        assert grid.domain_length == 2.0

    def test_set_positions_invalid(self, grid):
        """Test that non-monotonic positions raise error."""
        bad_x = np.array([0, 0.2, 0.15, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1])
        with pytest.raises(ValueError):
            grid.set_positions(bad_x)

    def test_update_positions(self, grid):
        """Test position update with velocity."""
        rho = np.ones(10)
        grid.initialize_mass(rho)

        u = np.ones(11) * 0.5  # Uniform velocity to the right
        dt = 0.1

        grid.update_positions(u, dt)

        # All positions should shift by 0.05
        expected = np.linspace(0.05, 1.05, 11)
        np.testing.assert_allclose(grid.x, expected, rtol=1e-10)

    def test_specific_volume_computation(self, grid):
        """Test specific volume computation."""
        rho = np.ones(10) * 2.0  # Uniform density = 2
        grid.initialize_mass(rho)

        tau = grid.compute_specific_volume()

        # tau = dx / dm = dx / (rho * dx) = 1/rho = 0.5
        np.testing.assert_allclose(tau, np.full(10, 0.5), rtol=1e-10)

    def test_density_computation(self, grid):
        """Test density computation from geometry."""
        rho_init = np.ones(10) * 2.0
        grid.initialize_mass(rho_init)

        rho = grid.compute_density()

        np.testing.assert_allclose(rho, rho_init, rtol=1e-10)

    def test_density_after_compression(self, grid):
        """Test that density increases after compression."""
        rho_init = np.ones(10)
        grid.initialize_mass(rho_init)

        # Compress the domain to half size
        new_x = np.linspace(0.0, 0.5, 11)
        grid.set_positions(new_x)

        rho = grid.compute_density()

        # Density should double
        np.testing.assert_allclose(rho, np.full(10, 2.0), rtol=1e-10)

    def test_cfl_timestep(self, grid):
        """Test CFL time step computation."""
        u = np.zeros(11)  # No flow velocity
        c = np.ones(10) * 340.0  # Sound speed

        dt = grid.get_cfl_timestep(u, c, cfl=0.5)

        # dt = CFL * dx / c = 0.5 * 0.1 / 340
        expected = 0.5 * 0.1 / 340.0
        assert abs(dt - expected) < 1e-10

    def test_grid_quality(self, grid):
        """Test grid quality metrics."""
        min_dx, max_dx, aspect = grid.check_quality()

        # For uniform grid, all should be equal
        assert abs(min_dx - 0.1) < 1e-10
        assert abs(max_dx - 0.1) < 1e-10
        assert abs(aspect - 1.0) < 1e-10

    def test_to_dict_from_dict(self, grid):
        """Test grid serialization and deserialization."""
        rho = np.ones(10)
        grid.initialize_mass(rho)

        data = grid.to_dict()
        grid_restored = LagrangianGrid.from_dict(data)

        np.testing.assert_allclose(grid_restored.x, grid.x)
        np.testing.assert_allclose(grid_restored.m, grid.m)
        np.testing.assert_allclose(grid_restored.dm, grid.dm)


class TestStretchedGrid:
    """Tests for stretched grids."""

    def test_stretched_grid_creation(self):
        """Test creation of stretched grid."""
        config = GridConfig(n_cells=10, x_min=0.0, x_max=1.0, stretch_factor=2.0)
        grid = LagrangianGrid(config)

        # First cell should be smaller than last cell
        dx = grid.dx
        assert dx[0] < dx[-1], "Stretched grid should have smaller cells at x_min"

    def test_stretched_grid_bounds(self):
        """Test that stretched grid has correct bounds."""
        config = GridConfig(n_cells=20, x_min=0.0, x_max=2.0, stretch_factor=1.5)
        grid = LagrangianGrid(config)

        assert abs(grid.x[0] - 0.0) < 1e-10
        assert abs(grid.x[-1] - 2.0) < 1e-10

    def test_stretched_grid_monotonic(self):
        """Test that stretched grid positions are monotonic."""
        config = GridConfig(n_cells=50, x_min=0.0, x_max=1.0, stretch_factor=3.0)
        grid = LagrangianGrid(config)

        assert np.all(np.diff(grid.x) > 0), "Positions must be strictly increasing"
