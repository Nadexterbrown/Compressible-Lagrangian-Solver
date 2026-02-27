"""
Unit tests for boundary conditions.
"""

import numpy as np
import pytest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from lagrangian_solver.equations.eos import IdealGasEOS
from lagrangian_solver.core.grid import LagrangianGrid, GridConfig
from lagrangian_solver.core.state import FlowState
from lagrangian_solver.boundary.base import BoundarySide, ThermalBCType
from lagrangian_solver.boundary.wall import SolidWallBC, SymmetryBC
from lagrangian_solver.boundary.piston import MovingPistonBC, sinusoidal_piston, ramp_piston
from lagrangian_solver.boundary.open import OpenBC, OutflowBC, InflowBC


@pytest.fixture
def eos():
    """Create standard ideal gas EOS."""
    return IdealGasEOS(gamma=1.4, R=287.05)


@pytest.fixture
def grid():
    """Create standard test grid."""
    config = GridConfig(n_cells=10, x_min=0.0, x_max=1.0)
    return LagrangianGrid(config)


@pytest.fixture
def uniform_state(grid, eos):
    """Create uniform flow state."""
    rho = np.ones(10) * 1.225
    u = np.zeros(11)
    p = np.ones(10) * 101325.0

    return FlowState.from_primitive(rho, u, p, grid.x.copy(), eos)


class TestSolidWallBC:
    """Tests for SolidWallBC class."""

    def test_initialization_adiabatic(self, eos):
        """Test adiabatic wall initialization."""
        bc = SolidWallBC(BoundarySide.LEFT, eos, ThermalBCType.ADIABATIC)
        assert bc.thermal_bc == ThermalBCType.ADIABATIC
        assert bc.wall_temperature is None

    def test_initialization_isothermal(self, eos):
        """Test isothermal wall initialization."""
        bc = SolidWallBC(
            BoundarySide.LEFT, eos, ThermalBCType.ISOTHERMAL, wall_temperature=300.0
        )
        assert bc.thermal_bc == ThermalBCType.ISOTHERMAL
        assert bc.wall_temperature == 300.0

    def test_isothermal_requires_temperature(self, eos):
        """Test that isothermal BC requires temperature."""
        with pytest.raises(ValueError):
            SolidWallBC(BoundarySide.LEFT, eos, ThermalBCType.ISOTHERMAL)

    def test_apply_sets_velocity_zero_left(self, eos, grid, uniform_state):
        """Test that apply sets left boundary velocity to zero."""
        # Set non-zero velocity
        uniform_state.u[0] = 10.0

        bc = SolidWallBC(BoundarySide.LEFT, eos)
        bc.apply(uniform_state, grid, t=0.0)

        assert uniform_state.u[0] == 0.0

    def test_apply_sets_velocity_zero_right(self, eos, grid, uniform_state):
        """Test that apply sets right boundary velocity to zero."""
        uniform_state.u[-1] = -5.0

        bc = SolidWallBC(BoundarySide.RIGHT, eos)
        bc.apply(uniform_state, grid, t=0.0)

        assert uniform_state.u[-1] == 0.0

    def test_compute_flux_pressure(self, eos, grid, uniform_state):
        """Test that flux returns wall pressure."""
        bc = SolidWallBC(BoundarySide.LEFT, eos)
        flux = bc.compute_flux(uniform_state, grid, t=0.0)

        assert flux.p_flux > 0
        assert flux.pu_flux == 0.0  # u=0 at wall
        assert flux.u_flux == 0.0

    def test_boundary_velocity_is_zero(self, eos):
        """Test that wall velocity is always zero."""
        bc = SolidWallBC(BoundarySide.LEFT, eos)
        assert bc.get_boundary_velocity(0.0) == 0.0
        assert bc.get_boundary_velocity(100.0) == 0.0


class TestSymmetryBC:
    """Tests for SymmetryBC (alias for adiabatic wall)."""

    def test_symmetry_is_adiabatic(self, eos):
        """Test that symmetry BC is adiabatic."""
        bc = SymmetryBC(BoundarySide.LEFT, eos)
        assert bc.thermal_bc == ThermalBCType.ADIABATIC


class TestMovingPistonBC:
    """Tests for MovingPistonBC class."""

    def test_constant_velocity(self, eos):
        """Test piston with constant velocity."""
        bc = MovingPistonBC(BoundarySide.LEFT, eos, velocity=10.0)
        assert bc.get_boundary_velocity(0.0) == 10.0
        assert bc.get_boundary_velocity(1.0) == 10.0

    def test_velocity_function(self, eos):
        """Test piston with velocity function."""
        bc = MovingPistonBC(BoundarySide.LEFT, eos, velocity=lambda t: 5.0 * t)
        assert bc.get_boundary_velocity(0.0) == 0.0
        assert bc.get_boundary_velocity(2.0) == 10.0

    def test_apply_sets_velocity(self, eos, grid, uniform_state):
        """Test that apply sets boundary velocity to piston velocity."""
        bc = MovingPistonBC(BoundarySide.LEFT, eos, velocity=15.0)
        bc.apply(uniform_state, grid, t=0.0)

        assert uniform_state.u[0] == 15.0

    def test_isothermal_piston(self, eos):
        """Test isothermal moving piston."""
        bc = MovingPistonBC(
            BoundarySide.LEFT,
            eos,
            velocity=5.0,
            thermal_bc=ThermalBCType.ISOTHERMAL,
            piston_temperature=400.0,
        )
        assert bc.piston_temperature == 400.0

    def test_porous_piston_deprecated(self, eos):
        """Test that porous piston emits deprecation warning but doesn't crash."""
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            bc = MovingPistonBC(
                BoundarySide.LEFT,
                eos,
                velocity=0.0,
                porous=True,
                permeability=1e-10,
                slip_coefficient=0.5,
            )
            # Check that a DeprecationWarning was issued
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "porous" in str(w[0].message).lower()

    def test_porous_parameters_ignored(self, eos):
        """Test that porous parameters are silently ignored (with warning)."""
        import warnings
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            # Should not raise - porous parameters are ignored
            bc = MovingPistonBC(BoundarySide.LEFT, eos, velocity=0.0, porous=True)


class TestVelocityProfiles:
    """Tests for piston velocity profile functions."""

    def test_sinusoidal_piston(self):
        """Test sinusoidal velocity profile."""
        profile = sinusoidal_piston(amplitude=1.0, frequency=1.0)

        # At t=0, sin(0) = 0
        assert abs(profile(0.0)) < 1e-10

        # At t=0.25, sin(pi/2) = 1
        assert abs(profile(0.25) - 1.0) < 1e-10

    def test_ramp_piston(self):
        """Test ramp velocity profile."""
        profile = ramp_piston(v_start=0.0, v_end=10.0, t_ramp=1.0)

        assert profile(0.0) == 0.0
        assert profile(0.5) == 5.0
        assert profile(1.0) == 10.0
        assert profile(2.0) == 10.0  # After ramp


class TestOpenBC:
    """Tests for open boundary conditions."""

    def test_outflow_bc(self, eos, grid, uniform_state):
        """Test outflow boundary condition."""
        bc = OutflowBC(BoundarySide.RIGHT, eos, back_pressure=101325.0)

        assert bc.back_pressure == 101325.0

        flux = bc.compute_flux(uniform_state, grid, t=0.0)
        assert flux.p_flux > 0

    def test_inflow_bc(self, eos, grid, uniform_state):
        """Test inflow boundary condition."""
        bc = InflowBC(
            BoundarySide.LEFT,
            eos,
            velocity=10.0,
            temperature=300.0,
            pressure=101325.0,
        )

        assert bc.inflow_velocity == 10.0
        assert bc.inflow_temperature == 300.0
        assert bc.inflow_pressure == 101325.0

    def test_flow_regime_detection(self, eos, grid):
        """Test flow regime detection."""
        from lagrangian_solver.boundary.open import FlowRegime

        # Create subsonic outflow state at RIGHT boundary
        # Flow leaving through right boundary means u > 0 (moving right)
        rho = np.ones(10) * 1.225
        u = np.full(11, 10.0)  # Flow to the right (outflow through right boundary)
        p = np.ones(10) * 101325.0
        state = FlowState.from_primitive(rho, u, p, grid.x.copy(), eos)

        bc = OpenBC(BoundarySide.RIGHT, eos)
        regime = bc.determine_regime(state)

        # At right boundary with u > 0, flow is outgoing (leaving through right)
        assert regime == FlowRegime.SUBSONIC_OUTFLOW

    def test_supersonic_extrapolation(self, eos, grid):
        """Test that supersonic outflow extrapolates all quantities."""
        from lagrangian_solver.boundary.open import FlowRegime

        # Create supersonic outflow state at RIGHT boundary
        # Flow leaving through right boundary at supersonic speed
        rho = np.ones(10) * 1.225
        u = np.full(11, 500.0)  # Supersonic flow to the right
        p = np.ones(10) * 101325.0
        state = FlowState.from_primitive(rho, u, p, grid.x.copy(), eos)

        bc = OpenBC(BoundarySide.RIGHT, eos)
        regime = bc.determine_regime(state)

        assert regime == FlowRegime.SUPERSONIC_OUTFLOW
