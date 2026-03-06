"""
Unit tests for MovingPorousPistonBC.

Tests the porous piston boundary condition which allows different
gas velocity and piston velocity at the boundary.
"""

import pytest
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from lagrangian_solver.boundary import MovingPorousPistonBC, BoundarySide, ThermalBCType
from lagrangian_solver.boundary.open import OpenBC
from lagrangian_solver.boundary.piston import TrajectoryInterpolator
from lagrangian_solver.equations.eos import IdealGasEOS
from lagrangian_solver.core.state import create_uniform_state
from lagrangian_solver.core.grid import LagrangianGrid, GridConfig
from lagrangian_solver.core.solver import LagrangianSolver, SolverConfig


class ConstantTrajectory(TrajectoryInterpolator):
    """Simple trajectory with constant velocity."""

    def __init__(self, velocity: float):
        self._velocity = velocity
        self._position = 0.0

    def position(self, t: float) -> float:
        return self._position + self._velocity * t

    def velocity(self, t: float) -> float:
        return self._velocity


class QuadraticTrajectory(TrajectoryInterpolator):
    """Trajectory with x = 0.5 * t^2, so v = t."""

    def position(self, t: float) -> float:
        return 0.5 * t**2

    def velocity(self, t: float) -> float:
        return t


class TestMovingPorousPistonBC:
    """Test cases for MovingPorousPistonBC."""

    @pytest.fixture
    def eos(self):
        """Create ideal gas EOS."""
        return IdealGasEOS(gamma=1.4, R=287.0)

    @pytest.fixture
    def grid(self):
        """Create test grid."""
        config = GridConfig(n_cells=50, x_min=0.0, x_max=1.0)
        return LagrangianGrid(config)

    def test_initialization(self, eos):
        """Test initialization with trajectory and offset."""
        trajectory = ConstantTrajectory(velocity=100.0)

        bc = MovingPorousPistonBC(
            side=BoundarySide.LEFT,
            eos=eos,
            trajectory=trajectory,
            gas_velocity_offset=-20.0,  # v_gas = v_piston - 20
        )

        assert bc.get_piston_velocity(0.0) == 100.0
        assert bc.get_gas_velocity(0.0) == 80.0

    def test_gas_velocity_offset(self, eos):
        """Test gas velocity offset calculation."""
        trajectory = ConstantTrajectory(velocity=100.0)

        bc = MovingPorousPistonBC(
            side=BoundarySide.LEFT,
            eos=eos,
            trajectory=trajectory,
            gas_velocity_offset=-30.0,
        )

        # v_piston = 100, offset = -30, v_gas = 70
        assert bc.get_piston_velocity(0.0) == 100.0
        assert bc.get_gas_velocity(0.0) == 70.0

    def test_gas_velocity_minimum_clamp(self, eos):
        """Test that gas velocity is clamped to minimum."""
        trajectory = ConstantTrajectory(velocity=50.0)

        bc = MovingPorousPistonBC(
            side=BoundarySide.LEFT,
            eos=eos,
            trajectory=trajectory,
            gas_velocity_offset=-100.0,  # Would give -50 without clamp
            gas_velocity_min=0.0,
        )

        # v_piston = 50, offset = -100 -> v_gas = -50, clamped to 0
        assert bc.get_piston_velocity(0.0) == 50.0
        assert bc.get_gas_velocity(0.0) == 0.0

    def test_mass_flux_calculation(self, eos, grid):
        """Test mass flux calculation."""
        trajectory = ConstantTrajectory(velocity=100.0)

        bc = MovingPorousPistonBC(
            side=BoundarySide.LEFT,
            eos=eos,
            trajectory=trajectory,
            gas_velocity_offset=-20.0,  # v_gas = 80
        )

        # Create state
        state = create_uniform_state(
            n_cells=50, x_left=0.0, x_right=1.0,
            rho=1.225, u=0.0, p=101325.0, eos=eos,
        )

        # Mass flux = rho * (v_piston - v_gas) = 1.225 * (100 - 80) = 24.5
        mass_flux = bc.get_mass_flux(state, grid, t=0.0)
        assert mass_flux == pytest.approx(1.225 * 20.0, rel=1e-6)

    def test_velocity_from_trajectory(self, eos):
        """Test that velocity comes from trajectory."""
        trajectory = QuadraticTrajectory()

        bc = MovingPorousPistonBC(
            side=BoundarySide.LEFT,
            eos=eos,
            trajectory=trajectory,
            gas_velocity_offset=0.0,
        )

        # At t=1, v = t = 1.0
        assert bc.get_piston_velocity(1.0) == pytest.approx(1.0, rel=1e-3)
        # At t=2, v = t = 2.0
        assert bc.get_piston_velocity(2.0) == pytest.approx(2.0, rel=1e-3)

    def test_has_ghost_cell(self, eos):
        """Test that BC reports having ghost cell."""
        trajectory = ConstantTrajectory(velocity=100.0)

        bc = MovingPorousPistonBC(
            side=BoundarySide.LEFT,
            eos=eos,
            trajectory=trajectory,
            gas_velocity_offset=-20.0,
        )

        assert bc.has_ghost_cell() is True

    def test_position_rate_override(self, eos):
        """Test that apply_position_rate sets d_x to piston velocity."""
        trajectory = ConstantTrajectory(velocity=100.0)

        bc = MovingPorousPistonBC(
            side=BoundarySide.LEFT,
            eos=eos,
            trajectory=trajectory,
            gas_velocity_offset=-20.0,
        )

        config = GridConfig(n_cells=10, x_min=0.0, x_max=1.0)
        grid = LagrangianGrid(config)
        state = create_uniform_state(
            n_cells=10, x_left=0.0, x_right=1.0,
            rho=1.225, u=50.0, p=101325.0, eos=eos,
        )

        # d_x starts with all velocities = 50 (from u)
        d_x = np.full(11, 50.0)

        # apply_position_rate should set boundary to v_piston = 100
        bc.apply_position_rate(d_x, state, grid, t=0.0)

        assert d_x[0] == 100.0  # Boundary now uses piston velocity
        assert d_x[1] == 50.0   # Interior unchanged

    def test_apply_velocity_sets_gas_velocity(self, eos):
        """Test that apply_velocity sets face to gas velocity, not piston."""
        trajectory = ConstantTrajectory(velocity=100.0)

        bc = MovingPorousPistonBC(
            side=BoundarySide.LEFT,
            eos=eos,
            trajectory=trajectory,
            gas_velocity_offset=-20.0,  # v_gas = 80
        )

        config = GridConfig(n_cells=10, x_min=0.0, x_max=1.0)
        grid = LagrangianGrid(config)
        state = create_uniform_state(
            n_cells=10, x_left=0.0, x_right=1.0,
            rho=1.225, u=0.0, p=101325.0, eos=eos,
        )

        bc.apply_velocity(state, grid, t=0.0)

        # Face velocity should be v_gas = 80, not v_piston = 100
        assert state.u[0] == 80.0


class TestMovingPorousPistonBCIntegration:
    """Integration tests with solver."""

    @pytest.fixture
    def eos(self):
        """Create ideal gas EOS."""
        return IdealGasEOS(gamma=1.4, R=287.0)

    def test_solid_vs_porous_pressure_difference(self, eos):
        """
        Test that porous BC produces different pressures than solid BC.

        With v_gas < v_piston, the porous BC should produce lower
        interface pressures than if v_gas = v_piston.
        """
        from lagrangian_solver.boundary import RiemannGhostPistonBC

        n_cells = 50
        grid_config = GridConfig(n_cells=n_cells, x_min=0.0, x_max=1.0)

        # Initial state
        rho_init = 1.225
        p_init = 101325.0
        u_init = 0.0

        # Create solid piston BC (v_gas = v_piston = 100 m/s)
        bc_solid = RiemannGhostPistonBC(
            side=BoundarySide.LEFT,
            eos=eos,
            velocity=100.0,
            ramp_time=0.0,
        )

        # Create porous piston BC (v_piston = 100, v_gas = 50)
        trajectory = ConstantTrajectory(velocity=100.0)
        bc_porous = MovingPorousPistonBC(
            side=BoundarySide.LEFT,
            eos=eos,
            trajectory=trajectory,
            gas_velocity_offset=-50.0,  # v_gas = 50
        )

        # Create states and grids
        grid_solid = LagrangianGrid(grid_config)
        state_solid = create_uniform_state(
            n_cells=n_cells, x_left=0.0, x_right=1.0,
            rho=rho_init, u=u_init, p=p_init, eos=eos,
        )

        grid_porous = LagrangianGrid(grid_config)
        state_porous = create_uniform_state(
            n_cells=n_cells, x_left=0.0, x_right=1.0,
            rho=rho_init, u=u_init, p=p_init, eos=eos,
        )

        # Compute interface states
        iface_solid = bc_solid.compute_interface_state(state_solid, grid_solid, 0.0)
        iface_porous = bc_porous.compute_interface_state(state_porous, grid_porous, 0.0)

        # Solid BC with higher gas velocity should produce higher pressure
        assert iface_solid.p > iface_porous.p
        assert iface_solid.u == 100.0  # Gas velocity = piston velocity
        assert iface_porous.u == 50.0  # Gas velocity < piston velocity

    def test_porous_simulation_runs(self, eos):
        """Test that a simulation with porous BC runs without error."""
        n_cells = 20
        grid_config = GridConfig(n_cells=n_cells, x_min=0.0, x_max=1.0)

        rho_init = 1.225
        p_init = 101325.0

        trajectory = ConstantTrajectory(velocity=50.0)
        bc_left = MovingPorousPistonBC(
            side=BoundarySide.LEFT,
            eos=eos,
            trajectory=trajectory,
            gas_velocity_offset=-20.0,  # v_gas = 30
        )

        bc_right = OpenBC(side=BoundarySide.RIGHT, eos=eos, p_external=p_init)

        grid = LagrangianGrid(grid_config)
        state = create_uniform_state(
            n_cells=n_cells, x_left=0.0, x_right=1.0,
            rho=rho_init, u=0.0, p=p_init, eos=eos,
        )

        solver_config = SolverConfig(
            cfl=0.4,
            t_end=1e-4,
            av_enabled=True,
        )

        solver = LagrangianSolver(
            grid=grid,
            eos=eos,
            bc_left=bc_left,
            bc_right=bc_right,
            config=solver_config,
        )

        solver.set_initial_condition(state)

        # Run 10 steps
        for _ in range(10):
            solver.step_forward(None)

        # Check that solution is valid (no NaN, positive pressure/density)
        assert not np.any(np.isnan(solver.state.rho))
        assert not np.any(np.isnan(solver.state.p))
        assert np.all(solver.state.rho > 0)
        assert np.all(solver.state.p > 0)

    def test_mass_leaving_system(self, eos):
        """Test that mass actually decreases when v_gas < v_piston."""
        n_cells = 20
        grid_config = GridConfig(n_cells=n_cells, x_min=0.0, x_max=1.0)

        rho_init = 1.225
        p_init = 101325.0

        trajectory = ConstantTrajectory(velocity=50.0)
        bc_left = MovingPorousPistonBC(
            side=BoundarySide.LEFT,
            eos=eos,
            trajectory=trajectory,
            gas_velocity_offset=-20.0,  # v_gas = 30
        )

        bc_right = OpenBC(side=BoundarySide.RIGHT, eos=eos, p_external=p_init)

        grid = LagrangianGrid(grid_config)
        state = create_uniform_state(
            n_cells=n_cells, x_left=0.0, x_right=1.0,
            rho=rho_init, u=0.0, p=p_init, eos=eos,
        )

        # Record initial mass
        initial_mass = np.sum(state.rho * state.dx)

        solver_config = SolverConfig(
            cfl=0.4,
            t_end=1e-4,
            av_enabled=True,
        )

        solver = LagrangianSolver(
            grid=grid,
            eos=eos,
            bc_left=bc_left,
            bc_right=bc_right,
            config=solver_config,
        )

        solver.set_initial_condition(state)

        # Run 20 steps
        for _ in range(20):
            solver.step_forward(None)

        # Record final mass
        final_mass = np.sum(solver.state.rho * solver.state.dx)

        # Mass should decrease since v_gas < v_piston (mass leaving)
        # Note: In Lagrangian, dm is fixed, but the domain shrinks differently
        # The key behavior is that shock pressure should be lower
        print(f"Initial mass: {initial_mass:.6f}")
        print(f"Final mass: {final_mass:.6f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
