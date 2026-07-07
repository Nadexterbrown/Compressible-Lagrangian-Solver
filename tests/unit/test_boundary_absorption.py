"""
Unit tests for boundary-cell absorption (drained porous piston cells).

When the porous piston drains its boundary cell below a mass threshold, the
cell is retired: absorbed conservatively into its neighbor with the cell
count reduced by one (docs/PISTON_CELL_COLLAPSE_FIX_PLAN.md Phase 2, after
GDTk L1D). These tests cover conservation, trigger policy, dt recovery, and
the cell-count floor.
"""

import pytest
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from lagrangian_solver.boundary import MovingPorousPistonBC, BoundarySide
from lagrangian_solver.boundary.piston import TrajectoryInterpolator
from lagrangian_solver.equations.eos import IdealGasEOS
from lagrangian_solver.core.state import create_uniform_state
from lagrangian_solver.core.grid import LagrangianGrid, GridConfig


class ConstantTrajectory(TrajectoryInterpolator):
    """Simple trajectory with constant velocity."""

    def __init__(self, velocity: float):
        self._velocity = velocity

    def position(self, t: float) -> float:
        return self._velocity * t

    def velocity(self, t: float) -> float:
        return self._velocity


N_CELLS = 20
RHO0 = 1.2
P0 = 101325.0


@pytest.fixture
def eos():
    return IdealGasEOS(gamma=1.4, R=287.0)


@pytest.fixture
def setup(eos):
    """Grid + uniform state + left porous BC (piston faster than gas -> drainage)."""
    grid = LagrangianGrid(GridConfig(n_cells=N_CELLS, x_min=0.0, x_max=1.0))
    state = create_uniform_state(
        n_cells=N_CELLS, x_left=0.0, x_right=1.0, rho=RHO0, u=0.0, p=P0, eos=eos,
    )
    grid.initialize_mass(state.rho)
    bc = MovingPorousPistonBC(
        side=BoundarySide.LEFT,
        eos=eos,
        trajectory=ConstantTrajectory(velocity=100.0),
        gas_velocity_offset=-50.0,  # u_gas < u_piston: boundary cell drains
    )
    return grid, state, bc


def drain_boundary_cell(grid, state, fraction):
    """Remove `fraction` of the boundary cell's mass, shrinking it geometrically
    (mimics porous drainage: mass leaves, the face moves in, density unchanged)."""
    dm0 = grid.dm[0]
    grid.add_boundary_mass(BoundarySide.LEFT, -fraction * dm0)
    # Face 1 moves so cell 0 keeps its density: dx0_new = dm0_new / rho0
    x = state.x.copy()
    x[1] = x[0] + grid.dm[0] / state.rho[0]
    state.x[:] = x
    grid.set_positions(x)


def total_mass(grid):
    return float(np.sum(grid.dm))


def total_energy(grid, state):
    u_c = 0.5 * (state.u[:-1] + state.u[1:])
    return float(np.sum(grid.dm * (state.e + 0.5 * u_c**2)))


class TestAbsorption:
    def test_absorbs_drained_cell(self, setup):
        grid, state, bc = setup
        drain_boundary_cell(grid, state, fraction=0.7)  # dm0 = 0.3*neighbor

        changed = bc.check_merge_split(grid, state)

        assert changed is True
        assert grid.n_cells == N_CELLS - 1
        assert state.n_cells == N_CELLS - 1
        assert len(state.x) == grid.n_faces

    def test_conservation(self, setup):
        grid, state, bc = setup
        drain_boundary_cell(grid, state, fraction=0.7)

        mass_before = total_mass(grid)
        energy_before = total_energy(grid, state)
        x_piston_before = state.x[0]
        x_right_before = state.x[-1]

        bc.check_merge_split(grid, state)

        assert total_mass(grid) == pytest.approx(mass_before, rel=1e-14)
        assert total_energy(grid, state) == pytest.approx(energy_before, rel=1e-12)
        # Domain boundaries must not move
        assert state.x[0] == x_piston_before
        assert state.x[-1] == x_right_before

    def test_state_consistency_after_absorption(self, setup):
        grid, state, bc = setup
        drain_boundary_cell(grid, state, fraction=0.7)
        bc.check_merge_split(grid, state)

        # Grid and state geometry agree
        np.testing.assert_allclose(state.x, grid.x)
        np.testing.assert_allclose(state.dm, grid.dm)
        # Positions strictly increasing
        assert np.all(np.diff(state.x) > 0)
        # Density consistent with geometry: rho = dm/dx
        np.testing.assert_allclose(state.rho, grid.dm / np.diff(state.x), rtol=1e-12)
        # EOS state valid and consistent (ideal gas: p = rho*R*T, e = p/((g-1)rho))
        assert np.all(state.p > 0) and np.all(state.T > 0) and np.all(state.e > 0)
        np.testing.assert_allclose(state.p, state.rho * 287.0 * state.T, rtol=1e-10)

    def test_no_absorption_without_drainage(self, setup):
        """Uniform compression (mass unchanged) must not retire cells."""
        grid, state, bc = setup
        # Compress the whole domain by 5x: dx shrinks, mass ratios unchanged
        x = state.x * 0.2
        state.x[:] = x
        grid.set_positions(x)

        changed = bc.check_merge_split(grid, state)

        assert changed is False
        assert grid.n_cells == N_CELLS

    def test_dt_constraint_recovers(self, setup):
        """The drainage-guard dt must grow after absorption (CFL recovery)."""
        grid, state, bc = setup
        bc.apply_velocity(state, grid, 0.0)  # sets internal time/velocities
        drain_boundary_cell(grid, state, fraction=0.9)  # very small cell

        dt_before = bc.get_max_dt_constraint(grid)
        bc.check_merge_split(grid, state)
        dt_after = bc.get_max_dt_constraint(grid)

        assert dt_after > 5.0 * dt_before

    def test_repeated_absorption_hits_floor(self, setup):
        """Repeated drainage must raise once the cell-count floor is reached."""
        grid, state, bc = setup
        with pytest.raises(RuntimeError, match="floor"):
            for _ in range(N_CELLS):
                drain_boundary_cell(grid, state, fraction=0.9)
                bc.check_merge_split(grid, state)

    def test_growth_direction_still_merge_splits(self, setup):
        """A boundary cell that GAINS mass keeps the equalizing merge-split."""
        grid, state, bc = setup
        dm0 = grid.dm[0]
        grid.add_boundary_mass(BoundarySide.LEFT, 1.5 * dm0)  # dm0 = 2.5x neighbor
        # Grow the cell geometrically to keep density sane
        x = state.x.copy()
        x[1] = x[0] + grid.dm[0] / state.rho[0]
        # Shift interior faces right of face 1 are unchanged; face1 moved right past x[2]?
        # Keep it simple: only valid if x stays monotonic
        if x[1] < x[2]:
            state.x[:] = x
            grid.set_positions(x)

        n_before = grid.n_cells
        changed = bc.check_merge_split(grid, state)

        assert changed is True
        assert grid.n_cells == n_before  # merge-split, not absorption
        # Equalized with neighbor
        assert grid.dm[0] == pytest.approx(grid.dm[1], rel=1e-12)
