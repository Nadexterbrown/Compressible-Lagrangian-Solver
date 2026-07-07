"""
Unit tests for time step control and clock honesty.

Covers the two-clock failure mode behind the 6/28 corrupted batch: an
explicit step_forward(dt) request that violates an internal constraint must
never be honored silently (docs/DT_CLAMP_REVERT_PLAN.md), and suggest_dt()
must already include every internal constraint so callers stepping with it
are never clamped.
"""

import pytest
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from lagrangian_solver.boundary import MovingPorousPistonBC, BoundarySide
from lagrangian_solver.boundary.open import OpenBC
from lagrangian_solver.boundary.piston import TrajectoryInterpolator
from lagrangian_solver.equations.eos import IdealGasEOS
from lagrangian_solver.core.state import create_uniform_state
from lagrangian_solver.core.grid import LagrangianGrid, GridConfig
from lagrangian_solver.core.solver import (
    CompatibleLagrangianSolver as LagrangianSolver,
    SolverConfig,
)


class ConstantTrajectory(TrajectoryInterpolator):
    def __init__(self, velocity: float):
        self._velocity = velocity

    def position(self, t: float) -> float:
        return self._velocity * t

    def velocity(self, t: float) -> float:
        return self._velocity


N_CELLS = 20
RHO0 = 1.225
P0 = 101325.0


def make_solver(strict_dt: bool = False):
    """Porous piston (draining) + open right boundary."""
    eos = IdealGasEOS(gamma=1.4, R=287.0)
    grid = LagrangianGrid(GridConfig(n_cells=N_CELLS, x_min=0.0, x_max=1.0))
    state = create_uniform_state(
        n_cells=N_CELLS, x_left=0.0, x_right=1.0, rho=RHO0, u=0.0, p=P0, eos=eos,
    )
    bc_left = MovingPorousPistonBC(
        side=BoundarySide.LEFT,
        eos=eos,
        trajectory=ConstantTrajectory(velocity=50.0),
        gas_velocity_offset=-20.0,  # drains the boundary cell
    )
    bc_right = OpenBC(side=BoundarySide.RIGHT, eos=eos, p_external=P0)
    solver = LagrangianSolver(
        grid=grid, eos=eos, bc_left=bc_left, bc_right=bc_right,
        config=SolverConfig(cfl=0.4, t_end=1.0, av_enabled=True, strict_dt=strict_dt),
    )
    solver.set_initial_condition(state)
    return solver, bc_left, grid


class TestSuggestDt:
    def test_includes_porous_guard(self):
        solver, bc_left, grid = make_solver()
        dt = solver.suggest_dt()
        assert dt <= bc_left.get_max_dt_constraint(grid)
        assert dt > 0

    def test_stepping_with_suggested_dt_never_clamps(self):
        solver, _, _ = make_solver()
        for _ in range(20):
            dt = solver.suggest_dt()
            info = solver.step_forward(dt)
            assert info.dt == dt
        assert solver.statistics.clamped_steps == 0

    def test_internal_dt_never_counts_as_clamped(self):
        solver, _, _ = make_solver()
        for _ in range(10):
            solver.step_forward(None)
        assert solver.statistics.clamped_steps == 0


def drain_boundary_cell(solver, fraction):
    """Shrink the boundary cell (mass out, density constant) so the drainage
    guard becomes the binding constraint - the geometry that produced the
    corrupted 6/28 batch - while the requested dt stays CFL-stable."""
    grid = solver._grid
    state = solver.state
    grid.add_boundary_mass(BoundarySide.LEFT, -fraction * grid.dm[0])
    x = state.x.copy()
    x[1] = x[0] + grid.dm[0] / state.rho[0]
    state.x[:] = x
    grid.set_positions(x)


class TestTwoClocks:
    def test_oversized_dt_warns_and_stays_honest(self):
        solver, bc_left, grid = make_solver()
        drain_boundary_cell(solver, fraction=0.95)  # guard << CFL dt
        dt_guard = bc_left.get_max_dt_constraint(grid)
        dt_request = 2.0 * dt_guard  # violates the guard, still CFL-stable

        with pytest.warns(RuntimeWarning, match="solver.time"):
            info = solver.step_forward(dt_request)

        # Honest reporting: integrated dt is the constrained one
        assert info.dt < dt_request
        assert info.dt == pytest.approx(dt_guard, rel=0.2)
        assert solver.statistics.clamped_steps == 1
        # The solver clock advanced by the ACTUAL dt, not the requested one
        assert solver.time == pytest.approx(info.dt)

    def test_external_clock_desync_is_detectable(self):
        """Reproduce the corrupted-batch pattern and confirm it is loud."""
        solver, bc_left, grid = make_solver()
        drain_boundary_cell(solver, fraction=0.95)
        t_external = 0.0
        dt_request = 2.0 * bc_left.get_max_dt_constraint(grid)

        with pytest.warns(RuntimeWarning):
            info = solver.step_forward(dt_request)
            t_external += dt_request  # the buggy pattern

        assert t_external > solver.time  # desync exists...
        assert solver.statistics.clamped_steps == 1  # ...and is counted
        assert t_external - solver.time == pytest.approx(dt_request - info.dt)

    def test_strict_dt_raises(self):
        solver, bc_left, grid = make_solver(strict_dt=True)
        drain_boundary_cell(solver, fraction=0.95)
        dt_request = 2.0 * bc_left.get_max_dt_constraint(grid)
        with pytest.raises(ValueError, match="strict_dt"):
            solver.step_forward(dt_request)


class TestPistonNodeFidelity:
    def test_node_tracks_trajectory_at_solver_time(self):
        """The piston node must sit at trajectory.position(solver.time).

        This is the invariant the corrupted batch violated: the node was at
        the trajectory evaluated on the SOLVER clock while records used the
        script clock. With the honest clock they are the same time base.
        Heun integration is exact for the linear-velocity trajectory used
        here, so the tolerance is tight.
        """
        solver, bc_left, grid = make_solver()
        traj = bc_left.trajectory

        for _ in range(200):
            solver.step_forward(solver.suggest_dt())

        assert solver.time > 0
        x_expected = traj.position(solver.time)
        assert grid.x[0] == pytest.approx(x_expected, abs=1e-12)
