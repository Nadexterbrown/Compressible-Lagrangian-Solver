"""
Adaptive mesh refinement for Lagrangian solver.

Implements cell splitting and merging to maintain resolution in regions
of expansion (rarefaction) and compression (shocks).

References:
    [Loubère2010] Loubère, Maire, Shashkov - "ReALE: A Reconnection-based
                  Arbitrary-Lagrangian-Eulerian Method" JCP 2010
    [Després2017] Chapter 7 - Rezoning and remapping for Lagrangian schemes
    [Galera2010] Galera, Breil, Maire - "A 2D unstructured multi-material
                 Cell-Centered Arbitrary Lagrangian-Eulerian scheme" JCP 2010
"""

from dataclasses import dataclass
from typing import Tuple, List, Optional
import numpy as np

from lagrangian_solver.core.state import FlowState, ConservedVariables
from lagrangian_solver.core.grid import LagrangianGrid
from lagrangian_solver.equations.eos import EOSBase


@dataclass
class AdaptiveConfig:
    """
    Configuration for adaptive mesh refinement.

    Attributes:
        enabled: Whether adaptive mesh is enabled
        refine_threshold: Split cells when size ratio > threshold (default 2.0)
        coarsen_threshold: Merge cells when size ratio < threshold (default 0.25)
        min_cells: Minimum number of cells (prevents over-coarsening)
        max_cells: Maximum number of cells (prevents runaway refinement)
        check_interval: Check for refinement every N steps (default 1)
        use_mass_ratio: If True, use mass ratio; if False, use spatial ratio
        smooth_transitions: Apply smoothing after refinement
        min_dt: Minimum time step - if dt drops below this, force coarsening
        min_dx: Minimum cell size - cells smaller than this get merged
        max_refine_level: Maximum refinement level (0 = original, 1 = split once, etc.)
        max_coarsen_level: Maximum coarsening level (negative, -1 = merged once, etc.)
        dt_floor: Absolute minimum time step (only applied if set, clips dt to this value)
        dt_floor_enabled: Whether to apply dt_floor (must be explicitly enabled)
    """

    enabled: bool = True
    refine_threshold: float = 2.0
    coarsen_threshold: float = 0.25
    min_cells: int = 20
    max_cells: int = 10000
    check_interval: int = 1
    use_mass_ratio: bool = False  # Use spatial (dx) ratio by default
    smooth_transitions: bool = True
    min_dt: Optional[float] = None  # Minimum allowed time step (triggers coarsening)
    min_dx: Optional[float] = None  # Minimum allowed cell size
    max_refine_level: int = 5  # Max times a cell can be split from original
    max_coarsen_level: int = -3  # Max times cells can be merged (negative)
    dt_floor: Optional[float] = None  # Absolute minimum time step (clips dt)
    dt_floor_enabled: bool = False  # Must explicitly enable dt_floor


@dataclass
class AdaptiveStats:
    """Statistics from adaptive mesh operation."""

    cells_split: int = 0
    cells_merged: int = 0
    cells_before: int = 0
    cells_after: int = 0
    max_size_ratio: float = 0.0
    min_size_ratio: float = 0.0
    max_level: int = 0  # Highest refinement level in mesh
    min_level: int = 0  # Lowest refinement level (most coarsened)


class AdaptiveMesh:
    """
    Adaptive mesh refinement manager for Lagrangian grids.

    Monitors cell quality and performs splitting/merging operations
    to maintain resolution throughout the simulation.

    Tracks refinement levels per cell:
        - Level 0: original cell
        - Level > 0: cell has been split (higher = more refined)
        - Level < 0: cell resulted from merging (lower = more coarsened)

    Reference: [Loubère2010] Section 3, [Després2017] Section 7.2
    """

    def __init__(self, config: AdaptiveConfig, eos: EOSBase, n_cells: int = 0):
        """
        Initialize adaptive mesh manager.

        Args:
            config: Adaptive mesh configuration
            eos: Equation of state for thermodynamic calculations
            n_cells: Initial number of cells (for level tracking)
        """
        self._config = config
        self._eos = eos
        self._step_count = 0
        # Track refinement level per cell (0 = original)
        self._cell_levels = np.zeros(n_cells, dtype=np.int32) if n_cells > 0 else None

    def initialize_levels(self, n_cells: int):
        """
        Initialize cell refinement levels to zero.

        Call this when setting up the solver with initial conditions.

        Args:
            n_cells: Number of cells in the initial grid
        """
        self._cell_levels = np.zeros(n_cells, dtype=np.int32)

    @property
    def cell_levels(self) -> Optional[np.ndarray]:
        """Current refinement level for each cell (read-only view)."""
        if self._cell_levels is not None:
            return self._cell_levels.view()
        return None

    @property
    def config(self) -> AdaptiveConfig:
        """Adaptive mesh configuration."""
        return self._config

    def compute_size_ratios(
        self, state: FlowState, grid: LagrangianGrid
    ) -> np.ndarray:
        """
        Compute cell size ratios relative to average.

        Args:
            state: Current flow state
            grid: Lagrangian grid

        Returns:
            Array of size ratios for each cell (ratio > 1 means larger than average)
        """
        if self._config.use_mass_ratio:
            # Use mass-based ratio (dm / dm_avg)
            dm = state.dm
            dm_avg = np.mean(dm)
            ratios = dm / dm_avg
        else:
            # Use spatial ratio (dx / dx_avg)
            dx = grid.dx
            dx_avg = np.mean(dx)
            ratios = dx / dx_avg

        return ratios

    def flag_cells(
        self, state: FlowState, grid: LagrangianGrid
    ) -> Tuple[List[int], List[int]]:
        """
        Flag cells for refinement or coarsening.

        Respects max_refine_level and max_coarsen_level limits.

        Args:
            state: Current flow state
            grid: Lagrangian grid

        Returns:
            Tuple of (cells_to_split, cells_to_merge)
            cells_to_merge contains pairs of adjacent cells
        """
        ratios = self.compute_size_ratios(state, grid)
        n_cells = len(ratios)

        # Initialize levels if not already done
        if self._cell_levels is None or len(self._cell_levels) != n_cells:
            self._cell_levels = np.zeros(n_cells, dtype=np.int32)

        cells_to_split = []
        cells_to_merge = []

        # Flag cells for splitting (too large)
        # Respect max_refine_level limit
        if n_cells < self._config.max_cells:
            for i in range(n_cells):
                if ratios[i] > self._config.refine_threshold:
                    # Check if cell has reached max refinement level
                    if self._cell_levels[i] < self._config.max_refine_level:
                        cells_to_split.append(i)

        # Flag cells for merging (too small)
        # Only merge adjacent cells, and avoid merging cells that will be split
        # Respect max_coarsen_level limit
        if n_cells > self._config.min_cells:
            i = 0
            while i < n_cells - 1:
                if (
                    ratios[i] < self._config.coarsen_threshold
                    and ratios[i + 1] < self._config.coarsen_threshold
                    and i not in cells_to_split
                    and (i + 1) not in cells_to_split
                ):
                    # Check if either cell has reached max coarsening level
                    merged_level = min(self._cell_levels[i], self._cell_levels[i + 1]) - 1
                    if merged_level >= self._config.max_coarsen_level:
                        cells_to_merge.append(i)  # Merge cell i with cell i+1
                        i += 2  # Skip the merged pair
                    else:
                        i += 1
                else:
                    i += 1

        return cells_to_split, cells_to_merge

    def split_cell(
        self,
        state: FlowState,
        grid: LagrangianGrid,
        cell_idx: int,
    ) -> Tuple[FlowState, LagrangianGrid]:
        """
        Split a single cell into two cells.

        Conservative splitting:
        - Mass is divided equally between new cells
        - Velocity at new face is interpolated
        - Internal energy is preserved (same value in both cells)
        - Pressure, temperature recomputed from EOS

        Reference: [Després2017] Section 7.2.1

        Args:
            state: Current flow state
            grid: Current grid
            cell_idx: Index of cell to split

        Returns:
            Tuple of (new_state, new_grid)
        """
        n_cells = state.n_cells
        idx = cell_idx

        # Original cell properties
        tau_old = state.tau[idx]
        rho_old = state.rho[idx]
        p_old = state.p[idx]
        e_old = state.e[idx]
        E_old = state.E[idx]
        dm_old = state.dm[idx]

        # Face velocities and positions
        u_left = state.u[idx]
        u_right = state.u[idx + 1]
        x_left = state.x[idx]
        x_right = state.x[idx + 1]

        # New midpoint
        x_mid = 0.5 * (x_left + x_right)
        u_mid = 0.5 * (u_left + u_right)

        # Create new arrays (n_cells + 1 cells, n_cells + 2 faces)
        new_n_cells = n_cells + 1
        new_n_faces = new_n_cells + 1

        # Cell-centered arrays
        new_tau = np.zeros(new_n_cells)
        new_rho = np.zeros(new_n_cells)
        new_p = np.zeros(new_n_cells)
        new_T = np.zeros(new_n_cells)
        new_e = np.zeros(new_n_cells)
        new_E = np.zeros(new_n_cells)
        new_c = np.zeros(new_n_cells)
        new_gamma = np.zeros(new_n_cells)
        new_s = np.zeros(new_n_cells)
        new_dm = np.zeros(new_n_cells)

        # Face-centered arrays
        new_x = np.zeros(new_n_faces)
        new_u = np.zeros(new_n_faces)
        new_m = np.zeros(new_n_faces)

        # Copy cells before the split
        new_tau[:idx] = state.tau[:idx]
        new_rho[:idx] = state.rho[:idx]
        new_p[:idx] = state.p[:idx]
        new_T[:idx] = state.T[:idx]
        new_e[:idx] = state.e[:idx]
        new_E[:idx] = state.E[:idx]
        new_c[:idx] = state.c[:idx]
        new_gamma[:idx] = state.gamma[:idx]
        new_s[:idx] = state.s[:idx]
        new_dm[:idx] = state.dm[:idx]

        # Insert two new cells at split location
        # Both cells have same thermodynamic state (conservative)
        dm_new = dm_old / 2.0  # Split mass equally

        for i in [idx, idx + 1]:
            new_tau[i] = tau_old
            new_rho[i] = rho_old
            new_p[i] = p_old
            new_T[i] = state.T[cell_idx]
            new_e[i] = e_old
            new_E[i] = E_old
            new_c[i] = state.c[cell_idx]
            new_gamma[i] = state.gamma[cell_idx]
            new_s[i] = state.s[cell_idx]
            new_dm[i] = dm_new

        # Copy cells after the split
        new_tau[idx + 2 :] = state.tau[idx + 1 :]
        new_rho[idx + 2 :] = state.rho[idx + 1 :]
        new_p[idx + 2 :] = state.p[idx + 1 :]
        new_T[idx + 2 :] = state.T[idx + 1 :]
        new_e[idx + 2 :] = state.e[idx + 1 :]
        new_E[idx + 2 :] = state.E[idx + 1 :]
        new_c[idx + 2 :] = state.c[idx + 1 :]
        new_gamma[idx + 2 :] = state.gamma[idx + 1 :]
        new_s[idx + 2 :] = state.s[idx + 1 :]
        new_dm[idx + 2 :] = state.dm[idx + 1 :]

        # Face positions and velocities
        new_x[: idx + 1] = state.x[: idx + 1]
        new_x[idx + 1] = x_mid
        new_x[idx + 2 :] = state.x[idx + 1 :]

        new_u[: idx + 1] = state.u[: idx + 1]
        new_u[idx + 1] = u_mid
        new_u[idx + 2 :] = state.u[idx + 1 :]

        # Recompute mass coordinates
        new_m[0] = state.m[0]
        new_m[1:] = new_m[0] + np.cumsum(new_dm)

        # Create new state
        new_state = FlowState(
            tau=new_tau,
            rho=new_rho,
            p=new_p,
            T=new_T,
            e=new_e,
            E=new_E,
            c=new_c,
            gamma=new_gamma,
            s=new_s,
            x=new_x,
            u=new_u,
            m=new_m,
            dm=new_dm,
        )

        # Update grid using resize method
        grid.resize(new_n_cells, new_x, new_m)

        # Update cell levels array (both new cells have parent level + 1)
        if self._cell_levels is not None:
            old_level = self._cell_levels[idx]
            new_levels = np.zeros(new_n_cells, dtype=np.int32)
            new_levels[:idx] = self._cell_levels[:idx]
            new_levels[idx] = old_level + 1  # First half of split cell
            new_levels[idx + 1] = old_level + 1  # Second half of split cell
            new_levels[idx + 2:] = self._cell_levels[idx + 1:]
            self._cell_levels = new_levels

        return new_state, grid

    def merge_cells(
        self,
        state: FlowState,
        grid: LagrangianGrid,
        cell_idx: int,
    ) -> Tuple[FlowState, LagrangianGrid]:
        """
        Merge two adjacent cells into one.

        Conservative merging:
        - Mass is summed
        - Momentum is summed (mass-weighted velocity)
        - Total energy is summed
        - Thermodynamic state recomputed from conserved variables

        Reference: [Després2017] Section 7.2.2

        Args:
            state: Current flow state
            grid: Current grid
            cell_idx: Index of first cell to merge (merges with cell_idx + 1)

        Returns:
            Tuple of (new_state, new_grid)
        """
        n_cells = state.n_cells
        idx = cell_idx

        if idx >= n_cells - 1:
            raise ValueError(f"Cannot merge cell {idx} with non-existent cell {idx+1}")

        # Properties of cells to merge
        dm1 = state.dm[idx]
        dm2 = state.dm[idx + 1]
        dm_merged = dm1 + dm2

        # Mass-weighted averaging for extensive quantities
        tau_merged = (dm1 * state.tau[idx] + dm2 * state.tau[idx + 1]) / dm_merged
        E_merged = (dm1 * state.E[idx] + dm2 * state.E[idx + 1]) / dm_merged

        # New arrays (n_cells - 1 cells)
        new_n_cells = n_cells - 1
        new_n_faces = new_n_cells + 1

        # Cell-centered arrays
        new_tau = np.zeros(new_n_cells)
        new_dm = np.zeros(new_n_cells)

        # Copy cells before merge
        new_tau[:idx] = state.tau[:idx]
        new_dm[:idx] = state.dm[:idx]

        # Merged cell
        new_tau[idx] = tau_merged
        new_dm[idx] = dm_merged

        # Copy cells after merge
        new_tau[idx + 1 :] = state.tau[idx + 2 :]
        new_dm[idx + 1 :] = state.dm[idx + 2 :]

        # Face-centered arrays - remove the middle face
        new_x = np.zeros(new_n_faces)
        new_u = np.zeros(new_n_faces)

        new_x[: idx + 1] = state.x[: idx + 1]
        new_x[idx + 1 :] = state.x[idx + 2 :]

        new_u[: idx + 1] = state.u[: idx + 1]
        new_u[idx + 1 :] = state.u[idx + 2 :]

        # Compute new total energy array
        new_E = np.zeros(new_n_cells)
        new_E[:idx] = state.E[:idx]
        new_E[idx] = E_merged
        new_E[idx + 1 :] = state.E[idx + 2 :]

        # Recompute mass coordinates
        new_m = np.zeros(new_n_faces)
        new_m[0] = state.m[0]
        new_m[1:] = new_m[0] + np.cumsum(new_dm)

        # Reconstruct full state from conserved variables
        conserved = ConservedVariables(tau=new_tau, u=new_u, E=new_E)
        new_state = FlowState.from_conserved(conserved, new_x, new_m, self._eos)

        # Update grid using resize method
        grid.resize(new_n_cells, new_x, new_m)

        # Update cell levels array (merged cell has min(parent levels) - 1)
        if self._cell_levels is not None:
            level1 = self._cell_levels[idx]
            level2 = self._cell_levels[idx + 1]
            merged_level = min(level1, level2) - 1
            new_levels = np.zeros(new_n_cells, dtype=np.int32)
            new_levels[:idx] = self._cell_levels[:idx]
            new_levels[idx] = merged_level
            new_levels[idx + 1:] = self._cell_levels[idx + 2:]
            self._cell_levels = new_levels

        return new_state, grid

    def adapt(
        self, state: FlowState, grid: LagrangianGrid, step: int,
        current_dt: Optional[float] = None
    ) -> Tuple[FlowState, LagrangianGrid, AdaptiveStats]:
        """
        Perform adaptive mesh refinement if needed.

        Args:
            state: Current flow state
            grid: Current grid
            step: Current time step number
            current_dt: Current time step size (for min_dt check)

        Returns:
            Tuple of (new_state, new_grid, stats)
        """
        stats = AdaptiveStats()
        stats.cells_before = state.n_cells

        # Initialize cell levels if needed
        if self._cell_levels is None or len(self._cell_levels) != state.n_cells:
            self._cell_levels = np.zeros(state.n_cells, dtype=np.int32)

        if not self._config.enabled:
            stats.cells_after = state.n_cells
            stats.max_level = int(np.max(self._cell_levels))
            stats.min_level = int(np.min(self._cell_levels))
            return state, grid, stats

        # Check if we should adapt this step
        if step % self._config.check_interval != 0:
            stats.cells_after = state.n_cells
            stats.max_level = int(np.max(self._cell_levels))
            stats.min_level = int(np.min(self._cell_levels))
            return state, grid, stats

        # Compute size ratios for statistics
        ratios = self.compute_size_ratios(state, grid)
        stats.max_size_ratio = float(np.max(ratios))
        stats.min_size_ratio = float(np.min(ratios))

        # Check for emergency coarsening due to small time step
        force_coarsen = False
        if (
            self._config.min_dt is not None
            and current_dt is not None
            and current_dt < self._config.min_dt
        ):
            force_coarsen = True

        # Flag cells
        cells_to_split, cells_to_merge = self.flag_cells(state, grid)

        # If forcing coarsening due to small time step, identify smallest cells
        # Respect max_coarsen_level limit
        if force_coarsen:
            # Find pairs of smallest adjacent cells to merge
            dx = grid.dx
            n_cells = state.n_cells
            # Sort cells by size and find smallest pairs
            sorted_indices = np.argsort(dx)
            for idx in sorted_indices:
                if idx < n_cells - 1 and len(cells_to_merge) < 5:
                    # Check if this cell or its neighbor is already flagged
                    already_flagged = False
                    for m in cells_to_merge:
                        if idx == m or idx + 1 == m or idx == m + 1:
                            already_flagged = True
                            break
                    # Check coarsening level limit
                    if not already_flagged:
                        merged_level = min(self._cell_levels[idx], self._cell_levels[idx + 1]) - 1
                        if merged_level >= self._config.max_coarsen_level:
                            cells_to_merge.append(idx)

        # Also check minimum cell size criterion
        if self._config.min_dx is not None:
            dx = grid.dx
            n_cells = state.n_cells
            for i in range(n_cells - 1):
                if dx[i] < self._config.min_dx and dx[i + 1] < self._config.min_dx:
                    already_flagged = False
                    for m in cells_to_merge:
                        if i == m or i + 1 == m or i == m + 1:
                            already_flagged = True
                            break
                    # Check coarsening level limit
                    if not already_flagged:
                        merged_level = min(self._cell_levels[i], self._cell_levels[i + 1]) - 1
                        if merged_level >= self._config.max_coarsen_level:
                            cells_to_merge.append(i)

        # Perform merging first (reduces cell count, simplifies split indexing)
        # Sort and process in reverse order to maintain valid indices
        cells_to_merge = sorted(set(cells_to_merge), reverse=True)
        for merge_idx in cells_to_merge:
            # Skip if index is now out of bounds (due to previous merges)
            if merge_idx >= state.n_cells - 1:
                continue
            if state.n_cells > self._config.min_cells:
                state, grid = self.merge_cells(state, grid, merge_idx)
                stats.cells_merged += 1

        # Recompute cells to split after merging (indices may have changed)
        # Respect max_refine_level limit
        if cells_to_split:
            ratios = self.compute_size_ratios(state, grid)
            cells_to_split = []
            for i in range(state.n_cells):
                if ratios[i] > self._config.refine_threshold:
                    if self._cell_levels[i] < self._config.max_refine_level:
                        cells_to_split.append(i)

        # Perform splitting (process in reverse order to maintain valid indices)
        for split_idx in reversed(cells_to_split):
            if state.n_cells < self._config.max_cells:
                state, grid = self.split_cell(state, grid, split_idx)
                stats.cells_split += 1

        # Apply smoothing if enabled
        if self._config.smooth_transitions and (
            stats.cells_split > 0 or stats.cells_merged > 0
        ):
            state = self._smooth_state(state, grid)

        stats.cells_after = state.n_cells

        # Update level statistics
        if self._cell_levels is not None and len(self._cell_levels) > 0:
            stats.max_level = int(np.max(self._cell_levels))
            stats.min_level = int(np.min(self._cell_levels))

        # Verify consistency between state and grid
        if state.n_cells != grid.n_cells:
            raise RuntimeError(
                f"State/grid mismatch after adaptation: "
                f"state.n_cells={state.n_cells}, grid.n_cells={grid.n_cells}"
            )

        return state, grid, stats

    def _smooth_state(
        self, state: FlowState, grid: LagrangianGrid
    ) -> FlowState:
        """
        Apply smoothing to reduce numerical artifacts after refinement.

        Uses simple Laplacian smoothing on velocity field.

        Args:
            state: Flow state after refinement
            grid: Current grid

        Returns:
            Smoothed flow state
        """
        # Light smoothing on velocity to reduce oscillations
        # Only smooth interior faces
        u_smooth = state.u.copy()
        n_faces = len(u_smooth)

        if n_faces > 3:
            # Simple 3-point smoothing: u_new = 0.25*u_{i-1} + 0.5*u_i + 0.25*u_{i+1}
            alpha = 0.1  # Smoothing factor (small to preserve accuracy)
            for i in range(1, n_faces - 1):
                u_smooth[i] = (1 - alpha) * state.u[i] + alpha * 0.5 * (
                    state.u[i - 1] + state.u[i + 1]
                )

        # Reconstruct state with smoothed velocity
        conserved = ConservedVariables(tau=state.tau.copy(), u=u_smooth, E=state.E.copy())
        return FlowState.from_conserved(conserved, state.x.copy(), state.m.copy(), self._eos)

    def get_level_distribution(self) -> dict:
        """
        Get distribution of cells across refinement levels.

        Returns:
            Dictionary mapping level -> count of cells at that level
        """
        if self._cell_levels is None:
            return {}

        distribution = {}
        for level in self._cell_levels:
            level_int = int(level)
            distribution[level_int] = distribution.get(level_int, 0) + 1
        return distribution

    def get_level_info(self) -> Tuple[int, int, float]:
        """
        Get summary level information.

        Returns:
            Tuple of (min_level, max_level, mean_level)
        """
        if self._cell_levels is None or len(self._cell_levels) == 0:
            return 0, 0, 0.0

        return (
            int(np.min(self._cell_levels)),
            int(np.max(self._cell_levels)),
            float(np.mean(self._cell_levels)),
        )


def create_adaptive_mesh(
    config: Optional[AdaptiveConfig] = None,
    eos: Optional[EOSBase] = None,
) -> AdaptiveMesh:
    """
    Factory function to create adaptive mesh manager.

    Args:
        config: Adaptive mesh configuration (uses defaults if None)
        eos: Equation of state (required)

    Returns:
        AdaptiveMesh instance
    """
    if eos is None:
        raise ValueError("EOS is required for adaptive mesh")

    if config is None:
        config = AdaptiveConfig()

    return AdaptiveMesh(config, eos)
