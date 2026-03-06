"""
Lagrangian grid management.

This module provides classes for managing a staggered Lagrangian grid
where positions move with the fluid while mass coordinates remain fixed.

References:
    [Despres2017] Chapter 3 - Lagrangian coordinates and grid motion

Grid Structure:
    Position (x):
         x_0      x_1      x_2      x_3      x_4
          |        |        |        |        |
          ▼        ▼        ▼        ▼        ▼
     ─────●────────●────────●────────●────────●─────
          │  Cell  │  Cell  │  Cell  │  Cell  │
          │   0    │   1    │   2    │   3    │
     ─────●────────●────────●────────●────────●─────
          ▲        ▲        ▲        ▲        ▲
        Face 0   Face 1   Face 2   Face 3   Face 4
       (BC Left)                          (BC Right)

    CELL-CENTERED (index i = 0 to N-1):
        tau[i], rho[i], p[i], T[i], e[i], E[i], c[i]

    FACE-CENTERED (index j = 0 to N):
        x[j], u[j], m[j]
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple
import numpy as np


@dataclass
class GridConfig:
    """
    Configuration for grid initialization.

    Attributes:
        n_cells: Number of computational cells
        x_min: Left boundary position [m]
        x_max: Right boundary position [m]
        stretch_factor: Grid stretching factor (1.0 = uniform)
    """

    n_cells: int
    x_min: float
    x_max: float
    stretch_factor: float = 1.0

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.n_cells < 1:
            raise ValueError(f"n_cells must be >= 1, got {self.n_cells}")
        if self.x_max <= self.x_min:
            raise ValueError(
                f"x_max must be > x_min: {self.x_max} <= {self.x_min}"
            )
        if self.stretch_factor <= 0:
            raise ValueError(
                f"stretch_factor must be > 0, got {self.stretch_factor}"
            )


class LagrangianGrid:
    """
    Lagrangian grid that moves with the fluid.

    In a Lagrangian formulation, cell masses are fixed and positions evolve
    with the fluid velocity. This class manages the grid geometry and provides
    methods for updating positions while maintaining mass conservation.

    Key properties:
        - Cell masses (dm) are constant in time
        - Face positions (x) evolve with velocity
        - Mass coordinates (m) are fixed Lagrangian coordinates

    Reference: [Despres2017] Section 3.1
    """

    def __init__(self, config: GridConfig):
        """
        Initialize the Lagrangian grid.

        Args:
            config: Grid configuration parameters
        """
        self._config = config
        self._n_cells = config.n_cells
        self._n_faces = config.n_cells + 1

        # Initialize uniform grid positions
        if abs(config.stretch_factor - 1.0) < 1e-10:
            self._x = np.linspace(config.x_min, config.x_max, self._n_faces)
        else:
            self._x = self._stretched_grid(
                config.x_min, config.x_max, self._n_faces, config.stretch_factor
            )

        # Mass coordinates (will be set when state is initialized)
        self._m = np.zeros(self._n_faces)
        self._dm = np.zeros(self._n_cells)
        self._mass_initialized = False

    @staticmethod
    def _stretched_grid(
        x_min: float, x_max: float, n_points: int, factor: float
    ) -> np.ndarray:
        """
        Create a stretched grid with geometric progression.

        Args:
            x_min: Left boundary
            x_max: Right boundary
            n_points: Number of grid points
            factor: Stretch factor (>1 = finer near x_min)

        Returns:
            Array of grid positions
        """
        # Use tanh-based stretching for smooth distribution
        xi = np.linspace(0, 1, n_points)
        if factor > 1:
            # Concentrate points near x_min
            xi_stretched = np.tanh(factor * xi) / np.tanh(factor)
        else:
            # Concentrate points near x_max
            xi_stretched = 1 - np.tanh(factor * (1 - xi)) / np.tanh(factor)

        return x_min + (x_max - x_min) * xi_stretched

    @property
    def n_cells(self) -> int:
        """Number of cells."""
        return self._n_cells

    @property
    def n_faces(self) -> int:
        """Number of faces (cell interfaces)."""
        return self._n_faces

    @property
    def x(self) -> np.ndarray:
        """Face positions [m] (read-only view)."""
        return self._x.view()

    @property
    def x_cell(self) -> np.ndarray:
        """Cell center positions [m]."""
        return 0.5 * (self._x[:-1] + self._x[1:])

    @property
    def dx(self) -> np.ndarray:
        """Cell widths [m]."""
        return np.diff(self._x)

    @property
    def m(self) -> np.ndarray:
        """Cumulative mass at faces [kg] (read-only view)."""
        return self._m.view()

    @property
    def dm(self) -> np.ndarray:
        """Cell masses [kg] (read-only view)."""
        return self._dm.view()

    @property
    def total_mass(self) -> float:
        """Total mass in the domain [kg]."""
        return self._m[-1] - self._m[0]

    @property
    def x_min(self) -> float:
        """Current left boundary position [m]."""
        return self._x[0]

    @property
    def x_max(self) -> float:
        """Current right boundary position [m]."""
        return self._x[-1]

    @property
    def domain_length(self) -> float:
        """Current domain length [m]."""
        return self._x[-1] - self._x[0]

    def initialize_mass(self, rho: np.ndarray):
        """
        Initialize mass coordinates from density distribution.

        This should be called once at the start of the simulation.
        After initialization, cell masses remain fixed.

        Args:
            rho: Cell-centered density [kg/m³]
        """
        if len(rho) != self._n_cells:
            raise ValueError(
                f"rho must have length n_cells={self._n_cells}, got {len(rho)}"
            )

        # Compute cell masses: dm = rho * dx (per unit area in 1D)
        self._dm = rho * self.dx

        # Cumulative mass
        self._m[0] = 0.0
        self._m[1:] = np.cumsum(self._dm)

        self._mass_initialized = True

    def set_positions(self, x: np.ndarray, enforce_minimum: bool = True):
        """
        Set face positions directly.

        Used during time integration to update grid positions.

        Args:
            x: New face positions [m]
            enforce_minimum: If True, enforce minimum cell width at boundary
                            only (for porous BC stability)

        Raises:
            ValueError: If positions are not monotonically increasing
        """
        if len(x) != self._n_faces:
            raise ValueError(
                f"x must have length n_faces={self._n_faces}, got {len(x)}"
            )

        x_safe = x.copy()

        if enforce_minimum:
            # Only fix the boundary cell if it's about to collapse
            # This allows the simulation to continue while protecting
            # against complete cell collapse
            min_dx = 1e-10  # Very small but non-zero

            # Check first cell (left boundary)
            if x_safe[1] - x_safe[0] < min_dx:
                x_safe[1] = x_safe[0] + min_dx

            # Check last cell (right boundary)
            if x_safe[-1] - x_safe[-2] < min_dx:
                x_safe[-2] = x_safe[-1] - min_dx

        # Final check
        if np.any(np.diff(x_safe) <= 0):
            # Try to identify the problematic cell
            diffs = np.diff(x_safe)
            bad_idx = np.where(diffs <= 0)[0]
            raise ValueError(
                f"Face positions must be strictly increasing. "
                f"Problem at cell(s): {bad_idx}, dx values: {diffs[bad_idx]}"
            )

        self._x[:] = x_safe

    def update_positions(self, u: np.ndarray, dt: float):
        """
        Update face positions based on velocity.

        dx/dt = u (Lagrangian position update)

        Args:
            u: Face velocities [m/s]
            dt: Time step [s]

        Reference: [Despres2017] Equation (3.5)
        """
        if len(u) != self._n_faces:
            raise ValueError(
                f"u must have length n_faces={self._n_faces}, got {len(u)}"
            )

        x_new = self._x + dt * u

        # Check that cells don't collapse
        if np.any(np.diff(x_new) <= 0):
            raise ValueError(
                "Cell collapse detected: positions would become non-monotonic"
            )

        self._x[:] = x_new

    def compute_specific_volume(self) -> np.ndarray:
        """
        Compute specific volume from current geometry and fixed masses.

        tau = dx / dm

        Returns:
            Cell-centered specific volume [m³/kg]
        """
        if not self._mass_initialized:
            raise RuntimeError("Mass not initialized. Call initialize_mass first.")

        return self.dx / self._dm

    def compute_density(self) -> np.ndarray:
        """
        Compute density from current geometry and fixed masses.

        rho = dm / dx = 1 / tau

        Returns:
            Cell-centered density [kg/m³]
        """
        return 1.0 / self.compute_specific_volume()

    def get_cfl_timestep(
        self, u: np.ndarray, c: np.ndarray, cfl: float = 0.5
    ) -> float:
        """
        Compute CFL-limited time step.

        In Lagrangian coordinates:
            dt <= CFL * min(dm / (rho * (|u_cell| + c)))

        But since dm = rho * dx, this simplifies to:
            dt <= CFL * min(dx / (|u_cell| + c))

        Reference: [Toro2009] Section 6.3

        Args:
            u: Face velocities [m/s]
            c: Cell sound speeds [m/s]
            cfl: CFL number (default 0.5 for 2nd order)

        Returns:
            Maximum stable time step [s]
        """
        # Cell-averaged velocity
        u_cell = 0.5 * np.abs(u[:-1] + u[1:])

        # Wave speed
        wave_speed = u_cell + c

        # Minimum ratio
        dx = self.dx
        dt_local = dx / np.maximum(wave_speed, 1e-10)

        return cfl * np.min(dt_local)

    def check_quality(self) -> Tuple[float, float, float]:
        """
        Check grid quality metrics.

        Returns:
            Tuple of (min_dx, max_dx, aspect_ratio)
            aspect_ratio = max_dx / min_dx (1.0 = uniform)
        """
        dx = self.dx
        min_dx = np.min(dx)
        max_dx = np.max(dx)
        aspect_ratio = max_dx / min_dx if min_dx > 0 else np.inf

        return min_dx, max_dx, aspect_ratio

    def to_dict(self) -> dict:
        """Export grid data to dictionary."""
        return {
            "n_cells": self._n_cells,
            "x": self._x.copy(),
            "m": self._m.copy(),
            "dm": self._dm.copy(),
        }

    def add_boundary_mass(self, side: "BoundarySide", dm: float) -> None:
        """
        Adjust boundary cell mass for porous BC.

        In a porous piston BC, mass can flow through the boundary when
        the gas velocity differs from the piston velocity. This method
        updates the boundary cell mass while maintaining consistency
        of the cumulative mass coordinate.

        Args:
            side: Which boundary (LEFT or RIGHT)
            dm: Mass change (positive = cell gains mass)

        Raises:
            ValueError: If boundary cell mass becomes non-positive

        Reference: Used by PorousGhostPistonBC for mass flux tracking
        """
        from lagrangian_solver.boundary.base import BoundarySide

        if side == BoundarySide.LEFT:
            self._dm[0] += dm
            # Update cumulative mass for all faces after cell 0
            self._m[1:] += dm
        else:
            self._dm[-1] += dm
            # Cumulative mass at right boundary doesn't change
            # (cumulative is measured from left)

        # Validate positive mass
        if side == BoundarySide.LEFT and self._dm[0] <= 0:
            raise ValueError(
                f"Left boundary cell mass became non-positive: {self._dm[0]}"
            )
        if side == BoundarySide.RIGHT and self._dm[-1] <= 0:
            raise ValueError(
                f"Right boundary cell mass became non-positive: {self._dm[-1]}"
            )

    @classmethod
    def from_dict(cls, data: dict) -> "LagrangianGrid":
        """Create grid from dictionary data."""
        n_cells = data["n_cells"]
        x = data["x"]

        config = GridConfig(
            n_cells=n_cells,
            x_min=x[0],
            x_max=x[-1],
        )

        grid = cls(config)
        grid._x[:] = x
        grid._m[:] = data["m"]
        grid._dm[:] = data["dm"]
        grid._mass_initialized = True

        return grid
