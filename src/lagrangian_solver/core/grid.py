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

    def set_positions(self, x: np.ndarray):
        """
        Set face positions directly.

        Used during time integration to update grid positions.
        Note: For resizing the grid (changing number of cells), use resize() instead.

        Args:
            x: New face positions [m]

        Raises:
            ValueError: If positions are not monotonically increasing
            ValueError: If array size doesn't match current grid size
        """
        # Check monotonicity
        if np.any(np.diff(x) <= 0):
            raise ValueError("Face positions must be strictly increasing")

        # Require exact size match - use resize() for changing cell count
        if len(x) != self._n_faces:
            raise ValueError(
                f"Position array size {len(x)} doesn't match grid size {self._n_faces}. "
                f"Use grid.resize() to change the number of cells."
            )

        self._x[:] = x

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

    def copy(self) -> "LagrangianGrid":
        """
        Create a deep copy of the grid.

        Returns:
            New LagrangianGrid with copied data
        """
        config = GridConfig(
            n_cells=self._n_cells,
            x_min=self._x[0],
            x_max=self._x[-1],
        )

        new_grid = LagrangianGrid(config)
        new_grid._x = self._x.copy()
        new_grid._m = self._m.copy()
        new_grid._dm = self._dm.copy()
        new_grid._mass_initialized = self._mass_initialized

        return new_grid

    def resize(self, n_cells: int, x: np.ndarray, m: np.ndarray):
        """
        Resize the grid to a new number of cells.

        Used by adaptive mesh refinement when splitting/merging cells.

        Args:
            n_cells: New number of cells
            x: New face positions [m] (length n_cells + 1)
            m: New cumulative mass at faces [kg] (length n_cells + 1)
        """
        n_faces = n_cells + 1

        if len(x) != n_faces:
            raise ValueError(f"x must have length {n_faces}, got {len(x)}")
        if len(m) != n_faces:
            raise ValueError(f"m must have length {n_faces}, got {len(m)}")

        self._n_cells = n_cells
        self._n_faces = n_faces
        self._x = x.copy()
        self._m = m.copy()
        self._dm = np.diff(m)
        self._mass_initialized = True

        # Verify consistency
        if len(self._dm) != self._n_cells:
            raise RuntimeError(
                f"dm array size mismatch after resize: "
                f"len(dm)={len(self._dm)}, n_cells={self._n_cells}"
            )
