"""
Linear and quadratic artificial viscosity for shock capturing.

This module implements combined linear (Landshoff) and quadratic (Von Neumann-
Richtmyer) artificial viscosity for capturing shocks in Lagrangian hydrodynamics
simulations.

References:
    [VNR1950] Von Neumann, J. & Richtmyer, R.D. (1950). "A method for the
              numerical calculation of hydrodynamic shocks." Journal of
              Applied Physics, 21(3), 232-237.
    [Landshoff1955] Landshoff, R. (1955). "A numerical method for treating
              fluid flow in the presence of shocks." Los Alamos Scientific
              Laboratory Report LA-1930.
    [Toro2009] Chapter 11 - Artificial viscosity methods

The artificial viscosity Q acts as an additional pressure in compression
zones, spreading the shock discontinuity over several grid cells while
ensuring proper entropy production.

The quadratic term (VNR) controls overall shock width and entropy production.
The linear term (Landshoff) damps high-frequency oscillations at the shock head.

Mathematical formulation:
    Q = c_1 * rho * c * dx * |du/dx| + c_2 * rho * dx^2 * (du/dx)^2
    for du/dx < 0 (compression), Q = 0 otherwise

where:
    c_1 = linear coefficient (~0.3, damps shock-head ringing)
    c_2 = quadratic coefficient (~2.0, spreads shock over ~3 cells)
    rho = density [kg/m^3]
    c = sound speed [m/s]
    dx = cell width [m]
    du/dx = velocity gradient [1/s]
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from lagrangian_solver.core.state import FlowState
    from lagrangian_solver.core.grid import LagrangianGrid


@dataclass
class ArtificialViscosityConfig:
    """
    Configuration for artificial viscosity.

    Attributes:
        c_linear: Linear viscosity coefficient (Landshoff, default 0.3).
                  Damps high-frequency oscillations at shock head.
                  Typical range: 0.06 - 0.5
        c_quad: Quadratic viscosity coefficient (VNR, default 2.0).
                Higher values spread the shock over more cells.
                Typical range: 1.0 - 4.0
        enabled: Whether artificial viscosity is active (default True).
    """

    c_linear: float = 0.3  # Landshoff linear term for shock-head damping
    c_quad: float = 2.0
    enabled: bool = True

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.c_linear < 0:
            raise ValueError(f"c_linear must be non-negative, got {self.c_linear}")
        if self.c_quad < 0:
            raise ValueError(f"c_quad must be non-negative, got {self.c_quad}")


class ArtificialViscosity:
    """
    Computes linear (Landshoff) + quadratic (VNR) artificial viscosity.

    The viscous stress Q is computed for each cell and added to the pressure
    in the momentum equation. This spreads shock discontinuities over several
    cells while dissipating kinetic energy into internal energy (entropy
    production), which is physically correct for shock waves.

    The linear term damps high-frequency oscillations at the shock head that
    the quadratic term alone cannot eliminate. This is the well-documented
    Landshoff correction to the original VNR method.

    Reference: [VNR1950], [Landshoff1955], [Toro2009] Section 11.1
    """

    def __init__(self, config: ArtificialViscosityConfig = None):
        """
        Initialize artificial viscosity calculator.

        Args:
            config: Viscosity configuration (default creates standard config)
        """
        self._config = config or ArtificialViscosityConfig()

    @property
    def config(self) -> ArtificialViscosityConfig:
        """Viscosity configuration."""
        return self._config

    @property
    def enabled(self) -> bool:
        """Whether artificial viscosity is enabled."""
        return self._config.enabled

    def compute_viscous_stress(
        self, state: "FlowState", grid: "LagrangianGrid"
    ) -> np.ndarray:
        """
        Compute artificial viscous stress Q for each cell.

        Combined linear (Landshoff) + quadratic (VNR) viscosity:
            Q = c_1 * rho * c * dx * |du/dx| + c_2 * rho * dx^2 * (du/dx)^2
            for du/dx < 0 (compression), Q = 0 otherwise

        The linear term damps shock-head oscillations; the quadratic term
        controls shock width and entropy production.

        Reference: [VNR1950] Eq. (8), [Landshoff1955], [Toro2009] Sec. 11.1

        Args:
            state: Current flow state with density, velocity, and sound speed
            grid: Lagrangian grid with cell widths

        Returns:
            Cell-centered viscous stress Q [Pa], shape (n_cells,)
        """
        if not self._config.enabled:
            return np.zeros(state.n_cells)

        n_cells = state.n_cells
        Q = np.zeros(n_cells)

        # Get cell properties
        rho = state.rho  # Cell-centered density
        c = state.c  # Cell-centered sound speed
        dx = grid.dx  # Cell widths
        u = state.u  # Face-centered velocity

        # Compute velocity gradient in each cell
        # du/dx = (u_{i+1} - u_i) / dx_i
        # where u_i is at left face, u_{i+1} is at right face
        du_dx = (u[1:] - u[:-1]) / dx

        # Apply viscosity only in compression (du/dx < 0)
        compression_mask = du_dx < 0

        # Get coefficients
        c_1 = self._config.c_linear
        c_2 = self._config.c_quad

        # Linear term (Landshoff): Q_lin = c_1 * rho * c * dx * |du/dx|
        # Damps high-frequency ringing at shock head
        Q_linear = c_1 * rho * c * dx * np.abs(du_dx)

        # Quadratic term (VNR): Q_quad = c_2 * rho * dx^2 * (du/dx)^2
        # Controls shock width and overall entropy production
        Q_quad = c_2 * rho * dx**2 * du_dx**2

        # Combined viscosity in compression zones only
        Q[compression_mask] = Q_linear[compression_mask] + Q_quad[compression_mask]

        return Q
