"""
Artificial heat conduction for contact discontinuity spreading in Lagrangian hydrodynamics.

Implements artificial thermal conductivity analogous to artificial viscosity.
While AV spreads shocks, artificial heat conduction spreads contact discontinuities
over 3-5 cells for numerical stability without excessive smearing.

References:
    [Noh1987] Noh, W.F. (1987). "Errors for calculations of strong shocks using
              an artificial viscosity and an artificial heat flux." Journal of
              Computational Physics, 72(1), 78-120.
    [VNR1950] Von Neumann & Richtmyer - original AV formulation (analogous approach)
    [Caramana1998] Compatible energy discretization

The artificial heat flux q is computed at cell faces and its divergence
is added to the internal energy equation:

    de/dt = -sigma * d_tau/dt - dq/dm

where:
    q_{j+1/2} = -kappa_{j+1/2} * (T_{j+1} - T_j) / (x_{j+1} - x_j)

and kappa is the artificial thermal conductivity that activates at
density gradients (the signature of contact discontinuities).

Mathematical formulation (analogous to AV):
    kappa_lin = kappa_1 * rho * c * dx * |d(ln rho)/dx|
    kappa_quad = kappa_2 * rho * dx^2 * |d(ln rho)/dx|^2

The linear term damps oscillations, the quadratic term controls spreading width.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from lagrangian_solver.core.state import FlowState
    from lagrangian_solver.core.grid import LagrangianGrid


@dataclass
class ArtificialHeatConductionConfig:
    """
    Configuration for artificial heat conduction.

    Attributes:
        kappa_linear: Linear coefficient (oscillation damping, default 0.1).
                      Analogous to Landshoff linear AV term.
                      Typical range: 0.05 - 0.3
        kappa_quad: Quadratic coefficient (contact width, default 0.5).
                    Analogous to VNR quadratic AV term.
                    Typical range: 0.1 - 1.0
        use_density_switch: If True, activate on density gradient (default).
                            If False, activate on temperature gradient.
        enabled: Whether artificial heat conduction is active (default False).
    """

    kappa_linear: float = 0.1
    kappa_quad: float = 0.5
    use_density_switch: bool = True
    enabled: bool = False

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.kappa_linear < 0:
            raise ValueError(f"kappa_linear must be non-negative, got {self.kappa_linear}")
        if self.kappa_quad < 0:
            raise ValueError(f"kappa_quad must be non-negative, got {self.kappa_quad}")


class ArtificialHeatConduction:
    """
    Computes artificial heat conduction for contact discontinuity spreading.

    The heat flux q is computed at cell faces and added to the energy
    equation as a conservative divergence. This spreads contact
    discontinuities (where density jumps but pressure is continuous)
    over several cells.

    The formulation uses density gradient as the activation switch
    because contacts are characterized by density jumps with constant
    pressure. This directly targets contact discontinuities without
    affecting shocks (which have pressure jumps).

    Energy conservation is maintained through the conservative formulation:
        d_e_heat = -(q[j+1/2] - q[j-1/2]) / dm

    Reference: [Noh1987], [Caramana1998]
    """

    def __init__(self, config: ArtificialHeatConductionConfig = None):
        """
        Initialize artificial heat conduction calculator.

        Args:
            config: Heat conduction configuration (default creates config with enabled=False)
        """
        self._config = config or ArtificialHeatConductionConfig()

    @property
    def config(self) -> ArtificialHeatConductionConfig:
        """Heat conduction configuration."""
        return self._config

    @property
    def enabled(self) -> bool:
        """Whether artificial heat conduction is enabled."""
        return self._config.enabled

    def compute_heat_flux(
        self, state: "FlowState", grid: "LagrangianGrid"
    ) -> np.ndarray:
        """
        Compute face-centered heat flux q [W/m^2].

        The heat flux is computed at each interior face:
            q_{j+1/2} = -kappa_{j+1/2} * (T_{j+1} - T_j) / (x_{j+1} - x_j)

        where the artificial thermal conductivity kappa is:
            kappa = kappa_lin + kappa_quad
            kappa_lin = kappa_1 * rho * c * dx * |d(ln rho)/dx|
            kappa_quad = kappa_2 * rho * dx^2 * |d(ln rho)/dx|^2

        The density gradient switch |d(ln rho)/dx| targets contacts where
        density jumps but pressure remains constant.

        Args:
            state: Current flow state with density, temperature, sound speed
            grid: Lagrangian grid with cell positions

        Returns:
            Face-centered heat flux [W/m^2], shape (n_faces,)
            Note: boundary faces (0 and n_cells) are set to zero (adiabatic)
        """
        if not self._config.enabled:
            return np.zeros(state.n_faces)

        n_cells = state.n_cells
        n_faces = state.n_faces
        q = np.zeros(n_faces)

        # Get cell properties
        rho = state.rho  # Cell-centered density
        T = state.T  # Cell-centered temperature
        c = state.c  # Cell-centered sound speed
        x_cell = grid.x_cell  # Cell centers

        # Get coefficients
        kappa_1 = self._config.kappa_linear
        kappa_2 = self._config.kappa_quad

        # Compute heat flux at interior faces (j = 1 to n_cells-1)
        for j in range(1, n_cells):
            # Left cell i = j-1, right cell i+1 = j
            rho_L = rho[j - 1]
            rho_R = rho[j]
            T_L = T[j - 1]
            T_R = T[j]
            c_L = c[j - 1]
            c_R = c[j]

            # Face position and spacing
            x_L = x_cell[j - 1]
            x_R = x_cell[j]
            dx_face = x_R - x_L

            if dx_face < 1e-15:
                continue  # Skip degenerate faces

            # Compute switch based on density gradient
            # |d(ln rho)/dx| = |1/rho * d_rho/dx| = |d_rho/dx| / rho_avg
            rho_avg = 0.5 * (rho_L + rho_R)
            d_rho_dx = (rho_R - rho_L) / dx_face
            switch = abs(d_rho_dx) / rho_avg  # |d(ln rho)/dx|

            # Face-averaged properties
            c_avg = 0.5 * (c_L + c_R)
            dx_avg = 0.5 * (grid.dx[j - 1] + grid.dx[j])

            # Artificial thermal conductivity at face
            # Linear term (oscillation damping)
            kappa_lin = kappa_1 * rho_avg * c_avg * dx_avg * switch

            # Quadratic term (contact width control)
            kappa_quad = kappa_2 * rho_avg * dx_avg**2 * switch**2

            kappa = kappa_lin + kappa_quad

            # Heat flux: q = -kappa * dT/dx
            dT_dx = (T_R - T_L) / dx_face
            q[j] = -kappa * dT_dx

        # Boundary faces: adiabatic (q = 0)
        q[0] = 0.0
        q[n_cells] = 0.0

        return q

    def compute_energy_source(
        self, state: "FlowState", grid: "LagrangianGrid"
    ) -> np.ndarray:
        """
        Compute cell-centered energy rate from heat conduction [J/(kg*s)].

        The energy source term is the conservative divergence of heat flux:
            d_e_heat = -dq/dm = -(q[j+1/2] - q[j-1/2]) / dm

        This formulation guarantees exact energy conservation because:
            Sum(d_e_heat * dm) = -(q[N-1/2] - q[1/2]) = 0 (for adiabatic BCs)

        Reference: [Noh1987], [Caramana1998]

        Args:
            state: Current flow state
            grid: Lagrangian grid

        Returns:
            Cell-centered energy source [J/(kg*s)], shape (n_cells,)
        """
        if not self._config.enabled:
            return np.zeros(state.n_cells)

        # Compute heat flux at faces
        q = self.compute_heat_flux(state, grid)

        # Conservative divergence: d_e = -(q[j+1] - q[j]) / dm
        # For cell i, left face is i, right face is i+1
        dm = grid.dm
        d_e_heat = -(q[1:] - q[:-1]) / dm

        return d_e_heat
