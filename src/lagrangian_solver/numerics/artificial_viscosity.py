"""
Artificial viscosity for shock capturing in Lagrangian hydrodynamics.

Implements the Von Neumann-Richtmyer quadratic viscosity with Landshoff
linear term, plus Noh's artificial heat conduction to prevent wall heating.

Wall Heating Problem:
    When artificial viscosity appears in the momentum equation but no
    corresponding artificial heat conduction appears in the energy equation,
    the system is thermodynamically inconsistent. This causes excess
    temperature accumulation at boundaries where shocks reflect (wall heating).

References:
    [Noh2001] Noh, W.F. "Errors for calculations of strong shocks using an
              artificial viscosity and an artificial heat flux"
              J. Comput. Physics 72(1), 78-120 (1987)
              See also: "Revisiting Wall Heating" J. Comput. Physics 169(1),
              405-407 (2001)

    [VonNeumannRichtmyer1950] Von Neumann, J. & Richtmyer, R.D. "A method for
              the numerical calculation of hydrodynamic shocks"
              J. Applied Physics 21, 232-237 (1950)

    [Landshoff1955] Landshoff, R. "A numerical method for treating fluid flow
              in the presence of shocks" LANL Report LA-1930 (1955)

    [Margolin2022] Margolin, L.G. "Artificial Viscosity - Then and Now"
              arXiv:2202.11084 (2022)

    [Toro2009] Toro, E.F. "Riemann Solvers and Numerical Methods for Fluid
              Dynamics" 3rd ed., Chapter 11 (Springer, 2009)
"""

from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from lagrangian_solver.core.state import FlowState
    from lagrangian_solver.core.grid import LagrangianGrid


@dataclass
class ArtificialViscosityConfig:
    """
    Configuration for artificial viscosity.

    The artificial viscous stress is:
        Q = rho * [c_quad * dx^2 * (du/dx)^2 + c_lin * c * dx * |du/dx|]
        for du/dx < 0 (compression), Q = 0 otherwise

    The artificial heat flux is (Noh's fix for wall heating):
        q = c_heat * rho * c * dx * dT/dx

    Attributes:
        c_quad: Quadratic viscosity coefficient (shock capturing)
                Typical value: 2.0 (spreads shock over ~3 cells)
        c_lin: Linear viscosity coefficient (post-shock oscillation damping)
               Typical value: 0.5
        c_heat: Artificial heat conduction coefficient (wall heating fix)
                Typical value: 0.1-1.0 (depends on application)
        enabled: Whether artificial viscosity is active

    Reference:
        [Margolin2022] Section 2.1 for coefficient recommendations
    """

    c_quad: float = 2.0
    c_lin: float = 0.5
    c_heat: float = 0.1
    enabled: bool = True


class ArtificialViscosity:
    """
    Computes artificial viscosity and heat conduction for shock capturing.

    The artificial viscosity serves two purposes:
    1. Spread shocks over several cells to prevent Gibbs oscillations
    2. Convert kinetic energy to internal energy at shocks (entropy production)

    The artificial heat conduction (Noh's fix) serves to:
    1. Redistribute heat away from boundaries where shocks reflect
    2. Make the system thermodynamically consistent

    Reference:
        [Noh2001] for wall heating physics
        [VonNeumannRichtmyer1950] for quadratic viscosity
        [Landshoff1955] for linear viscosity
    """

    def __init__(self, config: Optional[ArtificialViscosityConfig] = None):
        """
        Initialize artificial viscosity calculator.

        Args:
            config: Configuration parameters (uses defaults if None)
        """
        self._config = config or ArtificialViscosityConfig()

    @property
    def config(self) -> ArtificialViscosityConfig:
        """Configuration parameters."""
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

        Von Neumann-Richtmyer quadratic viscosity + Landshoff linear term:

            Q_i = rho_i * [c_Q * dx_i^2 * (du/dx)_i^2 + c_L * c_i * dx_i * |du/dx|_i]

        for (du/dx)_i < 0 (compression), Q_i = 0 otherwise.

        The velocity gradient is computed as:
            (du/dx)_i = (u_{i+1} - u_i) / dx_i

        Args:
            state: Current flow state
            grid: Lagrangian grid

        Returns:
            Viscous stress Q for each cell [Pa], shape (n_cells,)

        Reference:
            [VonNeumannRichtmyer1950] Equation (3)
            [Landshoff1955] Equation (12)
            [Toro2009] Section 11.3
        """
        if not self._config.enabled:
            return np.zeros(state.n_cells)

        n_cells = state.n_cells
        Q = np.zeros(n_cells)

        # Get cell widths
        dx = state.dx

        # Velocity gradient: du/dx = (u_{i+1} - u_i) / dx_i
        # u is face-centered, so u[i] is left face of cell i, u[i+1] is right face
        du = state.u[1:] - state.u[:-1]  # shape (n_cells,)
        du_dx = du / dx  # velocity gradient in each cell

        # Only apply in compression (du/dx < 0)
        compression = du_dx < 0

        # Von Neumann-Richtmyer quadratic term: rho * c_Q * dx^2 * (du/dx)^2
        Q_quad = (
            self._config.c_quad
            * state.rho
            * dx**2
            * du_dx**2
        )

        # Landshoff linear term: rho * c_L * c * dx * |du/dx|
        Q_lin = (
            self._config.c_lin
            * state.rho
            * state.c
            * dx
            * np.abs(du_dx)
        )

        # Total viscous stress (only in compression)
        Q[compression] = Q_quad[compression] + Q_lin[compression]

        return Q

    def compute_artificial_heat(
        self, state: "FlowState", grid: "LagrangianGrid"
    ) -> np.ndarray:
        """
        Compute artificial heat flux at cell faces.

        Noh's artificial heat conduction to prevent wall heating:

            q_{i+1/2} = c_H * rho_{i+1/2} * c_{i+1/2} * dx_{avg} * (T_{i+1} - T_i) / dx_{avg}
                      = c_H * rho_{i+1/2} * c_{i+1/2} * (T_{i+1} - T_i)

        This redistributes heat from hot regions (like the piston face after
        shock reflection) to cooler regions, eliminating the wall heating error.

        Args:
            state: Current flow state
            grid: Lagrangian grid

        Returns:
            Heat flux at each face [W/m^2], shape (n_faces,)
            Positive flux means heat flows in +x direction

        Reference:
            [Noh2001] Section 5 - Artificial heat flux
        """
        if not self._config.enabled or self._config.c_heat <= 0:
            return np.zeros(state.n_faces)

        n_faces = state.n_faces
        q_heat = np.zeros(n_faces)

        # Interior faces (1 to n_faces-2)
        for i in range(1, n_faces - 1):
            # Face i is between cells i-1 and i
            # Average properties at face
            rho_face = 0.5 * (state.rho[i - 1] + state.rho[i])
            c_face = 0.5 * (state.c[i - 1] + state.c[i])

            # Temperature gradient
            dT = state.T[i] - state.T[i - 1]

            # Cell widths on either side
            dx_avg = 0.5 * (state.dx[i - 1] + state.dx[i])

            # Artificial heat flux: q = c_H * rho * c * dT
            # (simplified form, dx terms cancel when used consistently)
            q_heat[i] = self._config.c_heat * rho_face * c_face * dT

        # Boundary faces: extrapolate or set to zero
        # For adiabatic boundaries, q = 0 at walls
        # q_heat[0] and q_heat[-1] remain zero

        return q_heat

    def compute_viscous_work(
        self, state: "FlowState", grid: "LagrangianGrid", Q: np.ndarray
    ) -> np.ndarray:
        """
        Compute viscous work rate (power) for energy equation contribution.

        The viscous work is:
            W_Q = Q * du/dx = -Q * d(tau)/dt / tau

        where tau = 1/rho is specific volume.

        In the energy equation, this appears as:
            dE/dt = -d(pu)/dm + d(Qu)/dm + d(q)/dm

        Args:
            state: Current flow state
            grid: Lagrangian grid
            Q: Viscous stress from compute_viscous_stress()

        Returns:
            Viscous work rate for each cell [W/kg], shape (n_cells,)

        Reference:
            [Toro2009] Section 11.3.2
        """
        dx = state.dx

        # Velocity gradient
        du_dx = (state.u[1:] - state.u[:-1]) / dx

        # Viscous work: Q * du/dx (converts kinetic to internal energy)
        return Q * du_dx
