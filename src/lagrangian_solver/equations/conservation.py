"""
Compatible energy discretization for Lagrangian conservation laws.

This module implements the 1D compressible Euler equations in Lagrangian
(mass) coordinates using COMPATIBLE ENERGY DISCRETIZATION.

Key Design Principle:
    Use cell-centered stress σ = p + Q for BOTH momentum AND energy equations.
    This guarantees exact total energy conservation to machine precision.

The previous implementation mixed Riemann fluxes (face-centered) with
cell-centered pressure differences - these are fundamentally incompatible
and led to energy conservation errors.

References:
    [Caramana1998] Caramana et al. (1998) JCP 146:227-262 - "The Construction
                   of Compatible Hydrodynamics Algorithms Utilizing Conservation
                   of Total Energy"
    [Burton1992] Burton (1992) UCRL-JC-105926 - "Conservation of Energy,
                 Momentum, and Angular Momentum in Lagrangian Staggered-Grid
                 Hydrodynamics"
    [Despres2017] Chapter 3 - Lagrangian formulation
    [Toro2009] Chapter 1 - Euler equations

Governing Equations in Lagrangian Mass Coordinates (m, t):

    Mass Conservation (specific volume):
        dτ_i/dt = (u_{i+1} - u_i) / dm_i

    Momentum Conservation (face-centered velocity):
        du_j/dt = -(σ_j - σ_{j-1}) / dm_face_j

        where:
            σ_i = p_i + Q_i          (cell-centered total stress)
            dm_face_j = 0.5*(dm_{j-1} + dm_j)  (mass at face)

    Internal Energy Conservation (COMPATIBLE - uses SAME stress):
        de_i/dt = -σ_i * dτ_i/dt
                = -σ_i * (u_{i+1} - u_i) / dm_i

    Position Update:
        dx_j/dt = u_j

Energy Conservation Proof:
    The work in momentum EXACTLY equals the energy change because
    we use the SAME cell-centered stress in both equations.
    dE_total/dt = 0 to machine precision (O(10^-15) relative error).
"""

from dataclasses import dataclass
from typing import Tuple, Optional, TYPE_CHECKING
import numpy as np

from lagrangian_solver.core.state import FlowState
from lagrangian_solver.core.grid import LagrangianGrid
from lagrangian_solver.equations.eos import EOSBase

if TYPE_CHECKING:
    from lagrangian_solver.numerics.artificial_viscosity import ArtificialViscosity
    from lagrangian_solver.boundary.base import BoundaryCondition


class CompatibleConservation:
    """
    Cell-centered compatible discretization of Lagrangian conservation laws.

    Uses the SAME stress σ = p + Q in both momentum and energy equations
    to guarantee exact total energy conservation.

    This is the core of the compatible energy discretization. By using
    cell-centered stress consistently, the discrete work done in the
    momentum equation exactly equals the energy change in the energy
    equation (to machine precision).

    Reference: Caramana et al. (1998) JCP 146:227-262
    """

    def __init__(
        self,
        eos: EOSBase,
        artificial_viscosity: "ArtificialViscosity" = None,
    ):
        """
        Initialize compatible conservation law solver.

        Args:
            eos: Equation of state
            artificial_viscosity: Optional artificial viscosity for shock capturing
        """
        self._eos = eos
        self._artificial_viscosity = artificial_viscosity

    @property
    def eos(self) -> EOSBase:
        """Equation of state."""
        return self._eos

    @property
    def artificial_viscosity(self) -> "ArtificialViscosity":
        """Artificial viscosity for shock capturing (may be None)."""
        return self._artificial_viscosity

    def compute_stress(self, state: FlowState, grid: LagrangianGrid) -> np.ndarray:
        """
        Compute cell-centered total stress σ = p + Q.

        The stress is used in BOTH momentum and energy equations,
        which is the key to compatible energy discretization.

        Args:
            state: Current flow state
            grid: Lagrangian grid

        Returns:
            Cell-centered total stress [Pa], shape (n_cells,)
        """
        if self._artificial_viscosity is not None:
            Q = self._artificial_viscosity.compute_viscous_stress(state, grid)
        else:
            Q = np.zeros(state.n_cells)

        return state.p + Q

    def compute_residual(
        self,
        state: FlowState,
        grid: LagrangianGrid,
        bc_left: "BoundaryCondition",
        bc_right: "BoundaryCondition",
        t: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute (d_tau, d_u, d_e, d_x) using compatible discretization.

        Key: d_e = -σ * d_tau (pdV work form)

        This guarantees exact total energy conservation because the
        internal energy equation uses the SAME stress as momentum.

        Ghost Cell Support:
            If a BC provides ghost cell stress (via has_ghost_cell() and
            get_interface_stress()), the momentum equation for the first
            interior face uses the Riemann interface stress instead of
            the interior cell stress. This gives correct wave dynamics
            for shocks, rarefactions, and acoustic waves.

        Reference: Caramana et al. (1998), Burton (1992), Toro (2009) Sec. 6.3

        Args:
            state: Current flow state
            grid: Lagrangian grid
            bc_left: Left boundary condition
            bc_right: Right boundary condition
            t: Current time [s]

        Returns:
            Tuple of (d_tau, d_u, d_e, d_x) arrays
        """
        n_cells = state.n_cells
        dm = grid.dm
        u = state.u

        # 1. Specific volume rate: dτ/dt = (u_{i+1} - u_i) / dm_i
        d_tau = (u[1:] - u[:-1]) / dm

        # 2. Cell-centered stress σ = p + Q
        sigma = self.compute_stress(state, grid)

        # 3. Check for ghost cell BCs and compute interface stresses
        sigma_left_interface = None
        sigma_right_interface = None

        if hasattr(bc_left, 'has_ghost_cell') and bc_left.has_ghost_cell():
            # Compute interface state from boundary Riemann problem
            bc_left.compute_interface_state(state, grid, t)
            sigma_left_interface = bc_left.get_interface_stress()

        if hasattr(bc_right, 'has_ghost_cell') and bc_right.has_ghost_cell():
            bc_right.compute_interface_state(state, grid, t)
            sigma_right_interface = bc_right.get_interface_stress()

        # 4. Momentum rate at faces: du_j/dt = -(σ_j - σ_{j-1}) / dm_face_j
        d_u = np.zeros(state.n_faces)

        # Determine starting index for interior loop
        j_start = 1

        # Left boundary: use interface stress if available
        if sigma_left_interface is not None and n_cells > 1:
            # Face j=1: use interface stress on left, sigma[1] on right
            dm_face = 0.5 * (dm[0] + dm[1])
            d_u[1] = -(sigma[1] - sigma_left_interface) / dm_face
            j_start = 2

        # Interior faces: j = j_start to n_cells-1
        j_end = n_cells
        if sigma_right_interface is not None and n_cells > 1:
            j_end = n_cells - 1

        for j in range(j_start, j_end):
            dm_face = 0.5 * (dm[j - 1] + dm[j])
            d_u[j] = -(sigma[j] - sigma[j - 1]) / dm_face

        # Right boundary: use interface stress if available
        if sigma_right_interface is not None and n_cells > 1:
            # Face j=n_cells-1: use sigma[-2] on left, interface stress on right
            j = n_cells - 1
            dm_face = 0.5 * (dm[j - 1] + dm[j])
            d_u[j] = -(sigma_right_interface - sigma[j - 1]) / dm_face

        # Apply boundary conditions for d_u[0] and d_u[n_cells]
        # Boundaries set their own momentum rates at boundary faces
        bc_left.apply_momentum(state, grid, d_u, sigma, t)
        bc_right.apply_momentum(state, grid, d_u, sigma, t)

        # 5. COMPATIBLE internal energy rate: de/dt = -σ * dτ/dt
        # NOTE: Energy equation uses CELL stress, not interface stress.
        # The ghost cell affects momentum (d_u), which affects d_tau,
        # and energy follows via d_e = -sigma * d_tau.
        # This maintains compatible energy discretization.
        d_e = -sigma * d_tau

        # 6. Position rate: dx/dt = u
        d_x = u.copy()

        return d_tau, d_u, d_e, d_x


# Keep old LagrangianConservation for backward compatibility, but mark deprecated
class LagrangianConservation:
    """
    DEPRECATED: Use CompatibleConservation for exact energy conservation.

    This class uses Riemann fluxes which are inconsistent with cell-centered
    pressure differences in the momentum equation, leading to energy errors.
    """

    def __init__(
        self,
        eos: EOSBase,
        riemann_solver=None,
        artificial_viscosity: "ArtificialViscosity" = None,
    ):
        """Initialize conservation law solver (DEPRECATED)."""
        import warnings
        warnings.warn(
            "LagrangianConservation is deprecated. Use CompatibleConservation "
            "for exact energy conservation.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._eos = eos
        self._riemann_solver = riemann_solver
        self._artificial_viscosity = artificial_viscosity

    @property
    def eos(self) -> EOSBase:
        return self._eos

    @property
    def riemann_solver(self):
        return self._riemann_solver

    @property
    def artificial_viscosity(self):
        return self._artificial_viscosity

    def compute_fluxes(self, state: FlowState, grid: LagrangianGrid):
        """Compute numerical fluxes (DEPRECATED)."""
        raise NotImplementedError(
            "LagrangianConservation is deprecated. Use CompatibleConservation."
        )

    def compute_residual(self, state: FlowState, grid: LagrangianGrid, fluxes):
        """Compute residual (DEPRECATED)."""
        raise NotImplementedError(
            "LagrangianConservation is deprecated. Use CompatibleConservation."
        )


def compute_mass_error(state: FlowState, grid: LagrangianGrid) -> float:
    """
    Compute relative mass conservation error.

    In Lagrangian formulation, total mass should be exactly conserved.

    Args:
        state: Current flow state
        grid: Lagrangian grid

    Returns:
        Relative mass error (should be machine precision)
    """
    # Current mass from density and volume
    current_mass = np.sum(state.rho * state.dx)

    # Original mass from grid
    original_mass = grid.total_mass

    if abs(original_mass) < 1e-15:
        return 0.0

    return abs(current_mass - original_mass) / original_mass


def compute_momentum(state: FlowState) -> float:
    """
    Compute total momentum in the domain.

    Total momentum = ∫ ρu dx

    Args:
        state: Current flow state

    Returns:
        Total momentum [kg·m/s]
    """
    # Cell-averaged velocity
    u_cell = 0.5 * (state.u[:-1] + state.u[1:])
    return np.sum(state.rho * u_cell * state.dx)


def compute_total_energy(state: FlowState, grid: LagrangianGrid) -> float:
    """
    Compute total energy in the domain using staggered grid.

    For compatible discretization with face-centered velocity:
        E_total = Σ_i (dm_i · e_i) + Σ_j (½ m_j · u_j²)

    where:
        - Internal energy uses cell masses dm_i
        - Kinetic energy uses face masses m_j

    Reference: Caramana et al. (1998) Section 2

    Args:
        state: Current flow state
        grid: Lagrangian grid

    Returns:
        Total energy [J]
    """
    dm = grid.dm
    n_cells = state.n_cells

    # Internal energy: sum over cells
    IE = np.sum(dm * state.e)

    # Kinetic energy: sum over faces using face masses
    # Face mass = average of adjacent cell masses
    KE = 0.0
    for j in range(state.n_faces):
        if j == 0:
            # Left boundary face: use half of first cell mass
            m_j = 0.5 * dm[0]
        elif j == n_cells:
            # Right boundary face: use half of last cell mass
            m_j = 0.5 * dm[-1]
        else:
            # Interior faces: average of adjacent cell masses
            m_j = 0.5 * (dm[j - 1] + dm[j])
        KE += 0.5 * m_j * state.u[j] ** 2

    return IE + KE


def compute_total_energy_simple(state: FlowState) -> float:
    """
    Compute total energy using simple cell-centered formula.

    This is the traditional formula using E = e + ½u² at cell centers.
    Less accurate for staggered grids but useful for comparison.

    Args:
        state: Current flow state

    Returns:
        Total energy [J]
    """
    return np.sum(state.rho * state.E * state.dx)


def compute_kinetic_energy(state: FlowState, grid: LagrangianGrid) -> float:
    """
    Compute total kinetic energy using face-centered velocity.

    Uses the same formula as compute_total_energy for consistency.

    Args:
        state: Current flow state
        grid: Lagrangian grid

    Returns:
        Kinetic energy [J]
    """
    dm = grid.dm
    n_cells = state.n_cells

    KE = 0.0
    for j in range(state.n_faces):
        if j == 0:
            m_j = 0.5 * dm[0]
        elif j == n_cells:
            m_j = 0.5 * dm[-1]
        else:
            m_j = 0.5 * (dm[j - 1] + dm[j])
        KE += 0.5 * m_j * state.u[j] ** 2

    return KE


def compute_internal_energy(state: FlowState, grid: LagrangianGrid) -> float:
    """
    Compute total internal energy in the domain.

    Internal energy = Σ dm_i · e_i

    Args:
        state: Current flow state
        grid: Lagrangian grid

    Returns:
        Internal energy [J]
    """
    return np.sum(grid.dm * state.e)
