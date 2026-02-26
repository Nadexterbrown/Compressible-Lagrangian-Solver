"""
Lagrangian conservation laws and flux computations.

This module implements the 1D compressible Euler equations in Lagrangian
(mass) coordinates.

References:
    [Despres2017] Chapter 3 - Lagrangian formulation
    [Toro2009] Chapter 1 - Euler equations

Governing Equations in Lagrangian Mass Coordinates (m, t):

    Mass Conservation (specific volume):
        ∂τ/∂t - ∂u/∂m = 0

    Momentum Conservation:
        ∂u/∂t + ∂p/∂m = 0

    Energy Conservation:
        ∂E/∂t + ∂(pu)/∂m = 0

    Position Update:
        ∂x/∂t = u

where:
    τ = 1/ρ is specific volume [m³/kg]
    u is velocity [m/s]
    E = e + ½u² is total specific energy [J/kg]
    p is pressure [Pa]
    m is mass coordinate [kg]
"""

from dataclasses import dataclass
from typing import Tuple
import numpy as np

from typing import TYPE_CHECKING

from lagrangian_solver.core.state import FlowState, ConservedVariables
from lagrangian_solver.core.grid import LagrangianGrid
from lagrangian_solver.equations.eos import EOSBase
from lagrangian_solver.numerics.riemann import (
    RiemannSolverBase,
    RiemannState,
)

if TYPE_CHECKING:
    from lagrangian_solver.numerics.artificial_viscosity import ArtificialViscosity


@dataclass
class LagrangianFlux:
    """
    Numerical fluxes at cell faces for Lagrangian equations.

    Reference: [Despres2017] Section 3.3

    Attributes:
        p_flux: Pressure at faces (momentum flux) [Pa]
        pu_flux: Pressure × velocity at faces (energy flux) [W/m² = Pa·m/s]
        u_flux: Velocity at faces (position update flux) [m/s]
    """

    p_flux: np.ndarray  # Shape (n_faces,)
    pu_flux: np.ndarray  # Shape (n_faces,)
    u_flux: np.ndarray  # Shape (n_faces,)


class LagrangianConservation:
    """
    Computes fluxes and residuals for Lagrangian conservation equations.

    The semi-discrete form of the equations is:

        dτ_i/dt = (u_{i+1} - u_i) / dm_i
        du_i/dt = -((p + Q)_{i+1} - (p + Q)_i) / dm_i   (face-centered)
        dE_i/dt = -(p_{i+1}u_{i+1} - p_i u_i) / dm_i

    where i is the cell index and face i is at the left of cell i.
    Q is the artificial viscous stress (Von Neumann-Richtmyer).

    Reference: [Despres2017] Section 3.4, [VNR1950]
    """

    def __init__(
        self,
        eos: EOSBase,
        riemann_solver: RiemannSolverBase,
        artificial_viscosity: "ArtificialViscosity" = None,
    ):
        """
        Initialize conservation law solver.

        Args:
            eos: Equation of state
            riemann_solver: Riemann solver for computing interface fluxes
            artificial_viscosity: Optional artificial viscosity for shock capturing
        """
        self._eos = eos
        self._riemann_solver = riemann_solver
        self._artificial_viscosity = artificial_viscosity

    @property
    def eos(self) -> EOSBase:
        """Equation of state."""
        return self._eos

    @property
    def riemann_solver(self) -> RiemannSolverBase:
        """Riemann solver for interface flux computation."""
        return self._riemann_solver

    @property
    def artificial_viscosity(self) -> "ArtificialViscosity":
        """Artificial viscosity for shock capturing (may be None)."""
        return self._artificial_viscosity

    def compute_fluxes(
        self, state: FlowState, grid: LagrangianGrid
    ) -> LagrangianFlux:
        """
        Compute numerical fluxes at all interior faces using Riemann solver.

        Boundary fluxes must be set separately using boundary conditions.

        Args:
            state: Current flow state
            grid: Lagrangian grid

        Returns:
            LagrangianFlux with pressure, energy, and velocity fluxes
        """
        n_cells = state.n_cells
        n_faces = n_cells + 1

        # Initialize flux arrays
        p_flux = np.zeros(n_faces)
        pu_flux = np.zeros(n_faces)
        u_flux = np.zeros(n_faces)

        # Compute fluxes at interior faces (1 to n_cells-1)
        for i in range(1, n_cells):
            # Left state (cell i-1)
            left = RiemannState(
                rho=state.rho[i - 1],
                u=state.u[i],  # Approximate: use face velocity
                p=state.p[i - 1],
                c=state.c[i - 1],
                e=state.e[i - 1],
            )

            # Right state (cell i)
            right = RiemannState(
                rho=state.rho[i],
                u=state.u[i],  # Approximate: use face velocity
                p=state.p[i],
                c=state.c[i],
                e=state.e[i],
            )

            # Solve Riemann problem to get interface pressure and velocity
            p_star, u_star = self._riemann_solver.compute_flux(left, right)

            p_flux[i] = p_star
            u_flux[i] = u_star  # Direct assignment - no division needed
            pu_flux[i] = p_star * u_star

        return LagrangianFlux(p_flux=p_flux, pu_flux=pu_flux, u_flux=u_flux)

    def compute_residual(
        self, state: FlowState, grid: LagrangianGrid, fluxes: LagrangianFlux
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the right-hand side of the semi-discrete equations.

        Returns the time derivatives:
            dτ/dt, du/dt, dE/dt, dx/dt

        When artificial viscosity is enabled, the momentum equation becomes:
            du/dt = -∂(p + Q)/∂m

        where Q is the Von Neumann-Richtmyer viscous stress.

        Reference: [Despres2017] Equations (3.15)-(3.17), [VNR1950]

        Args:
            state: Current flow state
            grid: Lagrangian grid
            fluxes: Pre-computed fluxes (including boundary fluxes)

        Returns:
            Tuple of (d_tau, d_u, d_E, d_x) arrays
        """
        n_cells = state.n_cells
        dm = grid.dm

        # Specific volume rate: dτ/dt = (u_{i+1} - u_i) / dm_i
        d_tau = np.zeros(n_cells)
        for i in range(n_cells):
            d_tau[i] = (fluxes.u_flux[i + 1] - fluxes.u_flux[i]) / dm[i]

        # Compute artificial viscous stress if enabled
        # Reference: [VNR1950], [Toro2009] Section 11.1
        if self._artificial_viscosity is not None:
            Q = self._artificial_viscosity.compute_viscous_stress(state, grid)
        else:
            Q = np.zeros(n_cells)

        # Total stress = pressure + artificial viscosity
        stress = state.p + Q

        # Velocity rate at faces: du/dt = -(stress_i - stress_{i-1}) / dm_face
        # Uses cell-centered stresses, not face pressures
        # For boundary faces, this is handled by boundary conditions
        d_u = np.zeros(state.n_faces)
        for i in range(1, n_cells):
            dm_avg = 0.5 * (dm[i - 1] + dm[i])
            d_u[i] = -(stress[i] - stress[i - 1]) / dm_avg

        # Energy rate: dE/dt = -(pu_{i+1} - pu_i) / dm_i
        #
        # NOTE: The artificial viscosity Q affects momentum but the energy
        # dissipation is handled implicitly through the velocity changes.
        # Adding explicit Q*u terms to the energy flux can cause instabilities.
        #
        # For now, use only the Riemann pressure flux for energy.
        # TODO: Implement compatible energy discretization (Burton 1992) for
        # proper AV-energy coupling.
        d_E = np.zeros(n_cells)
        for i in range(n_cells):
            d_E[i] = -(fluxes.pu_flux[i + 1] - fluxes.pu_flux[i]) / dm[i]

        # Position rate: dx/dt = u
        d_x = fluxes.u_flux.copy()

        return d_tau, d_u, d_E, d_x

    def compute_residual_direct(
        self, state: FlowState, grid: LagrangianGrid
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute residual using cell-centered pressure differences.

        This is a simpler approach that uses cell pressures directly
        rather than solving Riemann problems at each face.

        When artificial viscosity is enabled, the momentum equation becomes:
            du/dt = -∂(p + Q)/∂m

        Suitable for smooth flows without strong discontinuities.

        Args:
            state: Current flow state
            grid: Lagrangian grid

        Returns:
            Tuple of (d_tau, d_u, d_E, d_x) arrays
        """
        n_cells = state.n_cells
        dm = grid.dm

        # Velocity at faces
        u = state.u

        # Specific volume rate: dτ/dt = (u_{i+1} - u_i) / dm_i
        d_tau = (u[1:] - u[:-1]) / dm

        # Compute artificial viscous stress if enabled
        if self._artificial_viscosity is not None:
            Q = self._artificial_viscosity.compute_viscous_stress(state, grid)
        else:
            Q = np.zeros(n_cells)

        # Total stress = pressure + artificial viscosity
        stress = state.p + Q

        # Velocity rate at faces (interior only, boundaries set separately)
        d_u = np.zeros(state.n_faces)

        # For interior faces, use central difference on stress
        for i in range(1, n_cells):
            dm_face = 0.5 * (dm[i - 1] + dm[i])
            d_u[i] = -(stress[i] - stress[i - 1]) / dm_face

        # Energy rate using face velocities and pressures
        # NOTE: AV not included in energy flux - see compute_residual for explanation
        # Interpolate pressure to faces
        p_face = np.zeros(state.n_faces)
        p_face[0] = state.p[0]
        p_face[-1] = state.p[-1]
        p_face[1:-1] = 0.5 * (state.p[:-1] + state.p[1:])

        # dE/dt = -(pu_{i+1} - pu_i) / dm_i
        pu_face = p_face * u
        d_E = -(pu_face[1:] - pu_face[:-1]) / dm

        # Position rate: dx/dt = u
        d_x = u.copy()

        return d_tau, d_u, d_E, d_x


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


def compute_total_energy(state: FlowState) -> float:
    """
    Compute total energy in the domain.

    Total energy = ∫ ρE dx

    Args:
        state: Current flow state

    Returns:
        Total energy [J]
    """
    return np.sum(state.rho * state.E * state.dx)


def compute_kinetic_energy(state: FlowState) -> float:
    """
    Compute total kinetic energy in the domain.

    Kinetic energy = ∫ ½ρu² dx

    Args:
        state: Current flow state

    Returns:
        Kinetic energy [J]
    """
    u_cell = 0.5 * (state.u[:-1] + state.u[1:])
    return np.sum(0.5 * state.rho * u_cell**2 * state.dx)


def compute_internal_energy(state: FlowState) -> float:
    """
    Compute total internal energy in the domain.

    Internal energy = ∫ ρe dx

    Args:
        state: Current flow state

    Returns:
        Internal energy [J]
    """
    return np.sum(state.rho * state.e * state.dx)
