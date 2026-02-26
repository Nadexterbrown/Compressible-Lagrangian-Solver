"""
Flow state and conserved variables for Lagrangian formulation.

This module defines the data structures for storing the fluid state
on a staggered Lagrangian grid.

References:
    [Despres2017] Chapter 3 - Lagrangian formulation of conservation laws
    [Toro2009] Chapter 1 - Euler equations

Grid layout (staggered):
    Cell-centered (index i = 0 to N-1):
        tau[i], rho[i], p[i], T[i], e[i], E[i], c[i]

    Face-centered (index j = 0 to N):
        x[j], u[j], m[j]

    Face j is the left boundary of cell i=j and right boundary of cell i=j-1.
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from lagrangian_solver.equations.eos import EOSBase


@dataclass
class ConservedVariables:
    """
    Conserved variables in Lagrangian mass coordinates.

    In Lagrangian coordinates with mass as the spatial coordinate,
    the conserved variables per unit mass are:
        - τ (tau): Specific volume [m³/kg] = 1/ρ
        - u: Velocity [m/s]
        - E: Total specific energy [J/kg] = e + ½u²

    Reference: [Despres2017] Section 3.2, Equations (3.7)-(3.9)

    Attributes:
        tau: Specific volume (cell-centered) [m³/kg]
        u: Velocity (face-centered) [m/s]
        E: Total specific energy (cell-centered) [J/kg]
    """

    tau: np.ndarray  # Cell-centered, shape (n_cells,)
    u: np.ndarray  # Face-centered, shape (n_cells + 1,)
    E: np.ndarray  # Cell-centered, shape (n_cells,)

    def __post_init__(self):
        """Validate array shapes."""
        n_cells = len(self.tau)
        if len(self.E) != n_cells:
            raise ValueError(
                f"E must have same length as tau: {len(self.E)} != {n_cells}"
            )
        if len(self.u) != n_cells + 1:
            raise ValueError(
                f"u must have length n_cells+1: {len(self.u)} != {n_cells + 1}"
            )

    @property
    def n_cells(self) -> int:
        """Number of cells in the domain."""
        return len(self.tau)

    def copy(self) -> "ConservedVariables":
        """Create a deep copy of the conserved variables."""
        return ConservedVariables(
            tau=self.tau.copy(),
            u=self.u.copy(),
            E=self.E.copy(),
        )


@dataclass
class ConservedVariablesCompatible:
    """
    Conserved variables for compatible energy discretization.

    In compatible energy discretization, we solve for internal energy (e)
    directly rather than total energy (E). This avoids the numerical issues
    of extracting internal energy via subtraction (e = E - ½u²) which can
    lose precision at strong shocks.

    Reference:
        [Burton1992] UCRL-JC-105926 - Consistent Finite-Volume Discretization
        [Caramana1998] JCP 146 - Formulations of Artificial Viscosity

    Attributes:
        tau: Specific volume (cell-centered) [m³/kg]
        u: Velocity (face-centered) [m/s]
        e: Specific internal energy (cell-centered) [J/kg]
    """

    tau: np.ndarray  # Cell-centered, shape (n_cells,)
    u: np.ndarray  # Face-centered, shape (n_cells + 1,)
    e: np.ndarray  # Cell-centered, shape (n_cells,)

    def __post_init__(self):
        """Validate array shapes."""
        n_cells = len(self.tau)
        if len(self.e) != n_cells:
            raise ValueError(
                f"e must have same length as tau: {len(self.e)} != {n_cells}"
            )
        if len(self.u) != n_cells + 1:
            raise ValueError(
                f"u must have length n_cells+1: {len(self.u)} != {n_cells + 1}"
            )

    @property
    def n_cells(self) -> int:
        """Number of cells in the domain."""
        return len(self.tau)

    def copy(self) -> "ConservedVariablesCompatible":
        """Create a deep copy of the conserved variables."""
        return ConservedVariablesCompatible(
            tau=self.tau.copy(),
            u=self.u.copy(),
            e=self.e.copy(),
        )

    def to_total_energy(self) -> ConservedVariables:
        """
        Convert to total energy formulation.

        Returns:
            ConservedVariables with E = e + ½u_cell²
        """
        u_cell = 0.5 * (self.u[:-1] + self.u[1:])
        E = self.e + 0.5 * u_cell**2
        return ConservedVariables(
            tau=self.tau.copy(),
            u=self.u.copy(),
            E=E,
        )


@dataclass
class FlowState:
    """
    Complete flow state on a Lagrangian grid.

    Contains both conserved and derived thermodynamic variables.
    Cell-centered quantities are stored at cell centers (indices 0 to n_cells-1).
    Face-centered quantities are stored at cell interfaces (indices 0 to n_cells).

    Reference: [Despres2017] Chapter 3

    Attributes:
        # Cell-centered quantities
        tau: Specific volume [m³/kg]
        rho: Density [kg/m³]
        p: Pressure [Pa]
        T: Temperature [K]
        e: Specific internal energy [J/kg]
        E: Total specific energy [J/kg]
        c: Sound speed [m/s]
        gamma: Ratio of specific heats [-]
        s: Specific entropy [J/(kg·K)]

        # Face-centered quantities
        x: Position [m]
        u: Velocity [m/s]

        # Mass coordinates
        m: Cumulative mass at faces [kg]
        dm: Cell mass [kg]
    """

    # Cell-centered thermodynamic state
    tau: np.ndarray
    rho: np.ndarray
    p: np.ndarray
    T: np.ndarray
    e: np.ndarray
    E: np.ndarray
    c: np.ndarray
    gamma: np.ndarray
    s: np.ndarray  # Specific entropy

    # Face-centered kinematic state
    x: np.ndarray
    u: np.ndarray

    # Mass distribution
    m: np.ndarray  # Cumulative mass at faces
    dm: np.ndarray  # Cell masses

    def __post_init__(self):
        """Validate array shapes and consistency."""
        n_cells = len(self.tau)

        # Check cell-centered arrays
        for name, arr in [
            ("rho", self.rho),
            ("p", self.p),
            ("T", self.T),
            ("e", self.e),
            ("E", self.E),
            ("c", self.c),
            ("gamma", self.gamma),
            ("s", self.s),
            ("dm", self.dm),
        ]:
            if len(arr) != n_cells:
                raise ValueError(
                    f"{name} must have length n_cells={n_cells}, got {len(arr)}"
                )

        # Check face-centered arrays
        n_faces = n_cells + 1
        for name, arr in [("x", self.x), ("u", self.u), ("m", self.m)]:
            if len(arr) != n_faces:
                raise ValueError(
                    f"{name} must have length n_faces={n_faces}, got {len(arr)}"
                )

    @property
    def n_cells(self) -> int:
        """Number of cells in the domain."""
        return len(self.tau)

    @property
    def n_faces(self) -> int:
        """Number of faces (cell interfaces)."""
        return len(self.x)

    @property
    def dx(self) -> np.ndarray:
        """Cell widths [m]."""
        return np.diff(self.x)

    @property
    def total_mass(self) -> float:
        """Total mass in the domain [kg]."""
        return self.m[-1] - self.m[0]

    def get_conserved(self) -> ConservedVariables:
        """Extract conserved variables from the flow state."""
        return ConservedVariables(
            tau=self.tau.copy(),
            u=self.u.copy(),
            E=self.E.copy(),
        )

    def get_conserved_compatible(self) -> ConservedVariablesCompatible:
        """Extract compatible conserved variables (with internal energy) from state."""
        return ConservedVariablesCompatible(
            tau=self.tau.copy(),
            u=self.u.copy(),
            e=self.e.copy(),
        )

    def copy(self) -> "FlowState":
        """Create a deep copy of the flow state."""
        return FlowState(
            tau=self.tau.copy(),
            rho=self.rho.copy(),
            p=self.p.copy(),
            T=self.T.copy(),
            e=self.e.copy(),
            E=self.E.copy(),
            c=self.c.copy(),
            gamma=self.gamma.copy(),
            s=self.s.copy(),
            x=self.x.copy(),
            u=self.u.copy(),
            m=self.m.copy(),
            dm=self.dm.copy(),
        )

    @classmethod
    def from_conserved(
        cls,
        conserved: ConservedVariables,
        x: np.ndarray,
        m: np.ndarray,
        eos: EOSBase,
        enforce_positivity: bool = True,
    ) -> "FlowState":
        """
        Construct a FlowState from conserved variables using EOS.

        Args:
            conserved: ConservedVariables containing tau, u, E
            x: Face positions [m]
            m: Cumulative mass at faces [kg]
            eos: Equation of state for computing thermodynamic properties
            enforce_positivity: If True, clip non-physical values to small positive numbers

        Returns:
            Complete FlowState with all derived quantities
        """
        tau = conserved.tau.copy()
        u = conserved.u.copy()
        E = conserved.E.copy()

        n_cells = len(tau)

        # Positivity enforcement for robustness with strong shocks
        # Reference: [Toro2009] Section 6.5 - Positivity preserving schemes
        if enforce_positivity:
            # Minimum allowed specific volume (max density ~1e6 kg/m³)
            tau_min = 1e-10
            tau = np.maximum(tau, tau_min)

        # Compute density from specific volume
        rho = 1.0 / tau

        # Compute cell-averaged velocity (for kinetic energy)
        u_cell = 0.5 * (u[:-1] + u[1:])

        # Compute internal energy from total energy
        e = E - 0.5 * u_cell**2

        # NOTE: We do NOT enforce positivity on internal energy here.
        # Cantera and other real-gas EOS use formation enthalpy references
        # where internal energy can be legitimately negative.
        # For air at STP, Cantera returns e ≈ -85884 J/kg.
        # Positivity is enforced on PRESSURE instead (which must always be positive).

        # Use EOS to get pressure, temperature, sound speed, and entropy
        p = eos.pressure(rho, e)

        # Ensure positive pressure
        if enforce_positivity:
            p_min = 1e-10
            p = np.maximum(p, p_min)

        T = eos.temperature(rho, e)
        c = eos.sound_speed(rho, p)
        gamma = eos.get_gamma(rho, p)
        s = eos.entropy(rho, p)

        # Compute cell masses
        dm = np.diff(m)

        return cls(
            tau=tau,
            rho=rho,
            p=p,
            T=T,
            e=e,
            E=E,
            c=c,
            gamma=gamma,
            s=s,
            x=x,
            u=u,
            m=m,
            dm=dm,
        )

    @classmethod
    def from_conserved_compatible(
        cls,
        conserved: ConservedVariablesCompatible,
        x: np.ndarray,
        m: np.ndarray,
        eos: EOSBase,
        enforce_positivity: bool = True,
    ) -> "FlowState":
        """
        Construct a FlowState from compatible conserved variables (internal energy).

        This method is used with compatible energy discretization where internal
        energy (e) is evolved directly, avoiding the numerical issues of computing
        e = E - ½u² via subtraction.

        Reference:
            [Burton1992] UCRL-JC-105926 - Compatible energy discretization
            [Caramana1998] JCP 146 - Formulations of Artificial Viscosity

        Args:
            conserved: ConservedVariablesCompatible containing tau, u, e
            x: Face positions [m]
            m: Cumulative mass at faces [kg]
            eos: Equation of state for computing thermodynamic properties
            enforce_positivity: If True, clip non-physical values

        Returns:
            Complete FlowState with all derived quantities
        """
        tau = conserved.tau.copy()
        u = conserved.u.copy()
        e = conserved.e.copy()

        # Positivity enforcement for robustness with strong shocks
        if enforce_positivity:
            # Minimum allowed specific volume (max density ~1e6 kg/m³)
            tau_min = 1e-10
            tau = np.maximum(tau, tau_min)

        # Compute density from specific volume
        rho = 1.0 / tau

        # Internal energy is provided directly - no subtraction needed
        # This is the key advantage of compatible discretization

        # Use EOS to get pressure, temperature, sound speed, and entropy
        p = eos.pressure(rho, e)

        # Ensure positive pressure
        if enforce_positivity:
            p_min = 1e-10
            p = np.maximum(p, p_min)

        T = eos.temperature(rho, e)
        c = eos.sound_speed(rho, p)
        gamma = eos.get_gamma(rho, p)
        s = eos.entropy(rho, p)

        # Compute total energy from internal energy
        u_cell = 0.5 * (u[:-1] + u[1:])
        E = e + 0.5 * u_cell**2

        # Compute cell masses
        dm = np.diff(m)

        return cls(
            tau=tau,
            rho=rho,
            p=p,
            T=T,
            e=e,
            E=E,
            c=c,
            gamma=gamma,
            s=s,
            x=x,
            u=u,
            m=m,
            dm=dm,
        )

    @classmethod
    def from_primitive(
        cls,
        rho: np.ndarray,
        u: np.ndarray,
        p: np.ndarray,
        x: np.ndarray,
        eos: EOSBase,
    ) -> "FlowState":
        """
        Construct a FlowState from primitive variables.

        Args:
            rho: Density (cell-centered) [kg/m³]
            u: Velocity (face-centered) [m/s]
            p: Pressure (cell-centered) [Pa]
            x: Face positions [m]
            eos: Equation of state

        Returns:
            Complete FlowState
        """
        n_cells = len(rho)

        # Specific volume
        tau = 1.0 / rho

        # Internal energy from EOS
        e = eos.internal_energy(rho, p)

        # Cell-averaged velocity for kinetic energy
        u_cell = 0.5 * (u[:-1] + u[1:])

        # Total energy
        E = e + 0.5 * u_cell**2

        # Temperature, sound speed, and entropy from EOS
        T = eos.temperature(rho, e)
        c = eos.sound_speed(rho, p)
        gamma = eos.get_gamma(rho, p)
        s = eos.entropy(rho, p)

        # Compute mass coordinates
        dx = np.diff(x)
        dm = rho * dx  # Cell mass = density * volume (per unit area)
        m = np.zeros(n_cells + 1)
        m[1:] = np.cumsum(dm)

        return cls(
            tau=tau,
            rho=rho,
            p=p,
            T=T,
            e=e,
            E=E,
            c=c,
            gamma=gamma,
            s=s,
            x=x,
            u=u,
            m=m,
            dm=dm,
        )

    def update_from_conserved(
        self,
        tau: np.ndarray,
        u: np.ndarray,
        E: np.ndarray,
        x: np.ndarray,
        eos: EOSBase,
    ):
        """
        Update the flow state in-place from new conserved variables.

        Args:
            tau: New specific volume (cell-centered) [m³/kg]
            u: New velocity (face-centered) [m/s]
            E: New total specific energy (cell-centered) [J/kg]
            x: New face positions [m]
            eos: Equation of state
        """
        self.tau[:] = tau
        self.u[:] = u
        self.E[:] = E
        self.x[:] = x

        # Update derived quantities
        self.rho[:] = 1.0 / tau

        # Cell-averaged velocity
        u_cell = 0.5 * (u[:-1] + u[1:])

        # Internal energy
        self.e[:] = E - 0.5 * u_cell**2

        # EOS quantities
        self.p[:] = eos.pressure(self.rho, self.e)
        self.T[:] = eos.temperature(self.rho, self.e)
        self.c[:] = eos.sound_speed(self.rho, self.p)
        self.gamma[:] = eos.get_gamma(self.rho, self.p)
        self.s[:] = eos.entropy(self.rho, self.p)

        # Mass stays constant in Lagrangian formulation
        # Only positions change, masses dm are fixed

    def update_from_conserved_compatible(
        self,
        tau: np.ndarray,
        u: np.ndarray,
        e: np.ndarray,
        x: np.ndarray,
        eos: EOSBase,
    ):
        """
        Update the flow state in-place from compatible conserved variables.

        This method is used with compatible energy discretization where internal
        energy (e) is evolved directly.

        Reference: [Burton1992] UCRL-JC-105926

        Args:
            tau: New specific volume (cell-centered) [m³/kg]
            u: New velocity (face-centered) [m/s]
            e: New specific internal energy (cell-centered) [J/kg]
            x: New face positions [m]
            eos: Equation of state
        """
        self.tau[:] = tau
        self.u[:] = u
        self.e[:] = e
        self.x[:] = x

        # Update derived quantities
        self.rho[:] = 1.0 / tau

        # Cell-averaged velocity for total energy
        u_cell = 0.5 * (u[:-1] + u[1:])

        # Total energy from internal energy (computed, not primary variable)
        self.E[:] = e + 0.5 * u_cell**2

        # EOS quantities - internal energy is primary, no subtraction
        self.p[:] = eos.pressure(self.rho, self.e)
        self.T[:] = eos.temperature(self.rho, self.e)
        self.c[:] = eos.sound_speed(self.rho, self.p)
        self.gamma[:] = eos.get_gamma(self.rho, self.p)
        self.s[:] = eos.entropy(self.rho, self.p)

        # Mass stays constant in Lagrangian formulation


def create_uniform_state(
    n_cells: int,
    x_left: float,
    x_right: float,
    rho: float,
    u: float,
    p: float,
    eos: EOSBase,
) -> FlowState:
    """
    Create a uniform flow state.

    Args:
        n_cells: Number of cells
        x_left: Left boundary position [m]
        x_right: Right boundary position [m]
        rho: Uniform density [kg/m³]
        u: Uniform velocity [m/s]
        p: Uniform pressure [Pa]
        eos: Equation of state

    Returns:
        FlowState with uniform properties
    """
    # Create uniform face positions
    x = np.linspace(x_left, x_right, n_cells + 1)

    # Uniform cell-centered density and pressure
    rho_arr = np.full(n_cells, rho)
    p_arr = np.full(n_cells, p)

    # Uniform face-centered velocity
    u_arr = np.full(n_cells + 1, u)

    return FlowState.from_primitive(rho_arr, u_arr, p_arr, x, eos)


def create_riemann_state(
    n_cells: int,
    x_left: float,
    x_right: float,
    x_discontinuity: float,
    rho_L: float,
    u_L: float,
    p_L: float,
    rho_R: float,
    u_R: float,
    p_R: float,
    eos: EOSBase,
) -> FlowState:
    """
    Create a Riemann problem initial condition with a discontinuity.

    Args:
        n_cells: Number of cells
        x_left: Left boundary position [m]
        x_right: Right boundary position [m]
        x_discontinuity: Position of the initial discontinuity [m]
        rho_L, u_L, p_L: Left state (density, velocity, pressure)
        rho_R, u_R, p_R: Right state (density, velocity, pressure)
        eos: Equation of state

    Returns:
        FlowState with Riemann problem initial condition
    """
    # Create uniform face positions
    x = np.linspace(x_left, x_right, n_cells + 1)

    # Cell centers
    x_cell = 0.5 * (x[:-1] + x[1:])

    # Initialize arrays
    rho_arr = np.where(x_cell < x_discontinuity, rho_L, rho_R)
    p_arr = np.where(x_cell < x_discontinuity, p_L, p_R)

    # Velocity at faces
    u_arr = np.where(x < x_discontinuity, u_L, u_R)
    # Smooth transition at discontinuity
    disc_face = np.searchsorted(x, x_discontinuity)
    if 0 < disc_face < len(u_arr):
        u_arr[disc_face] = 0.5 * (u_L + u_R)

    return FlowState.from_primitive(rho_arr, u_arr, p_arr, x, eos)
