"""
Flame elongation functions and trajectory computation.

Implements various sigma(t) elongation functions from the Clavin-Tofaili
flame elongation model. The piston velocity is computed as:

    u_p = (sigma(t) - 1) * (rho_u / rho_b) * S_L

where sigma(t) is the instantaneous flame elongation factor.

Reference:
    Clavin, P., & Tofaili, H. (2021). Flame elongation model.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from flame_property_interpolator import FlamePropertyInterpolator, FlameProperties


class ElongationBase(ABC):
    """
    Abstract base class for flame elongation functions sigma(t).

    All elongation functions must provide:
    - sigma(t): Elongation factor at time t
    - dsigma_dt(t): Time derivative of sigma at time t
    - name: Human-readable name for the function
    """

    @abstractmethod
    def sigma(self, t: float) -> float:
        """
        Compute elongation factor sigma at time t.

        Args:
            t: Time [s]

        Returns:
            Elongation factor sigma (dimensionless, >= 1)
        """
        pass

    @abstractmethod
    def dsigma_dt(self, t: float) -> float:
        """
        Compute time derivative of sigma at time t.

        Args:
            t: Time [s]

        Returns:
            dsigma/dt [1/s]
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this elongation function."""
        pass


@dataclass
class PowerLawElongation(ElongationBase):
    """
    Power-law elongation function.

    sigma(t) = sigma_0 * (1 + (k*t)^n)

    For n=2, this gives quadratic acceleration typical of
    constant-acceleration flame stretch.

    Attributes:
        sigma_0: Initial elongation factor (default 1.0)
        k: Rate parameter [1/s] (default 1000.0)
        n: Power-law exponent (default 2.0)
    """

    sigma_0: float = 1.0
    k: float = 1000.0
    n: float = 2.0

    def sigma(self, t: float) -> float:
        """Compute sigma(t) = sigma_0 * (1 + (k*t)^n)."""
        return self.sigma_0 * (1.0 + (self.k * t) ** self.n)

    def dsigma_dt(self, t: float) -> float:
        """Compute dsigma/dt = sigma_0 * n * k * (k*t)^(n-1)."""
        if t <= 0 and self.n < 1:
            return 0.0
        return self.sigma_0 * self.n * self.k * (self.k * t) ** (self.n - 1)

    @property
    def name(self) -> str:
        return f"PowerLaw(sigma_0={self.sigma_0}, k={self.k}, n={self.n})"


@dataclass
class ExponentialElongation(ElongationBase):
    """
    Exponential elongation function.

    sigma(t) = sigma_0 * exp(B*t)

    This models exponential flame growth, typical of
    thermodiffusive instability.

    Attributes:
        sigma_0: Initial elongation factor (default 1.0)
        B: Growth rate [1/s] (default 1000.0)
    """

    sigma_0: float = 1.0
    B: float = 1000.0

    def sigma(self, t: float) -> float:
        """Compute sigma(t) = sigma_0 * exp(B*t)."""
        return self.sigma_0 * np.exp(self.B * t)

    def dsigma_dt(self, t: float) -> float:
        """Compute dsigma/dt = sigma_0 * B * exp(B*t)."""
        return self.sigma_0 * self.B * np.exp(self.B * t)

    @property
    def name(self) -> str:
        return f"Exponential(sigma_0={self.sigma_0}, B={self.B})"


@dataclass
class LinearElongation(ElongationBase):
    """
    Linear elongation function.

    sigma(t) = sigma_0 + k*t

    This models constant flame stretch rate.

    Attributes:
        sigma_0: Initial elongation factor (default 1.0)
        k: Linear rate [1/s] (default 1000.0)
    """

    sigma_0: float = 1.0
    k: float = 1000.0

    def sigma(self, t: float) -> float:
        """Compute sigma(t) = sigma_0 + k*t."""
        return self.sigma_0 + self.k * t

    def dsigma_dt(self, t: float) -> float:
        """Compute dsigma/dt = k."""
        return self.k

    @property
    def name(self) -> str:
        return f"Linear(sigma_0={self.sigma_0}, k={self.k})"


@dataclass
class ConstantElongation(ElongationBase):
    """
    Constant elongation (no time variation).

    sigma(t) = sigma_0

    Useful for steady-state analysis or debugging.

    Attributes:
        sigma_0: Constant elongation factor (default 1.0)
    """

    sigma_0: float = 1.0

    def sigma(self, t: float) -> float:
        """Return constant sigma_0."""
        return self.sigma_0

    def dsigma_dt(self, t: float) -> float:
        """Return 0 (no time variation)."""
        return 0.0

    @property
    def name(self) -> str:
        return f"Constant(sigma_0={self.sigma_0})"


class FlameElongationTrajectory:
    """
    Combines elongation function with flame property lookup.

    This class computes the piston velocity from the Clavin-Tofaili formula:
        u_p = (sigma(t) - 1) * (rho_u / rho_b) * S_L

    where:
        - sigma(t) is from the elongation function
        - rho_u, rho_b, S_L are from flame property interpolation

    The velocity can be computed at:
    1. Reference conditions (P_ref, T_ref) - for initial guess
    2. Local conditions (P, T) - for coupled iteration

    Attributes:
        elongation: The elongation function sigma(t)
        interpolator: Flame property interpolator
        P_ref, T_ref: Reference conditions for uncoupled velocity
    """

    def __init__(
        self,
        elongation: ElongationBase,
        flame_interpolator: FlamePropertyInterpolator,
        P_ref: float = 1e6,
        T_ref: float = 500.0,
    ):
        """
        Initialize trajectory with elongation function and flame data.

        Args:
            elongation: Elongation function sigma(t)
            flame_interpolator: Interpolator for flame properties
            P_ref: Reference pressure for uncoupled velocity [Pa]
            T_ref: Reference temperature for uncoupled velocity [K]
        """
        self._elongation = elongation
        self._interpolator = flame_interpolator
        self._P_ref = P_ref
        self._T_ref = T_ref

        # Cache reference properties for efficiency
        self._props_ref = self._interpolator.get_properties(P_ref, T_ref)

    @property
    def elongation(self) -> ElongationBase:
        """The elongation function."""
        return self._elongation

    @property
    def interpolator(self) -> FlamePropertyInterpolator:
        """The flame property interpolator."""
        return self._interpolator

    @property
    def P_ref(self) -> float:
        """Reference pressure [Pa]."""
        return self._P_ref

    @property
    def T_ref(self) -> float:
        """Reference temperature [K]."""
        return self._T_ref

    @property
    def properties_ref(self) -> FlameProperties:
        """Flame properties at reference conditions."""
        return self._props_ref

    def sigma(self, t: float) -> float:
        """Get elongation factor at time t."""
        return self._elongation.sigma(t)

    def dsigma_dt(self, t: float) -> float:
        """Get time derivative of elongation at time t."""
        return self._elongation.dsigma_dt(t)

    def velocity_uncoupled(self, t: float) -> float:
        """
        Compute piston velocity at reference (P_ref, T_ref).

        This is the "uncoupled" velocity that ignores the effect of
        piston motion on the local P and T. Useful as:
        1. Initial guess for coupled iteration
        2. Comparison baseline

        Args:
            t: Time [s]

        Returns:
            Piston velocity [m/s]
        """
        sig = self._elongation.sigma(t)
        return (sig - 1.0) * self._props_ref.density_ratio * self._props_ref.S_L

    def velocity(self, t: float, P: float, T: float) -> float:
        """
        Compute piston velocity at given (P, T) conditions.

        This is the "coupled" velocity that accounts for local
        thermodynamic conditions at the piston face.

        Args:
            t: Time [s]
            P: Local pressure [Pa]
            T: Local temperature [K]

        Returns:
            Piston velocity [m/s]
        """
        sig = self._elongation.sigma(t)
        props = self._interpolator.get_properties(P, T)
        return (sig - 1.0) * props.density_ratio * props.S_L

    def velocity_derivative_uncoupled(self, t: float) -> float:
        """
        Compute time derivative of uncoupled velocity.

        du_p/dt = d[(sigma-1) * (rho_u/rho_b) * S_L] / dt
                = dsigma/dt * (rho_u/rho_b) * S_L

        (at reference conditions where rho_u, rho_b, S_L are constant)

        Args:
            t: Time [s]

        Returns:
            Piston acceleration [m/s^2]
        """
        dsig_dt = self._elongation.dsigma_dt(t)
        return dsig_dt * self._props_ref.density_ratio * self._props_ref.S_L

    def __repr__(self) -> str:
        return (
            f"FlameElongationTrajectory(\n"
            f"  elongation={self._elongation.name},\n"
            f"  P_ref={self._P_ref:.2e} Pa,\n"
            f"  T_ref={self._T_ref:.1f} K,\n"
            f"  S_L_ref={self._props_ref.S_L:.2f} m/s,\n"
            f"  (rho_u/rho_b)_ref={self._props_ref.density_ratio:.2f}\n"
            f")"
        )
