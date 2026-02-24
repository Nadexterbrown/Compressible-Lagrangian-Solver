"""
Riemann solvers for 1D compressible flow.

This module provides exact and approximate Riemann solvers for computing
inter-cell fluxes in the Lagrangian formulation.

References:
    [Toro2009] Chapters 4-10 - Riemann solvers and approximate methods
    [Despres2017] Chapter 4 - Riemann problem in Lagrangian coordinates

Key feature of Lagrangian coordinates: The contact discontinuity is
stationary in mass coordinates, simplifying the Riemann problem structure.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple
import numpy as np

from lagrangian_solver.equations.eos import EOSBase, IdealGasEOS


class WaveType(Enum):
    """Type of wave in Riemann solution."""

    SHOCK = "shock"
    RAREFACTION = "rarefaction"


@dataclass
class RiemannState:
    """
    State on one side of a Riemann problem interface.

    Reference: [Toro2009] Section 4.1

    Attributes:
        rho: Density [kg/m³]
        u: Velocity [m/s]
        p: Pressure [Pa]
        c: Sound speed [m/s]
        e: Specific internal energy [J/kg]
    """

    rho: float
    u: float
    p: float
    c: float
    e: float

    @classmethod
    def from_primitive(
        cls, rho: float, u: float, p: float, eos: EOSBase
    ) -> "RiemannState":
        """Create state from primitive variables using EOS."""
        rho_arr = np.atleast_1d(rho)
        p_arr = np.atleast_1d(p)
        e = float(eos.internal_energy(rho_arr, p_arr)[0])
        c = float(eos.sound_speed(rho_arr, p_arr)[0])
        return cls(rho=rho, u=u, p=p, c=c, e=e)


@dataclass
class RiemannSolution:
    """
    Solution of the Riemann problem.

    Reference: [Toro2009] Section 4.2

    Attributes:
        p_star: Pressure in the star region [Pa]
        u_star: Velocity in the star region [m/s]
        rho_star_L: Density on left side of contact [kg/m³]
        rho_star_R: Density on right side of contact [kg/m³]
        wave_L: Type of left wave (shock or rarefaction)
        wave_R: Type of right wave (shock or rarefaction)
    """

    p_star: float
    u_star: float
    rho_star_L: float
    rho_star_R: float
    wave_L: WaveType
    wave_R: WaveType


class RiemannSolverBase(ABC):
    """
    Abstract base class for Riemann solvers.

    Reference: [Toro2009] Chapter 4
    """

    def __init__(self, eos: EOSBase):
        """
        Initialize solver with equation of state.

        Args:
            eos: Equation of state for computing thermodynamic properties
        """
        self._eos = eos

    @property
    def eos(self) -> EOSBase:
        """Equation of state."""
        return self._eos

    @abstractmethod
    def solve(self, left: RiemannState, right: RiemannState) -> RiemannSolution:
        """
        Solve the Riemann problem.

        Args:
            left: Left state
            right: Right state

        Returns:
            Solution containing star region values
        """
        pass

    @abstractmethod
    def compute_flux(
        self, left: RiemannState, right: RiemannState
    ) -> Tuple[float, float]:
        """
        Compute the numerical flux at the interface.

        In Lagrangian coordinates, the flux of momentum is pressure
        and the flux of energy is p*u.

        Args:
            left: Left state
            right: Right state

        Returns:
            Tuple of (pressure_flux, velocity_star)
        """
        pass


class ExactRiemannSolver(RiemannSolverBase):
    """
    Exact Riemann solver using Newton-Raphson iteration.

    Solves for the pressure in the star region using the pressure
    function that enforces velocity continuity across the contact.

    Reference: [Toro2009] Chapter 4, particularly Section 4.3

    The iteration solves f(p) = f_L(p) + f_R(p) + Δu = 0
    where f_K is the pressure function for wave K (K = L, R)
    and Δu = u_R - u_L.
    """

    def __init__(
        self,
        eos: EOSBase,
        tol: float = 1e-8,
        max_iter: int = 100,
        initial_guess_method: str = "pvrs",
    ):
        """
        Initialize exact Riemann solver.

        Args:
            eos: Equation of state
            tol: Convergence tolerance for pressure iteration
            max_iter: Maximum number of iterations
            initial_guess_method: Method for initial pressure guess
                'pvrs': Primitive Variable Riemann Solver (default)
                'two_rarefaction': Two-rarefaction approximation
                'two_shock': Two-shock approximation
                'mean': Arithmetic mean of left and right pressures

        Reference: [Toro2009] Section 4.3.1 for initial guess methods
        """
        super().__init__(eos)
        self._tol = tol
        self._max_iter = max_iter
        self._initial_guess_method = initial_guess_method

        # Track iteration statistics
        self.last_iteration_count = 0
        self.last_converged = False

        # Handle gamma for different EOS types
        if isinstance(eos, IdealGasEOS):
            self._gamma = eos.gamma
            self._use_variable_gamma = False
        else:
            self._gamma = 1.4  # Default fallback
            self._use_variable_gamma = True

    def _get_gamma(self, rho: float, p: float) -> float:
        """
        Get gamma for the given state.

        For IdealGasEOS, returns the constant gamma.
        For other EOS (e.g., CanteraEOS), queries the EOS for state-dependent gamma.

        Args:
            rho: Density [kg/m³]
            p: Pressure [Pa]

        Returns:
            Ratio of specific heats gamma = cp/cv
        """
        if not self._use_variable_gamma:
            return self._gamma
        if hasattr(self._eos, 'get_gamma'):
            return float(self._eos.get_gamma(np.atleast_1d(rho), np.atleast_1d(p))[0])
        return self._gamma

    def _pressure_function(
        self, p: float, rho_K: float, p_K: float, c_K: float
    ) -> Tuple[float, float]:
        """
        Compute pressure function f_K and its derivative for wave K.

        For shock waves (p > p_K):
            f_K = (p - p_K) * sqrt(A_K / (p + B_K))

        For rarefaction waves (p <= p_K):
            f_K = (2 c_K / (γ-1)) * ((p/p_K)^((γ-1)/(2γ)) - 1)

        Reference: [Toro2009] Equations (4.6) and (4.7)

        Args:
            p: Pressure in star region
            rho_K: Density on side K
            p_K: Pressure on side K
            c_K: Sound speed on side K

        Returns:
            Tuple of (f_K, df_K/dp)
        """
        gamma = self._gamma
        gm1 = gamma - 1
        gp1 = gamma + 1

        A_K = 2.0 / (gp1 * rho_K)
        B_K = gm1 / gp1 * p_K

        if p > p_K:
            # Shock wave
            sqrt_term = np.sqrt(A_K / (p + B_K))
            f = (p - p_K) * sqrt_term
            df = sqrt_term * (1.0 - 0.5 * (p - p_K) / (p + B_K))
        else:
            # Rarefaction wave
            p_ratio = p / p_K
            exponent = gm1 / (2 * gamma)
            f = 2.0 * c_K / gm1 * (p_ratio**exponent - 1.0)
            df = c_K / (gamma * p_K) * p_ratio ** (-(gp1) / (2 * gamma))

        return f, df

    def _initial_guess(
        self, left: RiemannState, right: RiemannState
    ) -> float:
        """
        Compute initial guess for star region pressure.

        Reference: [Toro2009] Section 4.3.1, Equations (4.46)-(4.49)
        """
        gamma = self._gamma
        gm1 = gamma - 1
        gp1 = gamma + 1

        rho_L, u_L, p_L, c_L = left.rho, left.u, left.p, left.c
        rho_R, u_R, p_R, c_R = right.rho, right.u, right.p, right.c

        if self._initial_guess_method == "mean":
            # Simple arithmetic mean
            return 0.5 * (p_L + p_R)

        elif self._initial_guess_method == "two_rarefaction":
            # Two-rarefaction approximation (good for low pressure ratios)
            # [Toro2009] Equation (4.46)
            exponent = gm1 / (2 * gamma)
            p_tr = (
                (c_L + c_R - 0.5 * gm1 * (u_R - u_L))
                / (c_L / p_L**exponent + c_R / p_R**exponent)
            ) ** (1.0 / exponent)
            return max(p_tr, 1e-10)

        elif self._initial_guess_method == "two_shock":
            # Two-shock approximation (good for high pressure ratios)
            # [Toro2009] Equation (4.47)
            p_avg = 0.5 * (p_L + p_R)
            A_L = 2.0 / (gp1 * rho_L)
            B_L = gm1 / gp1 * p_L
            A_R = 2.0 / (gp1 * rho_R)
            B_R = gm1 / gp1 * p_R

            g_L = np.sqrt(A_L / (p_avg + B_L))
            g_R = np.sqrt(A_R / (p_avg + B_R))

            p_ts = (g_L * p_L + g_R * p_R - (u_R - u_L)) / (g_L + g_R)
            return max(p_ts, 1e-10)

        else:  # pvrs (default)
            # Primitive Variable Riemann Solver - linearized solution
            # [Toro2009] Equation (4.48)
            rho_avg = 0.5 * (rho_L + rho_R)
            c_avg = 0.5 * (c_L + c_R)
            p_pvrs = 0.5 * (p_L + p_R) - 0.5 * (u_R - u_L) * rho_avg * c_avg
            return max(p_pvrs, 1e-10)

    def _compute_star_velocity(
        self, p_star: float, left: RiemannState, right: RiemannState
    ) -> float:
        """
        Compute velocity in star region from converged pressure.

        u* = 0.5 * (u_L + u_R) + 0.5 * (f_R - f_L)

        Reference: [Toro2009] Equation (4.9)
        """
        f_L, _ = self._pressure_function(p_star, left.rho, left.p, left.c)
        f_R, _ = self._pressure_function(p_star, right.rho, right.p, right.c)

        return 0.5 * (left.u + right.u) + 0.5 * (f_R - f_L)

    def _compute_star_densities(
        self, p_star: float, left: RiemannState, right: RiemannState
    ) -> Tuple[float, float, WaveType, WaveType]:
        """
        Compute densities in star region and determine wave types.

        Reference: [Toro2009] Equations (4.53)-(4.57)
        """
        gamma = self._gamma
        gm1 = gamma - 1
        gp1 = gamma + 1

        # Left wave
        if p_star > left.p:
            # Left shock
            wave_L = WaveType.SHOCK
            p_ratio = p_star / left.p
            rho_star_L = left.rho * (
                (p_ratio + gm1 / gp1) / (gm1 / gp1 * p_ratio + 1)
            )
        else:
            # Left rarefaction
            wave_L = WaveType.RAREFACTION
            rho_star_L = left.rho * (p_star / left.p) ** (1.0 / gamma)

        # Right wave
        if p_star > right.p:
            # Right shock
            wave_R = WaveType.SHOCK
            p_ratio = p_star / right.p
            rho_star_R = right.rho * (
                (p_ratio + gm1 / gp1) / (gm1 / gp1 * p_ratio + 1)
            )
        else:
            # Right rarefaction
            wave_R = WaveType.RAREFACTION
            rho_star_R = right.rho * (p_star / right.p) ** (1.0 / gamma)

        return rho_star_L, rho_star_R, wave_L, wave_R

    def solve(self, left: RiemannState, right: RiemannState) -> RiemannSolution:
        """
        Solve the Riemann problem using Newton-Raphson iteration.

        Reference: [Toro2009] Section 4.3, Algorithm 4.1
        """
        # For variable gamma EOS, compute averaged gamma from left/right states
        if self._use_variable_gamma:
            gamma_L = self._get_gamma(left.rho, left.p)
            gamma_R = self._get_gamma(right.rho, right.p)
            self._gamma = 0.5 * (gamma_L + gamma_R)

        # Check for vacuum generation
        du = right.u - left.u
        critical_velocity = (
            2.0 / (self._gamma - 1) * (left.c + right.c)
        )

        if du >= critical_velocity:
            raise ValueError(
                f"Vacuum generated: velocity difference {du:.4f} >= "
                f"critical velocity {critical_velocity:.4f}"
            )

        # Initial guess
        p_star = self._initial_guess(left, right)

        # Newton-Raphson iteration
        for iteration in range(self._max_iter):
            f_L, df_L = self._pressure_function(
                p_star, left.rho, left.p, left.c
            )
            f_R, df_R = self._pressure_function(
                p_star, right.rho, right.p, right.c
            )

            f = f_L + f_R + du
            df = df_L + df_R

            # Newton update
            dp = -f / df
            p_star_new = p_star + dp

            # Ensure positivity
            if p_star_new < 0:
                p_star_new = self._tol

            # Check convergence
            if abs(dp) < self._tol * abs(p_star):
                self.last_iteration_count = iteration + 1
                self.last_converged = True
                p_star = p_star_new
                break

            p_star = p_star_new
        else:
            self.last_iteration_count = self._max_iter
            self.last_converged = False

        # Compute star velocity
        u_star = self._compute_star_velocity(p_star, left, right)

        # Compute star densities and wave types
        rho_star_L, rho_star_R, wave_L, wave_R = self._compute_star_densities(
            p_star, left, right
        )

        return RiemannSolution(
            p_star=p_star,
            u_star=u_star,
            rho_star_L=rho_star_L,
            rho_star_R=rho_star_R,
            wave_L=wave_L,
            wave_R=wave_R,
        )

    def compute_flux(
        self, left: RiemannState, right: RiemannState
    ) -> Tuple[float, float]:
        """
        Compute numerical flux at interface.

        In Lagrangian coordinates at the contact (x/t = 0 in similarity
        coordinates), the flux is simply the star state values.

        Reference: [Despres2017] Section 4.2

        Returns:
            Tuple of (pressure_flux, velocity_star)
        """
        solution = self.solve(left, right)

        # Return p_star and u_star directly
        # The caller computes pu_flux = p_star * u_star
        return solution.p_star, solution.u_star

    def sample(
        self,
        left: RiemannState,
        right: RiemannState,
        x_t: float,
    ) -> Tuple[float, float, float]:
        """
        Sample the exact solution at a given x/t.

        Reference: [Toro2009] Section 4.5, Algorithm 4.2

        Args:
            left: Left state
            right: Right state
            x_t: Similarity variable (x - x_0) / t

        Returns:
            Tuple of (density, velocity, pressure) at x/t
        """
        solution = self.solve(left, right)
        gamma = self._gamma
        gm1 = gamma - 1
        gp1 = gamma + 1

        if x_t <= solution.u_star:
            # Left of contact discontinuity
            if solution.wave_L == WaveType.SHOCK:
                # Left shock
                # Shock speed
                S_L = left.u - left.c * np.sqrt(
                    (gp1 / (2 * gamma)) * (solution.p_star / left.p)
                    + gm1 / (2 * gamma)
                )
                if x_t <= S_L:
                    # Left of shock
                    return left.rho, left.u, left.p
                else:
                    # Right of shock (star region)
                    return solution.rho_star_L, solution.u_star, solution.p_star
            else:
                # Left rarefaction
                c_star_L = left.c * (solution.p_star / left.p) ** (gm1 / (2 * gamma))
                S_HL = left.u - left.c  # Head speed
                S_TL = solution.u_star - c_star_L  # Tail speed

                if x_t <= S_HL:
                    # Left of rarefaction fan
                    return left.rho, left.u, left.p
                elif x_t >= S_TL:
                    # Right of rarefaction fan (star region)
                    return solution.rho_star_L, solution.u_star, solution.p_star
                else:
                    # Inside rarefaction fan
                    u_fan = 2.0 / gp1 * (left.c + gm1 / 2 * left.u + x_t)
                    c_fan = 2.0 / gp1 * (left.c + gm1 / 2 * (left.u - x_t))
                    p_fan = left.p * (c_fan / left.c) ** (2 * gamma / gm1)
                    rho_fan = left.rho * (p_fan / left.p) ** (1 / gamma)
                    return rho_fan, u_fan, p_fan
        else:
            # Right of contact discontinuity
            if solution.wave_R == WaveType.SHOCK:
                # Right shock
                S_R = right.u + right.c * np.sqrt(
                    (gp1 / (2 * gamma)) * (solution.p_star / right.p)
                    + gm1 / (2 * gamma)
                )
                if x_t >= S_R:
                    # Right of shock
                    return right.rho, right.u, right.p
                else:
                    # Left of shock (star region)
                    return solution.rho_star_R, solution.u_star, solution.p_star
            else:
                # Right rarefaction
                c_star_R = right.c * (solution.p_star / right.p) ** (gm1 / (2 * gamma))
                S_HR = right.u + right.c  # Head speed
                S_TR = solution.u_star + c_star_R  # Tail speed

                if x_t >= S_HR:
                    # Right of rarefaction fan
                    return right.rho, right.u, right.p
                elif x_t <= S_TR:
                    # Left of rarefaction fan (star region)
                    return solution.rho_star_R, solution.u_star, solution.p_star
                else:
                    # Inside rarefaction fan
                    u_fan = 2.0 / gp1 * (-right.c + gm1 / 2 * right.u + x_t)
                    c_fan = 2.0 / gp1 * (right.c - gm1 / 2 * (right.u - x_t))
                    p_fan = right.p * (c_fan / right.c) ** (2 * gamma / gm1)
                    rho_fan = right.rho * (p_fan / right.p) ** (1 / gamma)
                    return rho_fan, u_fan, p_fan


class HLLCRiemannSolver(RiemannSolverBase):
    """
    HLLC (Harten-Lax-van Leer-Contact) approximate Riemann solver.

    The HLLC solver is a popular approximate Riemann solver that
    restores the contact wave missing in the original HLL solver.

    Reference: [Toro2009] Chapter 10, particularly Section 10.4
    """

    def __init__(self, eos: EOSBase):
        """
        Initialize HLLC solver.

        Args:
            eos: Equation of state
        """
        super().__init__(eos)

        if isinstance(eos, IdealGasEOS):
            self._gamma = eos.gamma
        else:
            self._gamma = 1.4

    def _wave_speed_estimates(
        self, left: RiemannState, right: RiemannState
    ) -> Tuple[float, float, float]:
        """
        Estimate wave speeds for HLLC solver.

        Uses the pressure-based wave speed estimates from Toro.

        Reference: [Toro2009] Section 10.5.1, Equations (10.67)-(10.70)

        Returns:
            Tuple of (S_L, S_star, S_R)
        """
        gamma = self._gamma
        gm1 = gamma - 1
        gp1 = gamma + 1

        # PVRS estimate for p_star
        rho_avg = 0.5 * (left.rho + right.rho)
        c_avg = 0.5 * (left.c + right.c)
        p_pvrs = max(
            0,
            0.5 * (left.p + right.p)
            - 0.5 * (right.u - left.u) * rho_avg * c_avg,
        )

        # Pressure-based wave speed corrections
        if p_pvrs <= left.p:
            q_L = 1.0
        else:
            q_L = np.sqrt(1 + gp1 / (2 * gamma) * (p_pvrs / left.p - 1))

        if p_pvrs <= right.p:
            q_R = 1.0
        else:
            q_R = np.sqrt(1 + gp1 / (2 * gamma) * (p_pvrs / right.p - 1))

        # Wave speeds
        S_L = left.u - left.c * q_L
        S_R = right.u + right.c * q_R

        # Contact wave speed
        S_star = (
            right.p - left.p
            + left.rho * left.u * (S_L - left.u)
            - right.rho * right.u * (S_R - right.u)
        ) / (
            left.rho * (S_L - left.u) - right.rho * (S_R - right.u)
        )

        return S_L, S_star, S_R

    def solve(self, left: RiemannState, right: RiemannState) -> RiemannSolution:
        """
        Solve the Riemann problem using HLLC approximation.

        Reference: [Toro2009] Section 10.4
        """
        S_L, S_star, S_R = self._wave_speed_estimates(left, right)

        # Star pressure from left state
        p_star_L = left.p + left.rho * (S_L - left.u) * (S_star - left.u)

        # Star pressure from right state (should be same)
        p_star_R = right.p + right.rho * (S_R - right.u) * (S_star - right.u)

        # Average for robustness
        p_star = 0.5 * (p_star_L + p_star_R)

        # Star densities
        rho_star_L = left.rho * (S_L - left.u) / (S_L - S_star)
        rho_star_R = right.rho * (S_R - right.u) / (S_R - S_star)

        # Determine wave types (approximate)
        wave_L = WaveType.SHOCK if p_star > left.p else WaveType.RAREFACTION
        wave_R = WaveType.SHOCK if p_star > right.p else WaveType.RAREFACTION

        return RiemannSolution(
            p_star=p_star,
            u_star=S_star,
            rho_star_L=rho_star_L,
            rho_star_R=rho_star_R,
            wave_L=wave_L,
            wave_R=wave_R,
        )

    def compute_flux(
        self, left: RiemannState, right: RiemannState
    ) -> Tuple[float, float]:
        """
        Compute HLLC numerical flux at interface.

        Reference: [Toro2009] Section 10.4, Equations (10.37)-(10.40)

        Returns:
            Tuple of (pressure_flux, velocity_star)
        """
        S_L, S_star, S_R = self._wave_speed_estimates(left, right)

        # In Lagrangian formulation at x/t = 0:
        if S_L >= 0:
            # All waves moving right - use left state
            return left.p, left.u
        elif S_R <= 0:
            # All waves moving left - use right state
            return right.p, right.u
        else:
            # Interface is in star region
            if S_star >= 0:
                # Left star region
                p_star = left.p + left.rho * (S_L - left.u) * (S_star - left.u)
            else:
                # Right star region
                p_star = right.p + right.rho * (S_R - right.u) * (S_star - right.u)

            # Return p_star and S_star (which is u_star)
            return p_star, S_star
