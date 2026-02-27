"""
Riemann solver for boundary conditions.

Solves the boundary Riemann problem: given an interior state and a
prescribed boundary velocity, find the interface pressure p* and
complete boundary state.

This is GENERAL - works for any wave type (shock, rarefaction, or acoustic).

The key equation solved is:
    u*(p*) = u_boundary

where u*(p*) is the velocity at the interface given interface pressure p*.
This is found by Newton iteration on the wave function.

References:
    [Toro2009] Section 4.3.3 - Pressure function in exact Riemann solver
    [Toro2009] Section 6.3 - Boundary conditions
    [Toro2009] Chapter 4 - Exact Riemann solver
"""

from dataclasses import dataclass
from typing import Tuple
import numpy as np

from lagrangian_solver.equations.eos import EOSBase


@dataclass
class BoundaryState:
    """
    Complete thermodynamic state at a boundary interface.

    This represents the solution to the boundary Riemann problem,
    valid for any wave type connecting interior to boundary.
    """
    rho: float      # Interface density [kg/m³]
    u: float        # Interface velocity [m/s] (= prescribed boundary velocity)
    p: float        # Interface pressure [Pa]
    e: float        # Interface internal energy [J/kg]
    c: float        # Interface sound speed [m/s]
    gamma: float    # Ratio of specific heats

    @property
    def sigma(self) -> float:
        """Interface stress (= pressure, no AV at boundary)."""
        return self.p


class BoundaryRiemannSolver:
    """
    Solves boundary Riemann problem for prescribed velocity.

    Given:
        - Interior state: (ρ_int, u_int, p_int) from adjacent cell
        - Boundary velocity: u_bc (prescribed)

    Find:
        - Interface pressure p* such that u* = u_bc
        - Complete interface state (ρ*, u_bc, p*, e*)

    This is general for any wave type connecting interior to boundary.

    Mathematical formulation:
        The velocity at the interface is related to pressure by the
        wave function f(p*):

        For left boundary:  u* = u_int - f(p*, p_int, ρ_int)
        For right boundary: u* = u_int + f(p*, p_int, ρ_int)

        where f is:
        - Shock (p* > p):      f = (p* - p) * A^(-1/2)
        - Rarefaction (p* ≤ p): f = (2c/(γ-1)) * [(p*/p)^((γ-1)/(2γ)) - 1]

    Reference: [Toro2009] Chapter 4, Section 6.3
    """

    def __init__(self, eos: EOSBase, tol: float = 1e-8, max_iter: int = 50):
        """
        Initialize boundary Riemann solver.

        Args:
            eos: Equation of state
            tol: Convergence tolerance for Newton iteration
            max_iter: Maximum iterations
        """
        self._eos = eos
        self._tol = tol
        self._max_iter = max_iter

    def solve_left_boundary(
        self,
        rho_int: float,
        u_int: float,
        p_int: float,
        u_bc: float,
    ) -> BoundaryState:
        """
        Solve boundary Riemann problem at LEFT boundary.

        The interior state is to the RIGHT of the boundary.
        The prescribed velocity u_bc is at the LEFT (boundary face).

        For a piston moving right (u_bc > 0 > u_int), this typically
        creates a right-running shock into the interior.

        Wave structure: [boundary] --wave--> [interior]

        Args:
            rho_int: Interior density [kg/m³]
            u_int: Interior velocity [m/s]
            p_int: Interior pressure [Pa]
            u_bc: Prescribed boundary velocity [m/s]

        Returns:
            BoundaryState at the interface
        """
        # Get thermodynamic properties at interior state
        gamma = float(self._eos.get_gamma(
            np.array([rho_int]), np.array([p_int])
        )[0])
        c_int = float(self._eos.sound_speed(
            np.array([rho_int]), np.array([p_int])
        )[0])

        # Solve for interface pressure
        p_star = self._solve_for_pressure(
            rho_int, u_int, p_int, c_int, gamma, u_bc, is_left_boundary=True
        )

        # Compute complete interface state
        return self._compute_interface_state(
            rho_int, p_int, c_int, gamma, u_bc, p_star
        )

    def solve_right_boundary(
        self,
        rho_int: float,
        u_int: float,
        p_int: float,
        u_bc: float,
    ) -> BoundaryState:
        """
        Solve boundary Riemann problem at RIGHT boundary.

        The interior state is to the LEFT of the boundary.
        The prescribed velocity u_bc is at the RIGHT (boundary face).

        Wave structure: [interior] --wave--> [boundary]

        Args:
            rho_int: Interior density [kg/m³]
            u_int: Interior velocity [m/s]
            p_int: Interior pressure [Pa]
            u_bc: Prescribed boundary velocity [m/s]

        Returns:
            BoundaryState at the interface
        """
        gamma = float(self._eos.get_gamma(
            np.array([rho_int]), np.array([p_int])
        )[0])
        c_int = float(self._eos.sound_speed(
            np.array([rho_int]), np.array([p_int])
        )[0])

        p_star = self._solve_for_pressure(
            rho_int, u_int, p_int, c_int, gamma, u_bc, is_left_boundary=False
        )

        return self._compute_interface_state(
            rho_int, p_int, c_int, gamma, u_bc, p_star
        )

    def _solve_for_pressure(
        self,
        rho: float,
        u: float,
        p: float,
        c: float,
        gamma: float,
        u_bc: float,
        is_left_boundary: bool,
    ) -> float:
        """
        Find p* such that u*(p*) = u_bc.

        For left boundary (wave travels rightward into domain):
            u_bc = u_int + f(p*)
            => f(p*) = u_bc - u_int = delta_u

        For right boundary (wave travels leftward into domain):
            u_bc = u_int - f(p*)
            => f(p*) = u_int - u_bc = delta_u

        Solve: f(p*) = delta_u using Newton iteration.

        Reference: [Toro2009] Section 4.3
        """
        if is_left_boundary:
            # Wave travels from boundary INTO domain (rightward)
            # Contact velocity: u_bc = u_int + f(p*)
            # For compression (u_bc > u_int): delta_u > 0, expect shock, p* > p
            delta_u = u_bc - u
        else:
            # Wave travels from domain TO boundary (leftward)
            # Contact velocity: u_bc = u_int - f(p*)
            # For compression (u_bc < u_int): delta_u > 0, expect shock, p* > p
            delta_u = u - u_bc

        # Initial guess based on wave type
        p_star = self._initial_guess(rho, p, c, gamma, delta_u)

        # Newton iteration
        for iteration in range(self._max_iter):
            f, df = self._wave_function_and_derivative(p_star, rho, p, c, gamma)

            residual = f - delta_u

            # Check convergence
            if abs(residual) < self._tol * (abs(delta_u) + c):
                break

            # Newton update with safeguards
            if abs(df) < 1e-30:
                # Near-zero derivative, use bisection-like step
                if residual > 0:
                    p_star *= 0.5
                else:
                    p_star *= 2.0
            else:
                dp = -residual / df
                p_star_new = p_star + dp

                # Ensure positive pressure with damping
                if p_star_new <= 0:
                    p_star *= 0.5
                else:
                    p_star = p_star_new

            # Absolute bounds
            p_star = max(p_star, 1e-10)
            p_star = min(p_star, p * 1e6)  # Prevent runaway

        return p_star

    def _initial_guess(
        self,
        rho: float,
        p: float,
        c: float,
        gamma: float,
        delta_u: float,
    ) -> float:
        """
        Compute initial guess for p* based on wave type.

        For compression (delta_u > 0): expect shock, p* > p
        For expansion (delta_u < 0): expect rarefaction, p* < p

        Uses acoustic approximation as starting point.
        """
        gm1 = gamma - 1

        if delta_u >= 0:
            # Compression: acoustic approximation for shock
            p_star = p + rho * c * delta_u
        else:
            # Expansion: isentropic approximation for rarefaction
            # From: delta_u = (2c/(γ-1)) * [(p*/p)^((γ-1)/(2γ)) - 1]
            # Solve for p*/p
            term = 1 + gm1 / 2 * delta_u / c
            if term > 0:
                p_star = p * term ** (2 * gamma / gm1)
            else:
                # Near-vacuum
                p_star = p * 0.01

        return max(p_star, 1e-10)

    def _wave_function_and_derivative(
        self,
        p_star: float,
        rho: float,
        p: float,
        c: float,
        gamma: float,
    ) -> Tuple[float, float]:
        """
        Compute wave function f(p*) and its derivative df/dp*.

        f(p*) gives the velocity change across a wave connecting
        state (rho, p) to interface pressure p*.

        For shock (p* > p):
            f = (p* - p) * sqrt(A)
            where A = 2 / [(gamma+1) * rho * (p* + (gamma-1)/(gamma+1) * p)]

        For rarefaction (p* <= p):
            f = (2c/(gamma-1)) * [(p*/p)^((gamma-1)/(2*gamma)) - 1]

        Reference: [Toro2009] Equations (4.6)-(4.8), (4.37)
        """
        gp1 = gamma + 1
        gm1 = gamma - 1

        if p_star > p:
            # Shock wave - Toro equations (4.7)-(4.8)
            # B = p* + (gamma-1)/(gamma+1) * p
            B = p_star + (gm1 / gp1) * p

            # A = 2 / [(gamma+1) * rho * B]
            A = 2.0 / (gp1 * rho * B)
            sqrt_A = np.sqrt(max(A, 1e-30))

            # f = (p* - p) * sqrt(A)
            f = (p_star - p) * sqrt_A

            # Derivative from Toro equation (4.37):
            # df/dp* = sqrt(A) * [1 - (p - p*)/(2*B)]
            #        = sqrt(A) * [1 + (p* - p)/(2*B)]
            df = sqrt_A * (1.0 + (p_star - p) / (2.0 * B))

        else:
            # Rarefaction wave - Toro equations (4.6), (4.56)
            pr = p_star / p  # Pressure ratio
            exponent = gm1 / (2.0 * gamma)

            # f = (2c/(gamma-1)) * (pr^exponent - 1)
            pr_exp = pr ** exponent
            f = (2.0 * c / gm1) * (pr_exp - 1.0)

            # df/dp* = (c / (gamma * p)) * pr^(-(gamma+1)/(2*gamma))
            # From Toro equation (4.56)
            df = (c / (gamma * p)) * (pr ** (-(gp1) / (2.0 * gamma)))

        return f, df

    def _compute_interface_state(
        self,
        rho_int: float,
        p_int: float,
        c_int: float,
        gamma: float,
        u_bc: float,
        p_star: float,
    ) -> BoundaryState:
        """
        Compute complete interface state from p*.

        The density relation depends on wave type:
        - Shock (p* > p): Rankine-Hugoniot relation
        - Rarefaction (p* ≤ p): Isentropic relation

        Reference: [Toro2009] Section 4.3
        """
        gp1 = gamma + 1
        gm1 = gamma - 1

        if p_star > p_int:
            # Shock wave: Rankine-Hugoniot density relation
            # ρ*/ρ = [(γ+1)p* + (γ-1)p] / [(γ-1)p* + (γ+1)p]
            pr = p_star / p_int
            rho_ratio = (gp1 * pr + gm1) / (gm1 * pr + gp1)
            rho_star = rho_int * rho_ratio
        else:
            # Rarefaction wave: isentropic density relation
            # ρ*/ρ = (p*/p)^(1/γ)
            rho_star = rho_int * (p_star / p_int) ** (1.0 / gamma)

        # Ensure positive density
        rho_star = max(rho_star, 1e-10)

        # Internal energy and sound speed from EOS
        e_star = float(self._eos.internal_energy(
            np.array([rho_star]), np.array([p_star])
        )[0])
        c_star = float(self._eos.sound_speed(
            np.array([rho_star]), np.array([p_star])
        )[0])

        return BoundaryState(
            rho=rho_star,
            u=u_bc,
            p=p_star,
            e=e_star,
            c=c_star,
            gamma=gamma,
        )
