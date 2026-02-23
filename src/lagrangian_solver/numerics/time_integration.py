"""
Time integration schemes for Lagrangian solver.

This module implements time integration methods for advancing the
Lagrangian conservation equations in time.

References:
    [Toro2009] Section 6.4 - Time stepping and stability
    [Despres2017] Section 5.2 - Time discretization for Lagrangian schemes
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Tuple, Optional
import numpy as np

from lagrangian_solver.core.state import FlowState, ConservedVariables
from lagrangian_solver.core.grid import LagrangianGrid
from lagrangian_solver.equations.eos import EOSBase


@dataclass
class TimeStepInfo:
    """
    Information about a time step.

    Attributes:
        dt: Time step size [s]
        cfl_used: CFL number used
        limiting_cell: Cell index that limited the time step
        max_wave_speed: Maximum wave speed in domain [m/s]
    """

    dt: float
    cfl_used: float
    limiting_cell: int
    max_wave_speed: float


class TimeIntegratorBase(ABC):
    """
    Abstract base class for time integration schemes.
    """

    def __init__(self, eos: EOSBase, cfl: float = 0.5):
        """
        Initialize time integrator.

        Args:
            eos: Equation of state for computing thermodynamic quantities
            cfl: CFL number for time step control (default 0.5 for 2nd order)
        """
        self._eos = eos
        self._cfl = cfl

    @property
    def eos(self) -> EOSBase:
        """Equation of state."""
        return self._eos

    @property
    def cfl(self) -> float:
        """CFL number."""
        return self._cfl

    @cfl.setter
    def cfl(self, value: float):
        """Set CFL number."""
        if value <= 0 or value > 1:
            raise ValueError(f"CFL must be in (0, 1], got {value}")
        self._cfl = value

    def compute_timestep(
        self, state: FlowState, grid: LagrangianGrid
    ) -> TimeStepInfo:
        """
        Compute stable time step based on CFL condition.

        In Lagrangian coordinates:
            dt <= CFL * min(dx / (|u| + c))

        Reference: [Toro2009] Section 6.3

        Args:
            state: Current flow state
            grid: Lagrangian grid

        Returns:
            TimeStepInfo with time step and diagnostic information
        """
        dx = grid.dx
        c = state.c

        # Cell-averaged velocity magnitude
        u_cell = 0.5 * np.abs(state.u[:-1] + state.u[1:])

        # Wave speed
        wave_speed = u_cell + c

        # Local time step limit
        dt_local = dx / np.maximum(wave_speed, 1e-10)

        # Find limiting cell
        limiting_cell = np.argmin(dt_local)

        # Global time step
        dt = self._cfl * np.min(dt_local)

        return TimeStepInfo(
            dt=dt,
            cfl_used=self._cfl,
            limiting_cell=int(limiting_cell),
            max_wave_speed=float(np.max(wave_speed)),
        )

    @abstractmethod
    def step(
        self,
        state: FlowState,
        grid: LagrangianGrid,
        dt: float,
        rhs_func: Callable[
            [FlowState, LagrangianGrid],
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        ],
    ) -> FlowState:
        """
        Advance the solution by one time step.

        Args:
            state: Current flow state
            grid: Lagrangian grid (positions will be updated in-place)
            dt: Time step size
            rhs_func: Function that computes (d_tau, d_u, d_E, d_x)

        Returns:
            New flow state at t + dt
        """
        pass


class HeunIntegrator(TimeIntegratorBase):
    """
    Heun's method (explicit trapezoidal) - 2nd order accurate.

    Also known as the improved Euler method or RK2.

    Predictor step:
        U* = U^n + Δt · f(U^n)

    Corrector step:
        U^{n+1} = U^n + (Δt/2) · [f(U^n) + f(U*)]

    Reference: [Toro2009] Section 6.4.2, [Despres2017] Section 5.2.1
    """

    def __init__(self, eos: EOSBase, cfl: float = 0.5):
        """
        Initialize Heun's method integrator.

        Args:
            eos: Equation of state
            cfl: CFL number (default 0.5 appropriate for 2nd order methods)
        """
        super().__init__(eos, cfl)

    def step(
        self,
        state: FlowState,
        grid: LagrangianGrid,
        dt: float,
        rhs_func: Callable[
            [FlowState, LagrangianGrid],
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        ],
    ) -> FlowState:
        """
        Advance solution using Heun's method.

        Args:
            state: Current flow state at time t
            grid: Lagrangian grid
            dt: Time step size
            rhs_func: Right-hand side function returning (d_tau, d_u, d_E, d_x)

        Returns:
            New flow state at time t + dt
        """
        # Store original values
        tau_n = state.tau.copy()
        u_n = state.u.copy()
        E_n = state.E.copy()
        x_n = grid.x.copy()

        # ============ Predictor Step ============
        # Compute RHS at current state
        d_tau_n, d_u_n, d_E_n, d_x_n = rhs_func(state, grid)

        # Predict intermediate values
        tau_star = tau_n + dt * d_tau_n
        u_star = u_n + dt * d_u_n
        E_star = E_n + dt * d_E_n
        x_star = x_n + dt * d_x_n

        # Update grid to intermediate position
        grid.set_positions(x_star)

        # Create intermediate state
        state_star = FlowState.from_conserved(
            ConservedVariables(tau=tau_star, u=u_star, E=E_star),
            x=x_star,
            m=grid.m,
            eos=self._eos,
        )

        # ============ Corrector Step ============
        # Compute RHS at predicted state
        d_tau_star, d_u_star, d_E_star, d_x_star = rhs_func(state_star, grid)

        # Average the two derivatives (trapezoidal rule)
        tau_new = tau_n + 0.5 * dt * (d_tau_n + d_tau_star)
        u_new = u_n + 0.5 * dt * (d_u_n + d_u_star)
        E_new = E_n + 0.5 * dt * (d_E_n + d_E_star)
        x_new = x_n + 0.5 * dt * (d_x_n + d_x_star)

        # Update grid to final position
        grid.set_positions(x_new)

        # Create final state
        state_new = FlowState.from_conserved(
            ConservedVariables(tau=tau_new, u=u_new, E=E_new),
            x=x_new,
            m=grid.m,
            eos=self._eos,
        )

        return state_new


class ForwardEulerIntegrator(TimeIntegratorBase):
    """
    Forward Euler method - 1st order accurate.

    Simple explicit method:
        U^{n+1} = U^n + Δt · f(U^n)

    Only 1st order accurate, primarily useful for debugging or
    comparison purposes. Use CFL <= 0.5 for stability.

    Reference: [Toro2009] Section 6.4.1
    """

    def __init__(self, eos: EOSBase, cfl: float = 0.3):
        """
        Initialize Forward Euler integrator.

        Args:
            eos: Equation of state
            cfl: CFL number (default 0.3, more restrictive for 1st order)
        """
        super().__init__(eos, cfl)

    def step(
        self,
        state: FlowState,
        grid: LagrangianGrid,
        dt: float,
        rhs_func: Callable[
            [FlowState, LagrangianGrid],
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        ],
    ) -> FlowState:
        """
        Advance solution using Forward Euler method.

        Args:
            state: Current flow state at time t
            grid: Lagrangian grid
            dt: Time step size
            rhs_func: Right-hand side function

        Returns:
            New flow state at time t + dt
        """
        # Compute RHS at current state
        d_tau, d_u, d_E, d_x = rhs_func(state, grid)

        # Update conserved variables
        tau_new = state.tau + dt * d_tau
        u_new = state.u + dt * d_u
        E_new = state.E + dt * d_E
        x_new = grid.x + dt * d_x

        # Update grid positions
        grid.set_positions(x_new)

        # Create new state
        state_new = FlowState.from_conserved(
            ConservedVariables(tau=tau_new, u=u_new, E=E_new),
            x=x_new,
            m=grid.m,
            eos=self._eos,
        )

        return state_new


class SSPRK3Integrator(TimeIntegratorBase):
    """
    Strong Stability Preserving Runge-Kutta 3rd order (SSP-RK3).

    A 3-stage, 3rd order method that preserves TVD/TVB properties.

    Stage 1: U^(1) = U^n + Δt · f(U^n)
    Stage 2: U^(2) = 3/4 U^n + 1/4 (U^(1) + Δt · f(U^(1)))
    Stage 3: U^{n+1} = 1/3 U^n + 2/3 (U^(2) + Δt · f(U^(2)))

    Reference: Gottlieb & Shu (1998), Math. Comp. 67, 73-85
    """

    def __init__(self, eos: EOSBase, cfl: float = 0.5):
        """
        Initialize SSP-RK3 integrator.

        Args:
            eos: Equation of state
            cfl: CFL number (effective CFL is cfl, not cfl/3)
        """
        super().__init__(eos, cfl)

    def step(
        self,
        state: FlowState,
        grid: LagrangianGrid,
        dt: float,
        rhs_func: Callable[
            [FlowState, LagrangianGrid],
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        ],
    ) -> FlowState:
        """
        Advance solution using SSP-RK3 method.
        """
        # Store original values
        tau_0 = state.tau.copy()
        u_0 = state.u.copy()
        E_0 = state.E.copy()
        x_0 = grid.x.copy()

        # ============ Stage 1 ============
        d_tau_0, d_u_0, d_E_0, d_x_0 = rhs_func(state, grid)

        tau_1 = tau_0 + dt * d_tau_0
        u_1 = u_0 + dt * d_u_0
        E_1 = E_0 + dt * d_E_0
        x_1 = x_0 + dt * d_x_0

        grid.set_positions(x_1)
        state_1 = FlowState.from_conserved(
            ConservedVariables(tau=tau_1, u=u_1, E=E_1),
            x=x_1,
            m=grid.m,
            eos=self._eos,
        )

        # ============ Stage 2 ============
        d_tau_1, d_u_1, d_E_1, d_x_1 = rhs_func(state_1, grid)

        tau_2 = 0.75 * tau_0 + 0.25 * (tau_1 + dt * d_tau_1)
        u_2 = 0.75 * u_0 + 0.25 * (u_1 + dt * d_u_1)
        E_2 = 0.75 * E_0 + 0.25 * (E_1 + dt * d_E_1)
        x_2 = 0.75 * x_0 + 0.25 * (x_1 + dt * d_x_1)

        grid.set_positions(x_2)
        state_2 = FlowState.from_conserved(
            ConservedVariables(tau=tau_2, u=u_2, E=E_2),
            x=x_2,
            m=grid.m,
            eos=self._eos,
        )

        # ============ Stage 3 ============
        d_tau_2, d_u_2, d_E_2, d_x_2 = rhs_func(state_2, grid)

        tau_new = (1 / 3) * tau_0 + (2 / 3) * (tau_2 + dt * d_tau_2)
        u_new = (1 / 3) * u_0 + (2 / 3) * (u_2 + dt * d_u_2)
        E_new = (1 / 3) * E_0 + (2 / 3) * (E_2 + dt * d_E_2)
        x_new = (1 / 3) * x_0 + (2 / 3) * (x_2 + dt * d_x_2)

        grid.set_positions(x_new)
        state_new = FlowState.from_conserved(
            ConservedVariables(tau=tau_new, u=u_new, E=E_new),
            x=x_new,
            m=grid.m,
            eos=self._eos,
        )

        return state_new
