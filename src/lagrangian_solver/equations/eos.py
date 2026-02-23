"""
Equation of State (EOS) implementations.

This module provides EOS classes for computing thermodynamic properties
required by the Lagrangian solver.

References:
    [Toro2009] Section 1.2 - Thermodynamic relations for ideal gases
    [Cantera] Goodwin et al. (2023) - Cantera thermodynamic library
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


@dataclass
class ThermodynamicState:
    """
    Complete thermodynamic state at a point.

    Attributes:
        rho: Density [kg/m³]
        p: Pressure [Pa]
        T: Temperature [K]
        e: Specific internal energy [J/kg]
        c: Sound speed [m/s]
        gamma: Ratio of specific heats (cp/cv) [-]
    """

    rho: float
    p: float
    T: float
    e: float
    c: float
    gamma: float


class EOSBase(ABC):
    """
    Abstract base class for equation of state implementations.

    The EOS provides closure relations between thermodynamic variables.
    All EOS implementations must provide methods to compute:
    - Pressure from density and internal energy
    - Temperature from density and internal energy
    - Sound speed from density and pressure
    - Complete state from any two independent variables
    """

    @abstractmethod
    def pressure(self, rho: np.ndarray, e: np.ndarray) -> np.ndarray:
        """
        Compute pressure from density and specific internal energy.

        Args:
            rho: Density [kg/m³]
            e: Specific internal energy [J/kg]

        Returns:
            Pressure [Pa]
        """
        pass

    @abstractmethod
    def temperature(self, rho: np.ndarray, e: np.ndarray) -> np.ndarray:
        """
        Compute temperature from density and specific internal energy.

        Args:
            rho: Density [kg/m³]
            e: Specific internal energy [J/kg]

        Returns:
            Temperature [K]
        """
        pass

    @abstractmethod
    def sound_speed(self, rho: np.ndarray, p: np.ndarray) -> np.ndarray:
        """
        Compute sound speed from density and pressure.

        Args:
            rho: Density [kg/m³]
            p: Pressure [Pa]

        Returns:
            Sound speed [m/s]
        """
        pass

    @abstractmethod
    def internal_energy(self, rho: np.ndarray, p: np.ndarray) -> np.ndarray:
        """
        Compute specific internal energy from density and pressure.

        Args:
            rho: Density [kg/m³]
            p: Pressure [Pa]

        Returns:
            Specific internal energy [J/kg]
        """
        pass

    @abstractmethod
    def get_gamma(self, rho: np.ndarray, p: np.ndarray) -> np.ndarray:
        """
        Get the ratio of specific heats.

        For ideal gas this is constant, for real gases it varies with state.

        Args:
            rho: Density [kg/m³]
            p: Pressure [Pa]

        Returns:
            Ratio of specific heats gamma = cp/cv [-]
        """
        pass

    @abstractmethod
    def entropy(self, rho: np.ndarray, p: np.ndarray) -> np.ndarray:
        """
        Compute specific entropy from density and pressure.

        Args:
            rho: Density [kg/m³]
            p: Pressure [Pa]

        Returns:
            Specific entropy [J/(kg·K)]
        """
        pass

    def complete_state(
        self, rho: float, p: Optional[float] = None, e: Optional[float] = None
    ) -> ThermodynamicState:
        """
        Compute complete thermodynamic state from density and one other variable.

        Args:
            rho: Density [kg/m³]
            p: Pressure [Pa] (optional, provide p or e)
            e: Specific internal energy [J/kg] (optional, provide p or e)

        Returns:
            Complete ThermodynamicState

        Raises:
            ValueError: If neither p nor e is provided, or both are provided
        """
        if (p is None) == (e is None):
            raise ValueError("Provide exactly one of p or e")

        rho_arr = np.atleast_1d(rho)

        if e is not None:
            e_arr = np.atleast_1d(e)
            p_arr = self.pressure(rho_arr, e_arr)
            p = float(p_arr[0])
            e = float(e_arr[0])
        else:
            p_arr = np.atleast_1d(p)
            e_arr = self.internal_energy(rho_arr, p_arr)
            p = float(p_arr[0])
            e = float(e_arr[0])

        T = float(self.temperature(rho_arr, e_arr)[0])
        c = float(self.sound_speed(rho_arr, p_arr)[0])
        gamma = float(self.get_gamma(rho_arr, p_arr)[0])

        return ThermodynamicState(rho=rho, p=p, T=T, e=e, c=c, gamma=gamma)


class IdealGasEOS(EOSBase):
    """
    Ideal gas equation of state.

    For an ideal gas with constant specific heats:
        p = ρ R T = (γ - 1) ρ e
        e = c_v T = p / (ρ (γ - 1))
        c = √(γ p / ρ)

    Reference: [Toro2009] Section 1.2, Equations (1.20)-(1.25)

    Attributes:
        gamma: Ratio of specific heats (default 1.4 for air)
        R: Specific gas constant [J/(kg·K)] (default 287.05 for air)
    """

    def __init__(self, gamma: float = 1.4, R: float = 287.05):
        """
        Initialize ideal gas EOS.

        Args:
            gamma: Ratio of specific heats cp/cv (default 1.4)
            R: Specific gas constant [J/(kg·K)] (default 287.05 for air)
        """
        if gamma <= 1.0:
            raise ValueError(f"gamma must be > 1, got {gamma}")
        if R <= 0:
            raise ValueError(f"R must be > 0, got {R}")

        self._gamma = gamma
        self._R = R
        self._cv = R / (gamma - 1)
        self._cp = gamma * self._cv

    @property
    def gamma(self) -> float:
        """Ratio of specific heats."""
        return self._gamma

    @property
    def R(self) -> float:
        """Specific gas constant [J/(kg·K)]."""
        return self._R

    @property
    def cv(self) -> float:
        """Specific heat at constant volume [J/(kg·K)]."""
        return self._cv

    @property
    def cp(self) -> float:
        """Specific heat at constant pressure [J/(kg·K)]."""
        return self._cp

    def pressure(self, rho: np.ndarray, e: np.ndarray) -> np.ndarray:
        """
        Compute pressure from density and specific internal energy.

        p = (γ - 1) ρ e

        Reference: [Toro2009] Equation (1.23)
        """
        rho = np.asarray(rho)
        e = np.asarray(e)
        return (self._gamma - 1) * rho * e

    def temperature(self, rho: np.ndarray, e: np.ndarray) -> np.ndarray:
        """
        Compute temperature from density and specific internal energy.

        T = e / c_v

        Reference: [Toro2009] Equation (1.21)
        """
        e = np.asarray(e)
        return e / self._cv

    def sound_speed(self, rho: np.ndarray, p: np.ndarray) -> np.ndarray:
        """
        Compute sound speed from density and pressure.

        c = √(γ p / ρ)

        Reference: [Toro2009] Equation (1.25)
        """
        rho = np.asarray(rho)
        p = np.asarray(p)
        # Ensure non-negative argument for sqrt (positivity preservation)
        arg = self._gamma * np.maximum(p, 1e-30) / np.maximum(rho, 1e-30)
        return np.sqrt(arg)

    def internal_energy(self, rho: np.ndarray, p: np.ndarray) -> np.ndarray:
        """
        Compute specific internal energy from density and pressure.

        e = p / (ρ (γ - 1))

        Reference: [Toro2009] Equation (1.23) rearranged
        """
        rho = np.asarray(rho)
        p = np.asarray(p)
        return p / (rho * (self._gamma - 1))

    def get_gamma(self, rho: np.ndarray, p: np.ndarray) -> np.ndarray:
        """Get the constant ratio of specific heats."""
        rho = np.asarray(rho)
        return np.full_like(rho, self._gamma, dtype=float)

    def temperature_from_rho_p(self, rho: np.ndarray, p: np.ndarray) -> np.ndarray:
        """
        Compute temperature from density and pressure.

        T = p / (ρ R)

        Reference: Ideal gas law
        """
        rho = np.asarray(rho)
        p = np.asarray(p)
        return p / (rho * self._R)

    def density_from_p_T(self, p: np.ndarray, T: np.ndarray) -> np.ndarray:
        """
        Compute density from pressure and temperature.

        ρ = p / (R T)

        Reference: Ideal gas law
        """
        p = np.asarray(p)
        T = np.asarray(T)
        return p / (self._R * T)

    def entropy(self, rho: np.ndarray, p: np.ndarray) -> np.ndarray:
        """
        Compute specific entropy from density and pressure.

        For ideal gas:
            s = cv * ln(T/T_ref) - R * ln(rho/rho_ref) + s_ref

        Simplified (relative entropy):
            s = cv * ln(p / rho^gamma) + const

        Using s = cv * ln(T) - R * ln(rho) form:

        Reference: Ideal gas thermodynamics
        """
        rho = np.asarray(rho)
        p = np.asarray(p)

        # Ensure positive values for log (positivity preservation)
        rho_safe = np.maximum(rho, 1e-30)
        p_safe = np.maximum(p, 1e-30)

        T = self.temperature_from_rho_p(rho_safe, p_safe)
        T_safe = np.maximum(T, 1e-30)

        # Specific entropy (relative to reference state)
        # s - s_ref = cv * ln(T/T_ref) - R * ln(rho/rho_ref)
        # Using T_ref = 300 K, rho_ref = 1.0 kg/m³
        T_ref = 300.0
        rho_ref = 1.0
        s = self._cv * np.log(T_safe / T_ref) - self._R * np.log(rho_safe / rho_ref)
        return s


class CanteraEOS(EOSBase):
    """
    Cantera-based equation of state for real gas mixtures.

    Uses Cantera for accurate thermodynamic property calculations including
    multi-species mixtures with temperature-dependent properties.

    Reference: [Cantera] Goodwin et al. (2023)

    Example usage following CLAUDE.md INPUT_PARAMS_CONFIG:
        INPUT_PARAMS_CONFIG = {
            'T': 503,
            'P': 10e5,
            'Phi': 1.0,
            'Fuel': 'H2',
            'Oxidizer': 'O2:1, N2:3.76',
            'mech': '../../chemical_mechanism/LiDryer.yaml',
        }
        eos = CanteraEOS(mechanism_file=INPUT_PARAMS_CONFIG['mech'])
        eos.set_mixture(fuel=INPUT_PARAMS_CONFIG['Fuel'],
                       oxidizer=INPUT_PARAMS_CONFIG['Oxidizer'],
                       phi=INPUT_PARAMS_CONFIG['Phi'])
    """

    def __init__(self, mechanism_file: str, phase_name: str = ""):
        """
        Initialize Cantera EOS with a mechanism file.

        Args:
            mechanism_file: Path to Cantera mechanism file (.yaml, .cti, or .xml)
            phase_name: Name of the phase to use (empty string for default)
        """
        try:
            import cantera as ct
        except ImportError:
            raise ImportError(
                "Cantera is required for CanteraEOS. "
                "Install with: pip install cantera"
            )

        self._ct = ct
        self._mechanism_file = mechanism_file

        if phase_name:
            self._gas = ct.Solution(mechanism_file, phase_name)
        else:
            self._gas = ct.Solution(mechanism_file)

        self._mixture_set = False

    @property
    def gas(self):
        """Access the underlying Cantera Solution object."""
        return self._gas

    def set_mixture(
        self,
        fuel: str,
        oxidizer: str,
        phi: float,
    ):
        """
        Set the mixture composition using equivalence ratio.

        Args:
            fuel: Fuel species string (e.g., 'H2')
            oxidizer: Oxidizer species string (e.g., 'O2:1, N2:3.76')
            phi: Equivalence ratio
        """
        self._gas.set_equivalence_ratio(phi, fuel, oxidizer)
        self._mixture_set = True

    def set_state_TP(self, T: float, P: float):
        """
        Set the thermodynamic state from temperature and pressure.

        Args:
            T: Temperature [K]
            P: Pressure [Pa]
        """
        self._gas.TP = T, P

    def set_state_rho_e(self, rho: float, e: float):
        """
        Set the thermodynamic state from density and internal energy.

        Args:
            rho: Density [kg/m³]
            e: Specific internal energy [J/kg]
        """
        self._gas.UV = e, 1.0 / rho

    def pressure(self, rho: np.ndarray, e: np.ndarray) -> np.ndarray:
        """
        Compute pressure from density and specific internal energy.

        Uses Cantera's internal solver to find consistent state.
        """
        rho = np.atleast_1d(rho)
        e = np.atleast_1d(e)
        p = np.zeros_like(rho, dtype=float)

        for i in range(len(rho)):
            self.set_state_rho_e(rho[i], e[i])
            p[i] = self._gas.P

        return p

    def temperature(self, rho: np.ndarray, e: np.ndarray) -> np.ndarray:
        """
        Compute temperature from density and specific internal energy.
        """
        rho = np.atleast_1d(rho)
        e = np.atleast_1d(e)
        T = np.zeros_like(rho, dtype=float)

        for i in range(len(rho)):
            self.set_state_rho_e(rho[i], e[i])
            T[i] = self._gas.T

        return T

    def sound_speed(self, rho: np.ndarray, p: np.ndarray) -> np.ndarray:
        """
        Compute sound speed from density and pressure.

        Uses Cantera's internal computation which accounts for real gas effects.
        """
        rho = np.atleast_1d(rho)
        p = np.atleast_1d(p)
        c = np.zeros_like(rho, dtype=float)

        for i in range(len(rho)):
            self._gas.DP = rho[i], p[i]
            # Cantera provides isentropic sound speed
            # c^2 = (∂p/∂ρ)_s = γ p / ρ for ideal gas
            # For real gas, use Cantera's internal calculation
            gamma = self._gas.cp / self._gas.cv
            c[i] = np.sqrt(gamma * p[i] / rho[i])

        return c

    def internal_energy(self, rho: np.ndarray, p: np.ndarray) -> np.ndarray:
        """
        Compute specific internal energy from density and pressure.
        """
        rho = np.atleast_1d(rho)
        p = np.atleast_1d(p)
        e = np.zeros_like(rho, dtype=float)

        for i in range(len(rho)):
            self._gas.DP = rho[i], p[i]
            e[i] = self._gas.int_energy_mass

        return e

    def get_gamma(self, rho: np.ndarray, p: np.ndarray) -> np.ndarray:
        """
        Get the ratio of specific heats at the given state.

        For real gases, gamma varies with temperature and composition.
        """
        rho = np.atleast_1d(rho)
        p = np.atleast_1d(p)
        gamma = np.zeros_like(rho, dtype=float)

        for i in range(len(rho)):
            self._gas.DP = rho[i], p[i]
            gamma[i] = self._gas.cp / self._gas.cv

        return gamma

    def get_R(self, rho: np.ndarray, p: np.ndarray) -> np.ndarray:
        """
        Get the specific gas constant at the given state.

        R = R_universal / M where M is the mixture molecular weight.
        """
        rho = np.atleast_1d(rho)
        p = np.atleast_1d(p)
        R = np.zeros_like(rho, dtype=float)

        for i in range(len(rho)):
            self._gas.DP = rho[i], p[i]
            R[i] = self._ct.gas_constant / self._gas.mean_molecular_weight

        return R

    def entropy(self, rho: np.ndarray, p: np.ndarray) -> np.ndarray:
        """
        Compute specific entropy from density and pressure using Cantera.

        Uses Cantera's accurate calculation of entropy.
        """
        rho = np.atleast_1d(rho)
        p = np.atleast_1d(p)
        s = np.zeros_like(rho, dtype=float)

        for i in range(len(rho)):
            self._gas.DP = rho[i], p[i]
            s[i] = self._gas.entropy_mass

        return s
