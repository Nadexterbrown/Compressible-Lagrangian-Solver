"""
Flame property interpolation from pre-computed Cantera data.

Loads flame_properties.csv and provides 2D P-T interpolation for
laminar flame speed (S_L), unburned density (rho_u), burned density (rho_b),
and other flame properties.

The interpolation is used in the flame-coupled piston boundary condition
to compute piston velocity from the Clavin-Tofaili formula:
    u_p = (sigma - 1) * (rho_u / rho_b) * S_L

Reference:
    Clavin, P., & Tofaili, H. (2021). Flame elongation model.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import warnings

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator


@dataclass
class FlameProperties:
    """
    Flame properties at a given (P, T) condition.

    Attributes:
        S_L: Laminar flame speed [m/s]
        rho_u: Unburned mixture density [kg/m^3]
        rho_b: Burned mixture density [kg/m^3]
        T_ad: Adiabatic flame temperature [K]
    """

    S_L: float
    rho_u: float
    rho_b: float
    T_ad: float

    @property
    def density_ratio(self) -> float:
        """Density ratio rho_u / rho_b."""
        return self.rho_u / self.rho_b


class FlamePropertyInterpolator:
    """
    2D interpolator for flame properties as functions of (P, T).

    Loads pre-computed flame data from CSV and builds interpolators
    using scipy.interpolate.RegularGridInterpolator for efficient
    2D interpolation on the structured P-T grid.

    Attributes:
        P_min, P_max: Pressure bounds [Pa]
        T_min, T_max: Temperature bounds [K]
        n_P, n_T: Number of grid points in each dimension
    """

    def __init__(
        self,
        csv_path: str,
        interpolation_method: str = "linear",
        bounds_error: bool = False,
        fill_value: Optional[float] = None,
    ):
        """
        Load CSV and build interpolators.

        Args:
            csv_path: Path to flame_properties.csv
            interpolation_method: Interpolation method ("linear" or "nearest")
            bounds_error: If True, raise error for out-of-bounds queries
            fill_value: Value for out-of-bounds queries (None = extrapolate)
        """
        self._csv_path = Path(csv_path)
        self._method = interpolation_method
        self._bounds_error = bounds_error
        self._fill_value = fill_value

        # Load and validate CSV data
        self._load_data()

        # Build interpolators
        self._build_interpolators()

    def _load_data(self) -> None:
        """Load CSV and extract P-T grid structure."""
        if not self._csv_path.exists():
            raise FileNotFoundError(f"Flame properties file not found: {self._csv_path}")

        # Load CSV
        df = pd.read_csv(self._csv_path)

        # Expected columns from flame_properties.csv
        required_cols = {"T [K]", "P [Pa]", "Su [m/s]", "rho_u [kg/m3]", "rho_b [kg/m3]", "T_ad [K]"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Extract unique P and T values (sorted)
        self._T_values = np.sort(df["T [K]"].unique())
        self._P_values = np.sort(df["P [Pa]"].unique())

        self._n_T = len(self._T_values)
        self._n_P = len(self._P_values)

        # Store bounds for validation
        self._T_min = self._T_values[0]
        self._T_max = self._T_values[-1]
        self._P_min = self._P_values[0]
        self._P_max = self._P_values[-1]

        # Reshape data to (n_P, n_T) grids
        # CSV is ordered by P (outer loop), T (inner loop)
        n_total = len(df)
        expected = self._n_P * self._n_T
        if n_total != expected:
            raise ValueError(f"Expected {expected} rows ({self._n_P}x{self._n_T}), got {n_total}")

        # Create 2D arrays for each property
        self._S_L_grid = np.zeros((self._n_P, self._n_T))
        self._rho_u_grid = np.zeros((self._n_P, self._n_T))
        self._rho_b_grid = np.zeros((self._n_P, self._n_T))
        self._T_ad_grid = np.zeros((self._n_P, self._n_T))

        # Fill grids by iterating through dataframe
        for _, row in df.iterrows():
            P = row["P [Pa]"]
            T = row["T [K]"]

            # Find indices
            i_P = np.searchsorted(self._P_values, P)
            i_T = np.searchsorted(self._T_values, T)

            # Handle floating point tolerance
            if i_P >= self._n_P or not np.isclose(self._P_values[i_P], P, rtol=1e-6):
                i_P = np.argmin(np.abs(self._P_values - P))
            if i_T >= self._n_T or not np.isclose(self._T_values[i_T], T, rtol=1e-6):
                i_T = np.argmin(np.abs(self._T_values - T))

            self._S_L_grid[i_P, i_T] = row["Su [m/s]"]
            self._rho_u_grid[i_P, i_T] = row["rho_u [kg/m3]"]
            self._rho_b_grid[i_P, i_T] = row["rho_b [kg/m3]"]
            self._T_ad_grid[i_P, i_T] = row["T_ad [K]"]

    def _build_interpolators(self) -> None:
        """Build RegularGridInterpolators for each property."""
        # RegularGridInterpolator expects (P, T) order
        # Grid axes must be strictly increasing
        self._S_L_interp = RegularGridInterpolator(
            (self._P_values, self._T_values),
            self._S_L_grid,
            method=self._method,
            bounds_error=self._bounds_error,
            fill_value=self._fill_value,
        )
        self._rho_u_interp = RegularGridInterpolator(
            (self._P_values, self._T_values),
            self._rho_u_grid,
            method=self._method,
            bounds_error=self._bounds_error,
            fill_value=self._fill_value,
        )
        self._rho_b_interp = RegularGridInterpolator(
            (self._P_values, self._T_values),
            self._rho_b_grid,
            method=self._method,
            bounds_error=self._bounds_error,
            fill_value=self._fill_value,
        )
        self._T_ad_interp = RegularGridInterpolator(
            (self._P_values, self._T_values),
            self._T_ad_grid,
            method=self._method,
            bounds_error=self._bounds_error,
            fill_value=self._fill_value,
        )

    @property
    def P_min(self) -> float:
        """Minimum pressure in dataset [Pa]."""
        return self._P_min

    @property
    def P_max(self) -> float:
        """Maximum pressure in dataset [Pa]."""
        return self._P_max

    @property
    def T_min(self) -> float:
        """Minimum temperature in dataset [K]."""
        return self._T_min

    @property
    def T_max(self) -> float:
        """Maximum temperature in dataset [K]."""
        return self._T_max

    @property
    def n_P(self) -> int:
        """Number of pressure grid points."""
        return self._n_P

    @property
    def n_T(self) -> int:
        """Number of temperature grid points."""
        return self._n_T

    @property
    def P_values(self) -> np.ndarray:
        """Pressure grid values [Pa]."""
        return self._P_values.copy()

    @property
    def T_values(self) -> np.ndarray:
        """Temperature grid values [K]."""
        return self._T_values.copy()

    def _check_bounds(self, P: float, T: float) -> None:
        """Warn if query is outside data bounds."""
        if P < self._P_min or P > self._P_max:
            warnings.warn(
                f"Pressure {P:.2e} Pa outside data bounds [{self._P_min:.2e}, {self._P_max:.2e}] Pa. "
                "Extrapolation may be inaccurate.",
                RuntimeWarning,
                stacklevel=3,
            )
        if T < self._T_min or T > self._T_max:
            warnings.warn(
                f"Temperature {T:.2f} K outside data bounds [{self._T_min:.2f}, {self._T_max:.2f}] K. "
                "Extrapolation may be inaccurate.",
                RuntimeWarning,
                stacklevel=3,
            )

    def get_properties(self, P: float, T: float) -> FlameProperties:
        """
        Interpolate all flame properties at given (P, T).

        Args:
            P: Pressure [Pa]
            T: Temperature [K]

        Returns:
            FlameProperties containing S_L, rho_u, rho_b, T_ad
        """
        self._check_bounds(P, T)

        pt = np.array([[P, T]])

        S_L = float(self._S_L_interp(pt)[0])
        rho_u = float(self._rho_u_interp(pt)[0])
        rho_b = float(self._rho_b_interp(pt)[0])
        T_ad = float(self._T_ad_interp(pt)[0])

        return FlameProperties(S_L=S_L, rho_u=rho_u, rho_b=rho_b, T_ad=T_ad)

    def get_flame_speed(self, P: float, T: float) -> float:
        """
        Get laminar flame speed S_L at given (P, T).

        This is a convenience method for efficient single-property lookup
        during iteration.

        Args:
            P: Pressure [Pa]
            T: Temperature [K]

        Returns:
            Laminar flame speed [m/s]
        """
        self._check_bounds(P, T)
        pt = np.array([[P, T]])
        return float(self._S_L_interp(pt)[0])

    def get_density_ratio(self, P: float, T: float) -> float:
        """
        Get density ratio rho_u / rho_b at given (P, T).

        This is used directly in the Clavin-Tofaili formula.

        Args:
            P: Pressure [Pa]
            T: Temperature [K]

        Returns:
            Density ratio (dimensionless)
        """
        self._check_bounds(P, T)
        pt = np.array([[P, T]])
        rho_u = float(self._rho_u_interp(pt)[0])
        rho_b = float(self._rho_b_interp(pt)[0])
        return rho_u / rho_b

    def get_properties_batch(
        self, P: np.ndarray, T: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Batch interpolation for arrays of (P, T) values.

        Args:
            P: Array of pressures [Pa]
            T: Array of temperatures [K]

        Returns:
            Tuple of (S_L, rho_u, rho_b, T_ad) arrays
        """
        P = np.atleast_1d(P)
        T = np.atleast_1d(T)

        if len(P) != len(T):
            raise ValueError("P and T arrays must have same length")

        pts = np.column_stack([P, T])

        S_L = self._S_L_interp(pts)
        rho_u = self._rho_u_interp(pts)
        rho_b = self._rho_b_interp(pts)
        T_ad = self._T_ad_interp(pts)

        return S_L, rho_u, rho_b, T_ad

    def __repr__(self) -> str:
        return (
            f"FlamePropertyInterpolator("
            f"P=[{self._P_min:.2e}, {self._P_max:.2e}] Pa, "
            f"T=[{self._T_min:.1f}, {self._T_max:.1f}] K, "
            f"grid={self._n_P}x{self._n_T})"
        )
