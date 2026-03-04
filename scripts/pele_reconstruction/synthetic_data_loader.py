"""
Loader for synthetic trajectory data.

Reads CSV files with columns: time, flame_position, flame_velocity, flame_gas_velocity
and provides interpolation compatible with PeleTrajectoryInterpolator interface.

The synthetic data can be generated from the synthetic_trajectory_gui.py tool
or from other sources.
"""

import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from scipy.interpolate import interp1d


@dataclass
class SyntheticTrajectoryData:
    """Container for synthetic trajectory data."""
    time: np.ndarray                    # [s]
    flame_position: np.ndarray          # [m]
    flame_velocity: np.ndarray          # [m/s]
    flame_gas_velocity: np.ndarray      # [m/s]

    # Optional fields (for compatibility with PeleTrajectoryData)
    burned_gas_velocity: Optional[np.ndarray] = None
    pressure: Optional[np.ndarray] = None
    temperature: Optional[np.ndarray] = None
    density: Optional[np.ndarray] = None

    def __post_init__(self):
        """Validate data consistency."""
        n = len(self.time)
        assert len(self.flame_position) == n, "flame_position length mismatch"
        assert len(self.flame_velocity) == n, "flame_velocity length mismatch"
        assert len(self.flame_gas_velocity) == n, "flame_gas_velocity length mismatch"


class SyntheticDataLoader:
    """
    Load synthetic trajectory data from CSV files.

    Expected CSV format:
        time,flame_position,flame_velocity,flame_gas_velocity
        0.0,0.0,0.0,0.0
        1e-6,0.001,100.0,80.0
        ...

    The CSV can optionally include additional columns:
        burned_gas_velocity, pressure, temperature, density
    """

    def __init__(self, data_path: str):
        """
        Initialize loader with path to CSV file.

        Parameters
        ----------
        data_path : str or Path
            Path to CSV file
        """
        self.data_path = Path(data_path)
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

    def load(self) -> SyntheticTrajectoryData:
        """
        Load trajectory data from CSV file.

        Returns
        -------
        SyntheticTrajectoryData
            Loaded trajectory data
        """
        # Read CSV with numpy
        with open(self.data_path, 'r') as f:
            header = f.readline().strip().split(',')

        # Load data
        data = np.loadtxt(self.data_path, delimiter=',', skiprows=1)

        # Map columns
        col_map = {name.strip(): i for i, name in enumerate(header)}

        # Required columns
        required = ['time', 'flame_position', 'flame_velocity', 'flame_gas_velocity']
        for col in required:
            if col not in col_map:
                raise ValueError(f"Required column '{col}' not found in CSV. "
                               f"Available: {list(col_map.keys())}")

        time = data[:, col_map['time']]
        flame_position = data[:, col_map['flame_position']]
        flame_velocity = data[:, col_map['flame_velocity']]
        flame_gas_velocity = data[:, col_map['flame_gas_velocity']]

        # Optional columns
        burned_gas_velocity = None
        if 'burned_gas_velocity' in col_map:
            burned_gas_velocity = data[:, col_map['burned_gas_velocity']]

        pressure = None
        if 'pressure' in col_map:
            pressure = data[:, col_map['pressure']]

        temperature = None
        if 'temperature' in col_map:
            temperature = data[:, col_map['temperature']]

        density = None
        if 'density' in col_map:
            density = data[:, col_map['density']]

        return SyntheticTrajectoryData(
            time=time,
            flame_position=flame_position,
            flame_velocity=flame_velocity,
            flame_gas_velocity=flame_gas_velocity,
            burned_gas_velocity=burned_gas_velocity,
            pressure=pressure,
            temperature=temperature,
            density=density,
        )


class SyntheticTrajectoryInterpolator:
    """
    Interpolator for synthetic trajectory data.

    Provides the same interface as PeleTrajectoryInterpolator for
    compatibility with boundary conditions.
    """

    def __init__(self, data: SyntheticTrajectoryData, extrapolate: bool = False):
        """
        Initialize interpolators.

        Parameters
        ----------
        data : SyntheticTrajectoryData
            Trajectory data to interpolate
        extrapolate : bool
            If True, allow extrapolation beyond data bounds
        """
        self._data = data
        self._extrapolate = extrapolate

        bounds_error = not extrapolate
        fill_value = "extrapolate" if extrapolate else None

        self._interp_position = interp1d(
            data.time, data.flame_position,
            kind='linear', bounds_error=bounds_error, fill_value=fill_value
        )
        self._interp_velocity = interp1d(
            data.time, data.flame_velocity,
            kind='linear', bounds_error=bounds_error, fill_value=fill_value
        )
        self._interp_gas_velocity = interp1d(
            data.time, data.flame_gas_velocity,
            kind='linear', bounds_error=bounds_error, fill_value=fill_value
        )

        # Optional interpolators
        if data.burned_gas_velocity is not None:
            self._interp_burned_gas_velocity = interp1d(
                data.time, data.burned_gas_velocity,
                kind='linear', bounds_error=bounds_error, fill_value=fill_value
            )
        else:
            self._interp_burned_gas_velocity = None

        if data.pressure is not None:
            self._interp_pressure = interp1d(
                data.time, data.pressure,
                kind='linear', bounds_error=bounds_error, fill_value=fill_value
            )
        else:
            self._interp_pressure = None

        if data.temperature is not None:
            self._interp_temperature = interp1d(
                data.time, data.temperature,
                kind='linear', bounds_error=bounds_error, fill_value=fill_value
            )
        else:
            self._interp_temperature = None

    @property
    def t_min(self) -> float:
        """Minimum time in data."""
        return float(self._data.time[0])

    @property
    def t_max(self) -> float:
        """Maximum time in data."""
        return float(self._data.time[-1])

    @property
    def data(self) -> SyntheticTrajectoryData:
        """Return underlying data."""
        return self._data

    def position(self, t: float) -> float:
        """Interpolate flame position at time t."""
        return float(self._interp_position(t))

    def velocity(self, t: float) -> float:
        """Interpolate flame velocity at time t."""
        return float(self._interp_velocity(t))

    def gas_velocity(self, t: float) -> float:
        """Interpolate flame gas velocity at time t."""
        return float(self._interp_gas_velocity(t))

    def burned_gas_velocity(self, t: float) -> Optional[float]:
        """
        Interpolate burned gas velocity at time t.

        If burned_gas_velocity data is not available, falls back to
        flame_gas_velocity (they may be equivalent in some datasets).
        """
        if self._interp_burned_gas_velocity is not None:
            return float(self._interp_burned_gas_velocity(t))
        else:
            # Fall back to flame_gas_velocity
            return float(self._interp_gas_velocity(t))

    def pressure(self, t: float) -> Optional[float]:
        """Interpolate pressure at time t."""
        if self._interp_pressure is None:
            return None
        return float(self._interp_pressure(t))

    def temperature(self, t: float) -> Optional[float]:
        """Interpolate temperature at time t."""
        if self._interp_temperature is None:
            return None
        return float(self._interp_temperature(t))

    def __repr__(self) -> str:
        return (
            f"SyntheticTrajectoryInterpolator("
            f"t=[{self.t_min:.6e}, {self.t_max:.6e}] s, "
            f"n_points={len(self._data.time)})"
        )


def load_synthetic_trajectory(csv_path: str, extrapolate: bool = False) -> SyntheticTrajectoryInterpolator:
    """
    Convenience function to load and create interpolator from CSV.

    Parameters
    ----------
    csv_path : str or Path
        Path to CSV file
    extrapolate : bool
        If True, allow extrapolation

    Returns
    -------
    SyntheticTrajectoryInterpolator
        Ready-to-use interpolator
    """
    loader = SyntheticDataLoader(csv_path)
    data = loader.load()
    return SyntheticTrajectoryInterpolator(data, extrapolate=extrapolate)


if __name__ == "__main__":
    # Test loading synthetic data
    from pathlib import Path

    data_path = Path(__file__).parent / "pele_data" / "synthetic_data" / "pele_collective_data.csv"

    if not data_path.exists():
        print(f"Test data not found: {data_path}")
    else:
        print(f"Loading: {data_path}")

        trajectory = load_synthetic_trajectory(str(data_path), extrapolate=False)

        print(f"\n{trajectory}")
        print(f"  Time range: [{trajectory.t_min*1e6:.1f}, {trajectory.t_max*1e6:.1f}] µs")

        data = trajectory.data
        print(f"  Flame position: [{data.flame_position.min()*100:.2f}, {data.flame_position.max()*100:.2f}] cm")
        print(f"  Flame velocity: [{data.flame_velocity.min():.1f}, {data.flame_velocity.max():.1f}] m/s")
        print(f"  Gas velocity: [{data.flame_gas_velocity.min():.1f}, {data.flame_gas_velocity.max():.1f}] m/s")

        # Test interpolation
        print(f"\nInterpolation test:")
        test_times = [0, 50e-6, 100e-6, 500e-6]
        for t in test_times:
            if t <= trajectory.t_max:
                x = trajectory.position(t)
                v = trajectory.velocity(t)
                v_gas = trajectory.gas_velocity(t)
                v_burned = trajectory.burned_gas_velocity(t)
                print(f"  t={t*1e6:6.1f} µs: x={x*100:.4f} cm, v={v:.1f} m/s, "
                      f"v_gas={v_gas:.1f} m/s, v_burned={v_burned:.1f} m/s")
