"""
Loader for experimental flame trajectory data.

Reads time-series data from processed experimental data files and provides
linear interpolation for flame position and velocity.

Reference: High-speed camera flame tracking data
"""

import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from scipy.interpolate import interp1d


@dataclass
class ExperimentalTrajectoryData:
    """Container for experimental flame trajectory data."""
    time: np.ndarray                    # [s]
    flame_position: np.ndarray          # [m]
    flame_velocity: np.ndarray          # [m/s] - flame front velocity

    def __post_init__(self):
        """Validate data consistency."""
        n = len(self.time)
        assert len(self.flame_position) == n, "flame_position length mismatch"
        assert len(self.flame_velocity) == n, "flame_velocity length mismatch"


class ExperimentalDataLoader:
    """
    Load experimental trajectory data from processed text files.

    Expected file format (whitespace-separated columns):
    - Line 1: Column numbers (0, 1, 2, 3)
    - Line 2: Column headers (Index, Time [s], Position [m], Velocity [m/s])
    - Lines 3+: Data

    Key columns (0-indexed):
        0: Index
        1: Time [s]
        2: Position [m]
        3: Velocity [m/s]
    """

    # Column indices (0-indexed in data, but we skip the Index column)
    COL_INDEX = 0
    COL_TIME = 1
    COL_POSITION = 2
    COL_VELOCITY = 3

    def __init__(self, filepath: str):
        """
        Initialize loader with data file path.

        Parameters
        ----------
        filepath : str
            Path to processed_data_run_*.txt file.
        """
        self.filepath = Path(filepath)
        if not self.filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")

    def load(self) -> ExperimentalTrajectoryData:
        """
        Load trajectory data from the file.

        Returns
        -------
        ExperimentalTrajectoryData
            Trajectory data.
        """
        # Read all lines
        with open(self.filepath, 'r') as f:
            lines = f.readlines()

        # Skip header lines (line 1: column numbers, line 2: column names)
        data_lines = lines[2:]

        # Parse data
        data = []
        for line in data_lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            values = [float(x) for x in line.split()]
            if len(values) >= 4:  # Need at least index, time, position, velocity
                data.append(values)

        data = np.array(data)

        # Extract columns
        time = data[:, self.COL_TIME]
        position = data[:, self.COL_POSITION]
        velocity = data[:, self.COL_VELOCITY]

        # Sort by time (should already be sorted, but ensure)
        sort_idx = np.argsort(time)
        time = time[sort_idx]
        position = position[sort_idx]
        velocity = velocity[sort_idx]

        return ExperimentalTrajectoryData(
            time=time,
            flame_position=position,
            flame_velocity=velocity,
        )


class ExperimentalTrajectoryInterpolator:
    """
    Interpolate experimental trajectory data for arbitrary times.

    Uses linear interpolation between data points.
    """

    def __init__(self, data: ExperimentalTrajectoryData, extrapolate: bool = False):
        """
        Initialize interpolators.

        Parameters
        ----------
        data : ExperimentalTrajectoryData
            Trajectory data to interpolate.
        extrapolate : bool, optional
            If True, allow extrapolation beyond data bounds.
            If False (default), raise error for out-of-bounds times.
        """
        self._data = data
        self._extrapolate = extrapolate

        # Create interpolators
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

    @property
    def t_min(self) -> float:
        """Minimum time in data."""
        return float(self._data.time[0])

    @property
    def t_max(self) -> float:
        """Maximum time in data."""
        return float(self._data.time[-1])

    @property
    def data(self) -> ExperimentalTrajectoryData:
        """Return underlying data."""
        return self._data

    def position(self, t: float) -> float:
        """Interpolate flame position at time t."""
        return float(self._interp_position(t))

    def velocity(self, t: float) -> float:
        """Interpolate flame velocity at time t."""
        return float(self._interp_velocity(t))

    def __repr__(self) -> str:
        return (
            f"ExperimentalTrajectoryInterpolator("
            f"t=[{self.t_min:.6e}, {self.t_max:.6e}] s, "
            f"n_points={len(self._data.time)})"
        )


def get_available_cases(base_dir: str) -> dict:
    """
    Find all available experimental cases.

    Parameters
    ----------
    base_dir : str
        Base directory (e.g., scripts/experiment_reconstruction)

    Returns
    -------
    dict
        Dictionary mapping (fuel, phi) to data file path.
        Example: {('c2h6', 'phi_0.8'): Path(...), ...}
    """
    base_path = Path(base_dir)
    cases = {}

    for fuel_dir in ['c2h6_cases', 'c4h10_cases']:
        fuel_path = base_path / fuel_dir / 'complete_data'
        if not fuel_path.exists():
            continue

        fuel = fuel_dir.replace('_cases', '')

        for phi_dir in fuel_path.iterdir():
            if not phi_dir.is_dir():
                continue

            # Find processed data file
            data_files = list(phi_dir.glob('processed_data_run_*.txt'))
            if data_files:
                # Use the first one found
                phi = phi_dir.name
                cases[(fuel, phi)] = data_files[0]

    return cases


if __name__ == "__main__":
    # Test loading data
    base_dir = Path(__file__).parent

    print("Finding available cases...")
    cases = get_available_cases(str(base_dir))

    print(f"\nFound {len(cases)} cases:")
    for (fuel, phi), filepath in sorted(cases.items()):
        print(f"  {fuel} {phi}: {filepath.name}")

    # Test loading one case
    if cases:
        (fuel, phi), filepath = list(cases.items())[0]
        print(f"\nLoading test case: {fuel} {phi}")

        loader = ExperimentalDataLoader(filepath)
        data = loader.load()

        print(f"  Loaded {len(data.time)} data points")
        print(f"  Time range: [{data.time[0]:.6e}, {data.time[-1]:.6e}] s")
        print(f"  Position range: [{data.flame_position.min():.4f}, {data.flame_position.max():.4f}] m")
        print(f"  Velocity range: [{data.flame_velocity.min():.1f}, {data.flame_velocity.max():.1f}] m/s")

        # Test interpolator
        interp = ExperimentalTrajectoryInterpolator(data)
        print(f"\n{interp}")
