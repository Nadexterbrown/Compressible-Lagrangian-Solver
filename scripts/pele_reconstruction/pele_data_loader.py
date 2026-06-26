"""
Loader for PeleC flame trajectory data.

Reads time-series data from PeleC simulation output files and provides
linear interpolation for flame position and velocity.

Reference: PeleC simulation output format
"""

import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple
from scipy.interpolate import interp1d


@dataclass
class PeleTrajectoryData:
    """Container for PeleC flame trajectory data."""
    time: np.ndarray                    # [s]
    flame_position: np.ndarray          # [m]
    flame_velocity: np.ndarray          # [m/s] - flame front velocity (S_f)
    flame_gas_velocity: np.ndarray      # [m/s] - gas velocity at flame (u_uf)

    # Optional additional data
    pressure: Optional[np.ndarray] = None           # [Pa]
    temperature: Optional[np.ndarray] = None        # [K]
    density: Optional[np.ndarray] = None            # [kg/m³] - rho_u (unburned gas at flame)
    sound_speed: Optional[np.ndarray] = None        # [m/s]
    burning_velocity: Optional[np.ndarray] = None   # [m/s] - flame burning velocity (S_c)
    burned_gas_velocity: Optional[np.ndarray] = None  # [m/s]
    burned_gas_density: Optional[np.ndarray] = None   # [kg/m³] - rho_b (burned gas behind flame)
    gas_offset_data: Optional[np.ndarray] = None    # [m/s] - arbitrary column for gas velocity offset

    def __post_init__(self):
        """Validate data consistency."""
        n = len(self.time)
        assert len(self.flame_position) == n, "flame_position length mismatch"
        assert len(self.flame_velocity) == n, "flame_velocity length mismatch"
        assert len(self.flame_gas_velocity) == n, "flame_gas_velocity length mismatch"


class PeleDataLoader:
    """
    Load PeleC trajectory data from raw text files.

    Expected file format (whitespace-separated columns):
    - Line 1: Column numbers (1, 2, 3, ...)
    - Line 2: Column headers
    - Lines 3+: Data

    Columns are identified by header names (not indices).
    """

    # Mapping from internal field names to expected column header names
    COLUMN_HEADERS = {
        'time': 'Time',
        'flame_position': 'Flame Position [m]',
        'flame_gas_velocity': 'Flame Gas Velocity [m / s]',
        'flame_temperature': 'Flame Thermodynamic Temperature [K]',
        'flame_pressure': 'Flame Thermodynamic Pressure [kg / m / s^2]',
        'flame_density': 'Flame Thermodynamic Density [kg / m^3]',  # rho_u - unburned gas density at flame
        'sound_speed': 'Flame Thermodynamic Sound Speed',
        'burning_velocity': 'Flame Burning Velocity [m / s]',  # S_c - flame burning velocity
        'flame_velocity': 'Flame Velocity [m / s]',  # S_f - flame front velocity
        'burned_gas_velocity': 'Burned Gas Gas Velocity [m / s]',
        'burned_gas_density': 'Burned Gas Thermodynamic Density [kg / m^3]',  # rho_b - burned gas density behind flame
    }

    # Time offsets for each data part (multi-part simulations)
    # Each part starts from t=0 in its file, but is actually a continuation
    DEFAULT_TIME_OFFSETS = {
        1: 0.0,
        2: 8.136537e-04,
        3: 8.136537e-04 + 8.6918377740062178e-04,
    }

    def __init__(self, data_dir: str, time_offsets: Optional[dict] = None,
                 gas_offset_column: Optional[str] = None):
        """
        Initialize loader with data directory path.

        Parameters
        ----------
        data_dir : str
            Path to directory containing part_1.txt, part_2.txt, etc.
        time_offsets : dict, optional
            Dictionary mapping part number to time offset [s].
            If None, uses DEFAULT_TIME_OFFSETS.
        gas_offset_column : str, optional
            Column header name for gas velocity offset data (e.g., "Flame Burning Velocity [m / s]").
            If specified, this column is loaded and stored in gas_offset_data.
        """
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        self.time_offsets = time_offsets if time_offsets is not None else self.DEFAULT_TIME_OFFSETS
        self.gas_offset_column = gas_offset_column
        self._header_map: Optional[dict] = None  # Column name -> index mapping

    def load(self, parts: Optional[List[int]] = None) -> PeleTrajectoryData:
        """
        Load trajectory data from one or more part files.

        Parameters
        ----------
        parts : list of int, optional
            Part numbers to load (e.g., [1, 2, 3]). If None, loads all available parts.

        Returns
        -------
        PeleTrajectoryData
            Combined trajectory data from all parts.
        """
        if parts is None:
            # Find all part files
            parts = self._find_available_parts()

        if not parts:
            raise FileNotFoundError(f"No part files found in {self.data_dir}")

        # Load and concatenate data from all parts
        all_data = []
        for part_num in sorted(parts):
            part_file = self.data_dir / f"part_{part_num}.txt"
            if not part_file.exists():
                raise FileNotFoundError(f"Part file not found: {part_file}")

            data = self._load_part(part_file)

            # Apply time offset for this part
            time_offset = self.time_offsets.get(part_num, 0.0)
            if time_offset != 0.0:
                data['time'] = data['time'] + time_offset

            all_data.append(data)

        # Concatenate arrays
        combined = self._concatenate_parts(all_data)

        return combined

    def _find_available_parts(self) -> List[int]:
        """Find all available part files."""
        parts = []
        for f in self.data_dir.glob("part_*.txt"):
            try:
                # Extract part number from filename
                stem = f.stem  # e.g., "part_1"
                num = int(stem.split("_")[1])
                parts.append(num)
            except (ValueError, IndexError):
                continue
        return sorted(parts)

    def _parse_headers(self, header_line: str) -> dict:
        """Parse header line to create column name -> index mapping."""
        # Remove leading '#' and split by multiple spaces (headers are space-padded)
        header_line = header_line.lstrip('#').strip()
        # Split on 2+ spaces to separate column names
        import re
        headers = re.split(r'\s{2,}', header_line)
        # Create mapping: header name -> 0-based index
        return {name.strip(): idx for idx, name in enumerate(headers)}

    def _get_column_index(self, field_name: str) -> int:
        """Get column index for a field using header mapping."""
        if self._header_map is None:
            raise ValueError("Header map not initialized - call _load_part first")

        header_name = self.COLUMN_HEADERS.get(field_name)
        if header_name is None:
            raise ValueError(f"Unknown field name: {field_name}")

        if header_name not in self._header_map:
            available = list(self._header_map.keys())
            raise ValueError(f"Column '{header_name}' not found for field '{field_name}'. Available: {available}")

        return self._header_map[header_name]

    def _load_part(self, filepath: Path) -> dict:
        """Load a single part file."""
        # Read all lines
        with open(filepath, 'r') as f:
            lines = f.readlines()

        # Parse header line (line 2, index 1) for column name mapping
        if self._header_map is None and len(lines) > 1:
            self._header_map = self._parse_headers(lines[1])

        # Skip header lines (line 1: column numbers, line 2: column names)
        data_lines = lines[2:]

        # Parse data
        data = []
        for line in data_lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            values = [float(x) for x in line.split()]
            data.append(values)

        data = np.array(data)

        # Extract columns using header-based lookup
        result = {
            'time': data[:, self._get_column_index('time')],
            'flame_position': data[:, self._get_column_index('flame_position')],
            'flame_velocity': data[:, self._get_column_index('flame_velocity')],
            'flame_gas_velocity': data[:, self._get_column_index('flame_gas_velocity')],
            'pressure': data[:, self._get_column_index('flame_pressure')],
            'temperature': data[:, self._get_column_index('flame_temperature')],
            'density': data[:, self._get_column_index('flame_density')],
            'sound_speed': data[:, self._get_column_index('sound_speed')],
            'burning_velocity': data[:, self._get_column_index('burning_velocity')],
            'burned_gas_velocity': data[:, self._get_column_index('burned_gas_velocity')],
            'burned_gas_density': data[:, self._get_column_index('burned_gas_density')],
        }

        # Load gas offset column if specified
        if self.gas_offset_column is not None:
            if self.gas_offset_column not in self._header_map:
                available = list(self._header_map.keys())
                raise ValueError(f"Column '{self.gas_offset_column}' not found. Available: {available}")
            col_idx = self._header_map[self.gas_offset_column]
            result['gas_offset_data'] = data[:, col_idx]

        return result

    def _concatenate_parts(self, parts: List[dict]) -> PeleTrajectoryData:
        """Concatenate multiple part dictionaries into a single PeleTrajectoryData."""
        # Concatenate all arrays
        time = np.concatenate([p['time'] for p in parts])
        flame_position = np.concatenate([p['flame_position'] for p in parts])
        flame_velocity = np.concatenate([p['flame_velocity'] for p in parts])
        flame_gas_velocity = np.concatenate([p['flame_gas_velocity'] for p in parts])
        pressure = np.concatenate([p['pressure'] for p in parts])
        temperature = np.concatenate([p['temperature'] for p in parts])
        density = np.concatenate([p['density'] for p in parts])
        sound_speed = np.concatenate([p['sound_speed'] for p in parts])
        burning_velocity = np.concatenate([p['burning_velocity'] for p in parts])
        burned_gas_velocity = np.concatenate([p['burned_gas_velocity'] for p in parts])
        burned_gas_density = np.concatenate([p['burned_gas_density'] for p in parts])

        # Gas offset data (optional)
        gas_offset_data = None
        if 'gas_offset_data' in parts[0]:
            gas_offset_data = np.concatenate([p['gas_offset_data'] for p in parts])

        # Sort by time (in case parts overlap or are out of order)
        sort_idx = np.argsort(time)

        # Remove duplicate times (keep first occurrence)
        _, unique_idx = np.unique(time[sort_idx], return_index=True)
        final_idx = sort_idx[unique_idx]

        return PeleTrajectoryData(
            time=time[final_idx],
            flame_position=flame_position[final_idx],
            flame_velocity=flame_velocity[final_idx],
            flame_gas_velocity=flame_gas_velocity[final_idx],
            pressure=pressure[final_idx],
            temperature=temperature[final_idx],
            density=density[final_idx],
            sound_speed=sound_speed[final_idx],
            burning_velocity=burning_velocity[final_idx],
            burned_gas_velocity=burned_gas_velocity[final_idx],
            burned_gas_density=burned_gas_density[final_idx],
            gas_offset_data=gas_offset_data[final_idx] if gas_offset_data is not None else None,
        )


class PeleTrajectoryInterpolator:
    """
    Interpolate PeleC trajectory data for arbitrary times.

    Uses linear interpolation between data points. Raises error
    for times outside the data range.
    """

    def __init__(self, data: PeleTrajectoryData, extrapolate: bool = False):
        """
        Initialize interpolators.

        Parameters
        ----------
        data : PeleTrajectoryData
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
        self._interp_gas_velocity = interp1d(
            data.time, data.flame_gas_velocity,
            kind='linear', bounds_error=bounds_error, fill_value=fill_value
        )

        # Optional interpolators
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

        if data.sound_speed is not None:
            self._interp_sound_speed = interp1d(
                data.time, data.sound_speed,
                kind='linear', bounds_error=bounds_error, fill_value=fill_value
            )
        else:
            self._interp_sound_speed = None

        if data.burned_gas_velocity is not None:
            self._interp_burned_gas_velocity = interp1d(
                data.time, data.burned_gas_velocity,
                kind='linear', bounds_error=bounds_error, fill_value=fill_value
            )
        else:
            self._interp_burned_gas_velocity = None

        if data.burned_gas_density is not None:
            self._interp_burned_gas_density = interp1d(
                data.time, data.burned_gas_density,
                kind='linear', bounds_error=bounds_error, fill_value=fill_value
            )
        else:
            self._interp_burned_gas_density = None

        if data.gas_offset_data is not None:
            self._interp_gas_offset = interp1d(
                data.time, data.gas_offset_data,
                kind='linear', bounds_error=bounds_error, fill_value=fill_value
            )
        else:
            self._interp_gas_offset = None

    @property
    def t_min(self) -> float:
        """Minimum time in data."""
        return float(self._data.time[0])

    @property
    def t_max(self) -> float:
        """Maximum time in data."""
        return float(self._data.time[-1])

    @property
    def data(self) -> PeleTrajectoryData:
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

    def sound_speed(self, t: float) -> Optional[float]:
        """Interpolate sound speed at time t."""
        if self._interp_sound_speed is None:
            return None
        return float(self._interp_sound_speed(t))

    def burned_gas_velocity(self, t: float) -> Optional[float]:
        """Interpolate burned gas velocity at time t."""
        if self._interp_burned_gas_velocity is None:
            return None
        return float(self._interp_burned_gas_velocity(t))

    def burned_gas_density(self, t: float) -> Optional[float]:
        """Interpolate burned gas density (rho_b) at time t."""
        if self._interp_burned_gas_density is None:
            return None
        return float(self._interp_burned_gas_density(t))

    def density(self, t: float) -> Optional[float]:
        """Interpolate unburned gas density (rho_u) at flame front at time t."""
        # The 'density' field in data is the flame thermodynamic density = rho_u
        if self._data.density is None:
            return None
        # Use linear interpolation
        return float(np.interp(t, self._data.time, self._data.density))

    def gas_offset(self, t: float) -> Optional[float]:
        """Interpolate gas velocity offset at time t."""
        if self._interp_gas_offset is None:
            return None
        return float(self._interp_gas_offset(t))

    def __repr__(self) -> str:
        return (
            f"PeleTrajectoryInterpolator("
            f"t=[{self.t_min:.6e}, {self.t_max:.6e}] s, "
            f"n_points={len(self._data.time)})"
        )


if __name__ == "__main__":
    # Test loading data
    import sys

    # Default to truncated_raw_data directory
    data_dir = Path(__file__).parent / "pele_data" / "truncated_raw_data"

    print(f"Loading data from: {data_dir}")

    loader = PeleDataLoader(data_dir)
    data = loader.load()

    print(f"\nLoaded {len(data.time)} data points")
    print(f"  Time range: [{data.time[0]:.6e}, {data.time[-1]:.6e}] s")
    print(f"  Position range: [{data.flame_position.min():.6e}, {data.flame_position.max():.6e}] m")
    print(f"  Flame velocity range: [{data.flame_velocity.min():.1f}, {data.flame_velocity.max():.1f}] m/s")
    print(f"  Gas velocity range: [{data.flame_gas_velocity.min():.1f}, {data.flame_gas_velocity.max():.1f}] m/s")
    print(f"  Sound speed range: [{data.sound_speed.min():.1f}, {data.sound_speed.max():.1f}] m/s")

    # Test interpolator
    interp = PeleTrajectoryInterpolator(data)
    print(f"\n{interp}")

    # Test interpolation at a few points
    test_times = [data.time[0], data.time[len(data.time)//2], data.time[-1]]
    print("\nInterpolation test:")
    for t in test_times:
        print(f"  t={t:.6e}: x={interp.position(t):.6e} m, v={interp.velocity(t):.1f} m/s")
