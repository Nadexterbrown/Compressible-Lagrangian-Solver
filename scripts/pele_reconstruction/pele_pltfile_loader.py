"""
PeleC pltfile loader for comparison with 1D solver results.

Uses yt to load AMR pltfile data and extract 1D rays for comparison.
Converts from PeleC CGS units to SI units.

Reference: LGDCS/scripts/pele_sim/pele_bds/piston_pele_bnds.py
"""

import os
import re
import numpy as np
from pathlib import Path
from glob import glob
from functools import reduce
from dataclasses import dataclass
from typing import List, Tuple, Optional


# Variable mapping for PeleC data (CGS units)
PELE_VAR_MAP = {
    'X': {'Name': 'x', 'Units': 'cm'},
    'Temperature': {'Name': 'Temp', 'Units': 'K'},
    'Pressure': {'Name': 'pressure', 'Units': 'g / cm / s^2'},
    'Density': {'Name': 'density', 'Units': 'g / cm^3'},
    'Velocity': {'Name': 'x_velocity', 'Units': 'cm / s'},
}

# Time offsets for multi-part PeleC simulations
PLTFILE_TIME_OFFSETS = {
    'Part-1': 0.0,
    'Part-2': 8.136537e-04,
    'Part-3': 8.136537e-04 + 8.6918377740062178e-04,
}


class UnitConverter:
    """Convert PeleC CGS units to SI units."""

    UNIT_MAP = {
        's': 1,
        'g': 1e-3,      # g -> kg
        'mol': 1e-3,
        'cm': 1e-2,     # cm -> m
        'K': 1,
        'kg': 1,
        'm': 1,
    }

    @classmethod
    def convert(cls, value: np.ndarray, var_name: str) -> np.ndarray:
        """Convert array from CGS to SI units."""
        unit_expr = cls._lookup_unit_expr(var_name)
        num_units, denom_units = cls._parse_units(unit_expr)
        num_factor = cls._convert_units(num_units, 1)
        denom_factor = cls._convert_units(denom_units, -1) if denom_units else 1
        return value * num_factor * denom_factor

    @classmethod
    def _lookup_unit_expr(cls, var_name: str) -> str:
        if var_name not in PELE_VAR_MAP:
            raise ValueError(f"Unknown variable '{var_name}'")
        unit_info = PELE_VAR_MAP[var_name]
        if isinstance(unit_info, dict) and 'Units' in unit_info:
            return unit_info['Units']
        raise ValueError(f"Units not defined for variable '{var_name}'")

    @classmethod
    def _parse_units(cls, unit_expr: str) -> Tuple[List[str], List[str]]:
        terms = re.split(r'\s*/\s*', unit_expr)
        num_units = terms[0].split()
        denom_units = [] if len(terms) == 1 else [t for term in terms[1:] for t in term.split()]
        return num_units, denom_units

    @classmethod
    def _convert_units(cls, units: List[str], exponent: int) -> float:
        if not units:
            return 1
        factors = []
        for unit in units:
            match = re.match(r"([a-zA-Z]+)(?:\^(-?\d+))?", unit)
            if not match:
                raise ValueError(f"Invalid unit format: {unit}")
            base_unit, exp = match.groups()
            if base_unit not in cls.UNIT_MAP:
                raise ValueError(f"Unknown unit '{base_unit}'")
            base_factor = cls.UNIT_MAP[base_unit]
            power = int(exp) if exp else 1
            factors.append(base_factor ** (power * exponent))
        return reduce(lambda x, y: x * y, factors, 1)


@dataclass
class PeleSnapshot:
    """Container for extracted PeleC snapshot data (in SI units)."""
    time: float              # [s] - with time offset applied
    time_raw: float          # [s] - raw time from pltfile
    part_name: str           # e.g., "Part-1"
    pltfile_name: str        # e.g., "plt200000"
    x: np.ndarray            # [m] - position
    u: np.ndarray            # [m/s] - velocity
    p: np.ndarray            # [Pa] - pressure
    rho: np.ndarray          # [kg/m³] - density
    T: Optional[np.ndarray]  # [K] - temperature (may be None)


def get_pltfile_number(pltfile_path: str) -> int:
    """Extract numerical index from pltfile path for sorting."""
    basename = os.path.basename(pltfile_path)
    match = re.search(r'plt(\d+)', basename)
    if match:
        return int(match.group(1))
    return 0


def get_sorted_pltfiles(pele_dir: str) -> List[str]:
    """Get all pltfiles from a directory, sorted numerically."""
    pltfiles = glob(os.path.join(pele_dir, 'plt*'))
    pltfiles = [p for p in pltfiles if os.path.isdir(p)]
    pltfiles.sort(key=get_pltfile_number)
    return pltfiles


def load_pele_pltfile(
    pltfile_path: str,
    extract_location: float,
    direction: str = 'x'
) -> Tuple:
    """
    Load a PeleC pltfile and extract orthogonal ray.

    Parameters
    ----------
    pltfile_path : str
        Path to pltfile directory
    extract_location : float
        Y-coordinate for ray extraction [m]
    direction : str
        Direction for ray ('x' or 'y')

    Returns
    -------
    ds : yt dataset
    dr : yt ray object
    """
    try:
        import yt
        yt.funcs.mylog.setLevel(50)  # Suppress yt output
    except ImportError:
        raise ImportError("yt package required. Install with: pip install yt")

    ds = yt.load(pltfile_path)

    if direction == 'x':
        dr = ds.ortho_ray(0, (extract_location, 0))
    else:
        dr = ds.ortho_ray(1, (0, extract_location))

    return ds, dr


def extract_pele_snapshot(
    pltfile_path: str,
    extract_location: float,
    time_offset: float = 0.0,
    part_name: str = "",
    direction: str = 'x'
) -> PeleSnapshot:
    """
    Extract and convert data from a PeleC pltfile.

    Parameters
    ----------
    pltfile_path : str
        Path to pltfile directory
    extract_location : float
        Y-coordinate for ray extraction [m]
    time_offset : float
        Time offset to add [s]
    part_name : str
        Part identifier (e.g., "Part-1")
    direction : str
        Direction for ray ('x' or 'y')

    Returns
    -------
    PeleSnapshot with SI units
    """
    ds, dr = load_pele_pltfile(pltfile_path, extract_location, direction)

    coord_field = 'x' if direction == 'x' else 'y'
    vel_field = 'x_velocity' if direction == 'x' else 'y_velocity'

    # Sort along ray coordinate
    ray_sort = np.argsort(dr['boxlib', coord_field].to_value().flatten())

    # Extract raw data (CGS)
    x_raw = dr['boxlib', coord_field][ray_sort].to_value().flatten()
    u_raw = dr['boxlib', vel_field][ray_sort].to_value().flatten()
    p_raw = dr['boxlib', 'pressure'][ray_sort].to_value().flatten()
    rho_raw = dr['boxlib', 'density'][ray_sort].to_value().flatten()

    # Temperature (may not exist in all pltfiles)
    try:
        T_raw = dr['boxlib', 'Temp'][ray_sort].to_value().flatten()
    except Exception:
        T_raw = None

    # Convert to SI
    x_si = UnitConverter.convert(x_raw, 'X')
    u_si = UnitConverter.convert(u_raw, 'Velocity')
    p_si = UnitConverter.convert(p_raw, 'Pressure')
    rho_si = UnitConverter.convert(rho_raw, 'Density')
    T_si = T_raw  # Already in K

    # Get time
    time_raw = float(ds.current_time.to_value())
    time_adjusted = time_raw + time_offset

    pltfile_name = os.path.basename(pltfile_path)

    return PeleSnapshot(
        time=time_adjusted,
        time_raw=time_raw,
        part_name=part_name,
        pltfile_name=pltfile_name,
        x=x_si,
        u=u_si,
        p=p_si,
        rho=rho_si,
        T=T_si,
    )


def load_all_pltfiles(
    pltfile_base_dir: str,
    extract_location: float = 0.0445,
    direction: str = 'x',
    time_offsets: Optional[dict] = None,
) -> List[PeleSnapshot]:
    """
    Load all PeleC pltfiles from Part-1, Part-2, Part-3 subdirectories.

    Parameters
    ----------
    pltfile_base_dir : str
        Base directory containing Part-1/, Part-2/, Part-3/ subdirs
    extract_location : float
        Y-coordinate for ray extraction [m]
    direction : str
        Direction for ray ('x' or 'y')
    time_offsets : dict, optional
        Time offsets per part. Defaults to PLTFILE_TIME_OFFSETS.

    Returns
    -------
    List of PeleSnapshot objects, sorted by time
    """
    if time_offsets is None:
        time_offsets = PLTFILE_TIME_OFFSETS

    base_path = Path(pltfile_base_dir)
    snapshots = []

    # Find all Part-* directories
    part_dirs = sorted(base_path.glob("Part-*"))

    if not part_dirs:
        print(f"No Part-* directories found in {base_path}")
        return []

    for part_dir in part_dirs:
        part_name = part_dir.name
        time_offset = time_offsets.get(part_name, 0.0)

        pltfiles = get_sorted_pltfiles(str(part_dir))

        print(f"  {part_name}: {len(pltfiles)} pltfiles (offset={time_offset*1e6:.1f} µs)")

        for pltfile in pltfiles:
            try:
                snapshot = extract_pele_snapshot(
                    pltfile,
                    extract_location,
                    time_offset=time_offset,
                    part_name=part_name,
                    direction=direction,
                )
                snapshots.append(snapshot)
                print(f"    Loaded {snapshot.pltfile_name}: t={snapshot.time*1e6:.1f} µs")
            except Exception as e:
                print(f"    WARNING: Failed to load {pltfile}: {e}")

    # Sort by adjusted time
    snapshots.sort(key=lambda s: s.time)

    return snapshots


if __name__ == "__main__":
    # Test loading pltfiles
    pltfile_dir = Path(__file__).parent / "pele_pltfiles"

    print(f"Loading pltfiles from: {pltfile_dir}")
    snapshots = load_all_pltfiles(str(pltfile_dir))

    print(f"\nLoaded {len(snapshots)} snapshots:")
    for s in snapshots:
        print(f"  {s.part_name}/{s.pltfile_name}: t={s.time*1e6:.1f} µs, "
              f"x=[{s.x.min():.3f}, {s.x.max():.3f}] m")
