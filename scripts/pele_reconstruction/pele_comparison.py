"""
Compare 1D Lagrangian solver results with PeleC pltfile snapshots.

Produces publication-quality comparison figures showing:
- Upper/lower velocity bounds with shaded regions
- Multi-time comparisons (vertical and horizontal layouts)
- Piston velocity vs time with comparison markers
- Individual time comparisons with PeleC data
- X-T diagrams
- Piston-aligned comparisons

Reference: LGDCS/scripts/pele_sim/pele_bds/truncated_bounds_study_results_old/
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass

from pele_pltfile_loader import load_all_pltfiles, PeleSnapshot


# Publication-quality plot settings (Times New Roman to match journal text)
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
mpl.rcParams['mathtext.fontset'] = 'stix'  # STIX fonts match Times New Roman
mpl.rcParams['text.usetex'] = False  # Use matplotlib's mathtext, not LaTeX
mpl.rcParams['axes.linewidth'] = 1.2
mpl.rcParams['xtick.major.width'] = 1.0
mpl.rcParams['ytick.major.width'] = 1.0
mpl.rcParams['axes.labelweight'] = 'normal'
mpl.rcParams['axes.titleweight'] = 'normal'

# Publication figure dimensions (Combustion and Flame)
# Full text width: 5.67 in (144 mm), Single column: 2.67 in (67.7 mm)
# Using EXACT dimensions ensures fonts appear at correct size when printed
FULL_PAGE_WIDTH = 5.67  # inches (144 mm) - exact CNF full text width
SINGLE_COL_WIDTH = 2.67  # inches (67.7 mm) - exact CNF single column width

# Font sizes for publication (increased for better readability)
# Journal main text is 10pt, but figures need larger fonts for reduction
FONT_SIZE_TITLE = 12
FONT_SIZE_LABEL = 11
FONT_SIZE_TICK = 10
FONT_SIZE_LEGEND = 10
LINE_WIDTH = 1.0
PLOT_DPI = 300


def get_figsize(n_cols: int, n_rows: int, column_mode: str = 'double') -> Tuple[float, float]:
    """
    Get figure size for publication based on column mode and panel layout.

    Args:
        n_cols: Number of panel columns in the figure
        n_rows: Number of panel rows in the figure
        column_mode: 'single' for single column, 'double' for full page width,
                     'presentation' for PowerPoint slides

    Returns:
        (width, height) in inches
    """
    if column_mode == 'single':
        width = SINGLE_COL_WIDTH
        height_per_row = 1.4  # Proportional scaling
    elif column_mode == 'presentation':
        width = 12.0  # Wide format for slides
        height_per_row = 2.8
    else:  # 'double'
        width = FULL_PAGE_WIDTH
        height_per_row = 1.5

    height = height_per_row * n_rows
    return (width, height)


# Font sizes that scale with column mode
def get_font_sizes(column_mode: str = 'double') -> dict:
    """Get font sizes appropriate for the column mode.

    Font sizes are increased for publication to ensure readability
    when figures are reduced to fit journal column widths.
    """
    if column_mode == 'single':
        # Single column: larger fonts since figure is narrower
        return {
            'title': 11,
            'label': 10,
            'tick': 9,
            'legend': 9,
            'linewidth': 0.9,
        }
    elif column_mode == 'presentation':
        # Large fonts for PowerPoint presentations
        return {
            'title': 22,
            'label': 20,
            'tick': 18,
            'legend': 16,
            'linewidth': 2.5,
        }
    else:
        # Double column (full page width)
        return {
            'title': FONT_SIZE_TITLE,
            'label': FONT_SIZE_LABEL,
            'tick': FONT_SIZE_TICK,
            'legend': FONT_SIZE_LEGEND,
            'linewidth': LINE_WIDTH,
        }

# Colors for different snapshot times (matching reference figures)
TIME_COLORS = ['black', '#E41A1C', '#377EB8', '#4DAF4A', '#984EA3']


def find_flame_position_hrr(pele_snap: 'PeleSnapshot', buffer_cells: int = 10) -> Tuple[float, float]:
    """
    Find flame position using maximum Heat Release Rate (HRR).

    Based on pele_processing flame detection method (PeleFlameAnalyzer.find_wave_position).
    Uses max HRR as the primary criterion for flame location.

    Args:
        pele_snap: PeleC snapshot with HRR data
        buffer_cells: Number of grid cells to add as buffer into reactants

    Returns:
        Tuple of (flame_position_m, buffer_position_m) where buffer_position is
        flame_position + buffer into the reactants (ahead of flame)
    """
    x = pele_snap.x
    hrr = pele_snap.hrr

    if hrr is None:
        # Fallback to temperature threshold if HRR not available
        T = pele_snap.T
        if T is not None:
            T_threshold = 2000.0  # K
            flame_indices = np.where(T >= T_threshold)[0]
            if len(flame_indices) > 0:
                # Rightmost point above threshold (flame front)
                flame_idx = flame_indices[-1]
            else:
                flame_idx = 0
        else:
            flame_idx = 0
    else:
        # Primary criterion: Maximum Heat Release Rate
        flame_idx = np.argmax(hrr)

    flame_x = x[flame_idx]

    # Calculate buffer distance (buffer_cells * grid spacing)
    if len(x) > 1:
        delta_x = np.abs(np.diff(x)).min()
    else:
        delta_x = 0.01  # Default fallback

    buffer_x = flame_x + buffer_cells * delta_x

    return flame_x, buffer_x


def compute_mass_coordinate(x: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """
    Compute Lagrangian (mass) coordinate from position and density.

    Mass coordinate: m(x) = ∫ρ dx from x[0] to x
    Uses trapezoidal integration.

    Args:
        x: Position array [m or cm]
        rho: Density array [kg/m³]

    Returns:
        Mass coordinate array (same length as x)
    """
    if len(x) < 2:
        return np.zeros_like(x)

    # Compute dx for each segment
    dx = np.diff(x)

    # Average density in each segment
    rho_avg = 0.5 * (rho[:-1] + rho[1:])

    # Mass in each segment
    dm = rho_avg * dx

    # Cumulative mass (starting from 0)
    m = np.zeros(len(x))
    m[1:] = np.cumsum(dm)

    return m


def compute_mass_coordinate_pele(pele_snap: 'PeleSnapshot') -> np.ndarray:
    """
    Compute mass coordinate for PeleC snapshot.

    Args:
        pele_snap: PeleC snapshot

    Returns:
        Mass coordinate array [kg/m²]
    """
    return compute_mass_coordinate(pele_snap.x, pele_snap.rho)


def compute_mass_coordinate_sim(snap: 'SimSnapshot') -> np.ndarray:
    """
    Compute mass coordinate for 1D simulation snapshot.

    Args:
        snap: Simulation snapshot

    Returns:
        Mass coordinate array [kg/m²]
    """
    return compute_mass_coordinate(snap.x_centers, snap.rho)


# Line styles
STYLE_UPPER = '-'   # Solid for upper bound (U_f)
STYLE_LOWER = '--'  # Dashed for lower bound (U_f - CJ)
STYLE_PELE = '--'   # Dashed for PeleC


@dataclass
class SimSnapshot:
    """Container for simulation snapshot data."""
    time: float
    step: int
    x_centers: np.ndarray
    u: np.ndarray
    p: np.ndarray
    rho: np.ndarray
    T: np.ndarray
    label: str
    path: Path


@dataclass
class SimTimeseries:
    """Container for full simulation timeseries."""
    t: np.ndarray
    u_piston: np.ndarray
    u_gas: np.ndarray  # Gas velocity at piston face
    x_piston: np.ndarray  # Piston position over time
    label: str
    path: Path


def get_snapshot_files(snapshot_dir: Path) -> List[Path]:
    """Get sorted list of snapshot file paths without loading data."""
    return sorted(snapshot_dir.glob("snapshot_*.npz"))


def load_snapshot_time(filepath: Path) -> Optional[Dict]:
    """Load only time info from a snapshot file."""
    try:
        data = np.load(filepath)
        t = float(data['t'])
        step = int(data['step']) if 'step' in data else int(filepath.stem.split('_')[1])
        return {'time': t, 'step': step, 'path': filepath}
    except Exception as e:
        print(f"  Warning: Could not read {filepath.name}: {e}")
        return None


def find_snapshots_for_times_coarse_to_fine(
    snapshot_files: List[Path],
    target_times: List[float],
    coarse_step: int = 100,
    fine_step: int = 10,
) -> List[Dict]:
    """
    Find snapshots matching target times using coarse-to-fine search.

    1. Load every coarse_step-th snapshot to get rough time mapping
    2. For each target time, identify the coarse region
    3. Load every fine_step-th snapshot in that region
    4. Find the exact nearest snapshot

    Args:
        snapshot_files: Sorted list of snapshot file paths
        target_times: List of target times to find snapshots for
        coarse_step: Step size for coarse scan (default 100)
        fine_step: Step size for fine scan (default 10)

    Returns:
        List of snapshot info dicts, one per target time
    """
    n_files = len(snapshot_files)
    if n_files == 0:
        return [None] * len(target_times)

    print(f"    Coarse scan: loading every {coarse_step}th of {n_files} snapshots...")

    # Coarse scan - load every coarse_step-th snapshot
    coarse_indices = list(range(0, n_files, coarse_step))
    # Always include the last file
    if coarse_indices[-1] != n_files - 1:
        coarse_indices.append(n_files - 1)

    coarse_snapshots = []
    for idx in coarse_indices:
        info = load_snapshot_time(snapshot_files[idx])
        if info:
            info['file_idx'] = idx
            coarse_snapshots.append(info)

    print(f"    Loaded {len(coarse_snapshots)} coarse snapshots")

    results = []

    for target_time in target_times:
        # Find coarse region containing target time
        coarse_times = np.array([s['time'] for s in coarse_snapshots])
        coarse_file_indices = np.array([s['file_idx'] for s in coarse_snapshots])

        # Find the two coarse snapshots bracketing the target time
        idx_after = np.searchsorted(coarse_times, target_time)
        idx_before = max(0, idx_after - 1)
        idx_after = min(len(coarse_snapshots) - 1, idx_after)

        # Get file index range for fine search
        start_file_idx = coarse_file_indices[idx_before]
        end_file_idx = coarse_file_indices[idx_after]

        # Expand range slightly to be safe
        start_file_idx = max(0, start_file_idx - fine_step)
        end_file_idx = min(n_files - 1, end_file_idx + fine_step)

        # Fine scan in this region
        fine_indices = list(range(start_file_idx, end_file_idx + 1, fine_step))
        if fine_indices[-1] != end_file_idx:
            fine_indices.append(end_file_idx)

        fine_snapshots = []
        for idx in fine_indices:
            info = load_snapshot_time(snapshot_files[idx])
            if info:
                info['file_idx'] = idx
                fine_snapshots.append(info)

        # Find nearest in fine scan
        fine_times = np.array([s['time'] for s in fine_snapshots])
        fine_file_indices = np.array([s['file_idx'] for s in fine_snapshots])

        nearest_fine_idx = np.argmin(np.abs(fine_times - target_time))

        # Now do exact search around the fine match
        fine_file_idx = fine_file_indices[nearest_fine_idx]
        exact_start = max(0, fine_file_idx - fine_step)
        exact_end = min(n_files - 1, fine_file_idx + fine_step)

        exact_snapshots = []
        for idx in range(exact_start, exact_end + 1):
            info = load_snapshot_time(snapshot_files[idx])
            if info:
                exact_snapshots.append(info)

        # Find the exact nearest
        exact_times = np.array([s['time'] for s in exact_snapshots])
        nearest_idx = np.argmin(np.abs(exact_times - target_time))
        results.append(exact_snapshots[nearest_idx])

    return results


def scan_snapshots(snapshot_dir: Path) -> List[Dict]:
    """Scan snapshot directory and return list of snapshot info.

    Note: This loads ALL snapshots which can be slow. For matching specific
    times, use find_snapshots_for_times_coarse_to_fine() instead.
    """
    snapshots = []
    snapshot_files = sorted(snapshot_dir.glob("snapshot_*.npz"))

    for f in snapshot_files:
        info = load_snapshot_time(f)
        if info:
            snapshots.append(info)

    return snapshots


def find_nearest_snapshot(snapshots: List[Dict], target_time: float) -> Dict:
    """Find snapshot with time nearest to target_time."""
    times = np.array([s['time'] for s in snapshots])
    idx = np.argmin(np.abs(times - target_time))
    return snapshots[idx]


def load_snapshot(snapshot_info: Dict, label: str) -> SimSnapshot:
    """Load a single snapshot .npz file."""
    data = np.load(snapshot_info['path'])
    return SimSnapshot(
        time=float(data['t']),
        step=snapshot_info['step'],
        x_centers=data['x_centers'],
        u=data['u'],
        p=data['p'],
        rho=data['rho'],
        T=data['T'],
        label=label,
        path=snapshot_info['path'],
    )


def load_timeseries(results_dir: Path, label: str) -> Optional[SimTimeseries]:
    """Load timeseries data from results directory."""
    ts_file = results_dir / "timeseries.npz"
    if not ts_file.exists():
        return None

    data = np.load(ts_file, allow_pickle=True)
    t = data['t']
    u_piston = data['u_piston']

    # Load gas velocity if available, otherwise use piston velocity
    if 'u_gas' in data.files:
        u_gas = data['u_gas']
    else:
        u_gas = u_piston.copy()  # Fallback for non-porous simulations

    # Compute piston position by integrating velocity
    x_piston = np.zeros_like(t)
    for i in range(1, len(t)):
        dt = t[i] - t[i-1]
        x_piston[i] = x_piston[i-1] + u_piston[i-1] * dt

    return SimTimeseries(
        t=t,
        u_piston=u_piston,
        u_gas=u_gas,
        x_piston=x_piston,
        label=label,
        path=results_dir,
    )


def plot_bounds_comparison_at_time(
    upper_snap: SimSnapshot,
    lower_snap: SimSnapshot,
    pele_snap: Optional[PeleSnapshot],
    output_file: str,
    shift_to_upper: bool = False,
    column_mode: str = 'double',
):
    """
    Plot comparison at a single time with shaded bounds.

    3-panel figure: Pressure, Velocity, Density
    Shows upper/lower bounds with shaded region, plus PeleC.

    Args:
        upper_snap: Upper bound simulation snapshot
        lower_snap: Lower bound simulation snapshot
        pele_snap: PeleC snapshot (optional)
        output_file: Output file path
        shift_to_upper: If True, shift lower bound to align piston with upper
        column_mode: 'single' for single column, 'double' for full page width
    """
    fonts = get_font_sizes(column_mode)
    lw = fonts['linewidth']

    if column_mode == 'single':
        # Stack vertically for single column
        fig, axes = plt.subplots(3, 1, figsize=(SINGLE_COL_WIDTH, 4.2))
    elif column_mode == 'presentation':
        # Large horizontal layout for PowerPoint
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    else:
        fig, axes = plt.subplots(1, 3, figsize=(FULL_PAGE_WIDTH, 1.9))

    time_s = upper_snap.time
    time_ms = time_s * 1e3

    # Get positions
    x_upper = upper_snap.x_centers
    x_lower = lower_snap.x_centers

    # Compute piston position shift if needed
    shift = 0.0
    if shift_to_upper:
        piston_upper = x_upper.min()
        piston_lower = x_lower.min()
        shift = piston_upper - piston_lower
        x_lower = x_lower + shift

    # Create common x grid for fill_between
    x_min = min(x_upper.min(), x_lower.min())
    x_max = max(x_upper.max(), x_lower.max())
    x_common = np.linspace(x_min, x_max, 500)

    # Interpolate to common grid
    def interp_safe(x_orig, y_orig, x_new):
        """Interpolate with bounds handling."""
        return np.interp(x_new, x_orig, y_orig, left=y_orig[0], right=y_orig[-1])

    # Pressure
    p_upper = interp_safe(x_upper, upper_snap.p, x_common)
    p_lower = interp_safe(x_lower, lower_snap.p, x_common)

    axes[0].fill_between(x_common, p_lower / 1e6, p_upper / 1e6,
                         alpha=0.3, color='lightblue', label='1D Model bounds')
    axes[0].plot(x_upper, upper_snap.p / 1e6, STYLE_UPPER, color='blue', lw=lw,
                 label=f'{upper_snap.label} (t={time_s:.6f}s)')
    axes[0].plot(x_lower, lower_snap.p / 1e6, STYLE_LOWER, color='red', lw=lw,
                 label=f'{lower_snap.label} (t={lower_snap.time:.6f}s)')

    # Velocity
    u_upper = interp_safe(x_upper, upper_snap.u, x_common)
    u_lower = interp_safe(x_lower, lower_snap.u, x_common)

    axes[1].fill_between(x_common, u_lower, u_upper,
                         alpha=0.3, color='lightblue', label='1D Model bounds')
    axes[1].plot(x_upper, upper_snap.u, STYLE_UPPER, color='blue', lw=lw,
                 label=f'{upper_snap.label} (t={time_s:.6f}s)')
    axes[1].plot(x_lower, lower_snap.u, STYLE_LOWER, color='red', lw=lw,
                 label=f'{lower_snap.label} (t={lower_snap.time:.6f}s)')

    # Density
    rho_upper = interp_safe(x_upper, upper_snap.rho, x_common)
    rho_lower = interp_safe(x_lower, lower_snap.rho, x_common)

    axes[2].fill_between(x_common, rho_lower, rho_upper,
                         alpha=0.3, color='lightblue', label='1D Model bounds')
    axes[2].plot(x_upper, upper_snap.rho, STYLE_UPPER, color='blue', lw=lw,
                 label=f'{upper_snap.label} (t={time_s:.6f}s)')
    axes[2].plot(x_lower, lower_snap.rho, STYLE_LOWER, color='red', lw=lw,
                 label=f'{lower_snap.label} (t={lower_snap.time:.6f}s)')

    # Compute y-axis limits from simulation data (before plotting PeleC)
    p_all = np.concatenate([upper_snap.p, lower_snap.p])
    u_all = np.concatenate([upper_snap.u, lower_snap.u])
    rho_all = np.concatenate([upper_snap.rho, lower_snap.rho])

    ylims = [
        (p_all.min() / 1e6 * 0.95, p_all.max() / 1e6 * 1.05),  # Pressure
        (u_all.min() * 0.95 if u_all.min() > 0 else u_all.min() * 1.05, u_all.max() * 1.05),  # Velocity
        (rho_all.min() * 0.95, rho_all.max() * 1.05),  # Density
    ]

    # Plot PeleC if available (filter points behind flame position using HRR)
    if pele_snap is not None:
        # Filter out points behind the flame using HRR-based detection
        flame_x, buffer_x = find_flame_position_hrr(pele_snap, buffer_cells=10)
        pele_mask = pele_snap.x >= buffer_x
        x_pele = pele_snap.x[pele_mask]
        p_pele = pele_snap.p[pele_mask]
        u_pele = pele_snap.u[pele_mask]
        rho_pele = pele_snap.rho[pele_mask]

        if len(x_pele) > 0:
            axes[0].plot(x_pele, p_pele / 1e6, 'k-', lw=lw + 0.3,
                         label=f'2D Simulation (t={pele_snap.time:.6f}s)')
            axes[1].plot(x_pele, u_pele, 'k-', lw=lw + 0.3,
                         label=f'2D Simulation (t={pele_snap.time:.6f}s)')
            axes[2].plot(x_pele, rho_pele, 'k-', lw=lw + 0.3,
                         label=f'2D Simulation (t={pele_snap.time:.6f}s)')

    # Labels and formatting
    titles = ['Pressure', 'Velocity', 'Density']
    ylabels = ['Pressure [Pa]', 'Velocity [m/s]', 'Density [kg/m³]']

    for i, ax in enumerate(axes):
        ax.set_xlabel('Position [m]', fontsize=fonts['label'])
        ax.set_ylabel(ylabels[i], fontsize=fonts['label'])
        ax.set_title(titles[i], fontsize=fonts['title'])
        ax.legend(fontsize=max(fonts['legend'] - 1, 5), loc='best')
        ax.tick_params(labelsize=fonts['tick'])
        ax.grid(True, alpha=0.3)
        ax.set_ylim(ylims[i])  # Set y-limits from simulation data

    # Get pltfile name from PeleC snapshot if available
    if pele_snap is not None:
        plt_name = f"{pele_snap.part_name}_{pele_snap.pltfile_name}" if pele_snap.part_name else pele_snap.pltfile_name
        fig.suptitle(f'1D Model Bounds vs 2D Simulation - {plt_name}', fontsize=fonts['title'] + 1, fontweight='bold')
    else:
        fig.suptitle(f'1D Model Bounds Comparison - t = {time_ms:.3f} ms', fontsize=fonts['title'] + 1, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_file, dpi=PLOT_DPI, bbox_inches='tight')
    plt.close(fig)


def plot_piston_velocity_bounds(
    upper_ts: SimTimeseries,
    lower_ts: SimTimeseries,
    comparison_times: List[float],
    output_file: str,
    column_mode: str = 'double',
):
    """
    Plot piston velocity vs time with shaded bounds.

    Shows U_f (upper bound) and U_f - CJ_def (lower bound) with markers
    at comparison times.

    Args:
        column_mode: 'single' for single column, 'double' for full page width
    """
    fonts = get_font_sizes(column_mode)
    lw = fonts['linewidth']
    ms = 6 if column_mode == 'single' else (14 if column_mode == 'presentation' else 10)  # Marker size

    if column_mode == 'single':
        fig, ax = plt.subplots(figsize=(SINGLE_COL_WIDTH, 2.0))
    elif column_mode == 'presentation':
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig, ax = plt.subplots(figsize=(FULL_PAGE_WIDTH, 2.5))

    t_upper_ms = upper_ts.t * 1e3
    t_lower_ms = lower_ts.t * 1e3

    # Create common time grid for fill_between
    t_min = max(upper_ts.t.min(), lower_ts.t.min())
    t_max = min(upper_ts.t.max(), lower_ts.t.max())
    t_common = np.linspace(t_min, t_max, 500)
    t_common_ms = t_common * 1e3

    u_upper = np.interp(t_common, upper_ts.t, upper_ts.u_piston)
    u_lower = np.interp(t_common, lower_ts.t, lower_ts.u_piston)

    # Check if single-bound mode (upper and lower are the same)
    single_bound = upper_ts.path == lower_ts.path

    if single_bound:
        # Single bound: plot one line, no legend
        ax.plot(t_upper_ms, upper_ts.u_piston, 'k-', lw=lw)
    else:
        # Two bounds: plot both with shaded region and legend
        ax.fill_between(t_common_ms, u_lower, u_upper, alpha=0.3, color='gray')
        ax.plot(t_upper_ms, upper_ts.u_piston, 'k-', lw=lw, label=r'$U_f$')
        ax.plot(t_lower_ms, lower_ts.u_piston, 'k--', lw=lw, label=r'$U_f - CJ_{def}$')

    # Markers at comparison times
    for i, t in enumerate(comparison_times):
        color = TIME_COLORS[i % len(TIME_COLORS)]

        # Upper bound marker (circle)
        u_upper_at_t = np.interp(t, upper_ts.t, upper_ts.u_piston)
        ax.plot(t * 1e3, u_upper_at_t, 'o', color=color, markersize=ms, markeredgecolor='black', markeredgewidth=0.5)

        if not single_bound:
            # Lower bound marker (square)
            u_lower_at_t = np.interp(t, lower_ts.t, lower_ts.u_piston)
            ax.plot(t * 1e3, u_lower_at_t, 's', color=color, markersize=ms-1, markeredgecolor='black', markeredgewidth=0.5)

    ax.set_xlabel('Time [ms]', fontsize=fonts['label'])
    ax.set_ylabel('Piston Velocity [m/s]', fontsize=fonts['label'])
    if not single_bound:
        ax.legend(fontsize=fonts['legend'], loc='lower right')
    ax.tick_params(labelsize=fonts['tick'])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=PLOT_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_file}")


def plot_gas_velocity_bounds(
    upper_ts: SimTimeseries,
    lower_ts: SimTimeseries,
    comparison_times: List[float],
    output_file: str,
    column_mode: str = 'double',
):
    """
    Plot gas velocity at piston face vs time with shaded bounds.

    Shows the actual gas velocity used in the Riemann solver at the boundary,
    which differs from piston velocity in porous simulations.

    Args:
        column_mode: 'single' for single column, 'double' for full page width
    """
    fonts = get_font_sizes(column_mode)
    lw = fonts['linewidth']
    ms = 6 if column_mode == 'single' else (14 if column_mode == 'presentation' else 10)  # Marker size

    if column_mode == 'single':
        fig, ax = plt.subplots(figsize=(SINGLE_COL_WIDTH, 2.0))
    elif column_mode == 'presentation':
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig, ax = plt.subplots(figsize=(FULL_PAGE_WIDTH, 2.5))

    t_upper_ms = upper_ts.t * 1e3
    t_lower_ms = lower_ts.t * 1e3

    # Create common time grid for fill_between
    t_min = max(upper_ts.t.min(), lower_ts.t.min())
    t_max = min(upper_ts.t.max(), lower_ts.t.max())
    t_common = np.linspace(t_min, t_max, 500)
    t_common_ms = t_common * 1e3

    u_upper = np.interp(t_common, upper_ts.t, upper_ts.u_gas)
    u_lower = np.interp(t_common, lower_ts.t, lower_ts.u_gas)

    # Check if single-bound mode (upper and lower are the same)
    single_bound = upper_ts.path == lower_ts.path

    if single_bound:
        # Single bound: plot one line, no legend
        ax.plot(t_upper_ms, upper_ts.u_gas, 'k-', lw=lw)
    else:
        # Two bounds: plot both with shaded region and legend
        ax.fill_between(t_common_ms, u_lower, u_upper, alpha=0.3, color='gray')
        ax.plot(t_upper_ms, upper_ts.u_gas, 'k-', lw=lw, label=r'$u_g = U_f$')
        ax.plot(t_lower_ms, lower_ts.u_gas, 'k--', lw=lw, label=r'$u_g = U_f - CJ_{def}$')

    # Markers at comparison times
    for i, t in enumerate(comparison_times):
        color = TIME_COLORS[i % len(TIME_COLORS)]

        # Upper bound marker (circle)
        u_upper_at_t = np.interp(t, upper_ts.t, upper_ts.u_gas)
        ax.plot(t * 1e3, u_upper_at_t, 'o', color=color, markersize=ms, markeredgecolor='black', markeredgewidth=0.5)

        if not single_bound:
            # Lower bound marker (square)
            u_lower_at_t = np.interp(t, lower_ts.t, lower_ts.u_gas)
            ax.plot(t * 1e3, u_lower_at_t, 's', color=color, markersize=ms-1, markeredgecolor='black', markeredgewidth=0.5)

    ax.set_xlabel('Time [ms]', fontsize=fonts['label'])
    ax.set_ylabel(r'$u(x=x_p, t)$ [m/s]', fontsize=fonts['label'])
    ax.set_title('Gas Velocity at Boundary', fontsize=fonts['title'])
    if not single_bound:
        ax.legend(fontsize=fonts['legend'], loc='lower right')
    ax.tick_params(labelsize=fonts['tick'])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=PLOT_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_file}")


def plot_multi_time_bounds_comparison(
    upper_snaps: List[SimSnapshot],
    lower_snaps: List[SimSnapshot],
    pele_snaps: List[PeleSnapshot],
    upper_ts: SimTimeseries,
    lower_ts: SimTimeseries,
    output_file: str,
    shift_to_upper: bool = False,
    column_mode: str = 'double',
    time_indices: Optional[List[int]] = None,
    lagrangian: bool = False,
):
    """
    Plot multi-time comparison with velocity bounds panel.

    Layout: 3 rows (Velocity, Temperature, Pressure) + velocity panel on right.
    Matches the reference multi_time_bounds_comparison.png format.

    Args:
        column_mode: 'single' for single column, 'double' for full page width
        time_indices: List of indices to plot. If None, plots all times.
        lagrangian: If True, use Lagrangian (mass) coordinates instead of position.
    """
    # Determine which time indices to plot
    if time_indices is None:
        time_indices = list(range(len(pele_snaps)))

    fonts = get_font_sizes(column_mode)
    lw = fonts['linewidth']
    ms = 5 if column_mode == 'single' else (12 if column_mode == 'presentation' else 8)  # Marker size

    if column_mode == 'single':
        # Single column: stack all 4 panels vertically
        fig = plt.figure(figsize=(SINGLE_COL_WIDTH, 6.5))
        gs = fig.add_gridspec(4, 1, hspace=0.45)
        ax_vel = fig.add_subplot(gs[0, 0])
        ax_temp = fig.add_subplot(gs[1, 0], sharex=ax_vel)
        ax_pres = fig.add_subplot(gs[2, 0], sharex=ax_vel)
        ax_piston = fig.add_subplot(gs[3, 0])
    elif column_mode == 'presentation':
        # Presentation: large figure for PowerPoint
        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(3, 2, width_ratios=[1.5, 1], height_ratios=[1, 1, 1],
                              hspace=0.25, wspace=0.35)
        ax_vel = fig.add_subplot(gs[0, 0])
        ax_temp = fig.add_subplot(gs[1, 0], sharex=ax_vel)
        ax_pres = fig.add_subplot(gs[2, 0], sharex=ax_vel)
        ax_piston = fig.add_subplot(gs[1:, 1])
    else:
        # Double column: 2x2 layout - profiles on left column, gas velocity bottom right
        fig = plt.figure(figsize=(FULL_PAGE_WIDTH, 4.0))
        gs = fig.add_gridspec(3, 2, width_ratios=[1.5, 1], height_ratios=[1, 1, 1],
                              hspace=0.08, wspace=0.3)
        ax_vel = fig.add_subplot(gs[0, 0])
        ax_temp = fig.add_subplot(gs[1, 0], sharex=ax_vel)
        ax_pres = fig.add_subplot(gs[2, 0], sharex=ax_vel)
        # Gas velocity panel spans bottom two rows on right
        ax_piston = fig.add_subplot(gs[1:, 1])

    profile_axes = [ax_vel, ax_temp, ax_pres]
    # Only include times that will be plotted for markers
    comparison_times = [pele_snaps[i].time for i in time_indices]

    # Plot each time (only those in time_indices)
    for t_idx in time_indices:
        upper = upper_snaps[t_idx]
        lower = lower_snaps[t_idx]
        pele = pele_snaps[t_idx]
        color = TIME_COLORS[t_idx % len(TIME_COLORS)]
        time_ms = pele.time * 1e3

        # Compute coordinates (Eulerian or Lagrangian)
        if lagrangian:
            # Lagrangian (mass) coordinates
            x_upper = compute_mass_coordinate_sim(upper)
            x_lower = compute_mass_coordinate_sim(lower)
        else:
            # Eulerian (position) coordinates in cm
            x_upper = upper.x_centers * 100
            x_lower = lower.x_centers * 100

        # Shift lower bound if requested
        if shift_to_upper and not lagrangian:
            shift = (upper.x_centers.min() - lower.x_centers.min()) * 100
            x_lower = x_lower + shift

        # Create common x grid for fill_between
        x_min = min(x_upper.min(), x_lower.min())
        x_max = max(x_upper.max(), x_lower.max())
        x_common = np.linspace(x_min, x_max, 500)

        def interp_safe(x_orig, y_orig, x_new):
            return np.interp(x_new, x_orig, y_orig, left=y_orig[0], right=y_orig[-1])

        # Velocity - solid lines for simulation
        u_upper = interp_safe(x_upper, upper.u, x_common)
        u_lower = interp_safe(x_lower, lower.u, x_common)
        ax_vel.fill_between(x_common, u_lower, u_upper, alpha=0.3, color=color)
        ax_vel.plot(x_upper, upper.u, '-', color=color, lw=lw)
        ax_vel.plot(x_lower, lower.u, '-', color=color, lw=lw)

        # Temperature - solid lines for simulation
        T_upper = interp_safe(x_upper, upper.T, x_common)
        T_lower = interp_safe(x_lower, lower.T, x_common)
        ax_temp.fill_between(x_common, T_lower, T_upper, alpha=0.3, color=color)
        ax_temp.plot(x_upper, upper.T, '-', color=color, lw=lw)
        ax_temp.plot(x_lower, lower.T, '-', color=color, lw=lw)

        # Pressure - solid lines for simulation
        p_upper = interp_safe(x_upper, upper.p / 1e6, x_common)
        p_lower = interp_safe(x_lower, lower.p / 1e6, x_common)
        ax_pres.fill_between(x_common, p_lower, p_upper, alpha=0.3, color=color)
        ax_pres.plot(x_upper, upper.p / 1e6, '-', color=color, lw=lw)
        ax_pres.plot(x_lower, lower.p / 1e6, '-', color=color, lw=lw)

        # PeleC data (filter points behind flame position using HRR) - dashed lines
        flame_x, buffer_x = find_flame_position_hrr(pele, buffer_cells=10)
        pele_mask = pele.x >= buffer_x
        if lagrangian:
            # Compute mass coordinate for PeleC starting from buffer position
            # This aligns with 1D model which starts from piston
            m_pele_full = compute_mass_coordinate_pele(pele)
            # Find mass at buffer position (start of filtered region)
            buffer_idx = np.searchsorted(pele.x, buffer_x)
            if buffer_idx < len(m_pele_full):
                m_offset = m_pele_full[buffer_idx]
            else:
                m_offset = 0
            x_pele = m_pele_full[pele_mask] - m_offset
        else:
            x_pele = pele.x[pele_mask] * 100  # Convert to cm for plotting
        if len(x_pele) > 0:
            ax_vel.plot(x_pele, pele.u[pele_mask], '--', color=color, lw=lw, alpha=0.8)
            if pele.T is not None:
                ax_temp.plot(x_pele, pele.T[pele_mask], '--', color=color, lw=lw, alpha=0.8)
            ax_pres.plot(x_pele, pele.p[pele_mask] / 1e6, '--', color=color, lw=lw, alpha=0.8)

    # Gas velocity at piston face panel (shows difference between solid and porous)
    t_upper_ms = upper_ts.t * 1e3
    t_lower_ms = lower_ts.t * 1e3

    t_min = max(upper_ts.t.min(), lower_ts.t.min())
    t_max = min(upper_ts.t.max(), lower_ts.t.max())
    t_common = np.linspace(t_min, t_max, 500)
    t_common_ms = t_common * 1e3

    # Use gas velocity (actual BC velocity) instead of piston velocity
    u_upper_interp = np.interp(t_common, upper_ts.t, upper_ts.u_gas)
    u_lower_interp = np.interp(t_common, lower_ts.t, lower_ts.u_gas)

    # Check if single-bound mode (upper and lower are the same)
    single_bound = upper_ts.path == lower_ts.path

    if single_bound:
        # Single bound: plot one line, no legend
        ax_piston.plot(t_upper_ms, upper_ts.u_gas, 'k-', lw=lw)
    else:
        # Two bounds: plot both with shaded region and legend
        ax_piston.fill_between(t_common_ms, u_lower_interp, u_upper_interp, alpha=0.3, color='gray')
        ax_piston.plot(t_upper_ms, upper_ts.u_gas, 'k-', lw=lw, label=r'$u_g = U_f$')
        ax_piston.plot(t_lower_ms, lower_ts.u_gas, 'k--', lw=lw, label=r'$u_g = U_f - CJ_{def}$')

    # Markers at comparison times (use original time_indices for color consistency)
    for orig_idx, t in zip(time_indices, comparison_times):
        color = TIME_COLORS[orig_idx % len(TIME_COLORS)]
        u_upper_at_t = np.interp(t, upper_ts.t, upper_ts.u_gas)
        ax_piston.plot(t * 1e3, u_upper_at_t, 'o', color=color, markersize=ms, markeredgecolor='black', markeredgewidth=0.5)
        if not single_bound:
            u_lower_at_t = np.interp(t, lower_ts.t, lower_ts.u_gas)
            ax_piston.plot(t * 1e3, u_lower_at_t, 's', color=color, markersize=ms-1, markeredgecolor='black', markeredgewidth=0.5)

    # Labels
    ax_vel.set_ylabel('Velocity [m/s]', fontsize=fonts['label'])
    ax_temp.set_ylabel('Temperature [K]', fontsize=fonts['label'])
    ax_pres.set_ylabel('Pressure [MPa]', fontsize=fonts['label'])
    x_label = r'Mass Coordinate [kg/m$^2$]' if lagrangian else 'Position [cm]'
    ax_pres.set_xlabel(x_label, fontsize=fonts['label'])

    ax_piston.set_xlabel('Time [ms]', fontsize=fonts['label'])
    ax_piston.set_ylabel(r'$u(x=x_p, t)$ [m/s]', fontsize=fonts['label'])
    if not single_bound:
        ax_piston.legend(fontsize=fonts['legend'] - 2, loc='upper left')

    # Hide x labels for upper profile plots (only for double column layout)
    if column_mode == 'double':
        plt.setp(ax_vel.get_xticklabels(), visible=False)
        plt.setp(ax_temp.get_xticklabels(), visible=False)

    # Legend for profile plots (solid = 1D model, dashed = 2D simulation)
    # Place legend outside plots at the top
    legend_lines = [
        Line2D([0], [0], color='gray', linestyle='-', lw=lw, label='1D Model'),
        Line2D([0], [0], color='gray', linestyle='--', lw=lw, label='2D Simulation'),
    ]
    for orig_idx, t in zip(time_indices, comparison_times):
        color = TIME_COLORS[orig_idx % len(TIME_COLORS)]
        legend_lines.append(Line2D([0], [0], color=color, linestyle='-', lw=lw,
                                    label=f't = {t*1e3:.2f} ms'))

    # Compute limits from ALL simulation data (consistent scaling for sequential overlays)
    all_u = np.concatenate([s.u for s in upper_snaps + lower_snaps])
    all_T = np.concatenate([s.T for s in upper_snaps + lower_snaps])
    all_p = np.concatenate([s.p for s in upper_snaps + lower_snaps]) / 1e6
    if lagrangian:
        all_x = np.concatenate([compute_mass_coordinate_sim(s) for s in upper_snaps + lower_snaps])
    else:
        all_x = np.concatenate([s.x_centers * 100 for s in upper_snaps + lower_snaps])  # cm

    # Set y-limits
    ax_vel.set_ylim(all_u.min() * 0.95 if all_u.min() > 0 else all_u.min() * 1.05, all_u.max() * 1.05)
    ax_temp.set_ylim(all_T.min() * 0.98, all_T.max() * 1.02)
    ax_pres.set_ylim(all_p.min() * 0.95, all_p.max() * 1.05)

    # Set x-limits (consistent across all sequential figures)
    x_margin = (all_x.max() - all_x.min()) * 0.02
    for ax in profile_axes:
        ax.set_xlim(all_x.min() - x_margin, all_x.max() + x_margin)

    for ax in profile_axes + [ax_piston]:
        ax.tick_params(labelsize=fonts['tick'])
        ax.grid(True, alpha=0.3)

    # Apply tight_layout and reserve space for legend at top
    plt.tight_layout()
    fig.subplots_adjust(top=0.92)

    # Place legend centered above the figure
    fig.legend(handles=legend_lines, fontsize=fonts['legend'] - 1, loc='upper center',
               ncol=3, bbox_to_anchor=(0.5, 0.99), frameon=True, fancybox=False,
               edgecolor='gray', columnspacing=1.2, handletextpad=0.4)
    plt.savefig(output_file, dpi=PLOT_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_file}")


def plot_multi_time_bounds_horizontal(
    upper_snaps: List[SimSnapshot],
    lower_snaps: List[SimSnapshot],
    pele_snaps: List[PeleSnapshot],
    output_file: str,
    shift_to_upper: bool = False,
    column_mode: str = 'double',
):
    """
    Plot multi-time comparison in horizontal layout.

    3 panels side by side: Velocity, Temperature, Pressure.

    Args:
        column_mode: 'single' for single column, 'double' for full page width
    """
    fonts = get_font_sizes(column_mode)
    lw = fonts['linewidth']

    if column_mode == 'single':
        # Stack vertically for single column
        fig, axes = plt.subplots(3, 1, figsize=(SINGLE_COL_WIDTH, 4.2))
    elif column_mode == 'presentation':
        # Large horizontal layout for PowerPoint
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    else:
        fig, axes = plt.subplots(1, 3, figsize=(FULL_PAGE_WIDTH, 2.0))
    ax_vel, ax_temp, ax_pres = axes.flatten() if column_mode == 'single' else axes

    comparison_times = [snap.time for snap in pele_snaps]

    for t_idx, (upper, lower, pele) in enumerate(zip(upper_snaps, lower_snaps, pele_snaps)):
        color = TIME_COLORS[t_idx % len(TIME_COLORS)]

        x_upper = upper.x_centers * 100
        x_lower = lower.x_centers * 100

        if shift_to_upper:
            shift = (upper.x_centers.min() - lower.x_centers.min()) * 100
            x_lower = x_lower + shift

        # Piston position lines
        for ax in [ax_vel, ax_temp, ax_pres]:
            ax.axvline(x_upper.min(), color=color, linestyle='--', alpha=0.5, lw=lw*0.7)

        x_min = min(x_upper.min(), x_lower.min())
        x_max = max(x_upper.max(), x_lower.max())
        x_common = np.linspace(x_min, x_max, 500)

        def interp_safe(x_orig, y_orig, x_new):
            return np.interp(x_new, x_orig, y_orig, left=y_orig[0], right=y_orig[-1])

        # Velocity - solid lines for simulation
        u_upper = interp_safe(x_upper, upper.u, x_common)
        u_lower = interp_safe(x_lower, lower.u, x_common)
        ax_vel.fill_between(x_common, u_lower, u_upper, alpha=0.3, color=color)
        ax_vel.plot(x_upper, upper.u, '-', color=color, lw=lw)
        ax_vel.plot(x_lower, lower.u, '-', color=color, lw=lw)

        # Temperature - solid lines for simulation
        T_upper = interp_safe(x_upper, upper.T, x_common)
        T_lower = interp_safe(x_lower, lower.T, x_common)
        ax_temp.fill_between(x_common, T_lower, T_upper, alpha=0.3, color=color)
        ax_temp.plot(x_upper, upper.T, '-', color=color, lw=lw)
        ax_temp.plot(x_lower, lower.T, '-', color=color, lw=lw)

        # Pressure - solid lines for simulation
        p_upper = interp_safe(x_upper, upper.p / 1e6, x_common)
        p_lower = interp_safe(x_lower, lower.p / 1e6, x_common)
        ax_pres.fill_between(x_common, p_lower, p_upper, alpha=0.3, color=color)
        ax_pres.plot(x_upper, upper.p / 1e6, '-', color=color, lw=lw)
        ax_pres.plot(x_lower, lower.p / 1e6, '-', color=color, lw=lw)

        # PeleC data (filter points behind flame position using HRR) - dashed lines
        flame_x, buffer_x = find_flame_position_hrr(pele, buffer_cells=10)
        pele_mask = pele.x >= buffer_x
        x_pele = pele.x[pele_mask] * 100  # Convert to cm for plotting
        if len(x_pele) > 0:
            ax_vel.plot(x_pele, pele.u[pele_mask], '--', color=color, lw=lw, alpha=0.8)
            if pele.T is not None:
                ax_temp.plot(x_pele, pele.T[pele_mask], '--', color=color, lw=lw, alpha=0.8)
            ax_pres.plot(x_pele, pele.p[pele_mask] / 1e6, '--', color=color, lw=lw, alpha=0.8)

    ax_vel.set_ylabel('Velocity [m/s]', fontsize=fonts['label'])
    ax_temp.set_ylabel('Temperature [K]', fontsize=fonts['label'])
    ax_pres.set_ylabel('Pressure [MPa]', fontsize=fonts['label'])

    # Compute y-limits from simulation data
    all_u = np.concatenate([s.u for s in upper_snaps + lower_snaps])
    all_T = np.concatenate([s.T for s in upper_snaps + lower_snaps])
    all_p = np.concatenate([s.p for s in upper_snaps + lower_snaps]) / 1e6

    ax_vel.set_ylim(all_u.min() * 0.95 if all_u.min() > 0 else all_u.min() * 1.05, all_u.max() * 1.05)
    ax_temp.set_ylim(all_T.min() * 0.98, all_T.max() * 1.02)
    ax_pres.set_ylim(all_p.min() * 0.95, all_p.max() * 1.05)

    for ax in [ax_vel, ax_temp, ax_pres]:
        ax.set_xlabel('Position [cm]', fontsize=fonts['label'])
        ax.tick_params(labelsize=fonts['tick'])
        ax.grid(True, alpha=0.3)

    # Legend (solid = 1D model, dashed = 2D simulation)
    legend_lines = [
        Line2D([0], [0], color='gray', linestyle='-', lw=lw, label='1D Model'),
        Line2D([0], [0], color='gray', linestyle='--', lw=lw, label='2D Simulation'),
    ]
    for t_idx, t in enumerate(comparison_times):
        color = TIME_COLORS[t_idx % len(TIME_COLORS)]
        legend_lines.append(Line2D([0], [0], color=color, linestyle='-', lw=lw,
                                    label=f't = {t*1e3:.2f} ms'))

    # Place legend at bottom (3 columns layout)
    plt.tight_layout()
    leg = fig.legend(handles=legend_lines, fontsize=fonts['legend'], loc='upper center',
                     ncol=3, bbox_to_anchor=(0.5, 0), frameon=True, fancybox=False,
                     edgecolor='gray', columnspacing=1.2, handletextpad=0.4)

    # Get actual legend height and adjust layout
    fig.canvas.draw()
    leg_bbox = leg.get_window_extent(fig.canvas.get_renderer())
    leg_height = leg_bbox.height / fig.dpi / fig.get_figheight()
    plt.subplots_adjust(bottom=leg_height + 0.02)
    plt.savefig(output_file, dpi=PLOT_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_file}")


def plot_accuracy(
    upper_snaps: List[SimSnapshot],
    lower_snaps: List[SimSnapshot],
    pele_snaps: List[PeleSnapshot],
    output_file: str,
    percent: bool = False,
    column_mode: str = 'single',
    show_boundaries: bool = False,
    nondim: bool = False,
    lagrangian: bool = False,
):
    """
    Plot accuracy/difference between 2D Simulation and 1D Model.

    Shows difference in velocity, temperature, and pressure profiles.
    Single bound: line plot. Two bounds: shaded region between bounds.

    Args:
        upper_snaps: Upper bound simulation snapshots
        lower_snaps: Lower bound simulation snapshots
        pele_snaps: PeleC (2D simulation) snapshots
        output_file: Output file path
        percent: If True, plot percent difference ((2D - 1D) / 2D), else absolute
        column_mode: 'single' for single column, 'double' for full page width
        show_boundaries: If True, show vertical dashed lines at piston and shock locations
        nondim: If True, non-dimensionalize x-axis between piston (0) and shock (1)
        lagrangian: If True, use Lagrangian (mass) coordinates instead of position
    """
    fonts = get_font_sizes(column_mode)
    lw = fonts['linewidth']

    # Check if single-bound mode
    single_bound = upper_snaps[0].path == lower_snaps[0].path

    # Stack vertically for single column (3 panels: velocity, temperature, pressure)
    if column_mode == 'single':
        fig, axes = plt.subplots(3, 1, figsize=(SINGLE_COL_WIDTH, 4.5))
    elif column_mode == 'presentation':
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    else:
        fig, axes = plt.subplots(1, 3, figsize=(FULL_PAGE_WIDTH, 2.0))

    ax_vel, ax_temp, ax_pres = axes.flatten() if column_mode == 'single' else axes

    # Store boundary positions for plotting vertical lines
    boundary_positions = []  # List of (color, piston_x, shock_x, x_min, x_max) tuples

    # Print diagnostic table header
    if single_bound:
        print("\n" + "=" * 140)
        print("ACCURACY REGION DIAGNOSTICS (Single Bound)")
        print("=" * 140)
        print(f"{'Time [ms]':>10} | {'Piston':>10} | {'Flame':>10} | {'PeleC Shock':>12} | {'Model Max':>10} | {'Terminator':>12}")
        print(f"{'':>10} | {'[cm]':>10} | {'[cm]':>10} | {'[cm]':>12} | {'[cm]':>10} | {'':>12}")
        print("-" * 140)
        print(f"{'':>10} | {'--- Error @ Piston Bound [%] ---':^34} | {'--- Error @ Shock Bound [%] ---':^34} | {'--- Error @ m=1 [%] ---':^34}")
        print(f"{'':>10} | {'Velocity':>10} | {'Temp':>10} | {'Pressure':>10} | {'Velocity':>10} | {'Temp':>10} | {'Pressure':>10} | {'Velocity':>10} | {'Temp':>10} | {'Pressure':>10}")
        print("-" * 140)
    else:
        print("\n" + "=" * 180)
        print("ACCURACY REGION DIAGNOSTICS (Upper/Lower Bounds)")
        print("=" * 180)
        print(f"{'Time [ms]':>10} | {'Piston':>10} | {'Flame':>10} | {'PeleC Shock':>12} | {'Model Max':>10} | {'Terminator':>12}")
        print(f"{'':>10} | {'[cm]':>10} | {'[cm]':>10} | {'[cm]':>12} | {'[cm]':>10} | {'':>12}")
        print("-" * 180)
        print(f"{'':>10} | {'Bound':>7} | {'--- Error @ Piston Bound [%] ---':^34} | {'--- Error @ Shock Bound [%] ---':^34} | {'--- Error @ m=1 [%] ---':^34}")
        print(f"{'':>10} | {'':>7} | {'Velocity':>10} | {'Temp':>10} | {'Pressure':>10} | {'Velocity':>10} | {'Temp':>10} | {'Pressure':>10} | {'Velocity':>10} | {'Temp':>10} | {'Pressure':>10}")
        print("-" * 180)

    for t_idx, (upper, lower, pele) in enumerate(zip(upper_snaps, lower_snaps, pele_snaps)):
        color = TIME_COLORS[t_idx % len(TIME_COLORS)]

        # Get pele position in cm
        x_pele_cm = pele.x * 100

        # Filter pele data to region covered by both 1D models (no extrapolation)
        x_upper_cm = upper.x_centers * 100
        x_lower_cm = lower.x_centers * 100

        # Valid region is the intersection of both model domains
        x_valid_min = max(x_upper_cm.min(), x_lower_cm.min())
        x_valid_max = min(x_upper_cm.max(), x_lower_cm.max())

        # Find flame location using HRR (max heat release rate)
        # Based on PeleFlameAnalyzer.find_wave_position method from pele_processing
        flame_x, buffer_x = find_flame_position_hrr(pele, buffer_cells=10)
        x_reactants_min = buffer_x * 100  # Convert to cm

        # Calculate grid spacing for far-end buffer
        if len(pele.x) > 1:
            delta_x_cm = np.abs(np.diff(pele.x)).min() * 100  # Convert to cm
        else:
            delta_x_cm = 0.01 * 100  # Fallback

        # Find shock front (where PeleC velocity becomes zero/near-zero)
        # Region of interest: between piston/flame and shock front
        #   - Piston/flame side: x_reactants_min = flame + 10*dx buffer
        #   - Shock side: x_reactants_max = shock_front - 10*dx buffer
        vel_threshold = 1.0  # m/s - velocity below this is considered "undisturbed"
        zero_vel_indices = np.where(np.abs(pele.u) < vel_threshold)[0]
        x_shock_front = None  # Will be set if shock is found
        if len(zero_vel_indices) > 0:
            # Find the first zero-velocity point (shock front) ahead of the flame
            zero_vel_x = pele.x[zero_vel_indices] * 100  # Convert to cm
            zero_vel_ahead = zero_vel_x[zero_vel_x > x_reactants_min]
            if len(zero_vel_ahead) > 0:
                x_shock_front = zero_vel_ahead.min()
                shock_buffer = 10 * delta_x_cm
                x_reactants_max = x_shock_front - shock_buffer
            else:
                x_reactants_max = x_valid_max
        else:
            x_reactants_max = x_valid_max

        # Only keep PeleC points between piston/flame and shock (excluding both)
        pele_mask = (x_pele_cm >= x_reactants_min) & (x_pele_cm <= x_reactants_max)

        # Store boundary positions for verification plot
        x_piston = x_upper_cm.min()  # Piston position
        x_shock_pele = x_shock_front if x_shock_front is not None else None
        x_model_max = x_valid_max  # 1D model domain boundary

        # Determine which terminates the comparison region
        if x_shock_pele is not None and x_shock_pele < x_model_max:
            terminator = "PeleC Shock"
            x_shock = x_shock_pele
        else:
            terminator = "Model Domain"
            x_shock = x_model_max

        # Compute percent error at boundaries for all 3 variables
        def compute_percent_error(pele_val, model_val, threshold=1.0):
            """Compute (PeleC - Model) / PeleC * 100, return NaN if near-zero."""
            if abs(pele_val) > threshold:
                return (pele_val - model_val) / pele_val * 100
            return np.nan

        # PeleC values at piston side (x_reactants_min)
        u_pele_piston = np.interp(x_reactants_min, x_pele_cm, pele.u)
        T_pele_piston = np.interp(x_reactants_min, x_pele_cm, pele.T) if pele.T is not None else np.nan
        p_pele_piston = np.interp(x_reactants_min, x_pele_cm, pele.p)

        # PeleC values at shock side (x_reactants_max)
        u_pele_shock = np.interp(x_reactants_max, x_pele_cm, pele.u)
        T_pele_shock = np.interp(x_reactants_max, x_pele_cm, pele.T) if pele.T is not None else np.nan
        p_pele_shock = np.interp(x_reactants_max, x_pele_cm, pele.p)

        # Upper bound model values
        u_upper_piston = np.interp(x_reactants_min, x_upper_cm, upper.u)
        T_upper_piston = np.interp(x_reactants_min, x_upper_cm, upper.T)
        p_upper_piston = np.interp(x_reactants_min, x_upper_cm, upper.p)
        u_upper_shock = np.interp(x_reactants_max, x_upper_cm, upper.u)
        T_upper_shock = np.interp(x_reactants_max, x_upper_cm, upper.T)
        p_upper_shock = np.interp(x_reactants_max, x_upper_cm, upper.p)

        # Errors for upper bound
        err_u_piston_upper = compute_percent_error(u_pele_piston, u_upper_piston, threshold=1.0)
        err_T_piston_upper = compute_percent_error(T_pele_piston, T_upper_piston, threshold=10.0)
        err_p_piston_upper = compute_percent_error(p_pele_piston, p_upper_piston, threshold=1000.0)
        err_u_shock_upper = compute_percent_error(u_pele_shock, u_upper_shock, threshold=1.0)
        err_T_shock_upper = compute_percent_error(T_pele_shock, T_upper_shock, threshold=10.0)
        err_p_shock_upper = compute_percent_error(p_pele_shock, p_upper_shock, threshold=1000.0)

        # Lower bound model values and errors (only if different from upper)
        if not single_bound:
            u_lower_piston = np.interp(x_reactants_min, x_lower_cm, lower.u)
            T_lower_piston = np.interp(x_reactants_min, x_lower_cm, lower.T)
            p_lower_piston = np.interp(x_reactants_min, x_lower_cm, lower.p)
            u_lower_shock = np.interp(x_reactants_max, x_lower_cm, lower.u)
            T_lower_shock = np.interp(x_reactants_max, x_lower_cm, lower.T)
            p_lower_shock = np.interp(x_reactants_max, x_lower_cm, lower.p)

            err_u_piston_lower = compute_percent_error(u_pele_piston, u_lower_piston, threshold=1.0)
            err_T_piston_lower = compute_percent_error(T_pele_piston, T_lower_piston, threshold=10.0)
            err_p_piston_lower = compute_percent_error(p_pele_piston, p_lower_piston, threshold=1000.0)
            err_u_shock_lower = compute_percent_error(u_pele_shock, u_lower_shock, threshold=1.0)
            err_T_shock_lower = compute_percent_error(T_pele_shock, T_lower_shock, threshold=10.0)
            err_p_shock_lower = compute_percent_error(p_pele_shock, p_lower_shock, threshold=1000.0)

        # Compute errors at mass coordinate m = 1 kg/m²
        # Mass coordinate for PeleC (starting from buffer position)
        m_pele_full = compute_mass_coordinate_pele(pele)
        buffer_idx = np.searchsorted(pele.x, buffer_x)
        m_offset = m_pele_full[buffer_idx] if buffer_idx < len(m_pele_full) else 0
        m_pele = m_pele_full - m_offset  # Offset so m=0 at buffer position

        # Mass coordinates for models (starting from piston)
        m_upper = compute_mass_coordinate_sim(upper)
        m_lower = compute_mass_coordinate_sim(lower) if not single_bound else m_upper

        # Check if m=1 is within the valid region
        m_target = 1.0  # kg/m²
        err_u_m1_upper = np.nan
        err_T_m1_upper = np.nan
        err_p_m1_upper = np.nan
        err_u_m1_lower = np.nan
        err_T_m1_lower = np.nan
        err_p_m1_lower = np.nan
        m1_status = "N/A"
        if m_pele.max() >= m_target and m_upper.max() >= m_target:
            # Interpolate PeleC values at m=1
            u_pele_m1 = np.interp(m_target, m_pele, pele.u)
            T_pele_m1 = np.interp(m_target, m_pele, pele.T) if pele.T is not None else np.nan
            p_pele_m1 = np.interp(m_target, m_pele, pele.p)
            # Interpolate upper model values at m=1
            u_upper_m1 = np.interp(m_target, m_upper, upper.u)
            T_upper_m1 = np.interp(m_target, m_upper, upper.T)
            p_upper_m1 = np.interp(m_target, m_upper, upper.p)
            # Compute percent errors for upper bound
            err_u_m1_upper = compute_percent_error(u_pele_m1, u_upper_m1, threshold=1.0)
            err_T_m1_upper = compute_percent_error(T_pele_m1, T_upper_m1, threshold=10.0)
            err_p_m1_upper = compute_percent_error(p_pele_m1, p_upper_m1, threshold=1000.0)
            m1_status = "OK"
            # Lower bound m=1 errors
            if not single_bound and m_lower.max() >= m_target:
                u_lower_m1 = np.interp(m_target, m_lower, lower.u)
                T_lower_m1 = np.interp(m_target, m_lower, lower.T)
                p_lower_m1 = np.interp(m_target, m_lower, lower.p)
                err_u_m1_lower = compute_percent_error(u_pele_m1, u_lower_m1, threshold=1.0)
                err_T_m1_lower = compute_percent_error(T_pele_m1, T_lower_m1, threshold=10.0)
                err_p_m1_lower = compute_percent_error(p_pele_m1, p_lower_m1, threshold=1000.0)
        elif m_pele.max() < m_target:
            m1_status = f"PeleC m_max={m_pele.max():.2f}"
        else:
            m1_status = f"Model m_max={m_upper.max():.2f}"

        # Print diagnostic rows
        time_ms = pele.time * 1e3
        flame_x_cm = flame_x * 100
        shock_pele_str = f"{x_shock_pele:.2f}" if x_shock_pele is not None else "N/A"

        def fmt_err(val):
            return f"{val:>10.1f}" if not np.isnan(val) else "       N/A"

        if single_bound:
            # Row 1: Positions and terminator
            print(f"{time_ms:>10.2f} | {x_piston:>10.2f} | {flame_x_cm:>10.2f} | {shock_pele_str:>12} | {x_model_max:>10.2f} | {terminator:>12}")
            # Row 2: Errors at boundaries and m=1
            print(f"{'':>10} | {fmt_err(err_u_piston_upper)} | {fmt_err(err_T_piston_upper)} | {fmt_err(err_p_piston_upper)} | {fmt_err(err_u_shock_upper)} | {fmt_err(err_T_shock_upper)} | {fmt_err(err_p_shock_upper)} | {fmt_err(err_u_m1_upper)} | {fmt_err(err_T_m1_upper)} | {fmt_err(err_p_m1_upper)}")
            print("-" * 140)
        else:
            # Row 1: Positions and terminator
            print(f"{time_ms:>10.2f} | {x_piston:>10.2f} | {flame_x_cm:>10.2f} | {shock_pele_str:>12} | {x_model_max:>10.2f} | {terminator:>12}")
            # Row 2: Upper bound errors
            print(f"{'':>10} | {'Upper':>7} | {fmt_err(err_u_piston_upper)} | {fmt_err(err_T_piston_upper)} | {fmt_err(err_p_piston_upper)} | {fmt_err(err_u_shock_upper)} | {fmt_err(err_T_shock_upper)} | {fmt_err(err_p_shock_upper)} | {fmt_err(err_u_m1_upper)} | {fmt_err(err_T_m1_upper)} | {fmt_err(err_p_m1_upper)}")
            # Row 3: Lower bound errors
            print(f"{'':>10} | {'Lower':>7} | {fmt_err(err_u_piston_lower)} | {fmt_err(err_T_piston_lower)} | {fmt_err(err_p_piston_lower)} | {fmt_err(err_u_shock_lower)} | {fmt_err(err_T_shock_lower)} | {fmt_err(err_p_shock_lower)} | {fmt_err(err_u_m1_lower)} | {fmt_err(err_T_m1_lower)} | {fmt_err(err_p_m1_lower)}")
            print("-" * 180)

        boundary_positions.append((color, x_piston, x_shock, x_reactants_min, x_reactants_max))

        x_pele_filtered = x_pele_cm[pele_mask]
        if len(x_pele_filtered) == 0:
            continue

        # Interpolate model data to pele positions (no extrapolation needed now)
        u_upper_interp = np.interp(x_pele_filtered, x_upper_cm, upper.u)
        u_lower_interp = np.interp(x_pele_filtered, x_lower_cm, lower.u)
        u_pele = pele.u[pele_mask]

        T_upper_interp = np.interp(x_pele_filtered, x_upper_cm, upper.T)
        T_lower_interp = np.interp(x_pele_filtered, x_lower_cm, lower.T)
        T_pele = pele.T[pele_mask] if pele.T is not None else None

        p_upper_interp = np.interp(x_pele_filtered, x_upper_cm, upper.p)
        p_lower_interp = np.interp(x_pele_filtered, x_lower_cm, lower.p)
        p_pele = pele.p[pele_mask]

        # Exclude points where velocity is zero or near-zero in PeleC
        # This filters out unphysical/unresolved regions in the 2D simulation
        vel_threshold = 1.0  # m/s - minimum velocity to include in accuracy calculation
        vel_nonzero_mask = np.abs(u_pele) >= vel_threshold

        x_pele_filtered = x_pele_filtered[vel_nonzero_mask]
        u_pele = u_pele[vel_nonzero_mask]
        u_upper_interp = u_upper_interp[vel_nonzero_mask]
        u_lower_interp = u_lower_interp[vel_nonzero_mask]
        T_upper_interp = T_upper_interp[vel_nonzero_mask]
        T_lower_interp = T_lower_interp[vel_nonzero_mask]
        T_pele = T_pele[vel_nonzero_mask] if T_pele is not None else None
        p_upper_interp = p_upper_interp[vel_nonzero_mask]
        p_lower_interp = p_lower_interp[vel_nonzero_mask]
        p_pele = p_pele[vel_nonzero_mask]

        if len(x_pele_filtered) == 0:
            continue

        # Compute differences
        if percent:
            # Percent difference: (2D - 1D) / 2D * 100
            # Use threshold to avoid division by near-zero values
            def safe_percent(pele_val, model_val, threshold_frac=0.01):
                """Compute percent difference, masking out near-zero denominators.

                Points where |pele_val| < threshold_frac * max(|pele_val|) are set to NaN.
                """
                result = np.full_like(pele_val, np.nan, dtype=float)
                max_abs = np.max(np.abs(pele_val))
                if max_abs == 0:
                    return result
                threshold = threshold_frac * max_abs
                valid = np.abs(pele_val) >= threshold
                if np.any(valid):
                    result[valid] = (pele_val[valid] - model_val[valid]) / pele_val[valid] * 100
                return result

            diff_u_upper = safe_percent(u_pele, u_upper_interp)
            diff_u_lower = safe_percent(u_pele, u_lower_interp)

            if T_pele is not None:
                diff_T_upper = safe_percent(T_pele, T_upper_interp)
                diff_T_lower = safe_percent(T_pele, T_lower_interp)
            else:
                diff_T_upper = diff_T_lower = None

            diff_p_upper = safe_percent(p_pele, p_upper_interp)
            diff_p_lower = safe_percent(p_pele, p_lower_interp)
        else:
            # Absolute difference: 2D - 1D
            diff_u_upper = u_pele - u_upper_interp
            diff_u_lower = u_pele - u_lower_interp

            if T_pele is not None:
                diff_T_upper = T_pele - T_upper_interp
                diff_T_lower = T_pele - T_lower_interp
            else:
                diff_T_upper = diff_T_lower = None

            diff_p_upper = (p_pele - p_upper_interp) / 1e6  # Convert to MPa
            diff_p_lower = (p_pele - p_lower_interp) / 1e6

        # Compute plot coordinates (Eulerian or Lagrangian, with optional non-dimensionalization)
        if lagrangian:
            # Compute mass coordinates for filtered PeleC data
            m_pele_full = compute_mass_coordinate_pele(pele)
            # Get indices of filtered points
            pele_indices = np.where(pele_mask)[0]
            vel_indices = pele_indices[vel_nonzero_mask]

            # Offset mass coordinate to start from buffer position (aligns with 1D model)
            buffer_idx = np.searchsorted(pele.x, buffer_x)
            if buffer_idx < len(m_pele_full):
                m_offset = m_pele_full[buffer_idx]
            else:
                m_offset = 0
            x_plot_raw = m_pele_full[vel_indices] - m_offset

            if nondim:
                # Non-dimensionalize mass coordinates
                m_min = x_plot_raw.min()
                m_max = x_plot_raw.max()
                if m_max > m_min:
                    x_plot = (x_plot_raw - m_min) / (m_max - m_min)
                else:
                    x_plot = x_plot_raw
            else:
                x_plot = x_plot_raw
        elif nondim:
            # Non-dimensionalize position coordinates
            x_piston = x_upper_cm.min()
            x_shock_pos = x_shock if x_shock is not None else x_valid_max
            if x_shock_pos > x_piston:
                x_plot = (x_pele_filtered - x_piston) / (x_shock_pos - x_piston)
            else:
                x_plot = x_pele_filtered  # Fallback if invalid
        else:
            x_plot = x_pele_filtered

        # Plot velocity difference
        if single_bound:
            ax_vel.plot(x_plot, diff_u_upper, '-', color=color, lw=lw)
        else:
            # Shaded region between bounds
            diff_u_min = np.minimum(diff_u_upper, diff_u_lower)
            diff_u_max = np.maximum(diff_u_upper, diff_u_lower)
            ax_vel.fill_between(x_plot, diff_u_min, diff_u_max, alpha=0.3, color=color)
            ax_vel.plot(x_plot, diff_u_upper, '-', color=color, lw=lw)
            ax_vel.plot(x_plot, diff_u_lower, '--', color=color, lw=lw)

        # Plot temperature difference
        if diff_T_upper is not None:
            if single_bound:
                ax_temp.plot(x_plot, diff_T_upper, '-', color=color, lw=lw)
            else:
                diff_T_min = np.minimum(diff_T_upper, diff_T_lower)
                diff_T_max = np.maximum(diff_T_upper, diff_T_lower)
                ax_temp.fill_between(x_plot, diff_T_min, diff_T_max, alpha=0.3, color=color)
                ax_temp.plot(x_plot, diff_T_upper, '-', color=color, lw=lw)
                ax_temp.plot(x_plot, diff_T_lower, '--', color=color, lw=lw)

        # Plot pressure difference
        if single_bound:
            ax_pres.plot(x_plot, diff_p_upper, '-', color=color, lw=lw)
        else:
            diff_p_min = np.minimum(diff_p_upper, diff_p_lower)
            diff_p_max = np.maximum(diff_p_upper, diff_p_lower)
            ax_pres.fill_between(x_plot, diff_p_min, diff_p_max, alpha=0.3, color=color)
            ax_pres.plot(x_plot, diff_p_upper, '-', color=color, lw=lw)
            ax_pres.plot(x_plot, diff_p_lower, '--', color=color, lw=lw)

    # Print table footer
    print("=" * 100)
    print()

    # Plot vertical dashed lines at piston and shock locations (for verification)
    if show_boundaries:
        for color, x_piston, x_shock, x_min, x_max in boundary_positions:
            for ax in [ax_vel, ax_temp, ax_pres]:
                # Piston location (solid line)
                ax.axvline(x_piston, color=color, linestyle=':', lw=1.0, alpha=0.7)
                # Shock location (solid line)
                ax.axvline(x_shock, color=color, linestyle=':', lw=1.0, alpha=0.7)
                # Included region boundaries (dashed lines)
                ax.axvline(x_min, color=color, linestyle='--', lw=1.0, alpha=0.5)
                ax.axvline(x_max, color=color, linestyle='--', lw=1.0, alpha=0.5)

    # Labels
    if percent:
        ax_vel.set_ylabel('Velocity Diff [%]', fontsize=fonts['label'])
        ax_temp.set_ylabel('Temperature Diff [%]', fontsize=fonts['label'])
        ax_pres.set_ylabel('Pressure Diff [%]', fontsize=fonts['label'])
        title_suffix = '(Percent)'
    else:
        ax_vel.set_ylabel('Velocity Diff [m/s]', fontsize=fonts['label'])
        ax_temp.set_ylabel('Temperature Diff [K]', fontsize=fonts['label'])
        ax_pres.set_ylabel('Pressure Diff [MPa]', fontsize=fonts['label'])
        title_suffix = '(Absolute)'

    # Add zero line for reference
    if lagrangian:
        if nondim:
            x_label = r'$(m - m_p) / (m_s - m_p)$'
        else:
            x_label = r'Mass Coordinate [kg/m$^2$]'
    elif nondim:
        x_label = r'$(x - x_p) / (x_s - x_p)$'
    else:
        x_label = 'Position [cm]'

    for ax in [ax_vel, ax_temp, ax_pres]:
        ax.axhline(0, color='gray', linestyle=':', lw=0.5, alpha=0.7)
        ax.set_xlabel(x_label, fontsize=fonts['label'])
        ax.tick_params(labelsize=fonts['tick'])
        ax.grid(True, alpha=0.3)

    # Set x-axis limits for non-dimensional plot
    if nondim:
        for ax in [ax_vel, ax_temp, ax_pres]:
            ax.set_xlim(0, 1)

    # Legend
    legend_lines = []
    if not single_bound:
        legend_lines.extend([
            Line2D([0], [0], color='gray', linestyle='-', lw=lw, label='Upper bound'),
            Line2D([0], [0], color='gray', linestyle='--', lw=lw, label='Lower bound'),
        ])
    for t_idx, pele in enumerate(pele_snaps):
        color = TIME_COLORS[t_idx % len(TIME_COLORS)]
        legend_lines.append(Line2D([0], [0], color=color, linestyle='-', lw=lw,
                                    label=f't = {pele.time*1e3:.2f} ms'))

    fig.suptitle(f'Accuracy: (2D Simulation) - (1D Model) {title_suffix}',
                 fontsize=fonts['title'], fontweight='bold')

    # Place legend at bottom (3 columns layout)
    plt.tight_layout()
    leg = fig.legend(handles=legend_lines, fontsize=fonts['legend'], loc='upper center',
                     ncol=3, bbox_to_anchor=(0.5, 0), frameon=True, fancybox=False,
                     edgecolor='gray', columnspacing=1.2, handletextpad=0.4)

    # Get actual legend height and adjust layout
    fig.canvas.draw()
    leg_bbox = leg.get_window_extent(fig.canvas.get_renderer())
    leg_height = leg_bbox.height / fig.dpi / fig.get_figheight()
    plt.subplots_adjust(bottom=leg_height + 0.02)
    plt.savefig(output_file, dpi=PLOT_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_file}")


def load_xt_data(results_dir: Path) -> Optional[Dict]:
    """
    Load X-T diagram data from a results directory.

    Assembles time, position, and field arrays from all snapshots.

    Args:
        results_dir: Path to results directory containing snapshots/

    Returns:
        Dict with 't', 'x', 'p', 'u', 'rho' arrays, or None if failed
    """
    snapshot_dir = results_dir / "snapshots"
    if not snapshot_dir.exists():
        print(f"  Warning: No snapshots directory found in {results_dir}")
        return None

    snapshot_files = sorted(snapshot_dir.glob("snapshot_*.npz"))
    if not snapshot_files:
        print(f"  Warning: No snapshot files found in {snapshot_dir}")
        return None

    print(f"  Loading {len(snapshot_files)} snapshots from {results_dir.name}...")

    t_list = []
    x_list = []
    p_list = []
    u_list = []
    rho_list = []

    for f in snapshot_files:
        try:
            data = np.load(f)
            t_list.append(float(data['t']))
            x_list.append(data['x_centers'])
            p_list.append(data['p'])
            # Cell-centered velocity from face velocities
            u_faces = data['u']
            u_cell = 0.5 * (u_faces[:-1] + u_faces[1:]) if len(u_faces) > 1 else u_faces
            u_list.append(u_cell)
            rho_list.append(data['rho'])
        except Exception as e:
            print(f"    Warning: Could not load {f.name}: {e}")
            continue

    if not t_list:
        return None

    return {
        't': np.array(t_list),
        'x': np.array(x_list, dtype=object),  # Variable length arrays
        'p': np.array(p_list, dtype=object),
        'u': np.array(u_list, dtype=object),
        'rho': np.array(rho_list, dtype=object),
    }


def plot_xt_diagrams(
    upper_ts_data: Dict,
    lower_ts_data: Dict,
    output_dir: Path,
    column_mode: str = 'double',
):
    """
    Plot X-T diagrams comparing upper and lower bounds.

    Side-by-side comparison for pressure, velocity, density.

    Args:
        upper_ts_data: Dict with 't', 'x', 'p', 'u', 'rho' arrays (from load_xt_data)
        lower_ts_data: Dict with 't', 'x', 'p', 'u', 'rho' arrays (from load_xt_data)
        output_dir: Output directory for figures
        column_mode: 'single' for single column, 'double' for full page width
    """
    fonts = get_font_sizes(column_mode)

    upper_ts = upper_ts_data
    lower_ts = lower_ts_data

    mode_suffix = f"_{column_mode}" if column_mode != 'double' else ""

    for var, units, vmin, vmax in [
        ('p', 'Pressure', None, None),
        ('u', 'Velocity', None, None),
        ('rho', 'Density', None, None),
    ]:
        if column_mode == 'single':
            # Stack vertically for single column
            fig, axes = plt.subplots(2, 1, figsize=(SINGLE_COL_WIDTH, 3.5))
        elif column_mode == 'presentation':
            # Large side-by-side for PowerPoint
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        else:
            fig, axes = plt.subplots(1, 2, figsize=(FULL_PAGE_WIDTH, 2.5))

        for ax, data, label in [(axes[0], upper_ts, 'Flame Velocity (No Porous)'),
                                 (axes[1], lower_ts, 'Flame Vel - CJ (Porous)')]:
            t = data['t']
            x_arr = data['x']
            var_arr = data[var]

            # Create meshgrid for pcolormesh
            n_times = len(t)
            if hasattr(x_arr[0], '__len__'):
                # Variable length arrays
                x_max_len = max(len(x) for x in x_arr)
                X = np.zeros((n_times, x_max_len))
                V = np.zeros((n_times, x_max_len))
                for i in range(n_times):
                    xi = x_arr[i]
                    vi = var_arr[i]
                    X[i, :len(xi)] = xi
                    V[i, :len(vi)] = vi
                    if len(xi) < x_max_len:
                        X[i, len(xi):] = xi[-1]
                        V[i, len(vi):] = vi[-1]
            else:
                X = x_arr
                V = var_arr

            T = np.tile(t * 1e3, (X.shape[1], 1)).T

            pcm = ax.pcolormesh(X, T, V, shading='auto', cmap='jet')
            plt.colorbar(pcm, ax=ax, label=units)

            ax.set_xlabel('Position x [m]', fontsize=fonts['label'])
            ax.set_ylabel('Time [ms]', fontsize=fonts['label'])
            ax.set_title(label, fontsize=fonts['title'])
            ax.tick_params(labelsize=fonts['tick'])

        fig.suptitle(f'x-t Diagram: {units}', fontsize=fonts['title'] + 1, fontweight='bold')
        plt.tight_layout()

        output_file = output_dir / f"xt_diagram_{var}{mode_suffix}.png"
        plt.savefig(output_file, dpi=PLOT_DPI, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {output_file}")


def plot_piston_results(
    upper_snaps: List[SimSnapshot],
    lower_snaps: List[SimSnapshot],
    upper_ts: SimTimeseries,
    lower_ts: SimTimeseries,
    output_file: str,
    column_mode: str = 'double',
):
    """
    Plot 6-panel piston results comparison.

    Shows density, velocity, pressure, temperature profiles and
    mesh movement and conservation errors.

    Args:
        column_mode: 'single' for single column, 'double' for full page width
    """
    fonts = get_font_sizes(column_mode)
    lw = fonts['linewidth']

    # Use the last snapshot for profile comparisons
    upper = upper_snaps[-1]
    lower = lower_snaps[-1]

    if column_mode == 'single':
        # 6 panels stacked vertically in 2 columns
        fig, axes = plt.subplots(3, 2, figsize=(SINGLE_COL_WIDTH, 5.0))
    elif column_mode == 'presentation':
        # Large 3x2 grid for PowerPoint
        fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    else:
        fig, axes = plt.subplots(3, 2, figsize=(FULL_PAGE_WIDTH, 5.5))

    time_s = upper.time

    x_upper = upper.x_centers
    x_lower = lower.x_centers

    # Density
    axes[0, 0].plot(x_upper, upper.rho, '-', color='blue', lw=lw, label='No Porous')
    axes[0, 0].plot(x_lower, lower.rho, '--', color='red', lw=lw, label='Porous')
    axes[0, 0].set_ylabel('Density [kg/m³]', fontsize=fonts['label'])
    axes[0, 0].set_title('Density Profile', fontsize=fonts['title'])
    axes[0, 0].legend(fontsize=fonts['legend'] - 1)

    # Velocity
    axes[0, 1].plot(x_upper, upper.u, '-', color='blue', lw=lw, label='No Porous')
    axes[0, 1].plot(x_lower, lower.u, '--', color='red', lw=lw, label='Porous')
    axes[0, 1].set_ylabel('Velocity [m/s]', fontsize=fonts['label'])
    axes[0, 1].set_title('Velocity Profile', fontsize=fonts['title'])
    axes[0, 1].legend(fontsize=fonts['legend'] - 1)

    # Pressure
    axes[1, 0].plot(x_upper, upper.p / 1e5, '-', color='blue', lw=lw, label='No Porous')
    axes[1, 0].plot(x_lower, lower.p / 1e5, '--', color='red', lw=lw, label='Porous')
    axes[1, 0].set_ylabel('Pressure [bar]', fontsize=fonts['label'])
    axes[1, 0].set_title('Pressure Profile', fontsize=fonts['title'])
    axes[1, 0].legend(fontsize=fonts['legend'] - 1)

    # Temperature
    axes[1, 1].plot(x_upper, upper.T, '-', color='blue', lw=lw, label='No Porous')
    axes[1, 1].plot(x_lower, lower.T, '--', color='red', lw=lw, label='Porous')
    axes[1, 1].set_ylabel('Temperature [K]', fontsize=fonts['label'])
    axes[1, 1].set_title('Temperature Profile', fontsize=fonts['title'])
    axes[1, 1].legend(fontsize=fonts['legend'] - 1)

    # Piston position vs time
    axes[2, 0].plot(upper_ts.t * 1e3, upper_ts.x_piston, '-', color='blue', lw=lw, label='No Porous')
    axes[2, 0].plot(lower_ts.t * 1e3, lower_ts.x_piston, '--', color='red', lw=lw, label='Porous')
    axes[2, 0].set_xlabel('Time [ms]', fontsize=fonts['label'])
    axes[2, 0].set_ylabel('Piston Position [m]', fontsize=fonts['label'])
    axes[2, 0].set_title('Piston Position', fontsize=fonts['title'])
    axes[2, 0].legend(fontsize=fonts['legend'] - 1)

    # Piston velocity vs time
    axes[2, 1].plot(upper_ts.t * 1e3, upper_ts.u_piston, '-', color='blue', lw=lw, label='No Porous')
    axes[2, 1].plot(lower_ts.t * 1e3, lower_ts.u_piston, '--', color='red', lw=lw, label='Porous')
    axes[2, 1].set_xlabel('Time [ms]', fontsize=fonts['label'])
    axes[2, 1].set_ylabel('Piston Velocity [m/s]', fontsize=fonts['label'])
    axes[2, 1].set_title('Piston Velocity', fontsize=fonts['title'])
    axes[2, 1].legend(fontsize=fonts['legend'] - 1)

    for ax in axes.flat:
        ax.tick_params(labelsize=fonts['tick'])
        ax.grid(True, alpha=0.3)

    # Set x labels for profile plots
    axes[0, 0].set_xlabel('Position x [m]', fontsize=fonts['label'])
    axes[0, 1].set_xlabel('Position x [m]', fontsize=fonts['label'])
    axes[1, 0].set_xlabel('Position x [m]', fontsize=fonts['label'])
    axes[1, 1].set_xlabel('Position x [m]', fontsize=fonts['label'])

    fig.suptitle(f'Piston-Driven Flow Comparison at t = {time_s:.6f} s', fontsize=fonts['title'] + 1, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_file, dpi=PLOT_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_file}")


def plot_velocity_comparison(
    upper_ts: SimTimeseries,
    lower_ts: SimTimeseries,
    output_file: str,
    column_mode: str = 'double',
):
    """
    Plot 4-panel velocity data comparison.

    Shows raw velocities, corrected velocities, gas BC velocity, and flame position.

    Args:
        column_mode: 'single' for single column, 'double' for full page width
    """
    fonts = get_font_sizes(column_mode)
    lw = fonts['linewidth']

    if column_mode == 'single':
        # Stack as 4x1 for single column
        fig, axes = plt.subplots(4, 1, figsize=(SINGLE_COL_WIDTH, 5.5))
        axes = axes.reshape(2, 2)  # Keep same indexing
    elif column_mode == 'presentation':
        # Large 2x2 grid for PowerPoint
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    else:
        fig, axes = plt.subplots(2, 2, figsize=(FULL_PAGE_WIDTH, 4.0))

    t_upper_ms = upper_ts.t * 1e3
    t_lower_ms = lower_ts.t * 1e3

    # Raw velocities (same as piston velocity for now)
    axes[0, 0].plot(t_upper_ms, upper_ts.u_piston, '-', color='blue', lw=lw, label='Upper')
    axes[0, 0].plot(t_lower_ms, lower_ts.u_piston, '-', color='red', lw=lw, label='Lower')
    axes[0, 0].set_ylabel('Velocity [m/s]', fontsize=fonts['label'])
    axes[0, 0].set_xlabel('Time [ms]', fontsize=fonts['label'])
    axes[0, 0].set_title('Raw Velocities', fontsize=fonts['title'])
    axes[0, 0].legend(fontsize=fonts['legend'] - 1)

    # Piston velocity (after correction)
    axes[0, 1].plot(t_upper_ms, upper_ts.u_piston, '-', color='blue', lw=lw, label='No Porous')
    axes[0, 1].plot(t_lower_ms, lower_ts.u_piston, '--', color='red', lw=lw, label='Porous')
    axes[0, 1].set_ylabel('Velocity [m/s]', fontsize=fonts['label'])
    axes[0, 1].set_xlabel('Time [ms]', fontsize=fonts['label'])
    axes[0, 1].set_title('Corrected Velocity', fontsize=fonts['title'])
    axes[0, 1].legend(fontsize=fonts['legend'] - 1)

    # Gas BC velocity (actual velocity used at boundary)
    axes[1, 0].plot(t_upper_ms, upper_ts.u_gas, '-', color='blue', lw=lw, label='No Porous')
    axes[1, 0].plot(t_lower_ms, lower_ts.u_gas, '--', color='red', lw=lw, label='Porous')
    axes[1, 0].set_ylabel('Gas Velocity [m/s]', fontsize=fonts['label'])
    axes[1, 0].set_xlabel('Time [ms]', fontsize=fonts['label'])
    axes[1, 0].set_title('Gas BC Velocity', fontsize=fonts['title'])
    axes[1, 0].legend(fontsize=fonts['legend'] - 1)

    # Flame position
    axes[1, 1].plot(t_upper_ms, upper_ts.x_piston, '-', color='blue', lw=lw, label='No Porous')
    axes[1, 1].plot(t_lower_ms, lower_ts.x_piston, '--', color='red', lw=lw, label='Porous')
    axes[1, 1].set_ylabel('Position [m]', fontsize=fonts['label'])
    axes[1, 1].set_xlabel('Time [ms]', fontsize=fonts['label'])
    axes[1, 1].set_title('Flame Position', fontsize=fonts['title'])
    axes[1, 1].legend(fontsize=fonts['legend'] - 1)

    for ax in axes.flat:
        ax.tick_params(labelsize=fonts['tick'])
        ax.grid(True, alpha=0.3)

    fig.suptitle('Velocity Data Comparison', fontsize=fonts['title'] + 1, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_file, dpi=PLOT_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_file}")


def compare_with_bounds(
    upper_results_dir: Path,
    lower_results_dir: Path,
    pltfile_dir: Path,
    output_dir: Path,
    upper_label: str = "Flame Velocity (No Porous)",
    lower_label: str = "Flame Vel - CJ (Porous)",
    extract_location: float = 0.0445,
    shift_to_upper: bool = False,
):
    """
    Full comparison with upper/lower bounds and PeleC data.

    Produces all comparison figures matching the reference directory.
    """
    print("=" * 70)
    print("BOUNDS COMPARISON")
    print("=" * 70)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load timeseries
    print("\nLoading timeseries...")
    upper_ts = load_timeseries(upper_results_dir, upper_label)
    lower_ts = load_timeseries(lower_results_dir, lower_label)

    if upper_ts is None or lower_ts is None:
        print("ERROR: Could not load timeseries data!")
        return

    print(f"  Upper: {len(upper_ts.t)} timesteps, t_max = {upper_ts.t.max()*1e3:.2f} ms")
    print(f"  Lower: {len(lower_ts.t)} timesteps, t_max = {lower_ts.t.max()*1e3:.2f} ms")

    # Load PeleC data first to get target times
    print(f"\nLoading PeleC pltfiles: {pltfile_dir}")
    pele_snaps = load_all_pltfiles(str(pltfile_dir), extract_location)
    print(f"  Loaded {len(pele_snaps)} PeleC snapshots")

    target_times = [pele.time for pele in pele_snaps]
    print(f"  Target times: {[f'{t*1e3:.2f}ms' for t in target_times]}")

    # Find matching snapshots using coarse-to-fine search
    print("\nFinding matching snapshots (coarse-to-fine search)...")
    upper_snap_dir = upper_results_dir / "snapshots"
    lower_snap_dir = lower_results_dir / "snapshots"

    upper_files = get_snapshot_files(upper_snap_dir)
    lower_files = get_snapshot_files(lower_snap_dir)

    print(f"  Upper: {len(upper_files)} snapshot files")
    print(f"  Lower: {len(lower_files)} snapshot files")

    print("  Searching upper snapshots...")
    upper_snaps_info = find_snapshots_for_times_coarse_to_fine(upper_files, target_times)

    print("  Searching lower snapshots...")
    lower_snaps_info = find_snapshots_for_times_coarse_to_fine(lower_files, target_times)

    # Load the matched snapshots
    print("\nLoading matched snapshots...")
    upper_matched = []
    lower_matched = []

    for i, pele in enumerate(pele_snaps):
        t_ms = pele.time * 1e3

        upper_snap = load_snapshot(upper_snaps_info[i], upper_label)
        lower_snap = load_snapshot(lower_snaps_info[i], lower_label)

        upper_matched.append(upper_snap)
        lower_matched.append(lower_snap)

        print(f"  t = {t_ms:.2f} ms: upper={upper_snap.time*1e3:.2f}ms, lower={lower_snap.time*1e3:.2f}ms")

    comparison_times = [pele.time for pele in pele_snaps]

    # Generate plots (X-T diagrams last since they require full timeseries)
    print("\nGenerating plots...")

    # 1. Velocity comparison (single column only)
    plot_velocity_comparison(
        upper_ts, lower_ts,
        str(output_dir / "velocity_comparison_single.png"),
        column_mode='single',
    )

    # 2. Multi-time bounds comparison - sequential figures (single column only, no shift)
    # Generate N figures where figure i shows times 0 through i (cumulative)
    n_times = len(pele_snaps)
    for i in range(n_times):
        time_indices = list(range(i + 1))  # [0], [0,1], [0,1,2], ...
        plot_multi_time_bounds_comparison(
            upper_matched, lower_matched, pele_snaps,
            upper_ts, lower_ts,
            str(output_dir / f"multi_time_bounds_comparison_single_{i+1}.png"),
            shift_to_upper=False,
            column_mode='single',
            time_indices=time_indices,
        )

    # 3. Accuracy plots (single column only)
    # Absolute difference
    plot_accuracy(
        upper_matched, lower_matched, pele_snaps,
        str(output_dir / "accuracy_absolute_single.png"),
        percent=False,
        column_mode='single',
    )

    # Absolute difference with boundary check lines
    plot_accuracy(
        upper_matched, lower_matched, pele_snaps,
        str(output_dir / "accuracy_absolute_single_check.png"),
        percent=False,
        column_mode='single',
        show_boundaries=True,
    )

    # Percent difference
    plot_accuracy(
        upper_matched, lower_matched, pele_snaps,
        str(output_dir / "accuracy_percent_single.png"),
        percent=True,
        column_mode='single',
    )

    # Percent difference with non-dimensionalized x-axis
    plot_accuracy(
        upper_matched, lower_matched, pele_snaps,
        str(output_dir / "accuracy_percent_single_nondim.png"),
        percent=True,
        column_mode='single',
        nondim=True,
    )

    # Lagrangian (mass) coordinate versions
    # Final multi-time bounds comparison in Lagrangian coordinates
    plot_multi_time_bounds_comparison(
        upper_matched, lower_matched, pele_snaps,
        upper_ts, lower_ts,
        str(output_dir / f"multi_time_bounds_comparison_single_{n_times}_lagrangian.png"),
        shift_to_upper=False,
        column_mode='single',
        time_indices=list(range(n_times)),
        lagrangian=True,
    )

    # Percent difference in Lagrangian (mass) coordinates
    plot_accuracy(
        upper_matched, lower_matched, pele_snaps,
        str(output_dir / "accuracy_percent_single_lagrangian.png"),
        percent=True,
        column_mode='single',
        lagrangian=True,
    )

    # 4. X-T diagrams - all column modes (load data once, use for all modes)
    print("\nGenerating X-T diagrams...")
    print("  Loading X-T data (once for all column modes)...")
    upper_xt_data = load_xt_data(upper_results_dir)
    lower_xt_data = load_xt_data(lower_results_dir)

    if upper_xt_data is not None and lower_xt_data is not None:
        for col_mode in ['single', 'double', 'presentation']:
            try:
                plot_xt_diagrams(upper_xt_data, lower_xt_data, output_dir, column_mode=col_mode)
            except Exception as e:
                print(f"  Warning: Could not generate X-T diagrams ({col_mode}): {e}")
    else:
        print("  Warning: Could not load X-T data, skipping X-T diagrams")

    print("\n" + "=" * 70)
    print("COMPARISON COMPLETE")
    print("=" * 70)
    print(f"Results saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare 1D solver bounds with PeleC data"
    )
    parser.add_argument("--upper-dir", type=str, required=True,
                        help="Upper bound results directory (e.g., results/upper_bound). "
                             "For single-case comparison, this is the only results dir needed.")
    parser.add_argument("--lower-dir", type=str, default=None,
                        help="Lower bound results directory (e.g., results/lower_bound). "
                             "Optional: if not provided, uses --upper-dir for single-case comparison.")
    parser.add_argument("--pltfile-dir", type=str, required=True,
                        help="Directory containing Part-*/ pltfile subdirs")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory")
    parser.add_argument("--upper-label", type=str, default="Flame Velocity (No Porous)",
                        help="Label for upper bound (or single case)")
    parser.add_argument("--lower-label", type=str, default=None,
                        help="Label for lower bound. If not provided, uses upper-label for single-case mode.")
    parser.add_argument("--extract-location", type=float, default=0.0445,
                        help="Y-coordinate for PeleC ray extraction [m]")
    parser.add_argument("--shift-to-upper", action="store_true",
                        help="Shift lower bound to align piston position with upper bound")

    args = parser.parse_args()

    # Single-case mode: use upper-dir for both if lower-dir not provided
    lower_dir = args.lower_dir if args.lower_dir else args.upper_dir
    lower_label = args.lower_label if args.lower_label else args.upper_label

    compare_with_bounds(
        upper_results_dir=Path(args.upper_dir),
        lower_results_dir=Path(lower_dir),
        pltfile_dir=Path(args.pltfile_dir),
        output_dir=Path(args.output_dir),
        upper_label=args.upper_label,
        lower_label=lower_label,
        extract_location=args.extract_location,
        shift_to_upper=args.shift_to_upper,
    )


if __name__ == "__main__":
    main()
