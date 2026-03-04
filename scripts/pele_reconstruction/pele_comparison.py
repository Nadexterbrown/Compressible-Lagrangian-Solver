"""
Compare 1D Lagrangian solver results with PeleC pltfile snapshots.

Supports loading multiple simulation datasets (e.g., solid and porous)
as upper/lower bounds with shaded regions between them.

Loads simulation snapshots from disk and finds nearest time matches
to PeleC pltfile data for comparison plots.

Reference: LGDCS/scripts/pele_sim/pele_bds/piston_pele_bnds.py
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from scipy.interpolate import interp1d

from pele_pltfile_loader import load_all_pltfiles, PeleSnapshot


# Publication-quality plot settings
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman']
mpl.rcParams['mathtext.fontset'] = 'stix'

FONT_SIZE_TITLE = 16
FONT_SIZE_LABEL = 14
FONT_SIZE_TICK = 12
FONT_SIZE_LEGEND = 11
LINE_WIDTH = 1.5
PLOT_DPI = 150

# Colors for different snapshot times (matching the example figure)
TIME_COLORS = ['#000000', '#E41A1C', '#377EB8', '#4DAF4A', '#984EA3',
               '#FF7F00', '#A65628', '#F781BF']


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
    path: Path


def scan_snapshots(snapshot_dir: Path) -> List[Tuple[float, int, Path]]:
    """
    Scan snapshot directory and return list of (time, index, path).

    Only loads time from each file for efficiency.
    """
    snapshots = []
    snapshot_files = sorted(snapshot_dir.glob("snapshot_*.npz"))

    for f in snapshot_files:
        try:
            data = np.load(f)
            t = float(data['t'])
            step = int(data['step']) if 'step' in data else int(f.stem.split('_')[1])
            snapshots.append((t, step, f))
        except Exception as e:
            print(f"  Warning: Could not read {f.name}: {e}")

    return snapshots


def find_nearest_snapshot(
    snapshots: List[Tuple[float, int, Path]],
    target_time: float
) -> Tuple[float, int, Path]:
    """Find snapshot with time nearest to target_time."""
    times = np.array([s[0] for s in snapshots])
    idx = np.argmin(np.abs(times - target_time))
    return snapshots[idx]


def load_snapshot(snapshot_path: Path) -> SimSnapshot:
    """Load a single snapshot .npz file."""
    data = np.load(snapshot_path)
    return SimSnapshot(
        time=float(data['t']),
        step=int(data['step']) if 'step' in data else 0,
        x_centers=data['x_centers'],
        u=data['u'],
        p=data['p'],
        rho=data['rho'],
        T=data['T'],
        path=snapshot_path,
    )


def mask_to_domain(
    x_pele: np.ndarray,
    pele_data: np.ndarray,
    x_min: float,
    x_max: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Mask PeleC data to simulation domain bounds."""
    mask = (x_pele >= x_min) & (x_pele <= x_max)
    return x_pele[mask], pele_data[mask]


def plot_single_comparison(
    sim_snapshot: SimSnapshot,
    pele_snapshot: PeleSnapshot,
    output_file: str,
):
    """
    Create comparison plot for a single time snapshot.

    1x3 subplot showing pressure, velocity, density.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Domain bounds from simulation
    x_min = sim_snapshot.x_centers.min()
    x_max = sim_snapshot.x_centers.max()

    # Convert to cm for plotting
    x_sim_cm = sim_snapshot.x_centers * 100
    x_pele_cm = pele_snapshot.x * 100

    # Mask PeleC data to simulation domain
    pele_mask = (pele_snapshot.x >= x_min) & (pele_snapshot.x <= x_max)
    x_pele_masked = x_pele_cm[pele_mask]

    # Time labels
    sim_label = f'1D Solver (t={sim_snapshot.time*1e6:.1f} µs)'
    pele_label = f'PeleC (t={pele_snapshot.time*1e6:.1f} µs)'

    # Pressure
    ax = axes[0]
    ax.plot(x_sim_cm, sim_snapshot.p / 1e6, 'b-', lw=LINE_WIDTH, label=sim_label)
    ax.plot(x_pele_masked, pele_snapshot.p[pele_mask] / 1e6, 'r--', lw=LINE_WIDTH, label=pele_label)
    ax.set_xlabel('x [cm]', fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel('Pressure [MPa]', fontsize=FONT_SIZE_LABEL)
    ax.set_title('Pressure', fontsize=FONT_SIZE_TITLE)
    ax.legend(fontsize=FONT_SIZE_LEGEND-2)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=FONT_SIZE_TICK)

    # Velocity
    ax = axes[1]
    ax.plot(x_sim_cm, sim_snapshot.u, 'b-', lw=LINE_WIDTH, label=sim_label)
    ax.plot(x_pele_masked, pele_snapshot.u[pele_mask], 'r--', lw=LINE_WIDTH, label=pele_label)
    ax.set_xlabel('x [cm]', fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel('Velocity [m/s]', fontsize=FONT_SIZE_LABEL)
    ax.set_title('Velocity', fontsize=FONT_SIZE_TITLE)
    ax.legend(fontsize=FONT_SIZE_LEGEND-2)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=FONT_SIZE_TICK)

    # Density
    ax = axes[2]
    ax.plot(x_sim_cm, sim_snapshot.rho, 'b-', lw=LINE_WIDTH, label=sim_label)
    ax.plot(x_pele_masked, pele_snapshot.rho[pele_mask], 'r--', lw=LINE_WIDTH, label=pele_label)
    ax.set_xlabel('x [cm]', fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel('Density [kg/m³]', fontsize=FONT_SIZE_LABEL)
    ax.set_title('Density', fontsize=FONT_SIZE_TITLE)
    ax.legend(fontsize=FONT_SIZE_LEGEND-2)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=FONT_SIZE_TICK)

    plt.suptitle(f'{pele_snapshot.part_name}/{pele_snapshot.pltfile_name}',
                 fontsize=FONT_SIZE_TITLE, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_file, dpi=PLOT_DPI, bbox_inches='tight')
    plt.close(fig)


def plot_multi_time_comparison(
    matched_pairs: List[Tuple[SimSnapshot, PeleSnapshot]],
    output_file: str,
    piston_history: Optional[Dict] = None,
):
    """
    Create publication-quality multi-time comparison plot.

    3x2 grid:
    - Left column: velocity, temperature, pressure profiles
    - Right column (middle row): piston velocity vs time

    Parameters
    ----------
    matched_pairs : list
        List of (SimSnapshot, PeleSnapshot) tuples
    output_file : str
        Output file path
    piston_history : dict, optional
        Contains 'times' and 'piston_velocity' arrays
    """
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(3, 2, figsize=(14, 12),
                              gridspec_kw={'width_ratios': [2, 1], 'wspace': 0.25, 'hspace': 0.3})

    n_times = len(matched_pairs)

    # Get global x-range
    x_min_global = min(sim.x_centers.min() for sim, _ in matched_pairs)
    x_max_global = max(sim.x_centers.max() for sim, _ in matched_pairs)
    x_range = x_max_global - x_min_global
    x_lim = (x_min_global * 100, (x_max_global + 0.2 * x_range) * 100)

    # Variables: (row, sim_attr, pele_attr, ylabel, scale)
    variables = [
        (0, 'u', 'u', 'Velocity [m/s]', 1.0),
        (1, 'T', 'T', 'Temperature [K]', 1.0),
        (2, 'p', 'p', 'Pressure [MPa]', 1e-6),
    ]

    for row, sim_attr, pele_attr, ylabel, scale in variables:
        ax = axes[row, 0]

        for i, (sim, pele) in enumerate(matched_pairs):
            color = TIME_COLORS[i % len(TIME_COLORS)]

            # Simulation (solid line)
            x_sim = sim.x_centers * 100
            y_sim = getattr(sim, sim_attr) * scale
            ax.plot(x_sim, y_sim, color=color, linestyle='-', lw=LINE_WIDTH)

            # PeleC (dashed line, masked to sim domain)
            x_min_sim = sim.x_centers.min()
            x_max_sim = sim.x_centers.max()
            pele_mask = (pele.x >= x_min_sim) & (pele.x <= x_max_sim)

            x_pele = pele.x[pele_mask] * 100
            y_pele_data = getattr(pele, pele_attr)
            if y_pele_data is not None:
                y_pele = y_pele_data[pele_mask] * scale
                ax.plot(x_pele, y_pele, color=color, linestyle='--', lw=LINE_WIDTH, alpha=0.8)

        ax.set_ylabel(ylabel, fontsize=FONT_SIZE_LABEL)
        ax.set_xlim(x_lim)
        ax.tick_params(labelsize=FONT_SIZE_TICK)
        ax.grid(True, alpha=0.3)

        if row == 2:
            ax.set_xlabel('Position [cm]', fontsize=FONT_SIZE_LABEL)

        # Legend in first row
        if row == 0:
            legend_elements = [
                Line2D([0], [0], color='gray', linestyle='-', lw=LINE_WIDTH, label='1D Solver'),
                Line2D([0], [0], color='gray', linestyle='--', lw=LINE_WIDTH, label='PeleC'),
                Line2D([0], [0], color='none', label=''),
            ]
            for i, (sim, pele) in enumerate(matched_pairs):
                color = TIME_COLORS[i % len(TIME_COLORS)]
                t_ms = pele.time * 1e3
                legend_elements.append(
                    Line2D([0], [0], color=color, linestyle='-', lw=LINE_WIDTH,
                           label=f't = {t_ms:.2f} ms')
                )
            ax.legend(handles=legend_elements, loc='upper right',
                     fontsize=FONT_SIZE_LEGEND, framealpha=0.9)

    # Right column: piston velocity (middle row only)
    for row in [0, 2]:
        axes[row, 1].axis('off')

    ax_piston = axes[1, 1]
    if piston_history is not None:
        t_hist = piston_history['times'] * 1e3  # ms
        v_hist = piston_history['piston_velocity']
        ax_piston.plot(t_hist, v_hist, 'k-', lw=LINE_WIDTH)

        # Mark snapshot times
        for i, (sim, pele) in enumerate(matched_pairs):
            color = TIME_COLORS[i % len(TIME_COLORS)]
            t_ms = pele.time * 1e3

            # Find nearest piston velocity
            idx = np.argmin(np.abs(piston_history['times'] - pele.time))
            v_at_t = v_hist[idx]

            ax_piston.plot(t_ms, v_at_t, 'o', color=color, markersize=10,
                          markeredgecolor='black', markeredgewidth=1.5)

        ax_piston.set_xlabel('Time [ms]', fontsize=FONT_SIZE_LABEL)
        ax_piston.set_ylabel('Piston Velocity [m/s]', fontsize=FONT_SIZE_LABEL)
        ax_piston.tick_params(labelsize=FONT_SIZE_TICK)
        ax_piston.grid(True, alpha=0.3)
        ax_piston.set_xlim(0, t_hist[-1] * 1.05)
    else:
        ax_piston.axis('off')

    plt.suptitle('1D Solver vs PeleC Comparison', fontsize=FONT_SIZE_TITLE + 2, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_file, dpi=PLOT_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved multi-time comparison: {output_file}")


def interpolate_to_common_grid(x1, y1, x2, y2):
    """Interpolate two datasets to a common x grid for fill_between."""
    x_min = max(x1.min(), x2.min())
    x_max = min(x1.max(), x2.max())

    # Create common grid
    n_points = max(len(x1), len(x2))
    x_common = np.linspace(x_min, x_max, n_points)

    # Interpolate both to common grid
    interp1 = interp1d(x1, y1, kind='linear', bounds_error=False, fill_value='extrapolate')
    interp2 = interp1d(x2, y2, kind='linear', bounds_error=False, fill_value='extrapolate')

    y1_common = interp1(x_common)
    y2_common = interp2(x_common)

    return x_common, y1_common, y2_common


def plot_bounds_comparison(
    matched_data: List[Dict],
    pele_snapshots: List[PeleSnapshot],
    output_file: str,
    piston_histories: Optional[List[Dict]] = None,
    dataset_labels: Optional[List[str]] = None,
    align_piston_positions: bool = False,
):
    """
    Create bounds comparison plot with multiple simulation datasets.

    Shows upper/lower bounds from different simulations with shaded region,
    overlaid with PeleC dashed lines.

    Parameters
    ----------
    matched_data : list of dict
        Each dict contains:
        - 'snapshots': list of SimSnapshot for each PeleC time
        - 'label': dataset label (e.g., 'solid', 'porous')
    pele_snapshots : list of PeleSnapshot
        PeleC data for each time
    output_file : str
        Output file path
    piston_histories : list of dict, optional
        Piston velocity history for each dataset
    dataset_labels : list of str, optional
        Labels for legend (e.g., ['S_f', 'S_f - CJ_def'])
    align_piston_positions : bool
        If True, shift datasets so all piston positions align with the
        maximum piston position (rightmost) for visual comparison.
    """
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    n_datasets = len(matched_data)
    n_times = len(pele_snapshots)

    if dataset_labels is None:
        dataset_labels = [d.get('label', f'Dataset {i+1}') for i, d in enumerate(matched_data)]

    # Compute position shifts for each dataset if aligning piston positions
    # Each dataset gets its own shift array (one shift per time snapshot)
    position_shifts = []
    if align_piston_positions and n_datasets >= 2:
        # For each time snapshot, find the max piston position across datasets
        for i in range(n_times):
            piston_positions = [data['snapshots'][i].x_centers.min() for data in matched_data]
            max_piston = max(piston_positions)
            shifts = [max_piston - pos for pos in piston_positions]
            position_shifts.append(shifts)
        print(f"  Position alignment enabled:")
        for j, data in enumerate(matched_data):
            avg_shift = np.mean([position_shifts[i][j] for i in range(n_times)])
            print(f"    Dataset '{dataset_labels[j]}': avg shift = {avg_shift*100:.2f} cm")
    else:
        # No shift
        for i in range(n_times):
            position_shifts.append([0.0] * n_datasets)

    # Create figure: 3 rows x 2 columns
    fig, axes = plt.subplots(3, 2, figsize=(14, 10),
                              gridspec_kw={'width_ratios': [2, 1], 'wspace': 0.25, 'hspace': 0.25})

    # Get global x-range from all datasets (with shifts applied)
    x_min_global = float('inf')
    x_max_global = float('-inf')
    for i, pele in enumerate(pele_snapshots):
        for j, data in enumerate(matched_data):
            snap = data['snapshots'][i]
            shift = position_shifts[i][j]
            x_min_global = min(x_min_global, snap.x_centers.min() + shift)
            x_max_global = max(x_max_global, snap.x_centers.max() + shift)

    x_lim = (0, x_max_global * 100 + 50)  # Extra space for legend

    # Variables: (row, attr, ylabel, scale)
    variables = [
        (0, 'u', 'Velocity [m/s]', 1.0),
        (1, 'T', 'Temperature [K]', 1.0),
        (2, 'p', 'Pressure [MPa]', 1e-6),
    ]

    for row, attr, ylabel, scale in variables:
        ax = axes[row, 0]

        # Plot initial state (black line at t=0)
        if row == 0:  # velocity
            ax.axhline(0, color='black', lw=LINE_WIDTH, zorder=1)
        elif row == 1:  # temperature
            ax.axhline(503, color='black', lw=LINE_WIDTH, zorder=1)
        elif row == 2:  # pressure
            ax.axhline(1.0, color='black', lw=LINE_WIDTH, zorder=1)

        for i, pele in enumerate(pele_snapshots):
            color = TIME_COLORS[(i + 1) % len(TIME_COLORS)]  # Skip black for t=0

            # Get simulation data from all datasets at this time
            sim_data_list = [data['snapshots'][i] for data in matched_data]
            shifts = position_shifts[i]

            # Get common x range for this time (with shifts applied)
            x_min_time = max(s.x_centers.min() + shifts[j] for j, s in enumerate(sim_data_list))
            x_max_time = min(s.x_centers.max() + shifts[j] for j, s in enumerate(sim_data_list))

            # Interpolate all datasets to common grid
            n_points = 500
            x_common = np.linspace(x_min_time, x_max_time, n_points)

            y_all = []
            for j, sim in enumerate(sim_data_list):
                shift = shifts[j]
                y_data = getattr(sim, attr) * scale
                # Interpolate using shifted x coordinates
                interp_func = interp1d(sim.x_centers + shift, y_data, kind='linear',
                                       bounds_error=False, fill_value='extrapolate')
                y_all.append(interp_func(x_common))

            y_all = np.array(y_all)
            y_min = np.min(y_all, axis=0)
            y_max = np.max(y_all, axis=0)

            x_cm = x_common * 100

            # Plot shaded region between bounds
            ax.fill_between(x_cm, y_min, y_max, color=color, alpha=0.3, zorder=2)

            # Plot bound lines
            ax.plot(x_cm, y_min, color=color, lw=LINE_WIDTH, linestyle='-', zorder=3)
            ax.plot(x_cm, y_max, color=color, lw=LINE_WIDTH, linestyle='-', zorder=3)

            # Plot vertical dashed line at piston position (use first dataset with shift)
            piston_x = (sim_data_list[0].x_centers.min() + shifts[0]) * 100
            ax.axvline(piston_x, color=color, lw=1, linestyle='--', alpha=0.7, zorder=1)

            # Plot PeleC (dashed line)
            pele_attr = attr
            pele_data = getattr(pele, pele_attr)
            if pele_data is not None:
                pele_mask = (pele.x >= x_min_time) & (pele.x <= x_max_time)
                x_pele = pele.x[pele_mask] * 100
                y_pele = pele_data[pele_mask] * scale
                ax.plot(x_pele, y_pele, color=color, lw=LINE_WIDTH, linestyle='--', zorder=4)

        ax.set_ylabel(ylabel, fontsize=FONT_SIZE_LABEL)
        ax.set_xlim(x_lim)
        ax.tick_params(labelsize=FONT_SIZE_TICK)

        if row == 2:
            ax.set_xlabel('Position [cm]', fontsize=FONT_SIZE_LABEL)

        # Legend in first row
        if row == 0:
            legend_elements = [
                Line2D([0], [0], color='gray', linestyle='-', lw=LINE_WIDTH, label='1D Solver'),
                Line2D([0], [0], color='gray', linestyle='--', lw=LINE_WIDTH, label='Pele'),
            ]
            # Add time colors
            legend_elements.append(Line2D([0], [0], color='none', label=''))
            legend_elements.append(
                Line2D([0], [0], color='black', linestyle='-', lw=LINE_WIDTH, label='t = 0.00 ms')
            )
            for i, pele in enumerate(pele_snapshots):
                color = TIME_COLORS[(i + 1) % len(TIME_COLORS)]
                t_ms = pele.time * 1e3
                legend_elements.append(
                    Line2D([0], [0], color=color, linestyle='-', lw=LINE_WIDTH,
                           label=f't = {t_ms:.2f} ms')
                )
            ax.legend(handles=legend_elements, loc='upper right',
                     fontsize=FONT_SIZE_LEGEND, framealpha=0.9)

    # Right column: piston velocity with bounds (middle row only)
    for row in [0, 2]:
        axes[row, 1].axis('off')

    ax_piston = axes[1, 1]

    if piston_histories and len(piston_histories) >= 2:
        # Get common time range
        t_min = max(ph['times'].min() for ph in piston_histories)
        t_max = min(ph['times'].max() for ph in piston_histories)

        n_points = 1000
        t_common = np.linspace(t_min, t_max, n_points)

        v_all = []
        for ph in piston_histories:
            interp_func = interp1d(ph['times'], ph['piston_velocity'], kind='linear',
                                   bounds_error=False, fill_value='extrapolate')
            v_all.append(interp_func(t_common))

        v_all = np.array(v_all)
        v_min = np.min(v_all, axis=0)
        v_max = np.max(v_all, axis=0)

        t_ms = t_common * 1e3

        # Plot shaded region
        ax_piston.fill_between(t_ms, v_min, v_max, color='gray', alpha=0.3)

        # Plot individual curves
        for j, (ph, label) in enumerate(zip(piston_histories, dataset_labels)):
            style = '-' if j == 0 else '--'
            ax_piston.plot(ph['times'] * 1e3, ph['piston_velocity'], 'k' + style, lw=LINE_WIDTH,
                          label=label)

        # Mark snapshot times with markers
        marker_styles = ['o', 's']  # circle, square
        for i, pele in enumerate(pele_snapshots):
            color = TIME_COLORS[(i + 1) % len(TIME_COLORS)]
            t_snap_ms = pele.time * 1e3

            for j, ph in enumerate(piston_histories):
                idx = np.argmin(np.abs(ph['times'] - pele.time))
                v_at_t = ph['piston_velocity'][idx]
                ax_piston.plot(t_snap_ms, v_at_t, marker_styles[j % 2], color=color,
                              markersize=8, markeredgecolor='black', markeredgewidth=1, zorder=5)

        ax_piston.set_xlabel('Time [ms]', fontsize=FONT_SIZE_LABEL)
        ax_piston.set_ylabel('Piston Velocity [m/s]', fontsize=FONT_SIZE_LABEL)
        ax_piston.tick_params(labelsize=FONT_SIZE_TICK)
        ax_piston.set_xlim(0, t_max * 1e3 * 1.02)
        ax_piston.legend(fontsize=FONT_SIZE_LEGEND, loc='lower right')

    elif piston_histories and len(piston_histories) == 1:
        # Single dataset - no bounds
        ph = piston_histories[0]
        ax_piston.plot(ph['times'] * 1e3, ph['piston_velocity'], 'k-', lw=LINE_WIDTH)
        ax_piston.set_xlabel('Time [ms]', fontsize=FONT_SIZE_LABEL)
        ax_piston.set_ylabel('Piston Velocity [m/s]', fontsize=FONT_SIZE_LABEL)
        ax_piston.tick_params(labelsize=FONT_SIZE_TICK)
    else:
        ax_piston.axis('off')

    plt.tight_layout()
    plt.savefig(output_file, dpi=PLOT_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved bounds comparison: {output_file}")


def compare_with_pele_bounds(
    snapshot_dirs: List[Path],
    pltfile_dir: Path,
    output_dir: Path,
    extract_location: float = 0.0445,
    dataset_labels: Optional[List[str]] = None,
    align_piston_positions: bool = False,
):
    """
    Compare multiple simulation datasets with PeleC as bounds.

    Parameters
    ----------
    snapshot_dirs : list of Path
        Directories containing simulation snapshots (e.g., [solid/snapshots, porous/snapshots])
    pltfile_dir : Path
        Directory containing Part-1/, Part-2/, Part-3/ pltfile subdirs
    output_dir : Path
        Output directory for comparison plots
    extract_location : float
        Y-coordinate for PeleC ray extraction [m]
    dataset_labels : list of str, optional
        Labels for datasets (e.g., ['S_f', 'S_f - CJ_def'])
    align_piston_positions : bool
        If True, shift datasets so piston positions align (for visual comparison)
    """
    print("=" * 70)
    print("PELE BOUNDS COMPARISON")
    print("=" * 70)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Scan all simulation snapshot directories
    all_snapshots = []
    for snap_dir in snapshot_dirs:
        print(f"\nScanning snapshots: {snap_dir}")
        snapshots = scan_snapshots(snap_dir)
        if not snapshots:
            print(f"  WARNING: No snapshots found in {snap_dir}")
            continue
        print(f"  Found {len(snapshots)} snapshots")
        print(f"  Time range: [{snapshots[0][0]*1e6:.1f}, {snapshots[-1][0]*1e6:.1f}] µs")
        all_snapshots.append({'dir': snap_dir, 'snapshots': snapshots})

    if len(all_snapshots) < 1:
        print("ERROR: No valid snapshot directories found!")
        return

    # Step 2: Load PeleC pltfiles
    print(f"\nLoading PeleC pltfiles: {pltfile_dir}")
    pele_snapshots = load_all_pltfiles(str(pltfile_dir), extract_location)
    if not pele_snapshots:
        print("ERROR: No PeleC pltfiles found!")
        return
    print(f"  Loaded {len(pele_snapshots)} PeleC snapshots")

    # Step 3: Match times and load snapshots for all datasets
    print("\nMatching snapshot times...")
    matched_data = []
    piston_histories = []

    for data_info in all_snapshots:
        snap_dir = data_info['dir']
        snapshots = data_info['snapshots']
        label = snap_dir.parent.name  # e.g., 'solid' or 'porous'

        matched_snaps = []
        for pele in pele_snapshots:
            nearest_t, nearest_step, nearest_path = find_nearest_snapshot(snapshots, pele.time)
            sim = load_snapshot(nearest_path)
            matched_snaps.append(sim)

            time_diff = abs(sim.time - pele.time) * 1e6
            print(f"  [{label}] PeleC t={pele.time*1e6:.1f} µs -> Sim t={sim.time*1e6:.1f} µs (Δ={time_diff:.1f} µs)")

        matched_data.append({'snapshots': matched_snaps, 'label': label})

        # Load piston history
        results_dir = snap_dir.parent
        piston_history = None
        piston_file = results_dir / "piston_history.npz"
        if piston_file.exists():
            ph = np.load(piston_file)
            piston_history = {'times': ph['times'], 'piston_velocity': ph['piston_velocity']}
        else:
            timeseries_file = results_dir / "timeseries.npz"
            if timeseries_file.exists():
                ts = np.load(timeseries_file, allow_pickle=True)
                piston_history = {'times': ts['t'], 'piston_velocity': ts['u_piston']}

        if piston_history is not None:
            piston_histories.append(piston_history)

    # Step 4: Generate bounds comparison plot
    print("\nGenerating bounds comparison plot...")
    output_file = output_dir / "bounds_comparison.png"
    plot_bounds_comparison(
        matched_data,
        pele_snapshots,
        str(output_file),
        piston_histories=piston_histories if piston_histories else None,
        dataset_labels=dataset_labels,
        align_piston_positions=align_piston_positions,
    )

    print("\n" + "=" * 70)
    print("BOUNDS COMPARISON COMPLETE")
    print("=" * 70)
    print(f"Results saved to: {output_dir}")


def compare_with_pele(
    snapshot_dir: Path,
    pltfile_dir: Path,
    output_dir: Path,
    extract_location: float = 0.0445,
):
    """
    Main comparison workflow (single dataset).

    Parameters
    ----------
    snapshot_dir : Path
        Directory containing simulation snapshots
    pltfile_dir : Path
        Directory containing Part-1/, Part-2/, Part-3/ pltfile subdirs
    output_dir : Path
        Output directory for comparison plots
    extract_location : float
        Y-coordinate for PeleC ray extraction [m]
    """
    print("=" * 70)
    print("PELE COMPARISON")
    print("=" * 70)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Scan simulation snapshots
    print(f"\nScanning snapshots: {snapshot_dir}")
    snapshots = scan_snapshots(snapshot_dir)
    if not snapshots:
        print("ERROR: No snapshots found!")
        return

    print(f"  Found {len(snapshots)} snapshots")
    print(f"  Time range: [{snapshots[0][0]*1e6:.1f}, {snapshots[-1][0]*1e6:.1f}] µs")

    # Step 2: Load PeleC pltfiles
    print(f"\nLoading PeleC pltfiles: {pltfile_dir}")
    pele_snapshots = load_all_pltfiles(str(pltfile_dir), extract_location)
    if not pele_snapshots:
        print("ERROR: No PeleC pltfiles found!")
        return

    print(f"  Loaded {len(pele_snapshots)} PeleC snapshots")

    # Step 3: Match times and load simulation snapshots
    print("\nMatching snapshot times...")
    matched_pairs = []

    for pele in pele_snapshots:
        nearest_t, nearest_step, nearest_path = find_nearest_snapshot(snapshots, pele.time)
        sim = load_snapshot(nearest_path)

        time_diff = abs(sim.time - pele.time) * 1e6
        print(f"  PeleC {pele.pltfile_name} (t={pele.time*1e6:.1f} µs) "
              f"-> Sim step {nearest_step} (t={sim.time*1e6:.1f} µs, Δt={time_diff:.1f} µs)")

        matched_pairs.append((sim, pele))

    # Step 4: Generate individual comparison plots
    print("\nGenerating individual comparison plots...")
    for sim, pele in matched_pairs:
        output_file = output_dir / f"comparison_{pele.part_name}_{pele.pltfile_name}.png"
        plot_single_comparison(sim, pele, str(output_file))
        print(f"  Saved: {output_file.name}")

    # Step 5: Load piston history for multi-time plot
    piston_history = None
    piston_file = output_dir / "piston_history.npz"
    if piston_file.exists():
        ph = np.load(piston_file)
        piston_history = {
            'times': ph['times'],
            'piston_velocity': ph['piston_velocity'],
        }
    else:
        # Try loading from timeseries
        timeseries_file = output_dir / "timeseries.npz"
        if timeseries_file.exists():
            ts = np.load(timeseries_file, allow_pickle=True)
            piston_history = {
                'times': ts['t'],
                'piston_velocity': ts['u_piston'],
            }

    # Step 6: Generate multi-time comparison plot
    print("\nGenerating multi-time comparison plot...")
    output_file = output_dir / "multi_time_comparison.png"
    plot_multi_time_comparison(matched_pairs, str(output_file), piston_history)

    print("\n" + "=" * 70)
    print("COMPARISON COMPLETE")
    print("=" * 70)
    print(f"Results saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare 1D solver results with PeleC pltfile snapshots"
    )
    parser.add_argument("--snapshot-dirs", type=str, nargs='+', required=True,
                        help="Directory(s) containing simulation snapshots. "
                             "Use multiple for bounds comparison (e.g., solid/snapshots porous/snapshots)")
    parser.add_argument("--pltfile-dir", type=str, required=True,
                        help="Directory containing Part-*/ pltfile subdirs")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for comparison plots")
    parser.add_argument("--extract-location", type=float, default=0.0445,
                        help="Y-coordinate for PeleC ray extraction [m]")
    parser.add_argument("--labels", type=str, nargs='+', default=None,
                        help="Labels for datasets (e.g., 'S_f' 'S_f-CJ_def')")
    parser.add_argument("--align-piston", action="store_true",
                        help="Shift datasets so piston positions align (for visual comparison)")

    args = parser.parse_args()

    snapshot_dirs = [Path(d) for d in args.snapshot_dirs]

    if len(snapshot_dirs) == 1:
        # Single dataset comparison
        compare_with_pele(
            snapshot_dir=snapshot_dirs[0],
            pltfile_dir=Path(args.pltfile_dir),
            output_dir=Path(args.output_dir),
            extract_location=args.extract_location,
        )
    else:
        # Multiple datasets - bounds comparison
        compare_with_pele_bounds(
            snapshot_dirs=snapshot_dirs,
            pltfile_dir=Path(args.pltfile_dir),
            output_dir=Path(args.output_dir),
            extract_location=args.extract_location,
            dataset_labels=args.labels,
            align_piston_positions=args.align_piston,
        )


if __name__ == "__main__":
    main()
