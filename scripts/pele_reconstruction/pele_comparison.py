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


# Publication-quality plot settings
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman']
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['axes.linewidth'] = 1.2
mpl.rcParams['xtick.major.width'] = 1.0
mpl.rcParams['ytick.major.width'] = 1.0

FONT_SIZE_TITLE = 16
FONT_SIZE_LABEL = 14
FONT_SIZE_TICK = 12
FONT_SIZE_LEGEND = 11
LINE_WIDTH = 1.5
PLOT_DPI = 150

# Colors for different snapshot times (matching reference figures)
TIME_COLORS = ['black', '#E41A1C', '#377EB8', '#4DAF4A', '#984EA3']

# Line styles
STYLE_UPPER = '-'   # Solid for upper bound (S_f)
STYLE_LOWER = '--'  # Dashed for lower bound (S_f - CJ)
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


def scan_snapshots(snapshot_dir: Path) -> List[Dict]:
    """Scan snapshot directory and return list of snapshot info."""
    snapshots = []
    snapshot_files = sorted(snapshot_dir.glob("snapshot_*.npz"))

    for f in snapshot_files:
        try:
            data = np.load(f)
            t = float(data['t'])
            step = int(data['step']) if 'step' in data else int(f.stem.split('_')[1])
            snapshots.append({'time': t, 'step': step, 'path': f})
        except Exception as e:
            print(f"  Warning: Could not read {f.name}: {e}")

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
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

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
                         alpha=0.3, color='lightblue', label='Simulation bounds')
    axes[0].plot(x_upper, upper_snap.p / 1e6, STYLE_UPPER, color='blue', lw=LINE_WIDTH,
                 label=f'{upper_snap.label} (t={time_s:.6f}s)')
    axes[0].plot(x_lower, lower_snap.p / 1e6, STYLE_LOWER, color='red', lw=LINE_WIDTH,
                 label=f'{lower_snap.label} (t={lower_snap.time:.6f}s)')

    # Velocity
    u_upper = interp_safe(x_upper, upper_snap.u, x_common)
    u_lower = interp_safe(x_lower, lower_snap.u, x_common)

    axes[1].fill_between(x_common, u_lower, u_upper,
                         alpha=0.3, color='lightblue', label='Simulation bounds')
    axes[1].plot(x_upper, upper_snap.u, STYLE_UPPER, color='blue', lw=LINE_WIDTH,
                 label=f'{upper_snap.label} (t={time_s:.6f}s)')
    axes[1].plot(x_lower, lower_snap.u, STYLE_LOWER, color='red', lw=LINE_WIDTH,
                 label=f'{lower_snap.label} (t={lower_snap.time:.6f}s)')

    # Density
    rho_upper = interp_safe(x_upper, upper_snap.rho, x_common)
    rho_lower = interp_safe(x_lower, lower_snap.rho, x_common)

    axes[2].fill_between(x_common, rho_lower, rho_upper,
                         alpha=0.3, color='lightblue', label='Simulation bounds')
    axes[2].plot(x_upper, upper_snap.rho, STYLE_UPPER, color='blue', lw=LINE_WIDTH,
                 label=f'{upper_snap.label} (t={time_s:.6f}s)')
    axes[2].plot(x_lower, lower_snap.rho, STYLE_LOWER, color='red', lw=LINE_WIDTH,
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

    # Plot PeleC if available (filter points behind flame position)
    if pele_snap is not None:
        # Filter out points behind the flame (use upper bound piston position as reference)
        flame_x = x_upper.min()
        pele_mask = pele_snap.x >= flame_x
        x_pele = pele_snap.x[pele_mask]
        p_pele = pele_snap.p[pele_mask]
        u_pele = pele_snap.u[pele_mask]
        rho_pele = pele_snap.rho[pele_mask]

        if len(x_pele) > 0:
            axes[0].plot(x_pele, p_pele / 1e6, 'k-', lw=LINE_WIDTH + 0.5,
                         label=f'Pele (t={pele_snap.time:.6f}s)')
            axes[1].plot(x_pele, u_pele, 'k-', lw=LINE_WIDTH + 0.5,
                         label=f'Pele (t={pele_snap.time:.6f}s)')
            axes[2].plot(x_pele, rho_pele, 'k-', lw=LINE_WIDTH + 0.5,
                         label=f'Pele (t={pele_snap.time:.6f}s)')

    # Labels and formatting
    titles = ['Pressure', 'Velocity', 'Density']
    ylabels = ['Pressure [Pa]', 'Velocity [m/s]', 'Density [kg/m³]']

    for i, ax in enumerate(axes):
        ax.set_xlabel('Position [m]', fontsize=FONT_SIZE_LABEL)
        ax.set_ylabel(ylabels[i], fontsize=FONT_SIZE_LABEL)
        ax.set_title(titles[i], fontsize=FONT_SIZE_TITLE)
        ax.legend(fontsize=FONT_SIZE_LEGEND - 2, loc='best')
        ax.tick_params(labelsize=FONT_SIZE_TICK)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(ylims[i])  # Set y-limits from simulation data

    # Get pltfile name from PeleC snapshot if available
    if pele_snap is not None:
        plt_name = f"{pele_snap.part_name}_{pele_snap.pltfile_name}" if pele_snap.part_name else pele_snap.pltfile_name
        fig.suptitle(f'Simulation Bounds vs PeleC - {plt_name}', fontsize=FONT_SIZE_TITLE + 2, fontweight='bold')
    else:
        fig.suptitle(f'Simulation Bounds Comparison - t = {time_ms:.3f} ms', fontsize=FONT_SIZE_TITLE + 2, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_file, dpi=PLOT_DPI, bbox_inches='tight')
    plt.close(fig)


def plot_piston_velocity_bounds(
    upper_ts: SimTimeseries,
    lower_ts: SimTimeseries,
    comparison_times: List[float],
    output_file: str,
):
    """
    Plot piston velocity vs time with shaded bounds.

    Shows S_f (upper bound) and S_f - CJ_def (lower bound) with markers
    at comparison times.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    t_upper_ms = upper_ts.t * 1e3
    t_lower_ms = lower_ts.t * 1e3

    # Create common time grid for fill_between
    t_min = max(upper_ts.t.min(), lower_ts.t.min())
    t_max = min(upper_ts.t.max(), lower_ts.t.max())
    t_common = np.linspace(t_min, t_max, 500)
    t_common_ms = t_common * 1e3

    u_upper = np.interp(t_common, upper_ts.t, upper_ts.u_piston)
    u_lower = np.interp(t_common, lower_ts.t, lower_ts.u_piston)

    # Shaded region
    ax.fill_between(t_common_ms, u_lower, u_upper, alpha=0.3, color='gray')

    # Bound lines
    ax.plot(t_upper_ms, upper_ts.u_piston, 'k-', lw=LINE_WIDTH, label=r'$S_f$')
    ax.plot(t_lower_ms, lower_ts.u_piston, 'k--', lw=LINE_WIDTH, label=r'$S_f - CJ_{def}$')

    # Markers at comparison times
    for i, t in enumerate(comparison_times):
        color = TIME_COLORS[i % len(TIME_COLORS)]

        # Upper bound marker (circle)
        u_upper_at_t = np.interp(t, upper_ts.t, upper_ts.u_piston)
        ax.plot(t * 1e3, u_upper_at_t, 'o', color=color, markersize=10, markeredgecolor='black', markeredgewidth=1)

        # Lower bound marker (square)
        u_lower_at_t = np.interp(t, lower_ts.t, lower_ts.u_piston)
        ax.plot(t * 1e3, u_lower_at_t, 's', color=color, markersize=9, markeredgecolor='black', markeredgewidth=1)

    ax.set_xlabel('Time [ms]', fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel('Piston Velocity [m/s]', fontsize=FONT_SIZE_LABEL)
    ax.legend(fontsize=FONT_SIZE_LEGEND, loc='lower right')
    ax.tick_params(labelsize=FONT_SIZE_TICK)
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
):
    """
    Plot gas velocity at piston face vs time with shaded bounds.

    Shows the actual gas velocity used in the Riemann solver at the boundary,
    which differs from piston velocity in porous simulations.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    t_upper_ms = upper_ts.t * 1e3
    t_lower_ms = lower_ts.t * 1e3

    # Create common time grid for fill_between
    t_min = max(upper_ts.t.min(), lower_ts.t.min())
    t_max = min(upper_ts.t.max(), lower_ts.t.max())
    t_common = np.linspace(t_min, t_max, 500)
    t_common_ms = t_common * 1e3

    u_upper = np.interp(t_common, upper_ts.t, upper_ts.u_gas)
    u_lower = np.interp(t_common, lower_ts.t, lower_ts.u_gas)

    # Shaded region
    ax.fill_between(t_common_ms, u_lower, u_upper, alpha=0.3, color='gray')

    # Bound lines
    ax.plot(t_upper_ms, upper_ts.u_gas, 'k-', lw=LINE_WIDTH, label=r'$u_{gas}$ (No Porous)')
    ax.plot(t_lower_ms, lower_ts.u_gas, 'k--', lw=LINE_WIDTH, label=r'$u_{gas}$ (Porous)')

    # Markers at comparison times
    for i, t in enumerate(comparison_times):
        color = TIME_COLORS[i % len(TIME_COLORS)]

        # Upper bound marker (circle)
        u_upper_at_t = np.interp(t, upper_ts.t, upper_ts.u_gas)
        ax.plot(t * 1e3, u_upper_at_t, 'o', color=color, markersize=10, markeredgecolor='black', markeredgewidth=1)

        # Lower bound marker (square)
        u_lower_at_t = np.interp(t, lower_ts.t, lower_ts.u_gas)
        ax.plot(t * 1e3, u_lower_at_t, 's', color=color, markersize=9, markeredgecolor='black', markeredgewidth=1)

    ax.set_xlabel('Time [ms]', fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel('Gas Velocity at Piston Face [m/s]', fontsize=FONT_SIZE_LABEL)
    ax.set_title('Gas Velocity at Boundary', fontsize=FONT_SIZE_TITLE)
    ax.legend(fontsize=FONT_SIZE_LEGEND, loc='lower right')
    ax.tick_params(labelsize=FONT_SIZE_TICK)
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
):
    """
    Plot multi-time comparison with velocity bounds panel.

    Layout: 3 rows (Velocity, Temperature, Pressure) + velocity panel on right.
    Matches the reference multi_time_bounds_comparison.png format.
    """
    fig = plt.figure(figsize=(14, 10))

    # Create grid: 3 rows x 2 columns, with right column spanning all rows
    gs = fig.add_gridspec(3, 2, width_ratios=[2, 1], hspace=0.1, wspace=0.25)

    ax_vel = fig.add_subplot(gs[0, 0])
    ax_temp = fig.add_subplot(gs[1, 0], sharex=ax_vel)
    ax_pres = fig.add_subplot(gs[2, 0], sharex=ax_vel)
    ax_piston = fig.add_subplot(gs[:, 1])

    profile_axes = [ax_vel, ax_temp, ax_pres]
    comparison_times = [snap.time for snap in pele_snaps]

    # Plot each time
    for t_idx, (upper, lower, pele) in enumerate(zip(upper_snaps, lower_snaps, pele_snaps)):
        color = TIME_COLORS[t_idx % len(TIME_COLORS)]
        time_ms = pele.time * 1e3

        x_upper = upper.x_centers * 100  # Convert to cm
        x_lower = lower.x_centers * 100

        # Shift lower bound if requested
        if shift_to_upper:
            shift = (upper.x_centers.min() - lower.x_centers.min()) * 100
            x_lower = x_lower + shift

        # Piston position vertical lines
        for ax in profile_axes:
            ax.axvline(x_upper.min(), color=color, linestyle='--', alpha=0.5, lw=1)

        # Create common x grid for fill_between
        x_min = min(x_upper.min(), x_lower.min())
        x_max = max(x_upper.max(), x_lower.max())
        x_common = np.linspace(x_min, x_max, 500)

        def interp_safe(x_orig, y_orig, x_new):
            return np.interp(x_new, x_orig, y_orig, left=y_orig[0], right=y_orig[-1])

        # Velocity
        u_upper = interp_safe(x_upper, upper.u, x_common)
        u_lower = interp_safe(x_lower, lower.u, x_common)
        ax_vel.fill_between(x_common, u_lower, u_upper, alpha=0.3, color=color)
        ax_vel.plot(x_upper, upper.u, '-', color=color, lw=LINE_WIDTH)
        ax_vel.plot(x_lower, lower.u, '--', color=color, lw=LINE_WIDTH)

        # Temperature
        T_upper = interp_safe(x_upper, upper.T, x_common)
        T_lower = interp_safe(x_lower, lower.T, x_common)
        ax_temp.fill_between(x_common, T_lower, T_upper, alpha=0.3, color=color)
        ax_temp.plot(x_upper, upper.T, '-', color=color, lw=LINE_WIDTH)
        ax_temp.plot(x_lower, lower.T, '--', color=color, lw=LINE_WIDTH)

        # Pressure
        p_upper = interp_safe(x_upper, upper.p / 1e6, x_common)
        p_lower = interp_safe(x_lower, lower.p / 1e6, x_common)
        ax_pres.fill_between(x_common, p_lower, p_upper, alpha=0.3, color=color)
        ax_pres.plot(x_upper, upper.p / 1e6, '-', color=color, lw=LINE_WIDTH)
        ax_pres.plot(x_lower, lower.p / 1e6, '--', color=color, lw=LINE_WIDTH)

        # PeleC data (filter points behind flame position)
        flame_x_cm = x_upper.min()  # Flame position in cm
        pele_mask = pele.x * 100 >= flame_x_cm
        x_pele = pele.x[pele_mask] * 100
        if len(x_pele) > 0:
            ax_vel.plot(x_pele, pele.u[pele_mask], ':', color=color, lw=LINE_WIDTH, alpha=0.8)
            if pele.T is not None:
                ax_temp.plot(x_pele, pele.T[pele_mask], ':', color=color, lw=LINE_WIDTH, alpha=0.8)
            ax_pres.plot(x_pele, pele.p[pele_mask] / 1e6, ':', color=color, lw=LINE_WIDTH, alpha=0.8)

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

    ax_piston.fill_between(t_common_ms, u_lower_interp, u_upper_interp, alpha=0.3, color='gray')
    ax_piston.plot(t_upper_ms, upper_ts.u_gas, 'k-', lw=LINE_WIDTH, label=r'$u_{gas}$ (No Porous)')
    ax_piston.plot(t_lower_ms, lower_ts.u_gas, 'k--', lw=LINE_WIDTH, label=r'$u_{gas}$ (Porous)')

    # Markers at comparison times
    for t_idx, t in enumerate(comparison_times):
        color = TIME_COLORS[t_idx % len(TIME_COLORS)]
        u_upper_at_t = np.interp(t, upper_ts.t, upper_ts.u_gas)
        u_lower_at_t = np.interp(t, lower_ts.t, lower_ts.u_gas)
        ax_piston.plot(t * 1e3, u_upper_at_t, 'o', color=color, markersize=8, markeredgecolor='black', markeredgewidth=1)
        ax_piston.plot(t * 1e3, u_lower_at_t, 's', color=color, markersize=7, markeredgecolor='black', markeredgewidth=1)

    # Labels
    ax_vel.set_ylabel('Velocity [m/s]', fontsize=FONT_SIZE_LABEL)
    ax_temp.set_ylabel('Temperature [K]', fontsize=FONT_SIZE_LABEL)
    ax_pres.set_ylabel('Pressure [MPa]', fontsize=FONT_SIZE_LABEL)
    ax_pres.set_xlabel('Position [cm]', fontsize=FONT_SIZE_LABEL)

    ax_piston.set_xlabel('Time [ms]', fontsize=FONT_SIZE_LABEL)
    ax_piston.set_ylabel('Gas Velocity at Piston Face [m/s]', fontsize=FONT_SIZE_LABEL)
    ax_piston.legend(fontsize=FONT_SIZE_LEGEND, loc='lower right')

    # Hide x labels for upper profile plots
    plt.setp(ax_vel.get_xticklabels(), visible=False)
    plt.setp(ax_temp.get_xticklabels(), visible=False)

    # Legend for profile plots
    legend_lines = [
        Line2D([0], [0], color='gray', linestyle='-', lw=LINE_WIDTH, label='CLTORC'),
        Line2D([0], [0], color='gray', linestyle='--', lw=LINE_WIDTH, label='Pele'),
    ]
    for t_idx, t in enumerate(comparison_times):
        color = TIME_COLORS[t_idx % len(TIME_COLORS)]
        legend_lines.append(Line2D([0], [0], color=color, linestyle='-', lw=LINE_WIDTH,
                                    label=f't = {t*1e3:.2f} ms'))

    ax_vel.legend(handles=legend_lines, fontsize=FONT_SIZE_LEGEND - 1, loc='upper right', ncol=2)

    # Compute y-limits from simulation data
    all_u = np.concatenate([s.u for s in upper_snaps + lower_snaps])
    all_T = np.concatenate([s.T for s in upper_snaps + lower_snaps])
    all_p = np.concatenate([s.p for s in upper_snaps + lower_snaps]) / 1e6

    ax_vel.set_ylim(all_u.min() * 0.95 if all_u.min() > 0 else all_u.min() * 1.05, all_u.max() * 1.05)
    ax_temp.set_ylim(all_T.min() * 0.98, all_T.max() * 1.02)
    ax_pres.set_ylim(all_p.min() * 0.95, all_p.max() * 1.05)

    for ax in profile_axes + [ax_piston]:
        ax.tick_params(labelsize=FONT_SIZE_TICK)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=PLOT_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_file}")


def plot_multi_time_bounds_horizontal(
    upper_snaps: List[SimSnapshot],
    lower_snaps: List[SimSnapshot],
    pele_snaps: List[PeleSnapshot],
    output_file: str,
    shift_to_upper: bool = False,
):
    """
    Plot multi-time comparison in horizontal layout.

    3 panels side by side: Velocity, Temperature, Pressure.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    ax_vel, ax_temp, ax_pres = axes

    comparison_times = [snap.time for snap in pele_snaps]

    for t_idx, (upper, lower, pele) in enumerate(zip(upper_snaps, lower_snaps, pele_snaps)):
        color = TIME_COLORS[t_idx % len(TIME_COLORS)]

        x_upper = upper.x_centers * 100
        x_lower = lower.x_centers * 100

        if shift_to_upper:
            shift = (upper.x_centers.min() - lower.x_centers.min()) * 100
            x_lower = x_lower + shift

        # Piston position lines
        for ax in axes:
            ax.axvline(x_upper.min(), color=color, linestyle='--', alpha=0.5, lw=1)

        x_min = min(x_upper.min(), x_lower.min())
        x_max = max(x_upper.max(), x_lower.max())
        x_common = np.linspace(x_min, x_max, 500)

        def interp_safe(x_orig, y_orig, x_new):
            return np.interp(x_new, x_orig, y_orig, left=y_orig[0], right=y_orig[-1])

        # Velocity
        u_upper = interp_safe(x_upper, upper.u, x_common)
        u_lower = interp_safe(x_lower, lower.u, x_common)
        ax_vel.fill_between(x_common, u_lower, u_upper, alpha=0.3, color=color)
        ax_vel.plot(x_upper, upper.u, '-', color=color, lw=LINE_WIDTH)
        ax_vel.plot(x_lower, lower.u, '--', color=color, lw=LINE_WIDTH)

        # Temperature
        T_upper = interp_safe(x_upper, upper.T, x_common)
        T_lower = interp_safe(x_lower, lower.T, x_common)
        ax_temp.fill_between(x_common, T_lower, T_upper, alpha=0.3, color=color)
        ax_temp.plot(x_upper, upper.T, '-', color=color, lw=LINE_WIDTH)
        ax_temp.plot(x_lower, lower.T, '--', color=color, lw=LINE_WIDTH)

        # Pressure
        p_upper = interp_safe(x_upper, upper.p / 1e6, x_common)
        p_lower = interp_safe(x_lower, lower.p / 1e6, x_common)
        ax_pres.fill_between(x_common, p_lower, p_upper, alpha=0.3, color=color)
        ax_pres.plot(x_upper, upper.p / 1e6, '-', color=color, lw=LINE_WIDTH)
        ax_pres.plot(x_lower, lower.p / 1e6, '--', color=color, lw=LINE_WIDTH)

        # PeleC data (filter points behind flame position)
        flame_x_cm = x_upper.min()
        pele_mask = pele.x * 100 >= flame_x_cm
        x_pele = pele.x[pele_mask] * 100
        if len(x_pele) > 0:
            ax_vel.plot(x_pele, pele.u[pele_mask], ':', color=color, lw=LINE_WIDTH, alpha=0.8)
            if pele.T is not None:
                ax_temp.plot(x_pele, pele.T[pele_mask], ':', color=color, lw=LINE_WIDTH, alpha=0.8)
            ax_pres.plot(x_pele, pele.p[pele_mask] / 1e6, ':', color=color, lw=LINE_WIDTH, alpha=0.8)

    ax_vel.set_ylabel('Velocity [m/s]', fontsize=FONT_SIZE_LABEL)
    ax_temp.set_ylabel('Temperature [K]', fontsize=FONT_SIZE_LABEL)
    ax_pres.set_ylabel('Pressure [MPa]', fontsize=FONT_SIZE_LABEL)

    # Compute y-limits from simulation data
    all_u = np.concatenate([s.u for s in upper_snaps + lower_snaps])
    all_T = np.concatenate([s.T for s in upper_snaps + lower_snaps])
    all_p = np.concatenate([s.p for s in upper_snaps + lower_snaps]) / 1e6

    ax_vel.set_ylim(all_u.min() * 0.95 if all_u.min() > 0 else all_u.min() * 1.05, all_u.max() * 1.05)
    ax_temp.set_ylim(all_T.min() * 0.98, all_T.max() * 1.02)
    ax_pres.set_ylim(all_p.min() * 0.95, all_p.max() * 1.05)

    for ax in axes:
        ax.set_xlabel('Position [cm]', fontsize=FONT_SIZE_LABEL)
        ax.tick_params(labelsize=FONT_SIZE_TICK)
        ax.grid(True, alpha=0.3)

    # Legend
    legend_lines = [
        Line2D([0], [0], color='gray', linestyle='-', lw=LINE_WIDTH, label='LGDCS'),
        Line2D([0], [0], color='gray', linestyle='--', lw=LINE_WIDTH, label='Pele'),
    ]
    for t_idx, t in enumerate(comparison_times):
        color = TIME_COLORS[t_idx % len(TIME_COLORS)]
        legend_lines.append(Line2D([0], [0], color=color, linestyle='-', lw=LINE_WIDTH,
                                    label=f't = {t*1e3:.2f} ms'))

    fig.legend(handles=legend_lines, fontsize=FONT_SIZE_LEGEND, loc='lower center',
               ncol=len(legend_lines), bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(output_file, dpi=PLOT_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_file}")


def plot_xt_diagrams(
    upper_results_dir: Path,
    lower_results_dir: Path,
    output_dir: Path,
):
    """
    Plot X-T diagrams comparing upper and lower bounds.

    Side-by-side comparison for pressure, velocity, density.
    """
    upper_ts = np.load(upper_results_dir / "timeseries.npz", allow_pickle=True)
    lower_ts = np.load(lower_results_dir / "timeseries.npz", allow_pickle=True)

    for var, units, vmin, vmax in [
        ('p', 'Pressure', None, None),
        ('u', 'Velocity', None, None),
        ('rho', 'Density', None, None),
    ]:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

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

            ax.set_xlabel('Position x [m]', fontsize=FONT_SIZE_LABEL)
            ax.set_ylabel('Time [ms]', fontsize=FONT_SIZE_LABEL)
            ax.set_title(label, fontsize=FONT_SIZE_TITLE)
            ax.tick_params(labelsize=FONT_SIZE_TICK)

        fig.suptitle(f'x-t Diagram: {units}', fontsize=FONT_SIZE_TITLE + 2, fontweight='bold')
        plt.tight_layout()

        output_file = output_dir / f"xt_diagram_{var}.png"
        plt.savefig(output_file, dpi=PLOT_DPI, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {output_file}")


def plot_piston_results(
    upper_snaps: List[SimSnapshot],
    lower_snaps: List[SimSnapshot],
    upper_ts: SimTimeseries,
    lower_ts: SimTimeseries,
    output_file: str,
):
    """
    Plot 6-panel piston results comparison.

    Shows density, velocity, pressure, temperature profiles and
    mesh movement and conservation errors.
    """
    # Use the last snapshot for profile comparisons
    upper = upper_snaps[-1]
    lower = lower_snaps[-1]

    fig, axes = plt.subplots(3, 2, figsize=(12, 12))

    time_s = upper.time

    x_upper = upper.x_centers
    x_lower = lower.x_centers

    # Density
    axes[0, 0].plot(x_upper, upper.rho, '-', color='blue', lw=LINE_WIDTH, label='Flame Velocity (No Porous)')
    axes[0, 0].plot(x_lower, lower.rho, '--', color='red', lw=LINE_WIDTH, label='Flame Vel - CJ (Porous)')
    axes[0, 0].set_ylabel('Density [kg/m³]', fontsize=FONT_SIZE_LABEL)
    axes[0, 0].set_title('Density Profile', fontsize=FONT_SIZE_TITLE)
    axes[0, 0].legend(fontsize=FONT_SIZE_LEGEND - 1)

    # Velocity
    axes[0, 1].plot(x_upper, upper.u, '-', color='blue', lw=LINE_WIDTH, label='Flame Velocity (No Porous)')
    axes[0, 1].plot(x_lower, lower.u, '--', color='red', lw=LINE_WIDTH, label='Flame Vel - CJ (Porous)')
    axes[0, 1].set_ylabel('Velocity [m/s]', fontsize=FONT_SIZE_LABEL)
    axes[0, 1].set_title('Velocity Profile', fontsize=FONT_SIZE_TITLE)
    axes[0, 1].legend(fontsize=FONT_SIZE_LEGEND - 1)

    # Pressure
    axes[1, 0].plot(x_upper, upper.p / 1e5, '-', color='blue', lw=LINE_WIDTH, label='Flame Velocity (No Porous)')
    axes[1, 0].plot(x_lower, lower.p / 1e5, '--', color='red', lw=LINE_WIDTH, label='Flame Vel - CJ (Porous)')
    axes[1, 0].set_ylabel('Pressure [bar]', fontsize=FONT_SIZE_LABEL)
    axes[1, 0].set_title('Pressure Profile', fontsize=FONT_SIZE_TITLE)
    axes[1, 0].legend(fontsize=FONT_SIZE_LEGEND - 1)

    # Temperature
    axes[1, 1].plot(x_upper, upper.T, '-', color='blue', lw=LINE_WIDTH, label='Flame Velocity (No Porous)')
    axes[1, 1].plot(x_lower, lower.T, '--', color='red', lw=LINE_WIDTH, label='Flame Vel - CJ (Porous)')
    axes[1, 1].set_ylabel('Temperature [K]', fontsize=FONT_SIZE_LABEL)
    axes[1, 1].set_title('Temperature Profile', fontsize=FONT_SIZE_TITLE)
    axes[1, 1].legend(fontsize=FONT_SIZE_LEGEND - 1)

    # Piston position vs time
    axes[2, 0].plot(upper_ts.t * 1e3, upper_ts.x_piston, '-', color='blue', lw=LINE_WIDTH, label='Flame Velocity (No Porous)')
    axes[2, 0].plot(lower_ts.t * 1e3, lower_ts.x_piston, '--', color='red', lw=LINE_WIDTH, label='Flame Vel - CJ (Porous)')
    axes[2, 0].set_xlabel('Time [ms]', fontsize=FONT_SIZE_LABEL)
    axes[2, 0].set_ylabel('Piston Position [m]', fontsize=FONT_SIZE_LABEL)
    axes[2, 0].set_title('Piston Position', fontsize=FONT_SIZE_TITLE)
    axes[2, 0].legend(fontsize=FONT_SIZE_LEGEND - 1)

    # Piston velocity vs time
    axes[2, 1].plot(upper_ts.t * 1e3, upper_ts.u_piston, '-', color='blue', lw=LINE_WIDTH, label='Flame Velocity (No Porous)')
    axes[2, 1].plot(lower_ts.t * 1e3, lower_ts.u_piston, '--', color='red', lw=LINE_WIDTH, label='Flame Vel - CJ (Porous)')
    axes[2, 1].set_xlabel('Time [ms]', fontsize=FONT_SIZE_LABEL)
    axes[2, 1].set_ylabel('Piston Velocity [m/s]', fontsize=FONT_SIZE_LABEL)
    axes[2, 1].set_title('Piston Velocity', fontsize=FONT_SIZE_TITLE)
    axes[2, 1].legend(fontsize=FONT_SIZE_LEGEND - 1)

    for ax in axes.flat:
        ax.tick_params(labelsize=FONT_SIZE_TICK)
        ax.grid(True, alpha=0.3)

    # Set x labels for bottom row
    axes[0, 0].set_xlabel('Position x [m]', fontsize=FONT_SIZE_LABEL)
    axes[0, 1].set_xlabel('Position x [m]', fontsize=FONT_SIZE_LABEL)
    axes[1, 0].set_xlabel('Position x [m]', fontsize=FONT_SIZE_LABEL)
    axes[1, 1].set_xlabel('Position x [m]', fontsize=FONT_SIZE_LABEL)

    fig.suptitle(f'Piston-Driven Flow Comparison at t = {time_s:.6f} s', fontsize=FONT_SIZE_TITLE + 2, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_file, dpi=PLOT_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_file}")


def plot_velocity_comparison(
    upper_ts: SimTimeseries,
    lower_ts: SimTimeseries,
    output_file: str,
):
    """
    Plot 4-panel velocity data comparison.

    Shows raw velocities, corrected velocities, gas BC velocity, and flame position.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    t_upper_ms = upper_ts.t * 1e3
    t_lower_ms = lower_ts.t * 1e3

    # Raw velocities (same as piston velocity for now)
    axes[0, 0].plot(t_upper_ms, upper_ts.u_piston, '-', color='blue', lw=LINE_WIDTH, label='upper_bound - Flame Vel')
    axes[0, 0].plot(t_lower_ms, lower_ts.u_piston, '-', color='red', lw=LINE_WIDTH, label='lower_bound - Flame Vel')
    axes[0, 0].set_ylabel('Velocity [m/s]', fontsize=FONT_SIZE_LABEL)
    axes[0, 0].set_xlabel('Time [ms]', fontsize=FONT_SIZE_LABEL)
    axes[0, 0].set_title('Raw Velocities', fontsize=FONT_SIZE_TITLE)
    axes[0, 0].legend(fontsize=FONT_SIZE_LEGEND - 1)

    # Piston velocity (after correction)
    axes[0, 1].plot(t_upper_ms, upper_ts.u_piston, '-', color='blue', lw=LINE_WIDTH, label='Flame Velocity (No Porous)')
    axes[0, 1].plot(t_lower_ms, lower_ts.u_piston, '--', color='red', lw=LINE_WIDTH, label='Flame Vel - CJ (Porous)')
    axes[0, 1].set_ylabel('Corrected Velocity [m/s]', fontsize=FONT_SIZE_LABEL)
    axes[0, 1].set_xlabel('Time [ms]', fontsize=FONT_SIZE_LABEL)
    axes[0, 1].set_title('Piston Velocity (after correction)', fontsize=FONT_SIZE_TITLE)
    axes[0, 1].legend(fontsize=FONT_SIZE_LEGEND - 1)

    # Gas BC velocity (actual velocity used at boundary)
    axes[1, 0].plot(t_upper_ms, upper_ts.u_gas, '-', color='blue', lw=LINE_WIDTH, label=r'$u_{gas}$ (No Porous)')
    axes[1, 0].plot(t_lower_ms, lower_ts.u_gas, '--', color='red', lw=LINE_WIDTH, label=r'$u_{gas}$ (Porous)')
    axes[1, 0].set_ylabel('Gas Velocity [m/s]', fontsize=FONT_SIZE_LABEL)
    axes[1, 0].set_xlabel('Time [ms]', fontsize=FONT_SIZE_LABEL)
    axes[1, 0].set_title('Gas BC Velocity at Piston Face', fontsize=FONT_SIZE_TITLE)
    axes[1, 0].legend(fontsize=FONT_SIZE_LEGEND - 1)

    # Flame position
    axes[1, 1].plot(t_upper_ms, upper_ts.x_piston, '-', color='blue', lw=LINE_WIDTH, label='Flame Velocity (No Porous)')
    axes[1, 1].plot(t_lower_ms, lower_ts.x_piston, '--', color='red', lw=LINE_WIDTH, label='Flame Vel - CJ (Porous)')
    axes[1, 1].set_ylabel('Position [m]', fontsize=FONT_SIZE_LABEL)
    axes[1, 1].set_xlabel('Time [ms]', fontsize=FONT_SIZE_LABEL)
    axes[1, 1].set_title('Flame Position', fontsize=FONT_SIZE_TITLE)
    axes[1, 1].legend(fontsize=FONT_SIZE_LEGEND - 1)

    for ax in axes.flat:
        ax.tick_params(labelsize=FONT_SIZE_TICK)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Velocity Data Comparison', fontsize=FONT_SIZE_TITLE + 2, fontweight='bold')
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

    # Scan snapshots
    print("\nScanning snapshots...")
    upper_snap_dir = upper_results_dir / "snapshots"
    lower_snap_dir = lower_results_dir / "snapshots"

    upper_snaps_info = scan_snapshots(upper_snap_dir)
    lower_snaps_info = scan_snapshots(lower_snap_dir)

    print(f"  Upper: {len(upper_snaps_info)} snapshots")
    print(f"  Lower: {len(lower_snaps_info)} snapshots")

    # Load PeleC data
    print(f"\nLoading PeleC pltfiles: {pltfile_dir}")
    pele_snaps = load_all_pltfiles(str(pltfile_dir), extract_location)
    print(f"  Loaded {len(pele_snaps)} PeleC snapshots")

    # Match snapshots at PeleC times
    print("\nMatching snapshots at PeleC times...")
    upper_matched = []
    lower_matched = []

    for pele in pele_snaps:
        t_ms = pele.time * 1e3

        upper_info = find_nearest_snapshot(upper_snaps_info, pele.time)
        lower_info = find_nearest_snapshot(lower_snaps_info, pele.time)

        upper_snap = load_snapshot(upper_info, upper_label)
        lower_snap = load_snapshot(lower_info, lower_label)

        upper_matched.append(upper_snap)
        lower_matched.append(lower_snap)

        print(f"  t = {t_ms:.2f} ms: upper={upper_snap.time*1e3:.2f}ms, lower={lower_snap.time*1e3:.2f}ms")

    comparison_times = [pele.time for pele in pele_snaps]

    # Helper function to generate all plots for a given shift setting
    def generate_plots(out_dir: Path, shift: bool, label: str):
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n--- Generating {label} plots ---")

        # Multi-time bounds comparison (vertical)
        plot_multi_time_bounds_comparison(
            upper_matched, lower_matched, pele_snaps,
            upper_ts, lower_ts,
            str(out_dir / "multi_time_bounds_comparison.png"),
            shift_to_upper=shift,
        )

        # Multi-time bounds comparison (horizontal)
        plot_multi_time_bounds_horizontal(
            upper_matched, lower_matched, pele_snaps,
            str(out_dir / "multi_time_bounds_comparison_horizontal.png"),
            shift_to_upper=shift,
        )

        # Individual time comparisons
        for i, (upper, lower, pele) in enumerate(zip(upper_matched, lower_matched, pele_snaps)):
            pele_name = f"{pele.part_name}_{pele.pltfile_name}" if pele.part_name else pele.pltfile_name
            output_file = out_dir / f"pele_comparison_{pele_name}.png"
            plot_bounds_comparison_at_time(upper, lower, pele, str(output_file), shift_to_upper=shift)
            print(f"  Saved: {output_file.name}")

    # Generate shared plots (not affected by shift)
    print("\nGenerating shared plots...")

    # 1. Piston velocity bounds
    plot_piston_velocity_bounds(
        upper_ts, lower_ts, comparison_times,
        str(output_dir / "piston_velocity_bounds.png")
    )

    # 1b. Gas velocity at piston face bounds
    plot_gas_velocity_bounds(
        upper_ts, lower_ts, comparison_times,
        str(output_dir / "gas_velocity_bounds.png")
    )

    # 2. Piston results
    plot_piston_results(
        upper_matched, lower_matched,
        upper_ts, lower_ts,
        str(output_dir / "piston_results.png")
    )

    # 3. Velocity comparison
    plot_velocity_comparison(
        upper_ts, lower_ts,
        str(output_dir / "velocity_comparison.png")
    )

    # 4. X-T diagrams
    try:
        plot_xt_diagrams(upper_results_dir, lower_results_dir, output_dir)
    except Exception as e:
        print(f"  Warning: Could not generate X-T diagrams: {e}")

    # Generate both original and aligned comparisons
    # Original (no shift)
    generate_plots(output_dir / "original", shift=False, label="original (no shift)")

    # Aligned (shift lower to match upper piston position)
    generate_plots(output_dir / "aligned", shift=True, label="aligned (shifted to upper)")

    print("\n" + "=" * 70)
    print("COMPARISON COMPLETE")
    print("=" * 70)
    print(f"Results saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare 1D solver bounds with PeleC data"
    )
    parser.add_argument("--upper-dir", type=str, required=True,
                        help="Upper bound results directory (e.g., results/upper_bound)")
    parser.add_argument("--lower-dir", type=str, required=True,
                        help="Lower bound results directory (e.g., results/lower_bound)")
    parser.add_argument("--pltfile-dir", type=str, required=True,
                        help="Directory containing Part-*/ pltfile subdirs")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory")
    parser.add_argument("--upper-label", type=str, default="Flame Velocity (No Porous)",
                        help="Label for upper bound")
    parser.add_argument("--lower-label", type=str, default="Flame Vel - CJ (Porous)",
                        help="Label for lower bound")
    parser.add_argument("--extract-location", type=float, default=0.0445,
                        help="Y-coordinate for PeleC ray extraction [m]")
    parser.add_argument("--shift-to-upper", action="store_true",
                        help="Shift lower bound to align piston position with upper bound")

    args = parser.parse_args()

    compare_with_bounds(
        upper_results_dir=Path(args.upper_dir),
        lower_results_dir=Path(args.lower_dir),
        pltfile_dir=Path(args.pltfile_dir),
        output_dir=Path(args.output_dir),
        upper_label=args.upper_label,
        lower_label=args.lower_label,
        extract_location=args.extract_location,
        shift_to_upper=args.shift_to_upper,
    )


if __name__ == "__main__":
    main()
