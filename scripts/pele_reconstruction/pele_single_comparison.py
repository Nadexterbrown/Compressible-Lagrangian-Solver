"""
Compare 1D Lagrangian solver results with a single PeleC pltfile snapshot.

Supports zoom functionality to focus on a region of interest.
Uses HRR-based flame detection for filtering.

Usage:
    python pele_single_comparison.py --upper-dir results/upper --lower-dir results/lower \
        --pltfile Part-2/plt123456 --zoom-max 150 --output-dir results/comparison
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
from pathlib import Path
from typing import Optional, Tuple

from pele_pltfile_loader import extract_pele_snapshot, PeleSnapshot, PLTFILE_TIME_OFFSETS
from pele_comparison import (
    get_snapshot_files,
    find_snapshots_for_times_coarse_to_fine,
    load_snapshot,
    load_timeseries,
    find_flame_position_hrr,
    get_font_sizes,
    compute_mass_coordinate,
    compute_mass_coordinate_pele,
    compute_mass_coordinate_sim,
    SimSnapshot,
    SimTimeseries,
    TIME_COLORS,
    FULL_PAGE_WIDTH,
    SINGLE_COL_WIDTH,
    PLOT_DPI,
)


# Publication-quality plot settings
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['text.usetex'] = False
mpl.rcParams['axes.linewidth'] = 1.2
mpl.rcParams['xtick.major.width'] = 1.0
mpl.rcParams['ytick.major.width'] = 1.0


def plot_single_comparison(
    upper_snap: SimSnapshot,
    lower_snap: SimSnapshot,
    pele_snap: PeleSnapshot,
    upper_ts: SimTimeseries,
    lower_ts: SimTimeseries,
    output_dir: Path,
    zoom_min: Optional[float] = None,
    zoom_max: Optional[float] = None,
    column_mode: str = 'single',
    lagrangian: bool = False,
):
    """
    Plot comparison for a single PeleC snapshot with zoom.

    Args:
        upper_snap: Upper bound simulation snapshot
        lower_snap: Lower bound simulation snapshot
        pele_snap: PeleC snapshot
        upper_ts: Upper bound timeseries
        lower_ts: Lower bound timeseries
        output_dir: Output directory
        zoom_min: Minimum x position [cm] or mass coordinate [kg/m²]. If None, uses piston location / 0.
        zoom_max: Maximum x position [cm] or mass coordinate [kg/m²]. If None, uses full domain.
        column_mode: 'single' or 'double' for figure sizing
        lagrangian: If True, plot in Lagrangian (mass) coordinates instead of position
    """
    fonts = get_font_sizes(column_mode)
    lw = fonts['linewidth']
    ms = 8  # Marker size

    # Check if single-bound mode
    single_bound = upper_snap.path == lower_snap.path

    # Create figure with 4 panels (3 profiles + piston velocity)
    if column_mode == 'single':
        fig = plt.figure(figsize=(SINGLE_COL_WIDTH, 7.0))
        gs = fig.add_gridspec(4, 1, hspace=0.45)
        ax_vel = fig.add_subplot(gs[0, 0])
        ax_temp = fig.add_subplot(gs[1, 0], sharex=ax_vel)
        ax_pres = fig.add_subplot(gs[2, 0], sharex=ax_vel)
        ax_piston = fig.add_subplot(gs[3, 0])
    else:
        fig = plt.figure(figsize=(FULL_PAGE_WIDTH, 4.0))
        gs = fig.add_gridspec(3, 2, width_ratios=[1.5, 1], height_ratios=[1, 1, 1],
                              hspace=0.08, wspace=0.3)
        ax_vel = fig.add_subplot(gs[0, 0])
        ax_temp = fig.add_subplot(gs[1, 0], sharex=ax_vel)
        ax_pres = fig.add_subplot(gs[2, 0], sharex=ax_vel)
        ax_piston = fig.add_subplot(gs[1:, 1])

    profile_axes = [ax_vel, ax_temp, ax_pres]
    color = TIME_COLORS[0]  # Black for single time
    time_ms = pele_snap.time * 1e3

    # Get coordinates (position in cm or mass coordinate in kg/m²)
    if lagrangian:
        x_upper = compute_mass_coordinate_sim(upper_snap)
        x_lower = compute_mass_coordinate_sim(lower_snap)
        x_label = 'Mass Coordinate [kg/m²]'
        coord_suffix = '_lagrangian'
    else:
        x_upper = upper_snap.x_centers * 100
        x_lower = lower_snap.x_centers * 100
        x_label = 'Position [cm]'
        coord_suffix = ''

    # Determine zoom limits
    if lagrangian:
        x_coord_min = 0.0  # Mass coordinate starts at 0
    else:
        x_coord_min = min(x_upper.min(), x_lower.min())

    if zoom_min is None:
        zoom_min = x_coord_min
    if zoom_max is None:
        zoom_max = max(x_upper.max(), x_lower.max())

    # Create common x grid for fill_between
    x_min = min(x_upper.min(), x_lower.min())
    x_max = max(x_upper.max(), x_lower.max())
    x_common = np.linspace(x_min, x_max, 500)

    def interp_safe(x_orig, y_orig, x_new):
        return np.interp(x_new, x_orig, y_orig, left=y_orig[0], right=y_orig[-1])

    # Velocity
    u_upper = interp_safe(x_upper, upper_snap.u, x_common)
    u_lower = interp_safe(x_lower, lower_snap.u, x_common)
    if not single_bound:
        ax_vel.fill_between(x_common, u_lower, u_upper, alpha=0.3, color=color)
    ax_vel.plot(x_upper, upper_snap.u, '-', color=color, lw=lw)
    if not single_bound:
        ax_vel.plot(x_lower, lower_snap.u, '-', color=color, lw=lw)

    # Temperature
    T_upper = interp_safe(x_upper, upper_snap.T, x_common)
    T_lower = interp_safe(x_lower, lower_snap.T, x_common)
    if not single_bound:
        ax_temp.fill_between(x_common, T_lower, T_upper, alpha=0.3, color=color)
    ax_temp.plot(x_upper, upper_snap.T, '-', color=color, lw=lw)
    if not single_bound:
        ax_temp.plot(x_lower, lower_snap.T, '-', color=color, lw=lw)

    # Pressure
    p_upper = interp_safe(x_upper, upper_snap.p / 1e6, x_common)
    p_lower = interp_safe(x_lower, lower_snap.p / 1e6, x_common)
    if not single_bound:
        ax_pres.fill_between(x_common, p_lower, p_upper, alpha=0.3, color=color)
    ax_pres.plot(x_upper, upper_snap.p / 1e6, '-', color=color, lw=lw)
    if not single_bound:
        ax_pres.plot(x_lower, lower_snap.p / 1e6, '-', color=color, lw=lw)

    # PeleC data (filter points behind flame position using HRR)
    flame_x, buffer_x = find_flame_position_hrr(pele_snap, buffer_cells=10)
    pele_mask = pele_snap.x >= buffer_x

    if lagrangian:
        # Compute mass coordinate for PeleC starting from buffer position
        m_pele_full = compute_mass_coordinate_pele(pele_snap)
        buffer_idx = np.searchsorted(pele_snap.x, buffer_x)
        m_offset = m_pele_full[buffer_idx] if buffer_idx < len(m_pele_full) else 0
        x_pele = m_pele_full[pele_mask] - m_offset
    else:
        x_pele = pele_snap.x[pele_mask] * 100  # Convert to cm

    if len(x_pele) > 0:
        ax_vel.plot(x_pele, pele_snap.u[pele_mask], '--', color=color, lw=lw, alpha=0.8)
        if pele_snap.T is not None:
            ax_temp.plot(x_pele, pele_snap.T[pele_mask], '--', color=color, lw=lw, alpha=0.8)
        ax_pres.plot(x_pele, pele_snap.p[pele_mask] / 1e6, '--', color=color, lw=lw, alpha=0.8)

    # Piston velocity panel - limit to comparison time + buffer
    t_plot_max = pele_snap.time * 1.1  # 10% buffer past comparison time

    # Mask timeseries to only show up to t_plot_max
    upper_t_mask = upper_ts.t <= t_plot_max
    lower_t_mask = lower_ts.t <= t_plot_max

    t_upper_plot = upper_ts.t[upper_t_mask]
    t_lower_plot = lower_ts.t[lower_t_mask]
    u_upper_plot = upper_ts.u_gas[upper_t_mask]
    u_lower_plot = lower_ts.u_gas[lower_t_mask]

    t_upper_ms = t_upper_plot * 1e3
    t_lower_ms = t_lower_plot * 1e3

    t_min = max(t_upper_plot.min(), t_lower_plot.min())
    t_max = min(t_upper_plot.max(), t_lower_plot.max())
    t_common = np.linspace(t_min, t_max, 500)
    t_common_ms = t_common * 1e3

    u_upper_interp = np.interp(t_common, t_upper_plot, u_upper_plot)
    u_lower_interp = np.interp(t_common, t_lower_plot, u_lower_plot)

    if single_bound:
        ax_piston.plot(t_upper_ms, u_upper_plot, 'k-', lw=lw)
    else:
        ax_piston.fill_between(t_common_ms, u_lower_interp, u_upper_interp, alpha=0.3, color='gray')
        ax_piston.plot(t_upper_ms, u_upper_plot, 'k-', lw=lw, label=r'$u_g = U_f$')
        ax_piston.plot(t_lower_ms, u_lower_plot, 'k--', lw=lw, label=r'$u_g = U_f - CJ_{def}$')

    # Marker at comparison time
    u_upper_at_t = np.interp(pele_snap.time, t_upper_plot, u_upper_plot)
    ax_piston.plot(time_ms, u_upper_at_t, 'o', color=color, markersize=ms,
                   markeredgecolor='black', markeredgewidth=0.5)
    if not single_bound:
        u_lower_at_t = np.interp(pele_snap.time, t_lower_plot, u_lower_plot)
        ax_piston.plot(time_ms, u_lower_at_t, 's', color=color, markersize=ms-1,
                       markeredgecolor='black', markeredgewidth=0.5)

    # Set piston panel x-limits
    ax_piston.set_xlim(0, t_plot_max * 1e3)

    # Labels
    ax_vel.set_ylabel('Velocity [m/s]', fontsize=fonts['label'])
    ax_temp.set_ylabel('Temperature [K]', fontsize=fonts['label'])
    ax_pres.set_ylabel('Pressure [MPa]', fontsize=fonts['label'])
    ax_pres.set_xlabel(x_label, fontsize=fonts['label'])

    ax_piston.set_xlabel('Time [ms]', fontsize=fonts['label'])
    ax_piston.set_ylabel(r'$u(x=x_p, t)$ [m/s]', fontsize=fonts['label'])
    if not single_bound:
        ax_piston.legend(fontsize=fonts['legend'] - 2, loc='upper left')

    # Apply zoom to x-axis
    for ax in profile_axes:
        ax.set_xlim(zoom_min, zoom_max)

    # Compute y-limits from data within zoom region
    zoom_mask_upper = (x_upper >= zoom_min) & (x_upper <= zoom_max)
    zoom_mask_lower = (x_lower >= zoom_min) & (x_lower <= zoom_max)
    zoom_mask_pele = (x_pele >= zoom_min) & (x_pele <= zoom_max) if len(x_pele) > 0 else np.array([])

    # Velocity y-limits
    u_in_zoom = []
    if zoom_mask_upper.any():
        u_in_zoom.append(upper_snap.u[zoom_mask_upper])
    if zoom_mask_lower.any():
        u_in_zoom.append(lower_snap.u[zoom_mask_lower])
    if len(zoom_mask_pele) > 0 and zoom_mask_pele.any():
        u_in_zoom.append(pele_snap.u[pele_mask][zoom_mask_pele])
    if u_in_zoom:
        u_in_zoom = np.concatenate(u_in_zoom)
        ax_vel.set_ylim(u_in_zoom.min() * 0.95 if u_in_zoom.min() > 0 else u_in_zoom.min() * 1.05,
                        u_in_zoom.max() * 1.05)

    # Temperature y-limits
    T_in_zoom = []
    if zoom_mask_upper.any():
        T_in_zoom.append(upper_snap.T[zoom_mask_upper])
    if zoom_mask_lower.any():
        T_in_zoom.append(lower_snap.T[zoom_mask_lower])
    if pele_snap.T is not None and len(zoom_mask_pele) > 0 and zoom_mask_pele.any():
        T_in_zoom.append(pele_snap.T[pele_mask][zoom_mask_pele])
    if T_in_zoom:
        T_in_zoom = np.concatenate(T_in_zoom)
        ax_temp.set_ylim(T_in_zoom.min() * 0.98, T_in_zoom.max() * 1.02)

    # Pressure y-limits
    p_in_zoom = []
    if zoom_mask_upper.any():
        p_in_zoom.append(upper_snap.p[zoom_mask_upper] / 1e6)
    if zoom_mask_lower.any():
        p_in_zoom.append(lower_snap.p[zoom_mask_lower] / 1e6)
    if len(zoom_mask_pele) > 0 and zoom_mask_pele.any():
        p_in_zoom.append(pele_snap.p[pele_mask][zoom_mask_pele] / 1e6)
    if p_in_zoom:
        p_in_zoom = np.concatenate(p_in_zoom)
        ax_pres.set_ylim(p_in_zoom.min() * 0.95, p_in_zoom.max() * 1.05)

    # Hide x labels for upper profile plots
    if column_mode == 'double':
        plt.setp(ax_vel.get_xticklabels(), visible=False)
        plt.setp(ax_temp.get_xticklabels(), visible=False)

    for ax in profile_axes + [ax_piston]:
        ax.tick_params(labelsize=fonts['tick'])
        ax.grid(True, alpha=0.3)

    # Legend
    legend_lines = [
        Line2D([0], [0], color='gray', linestyle='-', lw=lw, label='1D Model'),
        Line2D([0], [0], color='gray', linestyle='--', lw=lw, label='2D Simulation'),
        Line2D([0], [0], color=color, linestyle='-', lw=lw, label=f't = {time_ms:.2f} ms'),
    ]

    plt.tight_layout()
    fig.subplots_adjust(top=0.92)
    fig.legend(handles=legend_lines, fontsize=fonts['legend'] - 1, loc='upper center',
               ncol=3, bbox_to_anchor=(0.5, 0.99), frameon=True, fancybox=False,
               edgecolor='gray', columnspacing=1.2, handletextpad=0.4)

    # Save
    output_file = output_dir / f"single_comparison_t{time_ms:.2f}ms{coord_suffix}.png"
    plt.savefig(output_file, dpi=PLOT_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_file}")

    # Generate piston velocity trajectory plot (aspect ratio matches profile panels)
    fig_piston, ax_up = plt.subplots(figsize=(SINGLE_COL_WIDTH, 1.7))

    # Plot piston velocity trajectory
    if single_bound:
        ax_up.plot(t_upper_ms, upper_ts.u_piston[upper_t_mask], 'k-', lw=lw)
    else:
        ax_up.fill_between(t_common_ms,
                           np.interp(t_common, t_lower_plot, lower_ts.u_piston[lower_t_mask]),
                           np.interp(t_common, t_upper_plot, upper_ts.u_piston[upper_t_mask]),
                           alpha=0.3, color='gray')
        ax_up.plot(t_upper_ms, upper_ts.u_piston[upper_t_mask], 'k-', lw=lw, label=r'$u_g = U_f$')
        ax_up.plot(t_lower_ms, lower_ts.u_piston[lower_t_mask], 'k--', lw=lw, label=r'$u_g = U_f - CJ_{def}$')

    # Marker at comparison time
    u_p_upper_at_t = np.interp(pele_snap.time, t_upper_plot, upper_ts.u_piston[upper_t_mask])
    ax_up.plot(time_ms, u_p_upper_at_t, 'o', color=color, markersize=ms,
               markeredgecolor='black', markeredgewidth=0.5)
    if not single_bound:
        u_p_lower_at_t = np.interp(pele_snap.time, t_lower_plot, lower_ts.u_piston[lower_t_mask])
        ax_up.plot(time_ms, u_p_lower_at_t, 's', color=color, markersize=ms-1,
                   markeredgecolor='black', markeredgewidth=0.5)

    ax_up.set_xlim(0, t_plot_max * 1e3)
    ax_up.set_xlabel('Time [ms]', fontsize=fonts['label'])
    ax_up.set_ylabel(r'$u_p$ [m/s]', fontsize=fonts['label'])
    ax_up.tick_params(labelsize=fonts['tick'])
    ax_up.yaxis.set_major_locator(plt.MaxNLocator(nbins=6))
    ax_up.grid(True, alpha=0.3)
    if not single_bound:
        ax_up.legend(fontsize=fonts['legend'] - 2, loc='upper left')

    plt.tight_layout()
    piston_output_file = output_dir / f"piston_velocity_t{time_ms:.2f}ms{coord_suffix}.png"
    plt.savefig(piston_output_file, dpi=PLOT_DPI, bbox_inches='tight')
    plt.close(fig_piston)
    print(f"  Saved: {piston_output_file}")


def get_part_name_and_offset(pltfile_path: Path) -> Tuple[str, float]:
    """Extract part name and time offset from pltfile path."""
    path_str = str(pltfile_path)
    for part_name, offset in PLTFILE_TIME_OFFSETS.items():
        if part_name in path_str:
            return part_name, offset
    return "", 0.0


def compare_single_pltfile(
    upper_results_dir: Path,
    lower_results_dir: Path,
    pltfile_path: Path,
    output_dir: Path,
    upper_label: str = "Upper Bound",
    lower_label: str = "Lower Bound",
    extract_location: float = 0.0445,
    zoom_min: Optional[float] = None,
    zoom_max: Optional[float] = None,
    lagrangian: bool = False,
    lagrangian_min: Optional[float] = None,
    lagrangian_max: Optional[float] = None,
):
    """
    Compare 1D solver results with a single PeleC pltfile.

    Args:
        upper_results_dir: Upper bound results directory
        lower_results_dir: Lower bound results directory
        pltfile_path: Path to specific PeleC pltfile
        output_dir: Output directory
        upper_label: Label for upper bound
        lower_label: Label for lower bound
        extract_location: Y-coordinate for PeleC ray extraction [m]
        zoom_min: Minimum x position for zoom [cm]. If None, uses piston location.
        zoom_max: Maximum x position for zoom [cm]. If None, uses full domain.
        lagrangian: If True, also generate plots in Lagrangian (mass) coordinates.
        lagrangian_min: Minimum mass coordinate for Lagrangian zoom [kg/m²]. If None, uses 0.
        lagrangian_max: Maximum mass coordinate for Lagrangian zoom [kg/m²]. If None, uses full domain.
    """
    print("=" * 70)
    print("SINGLE PLTFILE COMPARISON")
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

    # Load PeleC snapshot
    print(f"\nLoading PeleC pltfile: {pltfile_path}")
    part_name, time_offset = get_part_name_and_offset(pltfile_path)
    print(f"  Part: {part_name}, Time offset: {time_offset*1e6:.1f} us")

    pele_snap = extract_pele_snapshot(
        str(pltfile_path),
        extract_location,
        time_offset=time_offset,
        part_name=part_name,
    )
    print(f"  Time: {pele_snap.time*1e3:.2f} ms")

    # Find matching 1D snapshots
    print("\nFinding matching 1D snapshots...")
    target_times = [pele_snap.time]

    upper_snap_dir = upper_results_dir / "snapshots"
    lower_snap_dir = lower_results_dir / "snapshots"

    upper_files = get_snapshot_files(upper_snap_dir)
    lower_files = get_snapshot_files(lower_snap_dir)

    print(f"  Upper: {len(upper_files)} snapshot files")
    print(f"  Lower: {len(lower_files)} snapshot files")

    upper_snaps_info = find_snapshots_for_times_coarse_to_fine(upper_files, target_times)
    lower_snaps_info = find_snapshots_for_times_coarse_to_fine(lower_files, target_times)

    upper_snap = load_snapshot(upper_snaps_info[0], upper_label)
    lower_snap = load_snapshot(lower_snaps_info[0], lower_label)

    print(f"  Matched: upper t={upper_snap.time*1e3:.2f} ms, lower t={lower_snap.time*1e3:.2f} ms")

    # Generate plot(s)
    print("\nGenerating comparison plot...")
    print(f"  Zoom: x = [{zoom_min if zoom_min else 'piston'}, {zoom_max if zoom_max else 'domain max'}] cm")

    # Eulerian (position) plot
    plot_single_comparison(
        upper_snap, lower_snap, pele_snap,
        upper_ts, lower_ts,
        output_dir,
        zoom_min=zoom_min,
        zoom_max=zoom_max,
        column_mode='single',
        lagrangian=False,
    )

    # Lagrangian (mass coordinate) plot
    if lagrangian:
        print("\nGenerating Lagrangian comparison plot...")
        print(f"  Zoom: m = [{lagrangian_min if lagrangian_min else '0'}, {lagrangian_max if lagrangian_max else 'domain max'}] kg/m²")
        plot_single_comparison(
            upper_snap, lower_snap, pele_snap,
            upper_ts, lower_ts,
            output_dir,
            zoom_min=lagrangian_min,
            zoom_max=lagrangian_max,
            column_mode='single',
            lagrangian=True,
        )

    print("\n" + "=" * 70)
    print("COMPARISON COMPLETE")
    print("=" * 70)
    print(f"Results saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare 1D solver with a single PeleC pltfile"
    )
    parser.add_argument("--upper-dir", type=str, required=True,
                        help="Upper bound results directory")
    parser.add_argument("--lower-dir", type=str, default=None,
                        help="Lower bound results directory (optional, defaults to upper-dir)")
    parser.add_argument("--pltfile", type=str, required=True,
                        help="Path to specific PeleC pltfile directory")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory")
    parser.add_argument("--upper-label", type=str, default="Upper Bound",
                        help="Label for upper bound")
    parser.add_argument("--lower-label", type=str, default=None,
                        help="Label for lower bound (defaults to upper-label)")
    parser.add_argument("--extract-location", type=float, default=0.0445,
                        help="Y-coordinate for PeleC ray extraction [m]")
    parser.add_argument("--zoom-min", type=float, default=None,
                        help="Minimum x position for zoom [cm]. Defaults to piston location.")
    parser.add_argument("--zoom-max", type=float, default=None,
                        help="Maximum x position for zoom [cm]. Defaults to full domain.")
    parser.add_argument("--lagrangian", action="store_true",
                        help="Also generate plots in Lagrangian (mass) coordinates")
    parser.add_argument("--lagrangian-min", type=float, default=None,
                        help="Minimum mass coordinate for Lagrangian zoom [kg/m²]. Defaults to 0.")
    parser.add_argument("--lagrangian-max", type=float, default=None,
                        help="Maximum mass coordinate for Lagrangian zoom [kg/m²]. Defaults to full domain.")

    args = parser.parse_args()

    # Handle single-bound mode
    lower_dir = args.lower_dir if args.lower_dir else args.upper_dir
    lower_label = args.lower_label if args.lower_label else args.upper_label

    compare_single_pltfile(
        upper_results_dir=Path(args.upper_dir),
        lower_results_dir=Path(lower_dir),
        pltfile_path=Path(args.pltfile),
        output_dir=Path(args.output_dir),
        upper_label=args.upper_label,
        lower_label=lower_label,
        extract_location=args.extract_location,
        zoom_min=args.zoom_min,
        zoom_max=args.zoom_max,
        lagrangian=args.lagrangian,
        lagrangian_min=args.lagrangian_min,
        lagrangian_max=args.lagrangian_max,
    )


if __name__ == "__main__":
    main()
