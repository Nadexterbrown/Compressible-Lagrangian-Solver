"""
Compare 1D Lagrangian solver results with PeleC pltfile snapshots.

Simple time-based comparison: at each PeleC time, plot all simulation
datasets and PeleC data on the same axes.

Reference: LGDCS/scripts/pele_sim/pele_bds/piston_pele_bnds.py
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
from typing import List, Optional, Dict
from dataclasses import dataclass

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

# Colors for different snapshot times
TIME_COLORS = ['#E41A1C', '#377EB8', '#4DAF4A', '#984EA3', '#FF7F00']

# Line styles for different datasets
DATASET_STYLES = ['-', '--', ':', '-.']


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


def plot_time_comparison(
    sim_snapshots: List[SimSnapshot],
    pele_snapshot: Optional[PeleSnapshot],
    output_file: str,
):
    """
    Plot comparison at a single time.

    All simulation datasets and PeleC data on same axes.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    time_ms = sim_snapshots[0].time * 1e3

    # Plot each simulation dataset
    for i, sim in enumerate(sim_snapshots):
        style = DATASET_STYLES[i % len(DATASET_STYLES)]
        x_cm = sim.x_centers * 100

        # Pressure
        axes[0].plot(x_cm, sim.p / 1e6, style, lw=LINE_WIDTH, label=sim.label)

        # Velocity
        axes[1].plot(x_cm, sim.u, style, lw=LINE_WIDTH, label=sim.label)

        # Temperature
        axes[2].plot(x_cm, sim.T, style, lw=LINE_WIDTH, label=sim.label)

    # Plot PeleC data if available
    if pele_snapshot is not None:
        x_pele_cm = pele_snapshot.x * 100
        axes[0].plot(x_pele_cm, pele_snapshot.p / 1e6, 'k:', lw=LINE_WIDTH, alpha=0.7, label='PeleC')
        axes[1].plot(x_pele_cm, pele_snapshot.u, 'k:', lw=LINE_WIDTH, alpha=0.7, label='PeleC')
        if pele_snapshot.T is not None:
            axes[2].plot(x_pele_cm, pele_snapshot.T, 'k:', lw=LINE_WIDTH, alpha=0.7, label='PeleC')

    # Labels
    axes[0].set_xlabel('Position [cm]', fontsize=FONT_SIZE_LABEL)
    axes[0].set_ylabel('Pressure [MPa]', fontsize=FONT_SIZE_LABEL)
    axes[0].set_title('Pressure', fontsize=FONT_SIZE_TITLE)
    axes[0].legend(fontsize=FONT_SIZE_LEGEND-1)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Position [cm]', fontsize=FONT_SIZE_LABEL)
    axes[1].set_ylabel('Velocity [m/s]', fontsize=FONT_SIZE_LABEL)
    axes[1].set_title('Velocity', fontsize=FONT_SIZE_TITLE)
    axes[1].legend(fontsize=FONT_SIZE_LEGEND-1)
    axes[1].grid(True, alpha=0.3)

    axes[2].set_xlabel('Position [cm]', fontsize=FONT_SIZE_LABEL)
    axes[2].set_ylabel('Temperature [K]', fontsize=FONT_SIZE_LABEL)
    axes[2].set_title('Temperature', fontsize=FONT_SIZE_TITLE)
    axes[2].legend(fontsize=FONT_SIZE_LEGEND-1)
    axes[2].grid(True, alpha=0.3)

    plt.suptitle(f't = {time_ms:.2f} ms', fontsize=FONT_SIZE_TITLE + 2, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_file, dpi=PLOT_DPI, bbox_inches='tight')
    plt.close(fig)


def plot_multi_time_comparison(
    all_matched: List[List[SimSnapshot]],
    pele_snapshots: List[PeleSnapshot],
    output_file: str,
    dataset_labels: List[str],
):
    """
    Plot multi-time comparison.

    3 rows (velocity, temperature, pressure) x 1 column.
    Each time gets a different color, each dataset gets a different line style.
    """
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    n_times = len(pele_snapshots)
    n_datasets = len(dataset_labels)

    # Plot each time
    for t_idx, pele in enumerate(pele_snapshots):
        color = TIME_COLORS[t_idx % len(TIME_COLORS)]

        # Plot each dataset at this time
        for d_idx, sim in enumerate(all_matched[t_idx]):
            style = DATASET_STYLES[d_idx % len(DATASET_STYLES)]
            x_cm = sim.x_centers * 100

            axes[0].plot(x_cm, sim.u, linestyle=style, color=color, lw=LINE_WIDTH)
            axes[1].plot(x_cm, sim.T, linestyle=style, color=color, lw=LINE_WIDTH)
            axes[2].plot(x_cm, sim.p / 1e6, linestyle=style, color=color, lw=LINE_WIDTH)

        # Plot PeleC
        x_pele_cm = pele.x * 100
        axes[0].plot(x_pele_cm, pele.u, ':', color=color, lw=LINE_WIDTH, alpha=0.7)
        if pele.T is not None:
            axes[1].plot(x_pele_cm, pele.T, ':', color=color, lw=LINE_WIDTH, alpha=0.7)
        axes[2].plot(x_pele_cm, pele.p / 1e6, ':', color=color, lw=LINE_WIDTH, alpha=0.7)

    # Labels
    axes[0].set_ylabel('Velocity [m/s]', fontsize=FONT_SIZE_LABEL)
    axes[1].set_ylabel('Temperature [K]', fontsize=FONT_SIZE_LABEL)
    axes[2].set_ylabel('Pressure [MPa]', fontsize=FONT_SIZE_LABEL)
    axes[2].set_xlabel('Position [cm]', fontsize=FONT_SIZE_LABEL)

    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=FONT_SIZE_TICK)

    # Legend for datasets (line styles)
    legend_datasets = []
    for d_idx, label in enumerate(dataset_labels):
        style = DATASET_STYLES[d_idx % len(DATASET_STYLES)]
        legend_datasets.append(Line2D([0], [0], color='gray', linestyle=style, lw=LINE_WIDTH, label=label))
    legend_datasets.append(Line2D([0], [0], color='gray', linestyle=':', lw=LINE_WIDTH, label='PeleC'))

    # Legend for times (colors)
    legend_times = []
    for t_idx, pele in enumerate(pele_snapshots):
        color = TIME_COLORS[t_idx % len(TIME_COLORS)]
        t_ms = pele.time * 1e3
        legend_times.append(Line2D([0], [0], color=color, linestyle='-', lw=LINE_WIDTH, label=f't={t_ms:.2f}ms'))

    axes[0].legend(handles=legend_datasets + legend_times, loc='upper right',
                   fontsize=FONT_SIZE_LEGEND-1, ncol=2)

    plt.suptitle('Time Comparison', fontsize=FONT_SIZE_TITLE + 2, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_file, dpi=PLOT_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_file}")


def compare_simulations(
    snapshot_dirs: List[Path],
    pltfile_dir: Path,
    output_dir: Path,
    extract_location: float = 0.0445,
    dataset_labels: Optional[List[str]] = None,
):
    """
    Compare simulation datasets with PeleC at each PeleC time.

    Simple time-based matching: find nearest snapshot from each dataset.
    """
    print("=" * 70)
    print("TIME-BASED COMPARISON")
    print("=" * 70)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Default labels from directory names
    if dataset_labels is None:
        dataset_labels = [d.parent.name for d in snapshot_dirs]

    # Scan all snapshot directories
    all_snapshots = []
    for i, snap_dir in enumerate(snapshot_dirs):
        print(f"\nScanning: {snap_dir}")
        snapshots = scan_snapshots(snap_dir)
        if not snapshots:
            print(f"  WARNING: No snapshots found!")
            continue
        print(f"  Found {len(snapshots)} snapshots")
        print(f"  Time range: [{snapshots[0]['time']*1e3:.2f}, {snapshots[-1]['time']*1e3:.2f}] ms")
        all_snapshots.append({
            'dir': snap_dir,
            'snapshots': snapshots,
            'label': dataset_labels[i] if i < len(dataset_labels) else f'Dataset {i+1}'
        })

    if not all_snapshots:
        print("ERROR: No valid snapshot directories!")
        return

    # Load PeleC data
    print(f"\nLoading PeleC pltfiles: {pltfile_dir}")
    pele_snapshots = load_all_pltfiles(str(pltfile_dir), extract_location)
    if not pele_snapshots:
        print("ERROR: No PeleC pltfiles found!")
        return
    print(f"  Loaded {len(pele_snapshots)} snapshots")

    # Match and plot for each PeleC time
    print("\nMatching and plotting...")
    all_matched = []  # For multi-time plot

    for pele in pele_snapshots:
        t_ms = pele.time * 1e3
        print(f"\n  t = {t_ms:.2f} ms:")

        matched_sims = []
        for data in all_snapshots:
            nearest = find_nearest_snapshot(data['snapshots'], pele.time)
            sim = load_snapshot(nearest, data['label'])
            matched_sims.append(sim)

            dt_us = abs(sim.time - pele.time) * 1e6
            piston_x = sim.x_centers.min() * 100
            print(f"    {data['label']}: t={sim.time*1e3:.2f}ms (Δt={dt_us:.1f}µs), piston={piston_x:.1f}cm")

        all_matched.append(matched_sims)

        # Individual time plot
        output_file = output_dir / f"comparison_t{t_ms:.2f}ms.png"
        plot_time_comparison(matched_sims, pele, str(output_file))
        print(f"    Saved: {output_file.name}")

    # Multi-time plot
    output_file = output_dir / "multi_time_comparison.png"
    plot_multi_time_comparison(all_matched, pele_snapshots, str(output_file),
                                [d['label'] for d in all_snapshots])

    print("\n" + "=" * 70)
    print("COMPARISON COMPLETE")
    print("=" * 70)
    print(f"Results saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare 1D solver results with PeleC at each PeleC time"
    )
    parser.add_argument("--snapshot-dirs", type=str, nargs='+', required=True,
                        help="Snapshot directories (e.g., results/solid/snapshots results/offset/snapshots)")
    parser.add_argument("--pltfile-dir", type=str, required=True,
                        help="Directory containing Part-*/ pltfile subdirs")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory")
    parser.add_argument("--extract-location", type=float, default=0.0445,
                        help="Y-coordinate for PeleC ray extraction [m]")
    parser.add_argument("--labels", type=str, nargs='+', default=None,
                        help="Labels for datasets")

    args = parser.parse_args()

    compare_simulations(
        snapshot_dirs=[Path(d) for d in args.snapshot_dirs],
        pltfile_dir=Path(args.pltfile_dir),
        output_dir=Path(args.output_dir),
        extract_location=args.extract_location,
        dataset_labels=args.labels,
    )


if __name__ == "__main__":
    main()
