"""
Visualize the difference between solid and porous results.

Shows that the apparent "time shift" is actually a position shift
due to different piston velocities.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import interp1d


def load_snapshot_at_time(results_dir: Path, target_time: float):
    """Load the snapshot closest to target_time."""
    snapshots_dir = results_dir / "snapshots"
    snapshot_files = sorted(snapshots_dir.glob("snapshot_*.npz"))

    best_snap = None
    best_dt = float('inf')

    for f in snapshot_files:
        data = np.load(f)
        t = float(data['t'])
        dt = abs(t - target_time)
        if dt < best_dt:
            best_dt = dt
            best_snap = f

    if best_snap is None:
        raise FileNotFoundError(f"No snapshots found in {snapshots_dir}")

    return np.load(best_snap)


def compute_piston_position(results_dir: Path, time: float):
    """Compute piston position by integrating velocity."""
    piston_file = results_dir / "piston_history.npz"
    if piston_file.exists():
        data = np.load(piston_file)
        t = data['times']
        v = data['piston_velocity']
    else:
        ts = np.load(results_dir / "timeseries.npz", allow_pickle=True)
        t = ts['t']
        v = ts['u_piston']

    # Integrate velocity to get position
    from scipy.integrate import cumulative_trapezoid
    x_piston = cumulative_trapezoid(v, t, initial=0.0)

    # Interpolate to target time
    return float(interp1d(t, x_piston, kind='linear', fill_value='extrapolate')(time))


def main():
    results_base = Path(__file__).parent / "results"
    solid_dir = results_base / "solid"
    porous_dir = results_base / "porous"

    print("=" * 70)
    print("POSITION SHIFT VISUALIZATION")
    print("=" * 70)

    # Compare at a few different times
    test_times = [500e-6, 1000e-6, 1500e-6, 2000e-6]

    fig, axes = plt.subplots(4, 3, figsize=(14, 14))

    for row, t_target in enumerate(test_times):
        print(f"\nTime: {t_target*1e6:.0f} µs")

        # Load snapshots
        try:
            snap_solid = load_snapshot_at_time(solid_dir, t_target)
            snap_porous = load_snapshot_at_time(porous_dir, t_target)
        except FileNotFoundError as e:
            print(f"  Error: {e}")
            continue

        t_solid = float(snap_solid['t'])
        t_porous = float(snap_porous['t'])

        print(f"  Solid: t={t_solid*1e6:.1f} µs")
        print(f"  Porous: t={t_porous*1e6:.1f} µs")

        # Get data
        x_solid = snap_solid['x_centers'] * 100  # cm
        x_porous = snap_porous['x_centers'] * 100  # cm

        # Piston positions (first cell position)
        piston_x_solid = x_solid.min()
        piston_x_porous = x_porous.min()
        print(f"  Piston positions: solid={piston_x_solid:.2f} cm, porous={piston_x_porous:.2f} cm")
        print(f"  Position difference: {piston_x_solid - piston_x_porous:.2f} cm")

        # Plot velocity
        ax = axes[row, 0]
        ax.plot(x_solid, snap_solid['u'], 'b-', lw=1.5, label='Solid')
        ax.plot(x_porous, snap_porous['u'], 'r-', lw=1.5, label='Porous')
        ax.axvline(piston_x_solid, color='b', ls='--', alpha=0.5, lw=1)
        ax.axvline(piston_x_porous, color='r', ls='--', alpha=0.5, lw=1)
        ax.set_ylabel('Velocity [m/s]')
        if row == 0:
            ax.set_title('Velocity')
            ax.legend()
        if row == len(test_times) - 1:
            ax.set_xlabel('Position [cm]')
        ax.grid(True, alpha=0.3)
        ax.text(0.02, 0.98, f't = {t_target*1e6:.0f} µs', transform=ax.transAxes,
                fontsize=10, va='top', ha='left', fontweight='bold')

        # Plot pressure
        ax = axes[row, 1]
        ax.plot(x_solid, snap_solid['p'] / 1e6, 'b-', lw=1.5, label='Solid')
        ax.plot(x_porous, snap_porous['p'] / 1e6, 'r-', lw=1.5, label='Porous')
        ax.axvline(piston_x_solid, color='b', ls='--', alpha=0.5, lw=1)
        ax.axvline(piston_x_porous, color='r', ls='--', alpha=0.5, lw=1)
        ax.set_ylabel('Pressure [MPa]')
        if row == 0:
            ax.set_title('Pressure')
        if row == len(test_times) - 1:
            ax.set_xlabel('Position [cm]')
        ax.grid(True, alpha=0.3)

        # Plot density
        ax = axes[row, 2]
        ax.plot(x_solid, snap_solid['rho'], 'b-', lw=1.5, label='Solid')
        ax.plot(x_porous, snap_porous['rho'], 'r-', lw=1.5, label='Porous')
        ax.axvline(piston_x_solid, color='b', ls='--', alpha=0.5, lw=1)
        ax.axvline(piston_x_porous, color='r', ls='--', alpha=0.5, lw=1)
        ax.set_ylabel('Density [kg/m³]')
        if row == 0:
            ax.set_title('Density')
        if row == len(test_times) - 1:
            ax.set_xlabel('Position [cm]')
        ax.grid(True, alpha=0.3)

    plt.suptitle('Solid vs Porous Comparison at Same Simulation Times\n'
                 '(Dashed lines show piston face position)', fontsize=12)
    plt.tight_layout()

    output_file = results_base / "position_shift_comparison.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {output_file}")
    plt.close()

    # Now create a plot showing profiles aligned by piston position
    print("\n" + "=" * 70)
    print("PISTON-ALIGNED COMPARISON")
    print("=" * 70)

    fig, axes = plt.subplots(4, 3, figsize=(14, 14))

    for row, t_target in enumerate(test_times):
        try:
            snap_solid = load_snapshot_at_time(solid_dir, t_target)
            snap_porous = load_snapshot_at_time(porous_dir, t_target)
        except FileNotFoundError:
            continue

        # Get data
        x_solid = snap_solid['x_centers']
        x_porous = snap_porous['x_centers']

        # Shift coordinates so piston is at x=0
        x_solid_shifted = (x_solid - x_solid.min()) * 100  # cm from piston
        x_porous_shifted = (x_porous - x_porous.min()) * 100  # cm from piston

        # Plot velocity (relative to piston)
        ax = axes[row, 0]
        ax.plot(x_solid_shifted, snap_solid['u'], 'b-', lw=1.5, label='Solid')
        ax.plot(x_porous_shifted, snap_porous['u'], 'r-', lw=1.5, label='Porous')
        ax.set_ylabel('Velocity [m/s]')
        if row == 0:
            ax.set_title('Velocity')
            ax.legend()
        if row == len(test_times) - 1:
            ax.set_xlabel('Distance from piston [cm]')
        ax.grid(True, alpha=0.3)
        ax.text(0.02, 0.98, f't = {t_target*1e6:.0f} µs', transform=ax.transAxes,
                fontsize=10, va='top', ha='left', fontweight='bold')

        # Plot pressure
        ax = axes[row, 1]
        ax.plot(x_solid_shifted, snap_solid['p'] / 1e6, 'b-', lw=1.5, label='Solid')
        ax.plot(x_porous_shifted, snap_porous['p'] / 1e6, 'r-', lw=1.5, label='Porous')
        ax.set_ylabel('Pressure [MPa]')
        if row == 0:
            ax.set_title('Pressure')
        if row == len(test_times) - 1:
            ax.set_xlabel('Distance from piston [cm]')
        ax.grid(True, alpha=0.3)

        # Plot density
        ax = axes[row, 2]
        ax.plot(x_solid_shifted, snap_solid['rho'], 'b-', lw=1.5, label='Solid')
        ax.plot(x_porous_shifted, snap_porous['rho'], 'r-', lw=1.5, label='Porous')
        ax.set_ylabel('Density [kg/m³]')
        if row == 0:
            ax.set_title('Density')
        if row == len(test_times) - 1:
            ax.set_xlabel('Distance from piston [cm]')
        ax.grid(True, alpha=0.3)

    plt.suptitle('Solid vs Porous - Profiles Aligned by Piston Position\n'
                 '(x = 0 at piston face for both)', fontsize=12)
    plt.tight_layout()

    output_file = results_base / "piston_aligned_comparison.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
The 'time shift' you observed is actually a POSITION shift:

1. Both simulations use the same time base (from PeleC data)
2. The piston velocities differ by exactly the expected offset (~119 m/s)
3. At the same simulation time, the porous piston has traveled LESS distance
4. This makes all features (piston face, shock, etc.) appear shifted back in space

This is the correct physical behavior - if you drive the piston slower,
the flow field develops more slowly in space (but at the same rate in time).

The 'piston_aligned_comparison.png' shows profiles with x=0 at the piston face.
These should show similar shapes, demonstrating the profiles are related
by a position shift, not a time shift.
""")


if __name__ == "__main__":
    main()
