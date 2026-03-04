"""
Verify that the fixed BC produces same piston positions for solid and porous modes.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


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


def main():
    results_base = Path(__file__).parent / "results"
    solid_dir = results_base / "solid_fixed"
    porous_dir = results_base / "porous_fixed"

    print("=" * 70)
    print("VERIFYING FIXED BC - PISTON POSITIONS")
    print("=" * 70)

    # Load piston histories
    ph_solid = np.load(solid_dir / "piston_history.npz")
    ph_porous = np.load(porous_dir / "piston_history.npz")

    t_solid = ph_solid['times']
    v_piston_solid = ph_solid['piston_velocity']
    v_gas_solid = ph_solid['gas_velocity']

    t_porous = ph_porous['times']
    v_piston_porous = ph_porous['piston_velocity']
    v_gas_porous = ph_porous['gas_velocity']

    print("\nPiston velocity comparison:")
    print(f"  Solid: v_piston range = [{v_piston_solid.min():.1f}, {v_piston_solid.max():.1f}] m/s")
    print(f"  Porous: v_piston range = [{v_piston_porous.min():.1f}, {v_piston_porous.max():.1f}] m/s")

    print("\nGas velocity comparison:")
    print(f"  Solid: v_gas range = [{v_gas_solid.min():.1f}, {v_gas_solid.max():.1f}] m/s")
    print(f"  Porous: v_gas range = [{v_gas_porous.min():.1f}, {v_gas_porous.max():.1f}] m/s")

    # Interpolate to common time grid
    from scipy.interpolate import interp1d
    t_min = max(t_solid.min(), t_porous.min())
    t_max = min(t_solid.max(), t_porous.max())
    t_common = np.linspace(t_min, t_max, 1000)

    v_piston_solid_interp = interp1d(t_solid, v_piston_solid, kind='linear')(t_common)
    v_piston_porous_interp = interp1d(t_porous, v_piston_porous, kind='linear')(t_common)

    v_gas_solid_interp = interp1d(t_solid, v_gas_solid, kind='linear')(t_common)
    v_gas_porous_interp = interp1d(t_porous, v_gas_porous, kind='linear')(t_common)

    # Check piston velocity difference (should be ~0 for both modes)
    piston_diff = v_piston_solid_interp - v_piston_porous_interp
    gas_diff = v_gas_solid_interp - v_gas_porous_interp

    print(f"\nPiston velocity difference (solid - porous):")
    print(f"  Mean: {piston_diff.mean():.2f} m/s")
    print(f"  Max: {np.abs(piston_diff).max():.2f} m/s")

    print(f"\nGas velocity difference (solid - porous):")
    print(f"  Mean: {gas_diff.mean():.2f} m/s (expected: ~119.2 m/s)")
    print(f"  Std: {gas_diff.std():.2f} m/s")

    # Check piston positions at sample times
    print("\nPiston positions at sample times:")
    test_times = [500e-6, 1000e-6, 1500e-6, 2000e-6]

    for t_target in test_times:
        try:
            snap_solid = load_snapshot_at_time(solid_dir, t_target)
            snap_porous = load_snapshot_at_time(porous_dir, t_target)

            x_piston_solid = snap_solid['x_centers'].min() * 100  # cm
            x_piston_porous = snap_porous['x_centers'].min() * 100  # cm

            print(f"  t={t_target*1e6:.0f} µs: solid={x_piston_solid:.2f} cm, "
                  f"porous={x_piston_porous:.2f} cm, diff={x_piston_solid - x_piston_porous:.2f} cm")
        except FileNotFoundError as e:
            print(f"  t={t_target*1e6:.0f} µs: Error - {e}")

    # Create comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Piston velocity vs time
    ax = axes[0, 0]
    ax.plot(t_solid * 1e6, v_piston_solid, 'b-', lw=1.5, label='Solid')
    ax.plot(t_porous * 1e6, v_piston_porous, 'r--', lw=1.5, label='Porous')
    ax.set_xlabel('Time [µs]')
    ax.set_ylabel('Piston Velocity [m/s]')
    ax.set_title('Piston Velocity (Grid Motion) - Should Match')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Gas velocity vs time
    ax = axes[0, 1]
    ax.plot(t_solid * 1e6, v_gas_solid, 'b-', lw=1.5, label='Solid')
    ax.plot(t_porous * 1e6, v_gas_porous, 'r--', lw=1.5, label='Porous')
    ax.set_xlabel('Time [µs]')
    ax.set_ylabel('Gas Velocity [m/s]')
    ax.set_title('Gas Velocity (Riemann BC) - Should Differ by Offset')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Piston velocity difference
    ax = axes[1, 0]
    ax.plot(t_common * 1e6, piston_diff, 'k-', lw=1.5)
    ax.axhline(0, color='r', ls='--', lw=1)
    ax.set_xlabel('Time [µs]')
    ax.set_ylabel('Difference [m/s]')
    ax.set_title('Piston Velocity Difference (Solid - Porous)')
    ax.grid(True, alpha=0.3)

    # Gas velocity difference
    ax = axes[1, 1]
    ax.plot(t_common * 1e6, gas_diff, 'k-', lw=1.5)
    ax.axhline(119.2, color='r', ls='--', lw=1, label='Expected (119.2 m/s)')
    ax.set_xlabel('Time [µs]')
    ax.set_ylabel('Difference [m/s]')
    ax.set_title('Gas Velocity Difference (Solid - Porous)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = results_base / "fixed_bc_verification.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nSaved verification plot: {output_file}")
    plt.close()


if __name__ == "__main__":
    main()
