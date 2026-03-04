"""
Diagnose time shift issue between solid and porous simulation results.

Compares piston velocities to verify they differ only by the offset,
not by a time shift.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_piston_history(results_dir: Path):
    """Load piston velocity history from a results directory."""
    piston_file = results_dir / "piston_history.npz"
    if piston_file.exists():
        data = np.load(piston_file)
        return data['times'], data['piston_velocity']

    timeseries_file = results_dir / "timeseries.npz"
    if timeseries_file.exists():
        data = np.load(timeseries_file, allow_pickle=True)
        return data['t'], data['u_piston']

    raise FileNotFoundError(f"No piston history found in {results_dir}")


def main():
    results_base = Path(__file__).parent / "results"
    solid_dir = results_base / "solid"
    porous_dir = results_base / "porous"

    print("=" * 70)
    print("TIME SHIFT DIAGNOSTIC")
    print("=" * 70)

    # Load both datasets
    print("\nLoading piston histories...")
    t_solid, v_solid = load_piston_history(solid_dir)
    t_porous, v_porous = load_piston_history(porous_dir)

    print(f"  Solid: {len(t_solid)} points, t=[{t_solid[0]*1e6:.1f}, {t_solid[-1]*1e6:.1f}] µs")
    print(f"  Porous: {len(t_porous)} points, t=[{t_porous[0]*1e6:.1f}, {t_porous[-1]*1e6:.1f}] µs")

    # Check if time arrays are the same
    print("\nChecking time alignment...")
    if len(t_solid) == len(t_porous):
        t_diff = np.abs(t_solid - t_porous)
        max_t_diff = np.max(t_diff)
        print(f"  Same length: YES")
        print(f"  Max time difference: {max_t_diff*1e9:.3f} ns")
    else:
        print(f"  Same length: NO ({len(t_solid)} vs {len(t_porous)})")
        # Find common time range
        t_min = max(t_solid.min(), t_porous.min())
        t_max = min(t_solid.max(), t_porous.max())
        print(f"  Common time range: [{t_min*1e6:.1f}, {t_max*1e6:.1f}] µs")

    # Interpolate to common time grid
    from scipy.interpolate import interp1d

    t_common_min = max(t_solid.min(), t_porous.min())
    t_common_max = min(t_solid.max(), t_porous.max())
    n_points = min(len(t_solid), len(t_porous))
    t_common = np.linspace(t_common_min, t_common_max, n_points)

    v_solid_interp = interp1d(t_solid, v_solid, kind='linear', fill_value='extrapolate')(t_common)
    v_porous_interp = interp1d(t_porous, v_porous, kind='linear', fill_value='extrapolate')(t_common)

    # Compute velocity difference
    v_diff = v_solid_interp - v_porous_interp

    print("\nVelocity difference analysis (solid - porous):")
    print(f"  Min difference: {v_diff.min():.2f} m/s")
    print(f"  Max difference: {v_diff.max():.2f} m/s")
    print(f"  Mean difference: {v_diff.mean():.2f} m/s")
    print(f"  Std deviation: {v_diff.std():.2f} m/s")

    # If the offset is constant, std deviation should be small
    if v_diff.std() < 5.0:
        print("\n  -> Difference is nearly constant (as expected)")
        print(f"     Expected offset (from args): see config.json")
    else:
        print("\n  -> Difference varies significantly!")
        print("     This suggests a timing issue or non-linear effect")

    # Check for time shift via cross-correlation
    print("\nCross-correlation analysis (detecting time shift)...")

    # Normalize velocities
    v_solid_norm = (v_solid_interp - v_solid_interp.mean()) / v_solid_interp.std()
    v_porous_norm = (v_porous_interp - v_porous_interp.mean()) / v_porous_interp.std()

    # Cross-correlation
    corr = np.correlate(v_solid_norm, v_porous_norm, mode='full')
    lag_samples = np.arange(-len(t_common) + 1, len(t_common))
    dt = t_common[1] - t_common[0]
    lag_time = lag_samples * dt

    # Find peak
    peak_idx = np.argmax(corr)
    peak_lag = lag_time[peak_idx]
    peak_corr = corr[peak_idx] / len(t_common)

    print(f"  Peak correlation: {peak_corr:.4f}")
    print(f"  Time lag at peak: {peak_lag*1e6:.2f} µs")

    if abs(peak_lag) < dt * 2:
        print("  -> Negligible time shift detected")
    else:
        print(f"  -> Significant time shift of {peak_lag*1e6:.2f} µs detected!")

    # Create diagnostic plots
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # Plot 1: Raw velocities
    ax = axes[0]
    ax.plot(t_solid * 1e6, v_solid, 'b-', lw=1.5, label='Solid')
    ax.plot(t_porous * 1e6, v_porous, 'r-', lw=1.5, label='Porous')
    ax.set_xlabel('Time [µs]')
    ax.set_ylabel('Piston Velocity [m/s]')
    ax.set_title('Piston Velocity vs Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Velocity difference
    ax = axes[1]
    ax.plot(t_common * 1e6, v_diff, 'k-', lw=1.5)
    ax.axhline(v_diff.mean(), color='r', ls='--', lw=1, label=f'Mean: {v_diff.mean():.1f} m/s')
    ax.set_xlabel('Time [µs]')
    ax.set_ylabel('Velocity Difference [m/s]')
    ax.set_title('Solid - Porous Velocity Difference')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Cross-correlation
    ax = axes[2]
    # Zoom in on relevant lag range
    lag_mask = np.abs(lag_time) < 50e-6  # ±50 µs
    ax.plot(lag_time[lag_mask] * 1e6, corr[lag_mask] / len(t_common), 'b-', lw=1.5)
    ax.axvline(peak_lag * 1e6, color='r', ls='--', lw=1, label=f'Peak lag: {peak_lag*1e6:.2f} µs')
    ax.set_xlabel('Time Lag [µs]')
    ax.set_ylabel('Normalized Correlation')
    ax.set_title('Cross-Correlation (detecting time shift)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = Path(__file__).parent / "results" / "time_shift_diagnostic.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nSaved diagnostic plot: {output_file}")
    plt.close()

    # Also check the velocity source data directly
    print("\n" + "=" * 70)
    print("CHECKING VELOCITY SOURCE DATA")
    print("=" * 70)

    from pele_data_loader import PeleDataLoader, PeleTrajectoryInterpolator

    data_dir = Path(__file__).parent / "pele_data" / "truncated_raw_data"
    loader = PeleDataLoader(data_dir)
    data = loader.load()
    trajectory = PeleTrajectoryInterpolator(data, extrapolate=False)

    # Check velocity at a few sample times
    print("\nFlame velocity from source data:")
    sample_times = [0.0, 100e-6, 200e-6, 500e-6, 1000e-6]
    for t in sample_times:
        if t >= trajectory.t_min and t <= trajectory.t_max:
            v_flame = trajectory.velocity(t)
            print(f"  t={t*1e6:6.0f} µs: v_flame = {v_flame:.1f} m/s")

    # Load configs to get velocity offset
    import json

    solid_config = solid_dir / "config.json"
    porous_config = porous_dir / "config.json"

    if solid_config.exists() and porous_config.exists():
        with open(solid_config) as f:
            cfg_solid = json.load(f)
        with open(porous_config) as f:
            cfg_porous = json.load(f)

        offset_solid = cfg_solid.get('velocity_offset', 0.0)
        offset_porous = cfg_porous.get('velocity_offset', 0.0)
        min_solid = cfg_solid.get('velocity_min', 0.0)
        min_porous = cfg_porous.get('velocity_min', 0.0)

        print("\nConfiguration:")
        print(f"  Solid:  offset={offset_solid} m/s, min={min_solid} m/s")
        print(f"  Porous: offset={offset_porous} m/s, min={min_porous} m/s")
        print(f"  Expected difference: {offset_solid - offset_porous} m/s")
        print(f"  Actual mean difference: {v_diff.mean():.1f} m/s")

        # Check when clamping kicks in
        print("\nClamping analysis:")
        v_flame_all = data.flame_velocity

        if offset_porous != 0:
            v_porous_expected = v_flame_all + offset_porous
            n_clamped = np.sum(v_porous_expected < min_porous)
            pct_clamped = 100.0 * n_clamped / len(v_flame_all)
            print(f"  Porous: {n_clamped} of {len(v_flame_all)} points clamped ({pct_clamped:.1f}%)")

            if n_clamped > 0:
                clamped_mask = v_porous_expected < min_porous
                t_clamped = data.time[clamped_mask]
                print(f"    Clamped time range: [{t_clamped.min()*1e6:.1f}, {t_clamped.max()*1e6:.1f}] µs")

    print("\n" + "=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
