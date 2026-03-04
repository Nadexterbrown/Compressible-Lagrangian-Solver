"""
Check if the solid and porous simulation results are actually identical.
"""

import numpy as np
from pathlib import Path


def main():
    results_base = Path(__file__).parent / "results"

    # Check both old and new results
    for suffix in ["", "_fixed"]:
        solid_dir = results_base / f"solid{suffix}"
        porous_dir = results_base / f"porous{suffix}"

        if not solid_dir.exists() or not porous_dir.exists():
            continue

        print(f"\n{'=' * 70}")
        print(f"COMPARING: solid{suffix} vs porous{suffix}")
        print(f"{'=' * 70}")

        # Load timeseries
        ts_solid = np.load(solid_dir / "timeseries.npz", allow_pickle=True)
        ts_porous = np.load(porous_dir / "timeseries.npz", allow_pickle=True)

        # Compare pressure at cell 0 over time
        p_solid = np.array([ts_solid['p'][i][0] for i in range(len(ts_solid['t']))])
        p_porous = np.array([ts_porous['p'][i][0] for i in range(len(ts_porous['t']))])

        t_solid = ts_solid['t']
        t_porous = ts_porous['t']

        # Find common time points
        n_compare = min(len(t_solid), len(t_porous), 100)

        print(f"\nPressure at cell 0 (first {n_compare} time points):")
        print(f"  t (µs)    p_solid (MPa)  p_porous (MPa)  Diff (MPa)")
        print(f"  {'-' * 55}")

        for i in range(0, n_compare, max(1, n_compare // 10)):
            diff = p_solid[i] - p_porous[i]
            print(f"  {t_solid[i]*1e6:7.1f}   {p_solid[i]/1e6:12.4f}   {p_porous[i]/1e6:12.4f}   {diff/1e6:10.6f}")

        # Check max pressure over time
        p_max_solid = np.array([ts_solid['p'][i].max() for i in range(len(ts_solid['t']))])
        p_max_porous = np.array([ts_porous['p'][i].max() for i in range(len(ts_porous['t']))])

        print(f"\nMax pressure difference:")
        print(f"  Mean |Δp_max| = {np.mean(np.abs(p_max_solid[:n_compare] - p_max_porous[:n_compare]))/1e6:.6f} MPa")
        print(f"  Max |Δp_max| = {np.max(np.abs(p_max_solid[:n_compare] - p_max_porous[:n_compare]))/1e6:.6f} MPa")

        # Check if essentially identical
        if np.max(np.abs(p_max_solid[:n_compare] - p_max_porous[:n_compare])) < 1e3:  # < 1 kPa
            print(f"\n  ** RESULTS ARE ESSENTIALLY IDENTICAL **")
        else:
            print(f"\n  Results are DIFFERENT")

        # Check piston velocity
        u_piston_solid = ts_solid['u_piston']
        u_piston_porous = ts_porous['u_piston']

        print(f"\nPiston velocity (grid motion):")
        print(f"  Mean |Δu_piston| = {np.mean(np.abs(u_piston_solid[:n_compare] - u_piston_porous[:n_compare])):.4f} m/s")

        # Check gas velocity if available
        if 'u_gas' in ts_solid.files and 'u_gas' in ts_porous.files:
            u_gas_solid = ts_solid['u_gas']
            u_gas_porous = ts_porous['u_gas']

            print(f"\nGas velocity (Riemann BC):")
            print(f"  Mean |Δu_gas| = {np.mean(np.abs(u_gas_solid[:n_compare] - u_gas_porous[:n_compare])):.2f} m/s")
            print(f"  This should be ~119.2 m/s for porous offset")
        else:
            print(f"\n  (u_gas not recorded in this dataset)")


if __name__ == "__main__":
    main()
