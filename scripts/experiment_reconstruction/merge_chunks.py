"""
Merge chunk files into a single timeseries.npz file and generate all outputs.

Use this script to consolidate incremental output chunks from an interrupted
or completed simulation run, and produce all standard output files (plots,
animations, snapshots).

Usage:
    python merge_chunks.py porous_results/c2h6_phi_1.0
    python merge_chunks.py porous_results/c2h6_phi_1.0 --delete-chunks
    python merge_chunks.py porous_results/c2h6_phi_1.0 --skip-animations
"""

import sys
import argparse
import json
import numpy as np
from pathlib import Path

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from run_experiment_reconstruction import (
    plot_final_state,
    plot_xt_diagrams,
    create_animation,
    save_snapshots,
    plot_velocity_comparison,
)
from experiment_data_loader import ExperimentalDataLoader


def merge_chunks(
    case_dir: Path,
    delete_chunks: bool = False,
    skip_animations: bool = False,
    skip_plots: bool = False,
) -> Path:
    """
    Merge all chunk files in a case directory into timeseries.npz and generate outputs.

    Parameters
    ----------
    case_dir : Path
        Path to case directory containing chunks/ subdirectory
    delete_chunks : bool
        If True, delete chunk files after successful merge
    skip_animations : bool
        If True, skip generating animations (faster)
    skip_plots : bool
        If True, skip all plot generation (only merge)

    Returns
    -------
    Path
        Path to merged timeseries.npz file
    """
    chunk_dir = case_dir / "chunks"

    if not chunk_dir.exists():
        raise FileNotFoundError(f"No chunks directory found: {chunk_dir}")

    chunk_files = sorted(chunk_dir.glob("chunk_*.npz"))

    if not chunk_files:
        raise FileNotFoundError(f"No chunk files found in: {chunk_dir}")

    print(f"Merging {len(chunk_files)} chunks from: {case_dir}")

    # Consolidate chunks
    keys = ['t', 'x', 'rho', 'u', 'p', 'e', 'T', 's', 'u_piston', 'T_b', 'rho_b']
    consolidated = {k: [] for k in keys}

    for chunk_file in chunk_files:
        print(f"  Loading: {chunk_file.name}")
        chunk = np.load(chunk_file, allow_pickle=True)
        for k in keys:
            if k in chunk:
                consolidated[k].append(chunk[k])
        chunk.close()

    # Concatenate arrays
    saved_data = {}
    for k, arrays in consolidated.items():
        if arrays:
            saved_data[k] = np.concatenate(arrays, axis=0)
        else:
            saved_data[k] = np.array([])

    n_snapshots = len(saved_data['t'])
    print(f"\nTotal snapshots: {n_snapshots}")

    if n_snapshots > 0:
        t_start = saved_data['t'][0] * 1e3
        t_end = saved_data['t'][-1] * 1e3
        print(f"Time range: {t_start:.3f} - {t_end:.3f} ms")

    # Load config
    config_file = case_dir / "config.json"
    if config_file.exists():
        with open(config_file) as f:
            config = json.load(f)
        print(f"Config loaded from: {config_file.name}")
    else:
        # Create minimal config from directory name
        dir_name = case_dir.name  # e.g., "c2h6_phi_1.0"
        parts = dir_name.split('_')
        fuel = parts[0] if parts else "unknown"
        phi_name = '_'.join(parts[1:]) if len(parts) > 1 else "phi_1.0"

        config = {
            "fuel": fuel,
            "phi_name": phi_name,
            "n_cells": len(saved_data['rho'][0]) if n_snapshots > 0 else 600,
            "domain_length": float(saved_data['x'][0][-1] - saved_data['x'][0][0]) if n_snapshots > 0 else 3.0,
            "rho_init": float(saved_data['rho'][0][0]) if n_snapshots > 0 else 1.0,
        }
        print("Warning: No config.json found, using defaults")

    # Save merged timeseries
    output_file = case_dir / "timeseries.npz"
    np.savez(output_file, **saved_data, config=config)
    print(f"\nSaved: {output_file}")

    # Save config if it didn't exist
    if not config_file.exists():
        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)
        print(f"Saved: {config_file}")

    # Generate outputs
    if not skip_plots and n_snapshots > 0:
        print(f"\nGenerating outputs...")

        # Final state plots
        print("  Generating final state plots...")
        plot_final_state(saved_data, config, str(case_dir / "final_state_x.png"), use_mass_coord=False)
        plot_final_state(saved_data, config, str(case_dir / "final_state_m.png"), use_mass_coord=True)
        print(f"    Saved: final_state_x.png, final_state_m.png")

        # X-T diagrams
        print("  Generating X-T diagrams...")
        plot_xt_diagrams(saved_data, config, str(case_dir), use_mass_coord=False)
        plot_xt_diagrams(saved_data, config, str(case_dir), use_mass_coord=True)

        # Animations
        if not skip_animations:
            print("  Generating animations...")
            create_animation(saved_data, config, str(case_dir / "animation_x.mp4"), use_mass_coord=False)
            create_animation(saved_data, config, str(case_dir / "animation_m.mp4"), use_mass_coord=True)
        else:
            print("  Skipping animations (--skip-animations)")

        # Snapshots
        print("  Saving snapshots...")
        save_snapshots(saved_data, config, str(case_dir), snapshot_interval=1)

        # Velocity comparison plot (requires trajectory data)
        if 'data_source' in config:
            try:
                data_file = Path(config['data_source'])
                if data_file.exists():
                    print("  Generating velocity comparison plot...")
                    loader = ExperimentalDataLoader(str(data_file))
                    traj_data = loader.load()
                    plot_velocity_comparison(saved_data, traj_data, str(case_dir / "velocity_comparison.png"))
                    print(f"    Saved: velocity_comparison.png")
                else:
                    print(f"  Skipping velocity comparison (data file not found: {data_file})")
            except Exception as e:
                print(f"  Skipping velocity comparison (error: {e})")
        else:
            print("  Skipping velocity comparison (no data_source in config)")

        print(f"\nAll outputs saved to: {case_dir}")

    # Optionally delete chunks
    if delete_chunks:
        print(f"\nDeleting {len(chunk_files)} chunk files...")
        for chunk_file in chunk_files:
            chunk_file.unlink()
        # Remove empty chunks directory
        if not any(chunk_dir.iterdir()):
            chunk_dir.rmdir()
            print(f"Removed empty directory: {chunk_dir}")
        print("Chunks deleted.")

    return output_file


def main():
    parser = argparse.ArgumentParser(
        description="Merge chunk files into timeseries.npz and generate outputs"
    )
    parser.add_argument(
        "case_dir",
        type=str,
        help="Path to case directory (e.g., porous_results/c2h6_phi_1.0)"
    )
    parser.add_argument(
        "--delete-chunks",
        action="store_true",
        help="Delete chunk files after successful merge"
    )
    parser.add_argument(
        "--skip-animations",
        action="store_true",
        help="Skip generating animations (faster)"
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip all plot generation (only merge chunks)"
    )

    args = parser.parse_args()

    case_dir = Path(args.case_dir)
    if not case_dir.exists():
        print(f"Error: Directory not found: {case_dir}")
        return 1

    try:
        merge_chunks(
            case_dir,
            delete_chunks=args.delete_chunks,
            skip_animations=args.skip_animations,
            skip_plots=args.skip_plots,
        )
        return 0
    except Exception as e:
        import traceback
        print(f"Error: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
