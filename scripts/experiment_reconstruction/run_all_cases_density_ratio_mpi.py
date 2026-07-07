"""
MPI Batch Runner for Experimental Reconstruction with Density Ratio BC
=======================================================================

Run multiple experimental reconstruction cases in parallel using MPI,
with the density ratio porous boundary condition.

The density ratio BC computes gas velocity from:
    u_g = ((sigma-1)/sigma) * U_f
where:
    sigma = rho_u / rho_b
    rho_u = unburned gas density from simulation (cell 0)
    rho_b = burned gas density from Cantera equilibrate('HP')

Usage:
------
    mpiexec -n <num_cases> python run_all_cases_density_ratio_mpi.py

Each MPI rank runs one or more cases. Cases are auto-detected from the
complete_data directories.

Available cases:
- C2H6: phi_0.8, phi_1.0, phi_1.3, phi_1.6
- C4H10: phi_0.8, phi_1.0, phi_1.3, phi_1.49, phi_1.6
"""

import sys
import os
import json
import argparse
from pathlib import Path
from datetime import datetime
import time

# Fix Windows MPI stdout encoding issue
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from experiment_data_loader import get_available_cases

# Base output directory - separate from solid piston results
BASE_OUTPUT_DIR = Path(__file__).parent / 'porous_results'

# Chemical mechanisms directory
MECH_DIR = Path(__file__).parent.parent.parent / 'src' / 'chemical_mechanisms'

# =============================================================================
# CASE CONFIGURATIONS
# =============================================================================

# Common simulation parameters
COMMON_PARAMS = {
    'n_cells': 1000,
    'cfl': 0.3,
    'av_linear': 0.3,
    'av_quad': 2.0,
    'velocity_scale': 1.0,
    'velocity_offset': 0.0,
    'velocity_min': None,
    # NOTE: dt_min was removed. Flooring dt above the porous drainage guard made
    # step_forward silently integrate a smaller dt than the script clock advanced
    # by, corrupting all recorded timelines (docs/DT_CLAMP_REVERT_PLAN.md).
}

# Mixture configurations per fuel type
# Experimental conditions: 0.3 atm (~30395 Pa), ambient temperature (~300 K)
# Pure oxygen experiments (not air)
# Mechanisms: Use full path from MECH_DIR or Cantera built-in name
MIXTURE_CONFIGS = {
    'c2h6': {
        'T': 300,           # Initial temperature [K]
        'P': 0.3 * 101325,  # Initial pressure [Pa] (0.3 atm)
        'fuel': 'C2H6',     # Cantera species name
        'oxidizer': 'O2:1', # Pure oxygen
        'mechanism': str(MECH_DIR / 'sandiego_mechCK.yaml'),
    },
    'c4h10': {
        'T': 300,           # Initial temperature [K]
        'P': 0.3 * 101325,  # Initial pressure [Pa] (0.3 atm)
        'fuel': 'C4H10',    # n-Butane (Cantera species name)
        'oxidizer': 'O2:1', # Pure oxygen
        'mechanism': str(MECH_DIR / 'sandiego_mechCK.yaml'),
    },
}

# Cases to run (set to None to run all, or list specific cases)
# Comment/uncomment to select which cases to run
CASES_TO_RUN = [
    ('c2h6', 'phi_0.8'),
    ('c2h6', 'phi_1.0'),
    ('c2h6', 'phi_1.3'),
    ('c2h6', 'phi_1.6'),
    ('c4h10', 'phi_0.8'),
    ('c4h10', 'phi_1.0'),
    ('c4h10', 'phi_1.3'),
    ('c4h10', 'phi_1.49'),
    ('c4h10', 'phi_1.6'),
]

# Domain lengths per case [meters]
# Based on maximum flame travel distance in experimental data
DOMAIN_LENGTHS = {
    # C2H6 cases
    ('c2h6', 'phi_0.8'): 2.4,
    ('c2h6', 'phi_1.0'): 2.5,
    ('c2h6', 'phi_1.3'): 1.8,
    ('c2h6', 'phi_1.6'): 2.1,
    # C4H10 cases
    ('c4h10', 'phi_0.8'): 2.1,
    ('c4h10', 'phi_1.0'): 1.7,
    ('c4h10', 'phi_1.3'): 1.3,
    ('c4h10', 'phi_1.49'): 1.3,
    ('c4h10', 'phi_1.6'): 1.4,
}

# Number of cells per case (optional override)
# If not specified, uses COMMON_PARAMS['n_cells']
N_CELLS = {
    # ('c2h6', 'phi_0.8'): 800,  # Example override
}

# Equivalence ratios
PHI_VALUES = {
    'phi_0.8': 0.8,
    'phi_1.0': 1.0,
    'phi_1.3': 1.3,
    'phi_1.49': 1.49,
    'phi_1.6': 1.6,
}

# =============================================================================
# END OF CASE CONFIGURATIONS
# =============================================================================


def run_single_case(case: dict, rank: int) -> dict:
    """
    Run a single simulation case with density ratio BC.

    Parameters
    ----------
    case : dict
        Case configuration with all parameters
    rank : int
        MPI rank running this case

    Returns
    -------
    dict
        Result dictionary with status and info
    """
    from run_experiment_reconstruction import run_reconstruction

    start_time = time.time()
    result = {
        'name': case['name'],
        'rank': rank,
        'success': False,
        'duration': 0,
        'error': None,
    }

    try:
        print(f"[Rank {rank}] START: {case['name']} (L={case['domain_length']:.1f}m, N={case['n_cells']}) [density ratio BC]")

        output_path = run_reconstruction(
            fuel=case['fuel'],
            phi_name=case['phi_name'],
            data_file=case['data_file'],
            domain_length=case['domain_length'],
            n_cells=case['n_cells'],
            cfl=case['cfl'],
            av_linear=case['av_linear'],
            av_quad=case['av_quad'],
            velocity_scale=case['velocity_scale'],
            velocity_offset=case['velocity_offset'],
            velocity_min=case['velocity_min'],
            output_dir=case['output_dir'],
            # Mixture configuration
            T_init=case['T_init'],
            P_init=case['P_init'],
            fuel_species=case['fuel_species'],
            oxidizer=case['oxidizer'],
            mechanism=case['mechanism'],
            # Enable density ratio BC
            use_density_ratio_bc=True,
        )

        result['duration'] = time.time() - start_time
        result['success'] = True
        result['output_dir'] = str(output_path)

        print(f"[Rank {rank}] DONE:  {case['name']} ({result['duration']:.1f}s)")

    except Exception as e:
        import traceback
        result['duration'] = time.time() - start_time
        result['error'] = str(e)
        result['traceback'] = traceback.format_exc()
        print(f"[Rank {rank}] ERROR: {case['name']}: {e}")
        traceback.print_exc()

    return result


def build_case_list(fuel_filter=None, phi_filter=None):
    """
    Build list of cases to run with full configuration.

    Parameters
    ----------
    fuel_filter : str, optional
        Only include cases for this fuel
    phi_filter : str, optional
        Only include cases for this phi

    Returns
    -------
    list of dict
        List of case configurations
    """
    base_dir = Path(__file__).parent
    all_cases = get_available_cases(str(base_dir))

    cases = []
    for (fuel, phi_name), data_file in sorted(all_cases.items()):
        # Filter by CASES_TO_RUN if defined
        if CASES_TO_RUN is not None and (fuel, phi_name) not in CASES_TO_RUN:
            continue
        if fuel_filter is not None and fuel != fuel_filter:
            continue
        if phi_filter is not None and phi_name != phi_filter:
            continue

        # Get mixture config for this fuel
        mixture = MIXTURE_CONFIGS.get(fuel, MIXTURE_CONFIGS['c2h6'])

        # Get domain length
        domain_length = DOMAIN_LENGTHS.get((fuel, phi_name), 3.0)

        # Get n_cells (case-specific or common)
        n_cells = N_CELLS.get((fuel, phi_name), COMMON_PARAMS['n_cells'])

        # Get phi value
        phi = PHI_VALUES.get(phi_name, 1.0)

        case = {
            'name': f"{fuel}_{phi_name}",
            'fuel': fuel,
            'phi_name': phi_name,
            'phi': phi,
            'data_file': str(data_file),
            'output_dir': str(BASE_OUTPUT_DIR / f"{fuel}_{phi_name}"),
            # Domain configuration
            'domain_length': domain_length,
            'n_cells': n_cells,
            # Mixture configuration
            'T_init': mixture['T'],
            'P_init': mixture['P'],
            'fuel_species': mixture['fuel'],
            'oxidizer': mixture['oxidizer'],
            'mechanism': mixture['mechanism'],
            # Solver parameters
            'cfl': COMMON_PARAMS['cfl'],
            'av_linear': COMMON_PARAMS['av_linear'],
            'av_quad': COMMON_PARAMS['av_quad'],
            'velocity_scale': COMMON_PARAMS['velocity_scale'],
            'velocity_offset': COMMON_PARAMS['velocity_offset'],
            'velocity_min': COMMON_PARAMS['velocity_min'],
            'dt_min': COMMON_PARAMS['dt_min'],
        }
        cases.append(case)

    return cases


def print_case_table(cases):
    """Print formatted table of cases."""
    print(f"\n{'Case':<20} {'Phi':>6} {'Domain':>8} {'Cells':>6} {'T0':>6} {'P0':>10}")
    print("-" * 60)
    for case in cases:
        print(f"{case['name']:<20} {case['phi']:>6.2f} {case['domain_length']:>7.1f}m "
              f"{case['n_cells']:>6d} {case['T_init']:>5.0f}K {case['P_init']/1e5:>9.3f}bar")


def main_mpi(fuel_filter=None, phi_filter=None):
    """Main MPI entry point."""
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        print("=" * 70)
        print("MPI BATCH RUNNER - Experimental Reconstruction (Density Ratio BC)")
        print("=" * 70)

        # Build case list
        cases = build_case_list(fuel_filter, phi_filter)
        n_cases = len(cases)

        if n_cases == 0:
            print("ERROR: No cases found matching filters")
            comm.Abort(1)

        print(f"\nDefined {n_cases} cases:")
        print_case_table(cases)

        print(f"\nRunning with {size} MPI ranks")
        print(f"Output directory: {BASE_OUTPUT_DIR}")
        print(f"BC type: density ratio porous")

        if size < n_cases:
            print(f"\nWARNING: More cases ({n_cases}) than ranks ({size})")
            print("Some ranks will run multiple cases sequentially.")
        elif size > n_cases:
            print(f"\nWARNING: More ranks ({size}) than cases ({n_cases})")
            print("Some ranks will be idle.")

        print("-" * 70)

        # Distribute cases to ranks (round-robin)
        rank_cases = [[] for _ in range(size)]
        for i, case in enumerate(cases):
            rank_cases[i % size].append(case)
    else:
        rank_cases = None

    # Scatter cases to all ranks
    my_cases = comm.scatter(rank_cases, root=0)

    # Each rank runs its assigned cases
    my_results = []
    for case in my_cases:
        result = run_single_case(case, rank)
        my_results.append(result)

    # Gather results back to rank 0
    all_results = comm.gather(my_results, root=0)

    # Rank 0 prints summary
    if rank == 0:
        results = [r for rank_results in all_results for r in rank_results]

        print("\n" + "=" * 70)
        print("BATCH COMPLETE")
        print("=" * 70)

        successful = sum(1 for r in results if r['success'])
        failed = len(results) - successful

        print(f"\nTotal runs: {len(results)}")
        print(f"Successful: {successful}")
        print(f"Failed:     {failed}")

        if successful > 0:
            print("\nSuccessful runs:")
            for r in results:
                if r['success']:
                    print(f"  - {r['name']}: {r['duration']:.1f}s")

        if failed > 0:
            print("\nFailed runs:")
            for r in results:
                if not r['success']:
                    err_msg = r['error'][:80] + "..." if len(r['error']) > 80 else r['error']
                    print(f"  - {r['name']} (rank {r['rank']}): {err_msg}")

        print(f"\nResults saved to: {BASE_OUTPUT_DIR}/")

        # Save batch summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'mpi_ranks': size,
            'n_cases': len(results),
            'n_success': successful,
            'n_failed': failed,
            'bc_type': 'density_ratio_porous',
            'results': results,
        }
        summary_file = BASE_OUTPUT_DIR / "batch_summary_mpi.json"
        BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"Summary saved: {summary_file}")


def main_serial(fuel_filter=None, phi_filter=None):
    """Serial execution for testing without MPI."""
    print("=" * 70)
    print("SERIAL BATCH RUNNER - Experimental Reconstruction (Density Ratio BC)")
    print("=" * 70)

    # Build case list
    cases = build_case_list(fuel_filter, phi_filter)
    n_cases = len(cases)

    if n_cases == 0:
        print("ERROR: No cases found matching filters")
        return

    print(f"\nDefined {n_cases} cases:")
    print_case_table(cases)

    print(f"\nOutput directory: {BASE_OUTPUT_DIR}")
    print(f"BC type: density ratio porous")
    print("-" * 70)

    results = []
    for i, case in enumerate(cases):
        print(f"\n[{i+1}/{n_cases}] Running {case['name']}...")
        result = run_single_case(case, rank=0)
        results.append(result)

    print("\n" + "=" * 70)
    print("BATCH COMPLETE")
    print("=" * 70)

    successful = sum(1 for r in results if r['success'])
    failed = len(results) - successful

    print(f"\nTotal runs: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed:     {failed}")

    print(f"\nResults saved to: {BASE_OUTPUT_DIR}/")

    # Save batch summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'mpi_ranks': 1,
        'n_cases': len(results),
        'n_success': successful,
        'n_failed': failed,
        'bc_type': 'density_ratio_porous',
        'results': results,
    }
    summary_file = BASE_OUTPUT_DIR / "batch_summary.json"
    BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Summary saved: {summary_file}")


def main_list():
    """List all cases without running."""
    print("=" * 70)
    print("CASE LIST - Experimental Reconstruction (Density Ratio BC)")
    print("=" * 70)

    cases = build_case_list()
    print_case_table(cases)
    print(f"\nTotal: {len(cases)} cases")
    print(f"Output directory: {BASE_OUTPUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MPI-parallel experimental reconstruction with density ratio BC"
    )
    parser.add_argument("--fuel", type=str, choices=['c2h6', 'c4h10'], default=None,
                        help="Run only cases for specified fuel")
    parser.add_argument("--phi", type=str, default=None,
                        help="Run only cases for specified phi (e.g., phi_1.0)")
    parser.add_argument("--serial", action="store_true",
                        help="Force serial execution")
    parser.add_argument("--list", action="store_true",
                        help="List cases without running")

    args = parser.parse_args()

    if args.list:
        main_list()
    elif args.serial:
        main_serial(args.fuel, args.phi)
    else:
        # Check if running with MPI
        try:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
            if comm.Get_size() > 1:
                main_mpi(args.fuel, args.phi)
            else:
                # Single rank - could be MPI with 1 process or no MPI
                print("Running in serial mode (use mpiexec -n N for parallel)")
                main_serial(args.fuel, args.phi)
        except ImportError:
            print("mpi4py not available, running in serial mode")
            main_serial(args.fuel, args.phi)
