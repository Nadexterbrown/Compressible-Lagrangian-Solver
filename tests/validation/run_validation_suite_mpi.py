"""
MPI Validation Suite: Pele + Experiment Reconstruction (post dt-clamp fix)
==========================================================================

Runs the six validation cases for the timestep-clamp / cell-collapse fix
(docs/PISTON_CELL_COLLAPSE_FIX_PLAN.md Phase 5) in parallel:

  Pele reconstruction (scripts/pele_reconstruction):
    1. pele_solid              - solid piston BC
    2. pele_fixed_offset       - porous, gas velocity = piston - 119.2 m/s (CJ offset)
    3. pele_consumption_speed  - porous, gas offset from flame burning velocity data
    4. pele_density_ratio      - porous, u_g = ((sigma-1)/sigma) * U_f

  Experiment reconstruction (scripts/experiment_reconstruction), C4H10 phi=0.8:
    5. exp_c4h10_phi0.8_solid
    6. exp_c4h10_phi0.8_density_ratio

All cases run WITHOUT dt_min (removed - see docs/DT_CLAMP_REVERT_PLAN.md),
with the honest solver clock and boundary-cell absorption active. Outputs go
to tests/validation/results/<case>/ - existing result directories elsewhere
are never touched.

Usage:
------
    mpiexec -n 6 python run_validation_suite_mpi.py          # one rank per case
    python run_validation_suite_mpi.py --serial              # sequential, no MPI
    python run_validation_suite_mpi.py --cases pele_solid exp_c4h10_phi0.8_solid

After the batch:
- check each case's config.json: clamped_steps MUST be 0
- run compare_pele_accuracy.py to validate the Pele cases against the
  reference results in scripts/pele_reconstruction/results/
"""

import sys
import json
import argparse
import time
from datetime import datetime
from pathlib import Path

# Fix Windows MPI stdout encoding issue
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

ROOT = Path(__file__).parent.parent.parent
PELE_DIR = ROOT / "scripts" / "pele_reconstruction"
EXP_DIR = ROOT / "scripts" / "experiment_reconstruction"
MECH_DIR = ROOT / "src" / "chemical_mechanisms"
OUT_DIR = Path(__file__).parent / "results"

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(PELE_DIR))
sys.path.insert(0, str(EXP_DIR))

# =============================================================================
# CASE CONFIGURATIONS
# =============================================================================

# Shared Pele parameters (match pele_porous_density_ratio, May 2026)
PELE_COMMON = dict(
    data_source="pele",
    data_path=str(PELE_DIR / "pele_data" / "truncated_raw_data"),
    domain_length=2.3,
    n_cells=1000,
    cfl=0.3,
    av_linear=0.3,
    av_quad=2.0,
    n_records=1000,
)

# Shared experiment parameters (match run_all_cases_density_ratio_mpi, no dt_min)
EXP_COMMON = dict(
    fuel="c4h10",
    phi_name="phi_0.8",
    domain_length=2.1,
    n_cells=1000,
    cfl=0.3,
    av_linear=0.3,
    av_quad=2.0,
    T_init=300,
    P_init=0.3 * 101325,
    fuel_species="C4H10",
    oxidizer="O2:1",
    mechanism=str(MECH_DIR / "sandiego_mechCK.yaml"),
)

CASES = [
    {"name": "pele_solid", "kind": "pele", "params": {}},
    {"name": "pele_fixed_offset", "kind": "pele",
     "params": {"gas_velocity_offset": -119.2}},
    {"name": "pele_consumption_speed", "kind": "pele",
     "params": {"gas_offset_data": "Flame Burning Velocity [m / s]"}},
    {"name": "pele_density_ratio", "kind": "pele",
     "params": {"gas_velocity_density_ratio": True}},
    {"name": "exp_c4h10_phi0.8_solid", "kind": "experiment",
     "params": {"use_density_ratio_bc": False}},
    {"name": "exp_c4h10_phi0.8_density_ratio", "kind": "experiment",
     "params": {"use_density_ratio_bc": True}},
]

# =============================================================================
# CASE RUNNER
# =============================================================================


def run_single_case(case: dict, rank: int) -> dict:
    """Run one validation case; returns a result record."""
    start = time.time()
    result = {
        "name": case["name"],
        "kind": case["kind"],
        "rank": rank,
        "success": False,
        "duration": 0.0,
        "error": None,
        "clamped_steps": None,
        "n_steps": None,
        "mass_leaked": None,
    }
    output_dir = OUT_DIR / case["name"]

    try:
        print(f"[Rank {rank}] START: {case['name']} ({case['kind']})", flush=True)

        if case["kind"] == "pele":
            from run_pele_reconstruction import run_reconstruction as run_pele
            run_pele(output_dir=str(output_dir), **PELE_COMMON, **case["params"])
        else:
            from run_experiment_reconstruction import run_reconstruction as run_exp
            run_exp(output_dir=str(output_dir), **EXP_COMMON, **case["params"])

        result["success"] = True

        # Pull diagnostics from the saved config (experiment driver records
        # clamped_steps/mass_leaked; the pele driver records a subset)
        cfg_file = output_dir / "config.json"
        if cfg_file.exists():
            cfg = json.loads(cfg_file.read_text())
            for key in ("clamped_steps", "n_steps", "mass_leaked"):
                result[key] = cfg.get(key)

        print(f"[Rank {rank}] DONE:  {case['name']} "
              f"({time.time()-start:.1f}s, clamped={result['clamped_steps']})", flush=True)

    except Exception as e:
        import traceback
        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()
        print(f"[Rank {rank}] ERROR: {case['name']}: {e}", flush=True)
        traceback.print_exc()

    result["duration"] = time.time() - start
    return result


def summarize(results, size):
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    print("\n" + "=" * 70)
    print("VALIDATION SUITE COMPLETE")
    print("=" * 70)
    print(f"Total: {len(results)} | OK: {len(successful)} | Failed: {len(failed)}")

    if successful:
        print(f"\n{'case':<34}{'wall [s]':>10}{'steps':>10}{'clamped':>9}{'leaked [kg]':>13}")
        for r in successful:
            leaked = f"{r['mass_leaked']:.4e}" if r["mass_leaked"] is not None else "-"
            print(f"{r['name']:<34}{r['duration']:>10.1f}"
                  f"{str(r['n_steps'] or '-'):>10}{str(r['clamped_steps'] if r['clamped_steps'] is not None else '-'):>9}"
                  f"{leaked:>13}")

    clamped_cases = [r for r in successful if (r["clamped_steps"] or 0) > 0]
    if clamped_cases:
        print("\nWARNING: nonzero clamped steps detected (should be 0 after the fix):")
        for r in clamped_cases:
            print(f"  - {r['name']}: {r['clamped_steps']}")

    if failed:
        print("\nFailed runs:")
        for r in failed:
            print(f"  - {r['name']} (rank {r['rank']}): {r['error']}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "timestamp": datetime.now().isoformat(),
        "mpi_ranks": size,
        "results": results,
    }
    summary_file = OUT_DIR / "validation_summary.json"
    summary_file.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nSummary saved: {summary_file}")


def main_mpi(cases):
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        print("=" * 70)
        print("MPI VALIDATION SUITE - Pele + Experiment Reconstruction")
        print("=" * 70)
        print(f"{len(cases)} cases on {size} ranks -> {OUT_DIR}")
        rank_cases = [[] for _ in range(size)]
        for i, case in enumerate(cases):
            rank_cases[i % size].append(case)
    else:
        rank_cases = None

    my_cases = comm.scatter(rank_cases, root=0)
    my_results = [run_single_case(c, rank) for c in my_cases]
    all_results = comm.gather(my_results, root=0)

    if rank == 0:
        summarize([r for rr in all_results for r in rr], size)


def main_serial(cases):
    print("SERIAL VALIDATION SUITE (no MPI)")
    results = [run_single_case(c, 0) for c in cases]
    summarize(results, 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the post-fix validation suite")
    parser.add_argument("--serial", action="store_true", help="Run without MPI")
    parser.add_argument("--cases", nargs="+", default=None,
                        help=f"Subset of cases. Available: {[c['name'] for c in CASES]}")
    args = parser.parse_args()

    selected = CASES
    if args.cases:
        unknown = set(args.cases) - {c["name"] for c in CASES}
        if unknown:
            sys.exit(f"Unknown case(s): {sorted(unknown)}")
        selected = [c for c in CASES if c["name"] in args.cases]

    if args.serial:
        main_serial(selected)
    else:
        main_mpi(selected)
