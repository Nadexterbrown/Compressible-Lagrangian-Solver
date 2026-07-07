"""
Regenerate post-processing (X-T diagrams, animations, snapshots, velocity
comparison) for validation cases whose plotting crashed, WITHOUT re-running
the simulations - everything is rebuilt from the saved timeseries.npz.

Background: the 7/6 validation batch completed all six simulations cleanly
(clamped_steps=0 everywhere), but the four porous cases crashed in
plot_xt_diagrams because pcolormesh rejects the NaN coordinate padding that
boundary-cell absorption introduces. The plot code is fixed (padded
coordinates are collapsed onto the piston face); this script re-runs just
the post-processing.

Usage:
    python regen_postprocessing.py                     # all incomplete cases
    python regen_postprocessing.py --cases pele_density_ratio
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
RES = HERE / "results"

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "pele_reconstruction"))
sys.path.insert(0, str(ROOT / "scripts" / "experiment_reconstruction"))

import run_pele_reconstruction as pele_mod
import run_experiment_reconstruction as exp_mod

CASES = {
    "pele_solid": "pele",
    "pele_fixed_offset": "pele",
    "pele_consumption_speed": "pele",
    "pele_density_ratio": "pele",
    "exp_c4h10_phi0.8_solid": "experiment",
    "exp_c4h10_phi0.8_density_ratio": "experiment",
}


def load(case_dir: Path):
    ts = np.load(case_dir / "timeseries.npz", allow_pickle=True)
    saved = {k: ts[k] for k in ts.files if k != "config"}
    cfg = json.loads((case_dir / "config.json").read_text())
    return saved, cfg


def regen(name: str, kind: str):
    d = RES / name
    if not (d / "timeseries.npz").exists():
        print(f"[skip] {name}: no timeseries.npz")
        return
    mod = pele_mod if kind == "pele" else exp_mod
    print(f"\n=== {name} ===")
    saved, cfg = load(d)
    mod.plot_xt_diagrams(saved, cfg, str(d), use_mass_coord=False)
    mod.plot_xt_diagrams(saved, cfg, str(d), use_mass_coord=True)
    mod.create_animation(saved, cfg, str(d / "animation_x.mp4"), use_mass_coord=False)
    mod.create_animation(saved, cfg, str(d / "animation_m.mp4"), use_mass_coord=True)
    mod.save_snapshots(saved, cfg, str(d), snapshot_interval=1)
    if kind == "experiment":
        from experiment_data_loader import ExperimentalDataLoader
        data = ExperimentalDataLoader(cfg["data_source"]).load()
        exp_mod.plot_velocity_comparison(saved, data, str(d / "velocity_comparison.png"))
    print(f"[done] {name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Regenerate validation post-processing")
    parser.add_argument("--cases", nargs="+", default=None,
                        help=f"Subset of cases. Available: {list(CASES)}")
    args = parser.parse_args()

    names = args.cases or [n for n in CASES
                           if not (RES / n / "T_xt.png").exists()
                           and (RES / n / "timeseries.npz").exists()]
    if not names:
        print("Nothing to regenerate (all cases have T_xt.png).")
    for n in names:
        if n not in CASES:
            sys.exit(f"Unknown case: {n}")
        regen(n, CASES[n])
