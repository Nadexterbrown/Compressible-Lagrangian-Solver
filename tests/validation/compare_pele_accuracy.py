"""
Accuracy / validation comparison: new Pele reconstruction runs vs references
=============================================================================

Compares the Pele cases produced by run_validation_suite_mpi.py (in
tests/validation/results/) against the corresponding pre-fix reference
results in scripts/pele_reconstruction/results/:

    new case                   reference
    ------------------------   -----------------------------------------
    pele_solid                 solid
    pele_fixed_offset          pele_porous_cj_extended
    pele_consumption_speed     pele_burning_vel_extended
    pele_density_ratio         pele_porous_density_ratio

The references were verified clean of the dt-clamp clock desync (deficits
<= 0.2%, see docs/DT_CLAMP_REVERT_PLAN.md), so the new runs should agree
with them up to (a) the unified CFL formula, (b) boundary-cell absorption
replacing merge-split near the piston face, and (c) any config differences
(printed as caveats - e.g. the solid reference used domain_length 2.5 m).

Metrics per pair:
- piston node trajectory: interpolated onto the common time base;
  max and final absolute difference [mm]
- final common-time p/T/rho profiles interpolated onto a common x grid:
  median and p95 relative difference
- cumulative mass_leaked (porous cases)
- step count and clamped_steps

Usage:
    python compare_pele_accuracy.py            # all four pairs
    python compare_pele_accuracy.py --pairs pele_density_ratio

Writes tests/validation/results/pele_accuracy_report.{json,md}.
"""

import sys
import json
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
NEW_DIR = HERE / "results"
REF_DIR = ROOT / "scripts" / "pele_reconstruction" / "results"

PAIRS = {
    "pele_solid": "solid",
    "pele_fixed_offset": "pele_porous_cj_extended",
    "pele_consumption_speed": "pele_burning_vel_extended",
    "pele_density_ratio": "pele_porous_density_ratio",
}

CONFIG_KEYS = ["domain_length", "n_cells", "cfl", "av_linear", "av_quad",
               "data_source", "gas_velocity_offset", "gas_offset_data"]


def load_case(path: Path):
    ts = np.load(path / "timeseries.npz", allow_pickle=True)
    cfg = json.loads((path / "config.json").read_text())
    return ts, cfg


def node_series(ts):
    """Piston node position series (x_piston if recorded, else x[:,0] with
    NaN-padding awareness: first finite entry per row)."""
    if "x_piston" in ts:
        return np.asarray(ts["t"]), np.asarray(ts["x_piston"])
    x = np.asarray(ts["x"], dtype=float)
    xp = np.empty(len(x))
    for i, row in enumerate(x):
        fin = np.isfinite(row)
        xp[i] = row[fin][0] if fin.any() else np.nan
    return np.asarray(ts["t"]), xp


def profile_at(ts, key, t_target):
    """Cell-centered profile (x_centers, values) at the snapshot nearest t_target,
    restricted to finite (active) cells."""
    t = np.asarray(ts["t"])
    i = int(np.argmin(np.abs(t - t_target)))
    x = np.asarray(ts["x"][i], dtype=float)
    v = np.asarray(ts[key][i], dtype=float)
    xc = 0.5 * (x[:-1] + x[1:])
    fin = np.isfinite(xc) & np.isfinite(v)
    return float(t[i]), xc[fin], v[fin]


def compare_pair(new_name: str, ref_name: str) -> dict:
    new_path, ref_path = NEW_DIR / new_name, REF_DIR / ref_name
    rec = {"new": new_name, "reference": ref_name, "ok": None, "caveats": [], "metrics": {}}

    for tag, p in [("new", new_path), ("reference", ref_path)]:
        if not (p / "timeseries.npz").exists():
            rec["ok"] = False
            rec["caveats"].append(f"missing {tag} results at {p}")
            return rec

    ts_n, cfg_n = load_case(new_path)
    ts_r, cfg_r = load_case(ref_path)

    # Config caveats
    for k in CONFIG_KEYS:
        a, b = cfg_n.get(k), cfg_r.get(k)
        if a != b and not (a is None and b is None):
            rec["caveats"].append(f"config {k}: new={a} vs ref={b}")

    def pct_change(new, ref):
        """Signed percent change of `new` w.r.t. the reference."""
        if new is None or ref is None or ref == 0:
            return None
        return float(100.0 * (new - ref) / ref)

    # Diagnostics
    rec["metrics"]["n_steps"] = {"new": cfg_n.get("n_steps"), "ref": cfg_r.get("n_steps"),
                                 "pct_change": pct_change(cfg_n.get("n_steps"), cfg_r.get("n_steps"))}
    rec["metrics"]["clamped_steps_new"] = cfg_n.get("clamped_steps")
    if "mass_leaked" in cfg_n or "mass_leaked" in cfg_r:
        rec["metrics"]["mass_leaked"] = {"new": cfg_n.get("mass_leaked"),
                                         "ref": cfg_r.get("mass_leaked"),
                                         "pct_change": pct_change(cfg_n.get("mass_leaked"),
                                                                  cfg_r.get("mass_leaked"))}

    # Piston node trajectory over the common time window
    t_n, x_n = node_series(ts_n)
    t_r, x_r = node_series(ts_r)
    t_lo, t_hi = max(t_n[0], t_r[0]), min(t_n[-1], t_r[-1])
    t_common = np.linspace(t_lo, t_hi, 500)
    dn = np.interp(t_common, t_n, x_n)
    dr = np.interp(t_common, t_r, x_r)
    node_diff = np.abs(dn - dr) * 1e3  # mm
    rec["metrics"]["node_max_diff_mm"] = float(np.max(node_diff))
    rec["metrics"]["node_final_diff_mm"] = float(node_diff[-1])
    rec["metrics"]["node_final_pct_change"] = pct_change(dn[-1], dr[-1])
    rec["metrics"]["common_t_end_ms"] = float(t_hi * 1e3)

    # Final common-time profiles on a shared x grid
    for key in ("p", "T", "rho"):
        t_used, xc_n, v_n = profile_at(ts_n, key, t_hi)
        _, xc_r, v_r = profile_at(ts_r, key, t_hi)
        x_lo, x_hi = max(xc_n[0], xc_r[0]), min(xc_n[-1], xc_r[-1])
        xg = np.linspace(x_lo, x_hi, 800)
        vn = np.interp(xg, xc_n, v_n)
        vr = np.interp(xg, xc_r, v_r)
        rel = np.abs(vn - vr) / np.maximum(np.abs(vr), 1e-30)
        rec["metrics"][f"{key}_median_rel"] = float(np.median(rel))
        rec["metrics"][f"{key}_p95_rel"] = float(np.percentile(rel, 95))
        # Signed percent change w.r.t. the reference profile: median of the
        # pointwise change (bulk shift direction) and change of the profile
        # mean (integral quantity)
        signed_pct = 100.0 * (vn - vr) / np.maximum(np.abs(vr), 1e-30)
        rec["metrics"][f"{key}_median_pct_change"] = float(np.median(signed_pct))
        rec["metrics"][f"{key}_mean_pct_change"] = pct_change(float(np.mean(vn)), float(np.mean(vr)))

    # Pass criteria: node within 0.5% of travel, median profiles within 5%,
    # zero clamped steps in the new run.
    travel = max(abs(dr[-1] - dr[0]) * 1e3, 1e-9)
    checks = {
        "node_within_0.5pct": bool(rec["metrics"]["node_max_diff_mm"] <= 0.005 * travel + 0.5),
        "profiles_within_5pct": bool(all(rec["metrics"][f"{k}_median_rel"] <= 0.05
                                         for k in ("p", "T", "rho"))),
        "zero_clamped_steps": bool((cfg_n.get("clamped_steps") or 0) == 0),
    }
    rec["checks"] = checks
    rec["ok"] = all(checks.values())
    return rec


def main():
    parser = argparse.ArgumentParser(description="Compare new Pele runs vs references")
    parser.add_argument("--pairs", nargs="+", default=None,
                        help=f"Subset of new-case names. Available: {list(PAIRS)}")
    args = parser.parse_args()

    selected = PAIRS
    if args.pairs:
        unknown = set(args.pairs) - set(PAIRS)
        if unknown:
            sys.exit(f"Unknown pair(s): {sorted(unknown)}")
        selected = {k: PAIRS[k] for k in args.pairs}

    records = []
    for new_name, ref_name in selected.items():
        print(f"\n=== {new_name}  vs  {ref_name} ===")
        rec = compare_pair(new_name, ref_name)
        records.append(rec)
        for c in rec["caveats"]:
            print(f"  caveat: {c}")
        if rec["metrics"]:
            m = rec["metrics"]
            if "node_max_diff_mm" in m:
                print(f"  node: final pct change={m['node_final_pct_change']:+.4f}% "
                      f"(max diff={m['node_max_diff_mm']:.3f} mm; "
                      f"common window ends {m['common_t_end_ms']:.3f} ms)")
            for k in ("p", "T", "rho"):
                if f"{k}_median_pct_change" in m:
                    print(f"  {k:3s}: median pct change={m[f'{k}_median_pct_change']:+.4f}%, "
                          f"mean pct change={m[f'{k}_mean_pct_change']:+.4f}%  "
                          f"(|median rel|={m[f'{k}_median_rel']:.2e}, p95={m[f'{k}_p95_rel']:.2e})")
            if "mass_leaked" in m:
                ml = m["mass_leaked"]
                pct = f"{ml['pct_change']:+.3f}%" if ml.get("pct_change") is not None else "n/a"
                print(f"  mass_leaked: pct change={pct} (new={ml['new']} ref={ml['ref']})")
            if "n_steps" in m and m["n_steps"].get("pct_change") is not None:
                print(f"  n_steps: pct change={m['n_steps']['pct_change']:+.1f}% "
                      f"(new={m['n_steps']['new']} ref={m['n_steps']['ref']})")
        if rec["ok"] is not None:
            print(f"  RESULT: {'PASS' if rec['ok'] else 'FAIL'}  {rec.get('checks', '')}")

    n_fail = sum(1 for r in records if not r["ok"])
    print("\n" + "=" * 60)
    print(f"ACCURACY VALIDATION: {len(records) - n_fail}/{len(records)} pairs passed")

    NEW_DIR.mkdir(parents=True, exist_ok=True)
    report = {"timestamp": datetime.now().isoformat(), "pairs": records}
    (NEW_DIR / "pele_accuracy_report.json").write_text(json.dumps(report, indent=2))

    lines = [f"# Pele reconstruction accuracy report ({report['timestamp']})", ""]
    for r in records:
        lines.append(f"## {r['new']} vs {r['reference']}: "
                     f"{'PASS' if r['ok'] else 'FAIL'}")
        for c in r["caveats"]:
            lines.append(f"- caveat: {c}")
        for k, v in r["metrics"].items():
            lines.append(f"- {k}: {v}")
        lines.append("")
    (NEW_DIR / "pele_accuracy_report.md").write_text("\n".join(lines))
    print(f"Report saved: {NEW_DIR / 'pele_accuracy_report.md'}")

    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
