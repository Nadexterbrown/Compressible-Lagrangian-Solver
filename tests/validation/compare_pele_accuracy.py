"""
Accuracy / validation comparison: new Pele reconstruction runs vs references
=============================================================================

Compares the Pele cases produced by run_validation_suite_mpi.py (in
tests/validation/results/) against the pre-fix reference results in
scripts/pele_reconstruction/results/:

    new case                   reference
    ------------------------   -----------------------------------------
    pele_solid                 solid
    pele_fixed_offset          pele_porous_cj_extended
    pele_consumption_speed     pele_burning_vel_extended
    pele_density_ratio         pele_porous_density_ratio

All comparison quantities are SIGNED PERCENT DIFFERENCE w.r.t. the
reference: (new - ref) / ref * 100.

Reported per pair:
- n_steps            : cost of the run
- node final         : piston node position at the end of the common window
- piston-face state  : p/T/rho of the first active cell (final value at the
                       common end time, and median over the common window)
- domain statistics  : spatial mean/std/min/max of p/T/rho over the active
                       domain at the final common time, each compared as a
                       percent difference of the statistic

Usage:
    python compare_pele_accuracy.py
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

VARS = ("p", "T", "rho")
CONFIG_KEYS = ["domain_length", "n_cells", "cfl", "data_source"]


def pct(new, ref):
    """Signed percent difference of `new` w.r.t. `ref`."""
    if new is None or ref is None or ref == 0 or not np.isfinite(ref):
        return None
    return float(100.0 * (new - ref) / ref)


def load_case(path: Path):
    ts = np.load(path / "timeseries.npz", allow_pickle=True)
    cfg = json.loads((path / "config.json").read_text())
    return ts, cfg


def node_series(ts):
    """Piston node position series (first finite face per row)."""
    if "x_piston" in ts:
        return np.asarray(ts["t"]), np.asarray(ts["x_piston"])
    x = np.asarray(ts["x"], dtype=float)
    xp = np.empty(len(x))
    for i, row in enumerate(x):
        fin = np.isfinite(row)
        xp[i] = row[fin][0] if fin.any() else np.nan
    return np.asarray(ts["t"]), xp


def face_series(ts, key):
    """Piston-face (first active cell) value of `key` over time."""
    t = np.asarray(ts["t"])
    rows = np.asarray(ts[key], dtype=float)
    v = np.empty(len(rows))
    for i, row in enumerate(rows):
        fin = np.isfinite(row)
        v[i] = row[fin][0] if fin.any() else np.nan
    return t, v


def domain_values_at(ts, key, t_target):
    """Active-domain cell values of `key` at the snapshot nearest t_target."""
    t = np.asarray(ts["t"])
    i = int(np.argmin(np.abs(t - t_target)))
    v = np.asarray(ts[key][i], dtype=float)
    return v[np.isfinite(v)]


def spatial_stats(v):
    return {"mean": float(np.mean(v)), "std": float(np.std(v)),
            "min": float(np.min(v)), "max": float(np.max(v))}


def compare_pair(new_name: str, ref_name: str) -> dict:
    new_path, ref_path = NEW_DIR / new_name, REF_DIR / ref_name
    rec = {"new": new_name, "reference": ref_name, "ok": None,
           "caveats": [], "metrics": {}}

    for tag, p in [("new", new_path), ("reference", ref_path)]:
        if not (p / "timeseries.npz").exists():
            rec["ok"] = False
            rec["caveats"].append(f"missing {tag} results at {p}")
            return rec

    ts_n, cfg_n = load_case(new_path)
    ts_r, cfg_r = load_case(ref_path)
    m = rec["metrics"]

    for k in CONFIG_KEYS:
        a, b = cfg_n.get(k), cfg_r.get(k)
        if a != b and not (a is None and b is None):
            rec["caveats"].append(f"config {k}: new={a} vs ref={b}")

    m["clamped_steps_new"] = cfg_n.get("clamped_steps")

    # --- n_steps ---
    m["n_steps_pct"] = pct(cfg_n.get("n_steps"), cfg_r.get("n_steps"))

    # --- node final position over the common time window ---
    t_n, x_n = node_series(ts_n)
    t_r, x_r = node_series(ts_r)
    t_lo, t_hi = max(t_n[0], t_r[0]), min(t_n[-1], t_r[-1])
    t_common = np.linspace(t_lo, t_hi, 500)
    m["common_t_end_ms"] = float(t_hi * 1e3)
    m["node_final_pct"] = pct(float(np.interp(t_hi, t_n, x_n)),
                              float(np.interp(t_hi, t_r, x_r)))

    # --- state at the piston face ---
    m["face"] = {}
    for key in VARS:
        tf_n, vf_n = face_series(ts_n, key)
        tf_r, vf_r = face_series(ts_r, key)
        fn = np.interp(t_common, tf_n, vf_n)
        fr = np.interp(t_common, tf_r, vf_r)
        valid = np.isfinite(fn) & np.isfinite(fr) & (np.abs(fr) > 1e-30)
        series = 100.0 * (fn[valid] - fr[valid]) / fr[valid]
        m["face"][key] = {
            "final_pct": float(series[-1]) if series.size else None,
            "median_pct": float(np.median(series)) if series.size else None,
        }

    # --- spatial statistics over the entire (active) domain at t_hi ---
    m["domain_stats"] = {}
    for key in VARS:
        sn = spatial_stats(domain_values_at(ts_n, key, t_hi))
        sr = spatial_stats(domain_values_at(ts_r, key, t_hi))
        m["domain_stats"][key] = {stat: pct(sn[stat], sr[stat])
                                  for stat in ("mean", "std", "min", "max")}

    # --- pass criteria ---
    checks = {
        "zero_clamped_steps": bool((cfg_n.get("clamped_steps") or 0) == 0),
        "node_within_0.5pct": bool(abs(m["node_final_pct"] or 0) <= 0.5),
        "face_median_within_5pct": bool(all(
            abs(m["face"][k]["median_pct"] or 0) <= 5.0 for k in VARS)),
        "domain_mean_within_5pct": bool(all(
            abs(m["domain_stats"][k]["mean"] or 0) <= 5.0 for k in VARS)),
    }
    rec["checks"] = checks
    rec["ok"] = all(checks.values())
    return rec


def fmt(v, digits=4):
    return f"{v:+.{digits}f}%" if v is not None else "n/a"


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
        rec = compare_pair(new_name, ref_name)
        records.append(rec)
        m = rec["metrics"]

        print(f"\n=== {new_name}  vs  {ref_name}: {'PASS' if rec['ok'] else 'FAIL'} ===")
        for c in rec["caveats"]:
            print(f"  caveat: {c}")
        if not m:
            continue
        print(f"  n_steps:    {fmt(m['n_steps_pct'], 1)}")
        print(f"  node final: {fmt(m['node_final_pct'])}   (common window ends {m['common_t_end_ms']:.3f} ms)")
        print(f"  piston-face state (final / median over window):")
        for k in VARS:
            f = m["face"][k]
            print(f"    {k:3s}: {fmt(f['final_pct'])} / {fmt(f['median_pct'])}")
        print(f"  domain spatial stats at final common time (mean / std / min / max):")
        for k in VARS:
            d = m["domain_stats"][k]
            print(f"    {k:3s}: {fmt(d['mean'])} / {fmt(d['std'])} / {fmt(d['min'])} / {fmt(d['max'])}")

    n_fail = sum(1 for r in records if not r["ok"])
    print("\n" + "=" * 60)
    print(f"ACCURACY VALIDATION: {len(records) - n_fail}/{len(records)} pairs passed")

    NEW_DIR.mkdir(parents=True, exist_ok=True)
    report = {"timestamp": datetime.now().isoformat(), "pairs": records}
    (NEW_DIR / "pele_accuracy_report.json").write_text(json.dumps(report, indent=2))

    lines = [f"# Pele reconstruction accuracy report ({report['timestamp']})",
             "", "All values are signed percent difference w.r.t. the reference:",
             "(new - ref) / ref * 100", ""]
    for r in records:
        lines.append(f"## {r['new']} vs {r['reference']}: {'PASS' if r['ok'] else 'FAIL'}")
        for c in r["caveats"]:
            lines.append(f"- caveat: {c}")
        m = r["metrics"]
        if m:
            lines.append(f"- n_steps: {fmt(m['n_steps_pct'], 1)}")
            lines.append(f"- node final: {fmt(m['node_final_pct'])}")
            lines.append("")
            lines.append("| variable | face final | face median | domain mean | domain std | domain min | domain max |")
            lines.append("|---|---|---|---|---|---|---|")
            for k in VARS:
                f, d = m["face"][k], m["domain_stats"][k]
                lines.append(f"| {k} | {fmt(f['final_pct'])} | {fmt(f['median_pct'])} | "
                             f"{fmt(d['mean'])} | {fmt(d['std'])} | {fmt(d['min'])} | {fmt(d['max'])} |")
        lines.append("")
    (NEW_DIR / "pele_accuracy_report.md").write_text("\n".join(lines))
    print(f"Report saved: {NEW_DIR / 'pele_accuracy_report.md'}")

    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()