# WARNING: Corrupted timelines — do not use for analysis

All 9 density-ratio cases in this directory (run 2026-06-28/29 with
`dt_min = 1e-8`) have **corrupted recorded timelines**: the piston position
lags the trajectory by **7–31%** by end of run.

## Defect

The batch script floored the requested timestep at 10 ns while the solver's
porous drainage guard (`get_max_dt_constraint`) silently reduced the
integrated dt (down to ~4 ns; 83.5% of steps clamped after t ≈ 3.0 ms in the
verified reproduction). The script advanced its own clock by the *requested*
dt, so every recorded `t` beyond the clamp onset is inflated. The *physics*
of each snapshot is internally consistent — only the time stamps (and
anything derived from them, e.g. apparent piston position vs. time) are
wrong.

Full forensics: `docs/DT_CLAMP_REVERT_PLAN.md`.
Fix: `docs/PISTON_CELL_COLLAPSE_FIX_PLAN.md` (branch `fix/piston-cell-collapse`).
The exact scripts that produced this batch are bookmarked in commit `b783189`.

## Status of other result sets

- All **solid-piston** experiment reconstructions: clean (clamp never binds).
- All **Pele** reconstructions (`scripts/pele_reconstruction/results/*`):
  clean, verified ≤ 0.2% (density-ratio batch 0.005%).

## Re-runs

Corrected runs are produced by `tests/validation/run_validation_suite_mpi.py`
into `tests/validation/results/` — never overwrite this directory; it is the
reproduction evidence for the defect.
