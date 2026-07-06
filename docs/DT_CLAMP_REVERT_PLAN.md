/mod# Timestep-Clamp Clock Desync: Commit Forensics, Revert Scope, and Fix Plan

Date: 2026-07-06
Context: The 9 experiment-reconstruction density-ratio cases (run 6/28–29) show 7–31%
piston-position deficit caused by a clock desync between the driver script and the solver.
Pele reconstructions are unaffected (verified ≤0.2%, density-ratio batch 0.005%).

## 1. Which commit contains the error

There is no single guilty commit — the corruption requires three layers, each individually
defensible, that only fail in combination:

| Layer | Where | Commit | Date |
|---|---|---|---|
| 1. Silent dt clamp in `step_forward()` | `src/lagrangian_solver/core/solver.py:388-397` | **`d40da3e`** "Add MovingPorousPistonBC with merge-split cell management" | 2026-03-06 |
| 2. Drainage guard on the BC actually used by the experiment runs | `MovingPorousPistonBC.get_max_dt_constraint`, `src/lagrangian_solver/boundary/piston.py:1642` | **`7bce7aa`** "Add data-driven gas velocity options for porous piston BC" | 2026-05-03 |
| 3. `dt_min = 1e-8` floor + manual script clock (`t += dt` with the *requested* dt) | `scripts/experiment_reconstruction/run_all_cases_density_ratio_mpi.py:65`, `run_experiment_reconstruction.py:767-768, 800` | **uncommitted** (new files, still `AM` in git status) | ~2026-06-28 |

Supporting detail:

- `d40da3e` introduced the *pattern* that violates the `step_forward(dt)` API: a
  caller-supplied dt is silently reduced (`dt = min(dt, dt_porous)`), communicated only
  through the usually-ignored `TimeStepInfo` return value. It also added
  `PorousGhostPistonBC.get_max_dt_constraint` (piston.py:1243), but that class is not used
  by the affected runs.
- `7bce7aa` added the same guard method to `MovingPorousPistonBC` — this is the commit that
  *armed* the clamp for the experiment-reconstruction runs (guard: drain ≤10% of boundary
  cell width per step, piston.py:1655-1665).
- The solver-config floor `SolverConfig.dt_min` (`d7fa4cf`, 2026-02-23, solver.py:384-386)
  is benign on its own — the solver's internal clock always advances by the dt actually
  integrated. Note the ordering: the floor is applied *before* the porous clamp, so the
  guard always wins inside the solver. The two directives are contradictory by design.
- The trigger is layer 3: the batch script requests dt = max(CFL dt, 10 ns), the solver
  integrates min(requested, guard) — as small as ~4 ns — and the script advances its own
  clock by the requested dt anyway. Live repro confirmed 83.5% of steps clamped after
  t ≈ 3.0 ms, reproducing the observed lag.
- Pele runs are clean because `scripts/pele_reconstruction/run_pele_reconstruction.py` has
  no dt_min: its requested CFL dt is ~3–4× smaller than the guard, so the clamp never binds.

## 2. How much of a revert is required

**Do not revert the commits.** `d40da3e` and `7bce7aa` carry the entire porous-BC
infrastructure (merge-split cell management, data-driven gas velocity, PeleDataLoader
extensions) that all subsequent work depends on. The erroneous code is surgical:

### Remove (uncommitted script code — just delete)
1. `run_all_cases_density_ratio_mpi.py:65` — `'dt_min': 1e-8` entry, and the
   `dt_min=case['dt_min']` pass-through at line 190.
2. `run_experiment_reconstruction.py:767-768` — the script-side floor
   (`if dt_min is not None and dt < dt_min: dt = dt_min`).
   Optionally keep the `--dt_min` CLI plumbing but default it to None and warn loudly.
3. `run_experiment_reconstruction.py:709` — stop passing `dt_min` into `SolverConfig`
   (the solver-side floor is immediately overridden by the porous guard anyway).

### Fix (committed solver code — modify, don't delete)
4. `solver.py:388-397` — keep the porous constraint but make it **loud**: when
   `step_forward` reduces a caller-supplied dt, emit a `warnings.warn` (once per N steps
   to avoid spam) carrying requested vs. actual dt. The guard itself is physically
   motivated and must stay until item 8 lands.

### Replace (the honest-clock pattern, all three driver scripts)
5. `run_experiment_reconstruction.py`, `run_pele_reconstruction.py`,
   `run_oscillating_piston.py`: replace the manual clock
   (`t += dt` on the requested dt) with the solver's clock:
   `info = solver.step_forward(dt)` then `t = solver.time`. Record snapshots and evaluate
   `u_piston`/`u_gas` diagnostics at `solver.time`. Add
   `assert abs(t - solver.time) < 1e-12` as a tripwire.
6. Fix the tautological validation plot: `velocity_comparison.png` currently plots
   `trajectory.velocity(script_t)` against the experimental flame velocity — it matches by
   construction. Plot the actual node position `x[:,0]` (and its numerical derivative)
   against `trajectory.position/velocity` instead.
7. Log cumulative `mass_leaked` and clamp-event counts into `config.json` for every run.

## 3. Plan for the problems the erroneous code was trying to solve

The two band-aids exist for real reasons; removing them re-exposes those problems:

- **`get_max_dt_constraint` (keep, relocate)** was added to prevent instability when the
  boundary cell drains >10% of its width in one step.
- **`dt_min = 1e-8` (delete, replace with real fix)** was added because runs "stall": the
  merge-split cell management keeps 11–17 µm cells alive near the boundary, which throttle
  the *global* CFL dt for the entire run (147k steps vs. 62k for solid) — the floor tried
  to push through this, which is exactly what armed the clamp conflict. You cannot honor
  both a floor and the guard; the contradiction must be resolved at the grid level.

### Structural fix (ordered)
8. **Boundary-cell absorption instead of merge-split residue.** When the boundary cell
   drains below a threshold (e.g., 25% of the initial cell mass), remove it: absorb its
   remaining mass/momentum/energy conservatively into its neighbor and reduce the active
   cell count by one — the approach used by GDTk's L1D for piston-adjacent cells. This
   eliminates the µm-scale cells that both (a) collapse the CFL dt (the "stalling" that
   motivated dt_min) and (b) trip the drainage guard. Cite GDTk L1D in
   `docs/CITATIONS.md` per project policy.
9. **Move the porous constraint into `compute_timestep()`** so the CFL suggestion already
   includes it. Then `step_forward(None)` needs no post-hoc clamping, and callers that pass
   an explicit dt get either exact honoring or a loud warning (item 4).
10. **Stop duplicating the CFL formula in scripts.** The driver scripts recompute
    `dt = cfl * min(dx / (c + |u|))` (e.g., `run_experiment_reconstruction.py:762-764`,
    `run_pele_reconstruction.py:783`) — a second latent desync source. Call
    `solver.step_forward(None)` (or a public `solver.suggest_dt()`) instead.
11. **Unit test with two clocks.** The porous tests currently call `step_forward(None)`
    only. Add a test that drives `step_forward(dt)` with an explicit dt while a guard is
    active and asserts the caller can detect the reduction (warning raised, honest
    `TimeStepInfo.dt`, `solver.time` consistent).

### Validation & re-run
12. Re-run one corrupted case (c4h10 φ=0.8) with fixes 1–7: expect the porous piston node
    to track the trajectory exactly (as solid does), with clamp warnings gone after item 8.
13. Re-run all 9 experiment density-ratio cases. Expect state differences between porous
    and solid to shrink to genuine porous physics (modestly lower p/T from mass removal).
14. Spot-re-check `pele_burning_vel_extended` (the one Pele run with a nonzero 2.6 mm /
    0.2% deficit) after item 8.

## Verified-clean results (no action needed)
- All solid-piston experiment reconstructions.
- All six Pele porous reconstructions (`scripts/pele_reconstruction/results/*`): generated
  Mar–May, no dt_min, guard never binds; `pele_porous_density_ratio` re-verified 2026-07-06
  at 0.005% deficit.
