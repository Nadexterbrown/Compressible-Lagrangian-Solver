# Implementation Plan: Clock-Desync Fixes + Piston-Face Cell Collapse

Date: 2026-07-06
Companion doc: `DT_CLAMP_REVERT_PLAN.md` (commit forensics and revert scope)

## Problem statement

Two coupled problems:

1. **Correctness bug (acute):** the script/solver clock desync corrupted the 9
   experiment-reconstruction density-ratio cases (7–31% piston-position deficit).
2. **Performance root cause (chronic):** the porous piston drains its boundary cell, and
   the merge-split cell management leaves a train of ~11–17 µm cells near the piston face.
   These throttle the *global* CFL timestep to sub-nanosecond values, pushing runs past
   12+ hours (terminated before completion). `dt_min = 1e-8` was added to push through
   this — which is exactly what armed the silent-clamp conflict and caused problem 1.

The fix must remove the band-aids AND remove the reason they were needed.

### Why merge-split cannot solve the collapse (diagnosis)

`MovingPorousPistonBC.check_merge_split` (piston.py:1882) merges the boundary cell
pairwise with its neighbor and **splits the result back into two equal cells**
(piston.py:1984). Consequences:

- The cell count never decreases. Mass keeps draining, so the equalized pair keeps
  shrinking; recursion propagates the smallness inward (up to half the grid).
- Every merge-split re-averages state over the near-boundary region every step →
  numerical diffusion and repeated EOS re-evaluations.
- The global CFL dt is min over all cells → the whole simulation runs at the pace of the
  smallest boundary-region cell (147k steps vs 62k for the equivalent solid run).
- The drainage guard (`get_max_dt_constraint`, ≤10% of boundary-cell width per step)
  shrinks along with the cell — the two problems feed each other.

The structural answer is **true cell-count reduction**: when the boundary cell has drained
below a threshold, absorb its remaining contents into its neighbor and retire it. This is
the approach used by GDTk's L1D for piston-adjacent cell management (Jacobs, GDTk,
https://gdtk.uqcloud.net — cite in `docs/CITATIONS.md` with file/line per project policy;
conservative-remap math follows Benson 1992, already cited at piston.py:2012).

Blocking constraint: `LagrangianGrid` is fixed-size — `n_cells` set at construction,
all arrays preallocated (grid.py:90-103), no removal API. FlowState likewise. Phase 2
addresses this first.

---

## Phase 0 — Safety baseline (no behavior changes)

0.1 Commit the current uncommitted scripts **as-is** on `main`
    ("Batch scripts as used for 6/28 density-ratio runs — contains dt_min clock-desync
    bug, see docs/DT_CLAMP_REVERT_PLAN.md"). This bookmarks the exact code that produced
    the corrupted batch.
0.2 Create branch `fix/piston-cell-collapse`.
0.3 Run the full test suite; record pass/fail baseline (later failures must be
    attributable to our changes).
0.4 Golden runs for equivalence gating (short, minutes not hours):
    a. Solid piston, c4h10 φ=0.8, truncated t_end (~0.5 ms).
    b. Porous (density-ratio) same case, truncated, **no dt_min** — clamp never binds.
    Save outputs outside the results tree; these must reproduce bit-identically after
    Phase 1.

## Phase 1 — Correctness fixes (the acute bug; no numerics changes)

1.1 Remove `'dt_min': 1e-8` from `run_all_cases_density_ratio_mpi.py:65` and its
    pass-through (line 190).
1.2 Remove the script-side floor in `run_experiment_reconstruction.py:767-768`; stop
    passing `dt_min` into `SolverConfig` (line 709). Keep the CLI flag but make it error
    out with a pointer to this doc (prevents silent reintroduction).
1.3 Honest clock in all three drivers (`run_experiment_reconstruction.py`,
    `run_pele_reconstruction.py`, `run_oscillating_piston.py`):
    `info = solver.step_forward(dt)` then `t = solver.time`; record snapshots and evaluate
    `u_piston`/`u_gas` at `solver.time`; tripwire `assert abs(t - solver.time) < 1e-12`.
1.4 Loud clamp: in `step_forward` (solver.py:388-397), `warnings.warn` when a
    caller-supplied dt is reduced (rate-limited, with requested vs actual dt and
    cumulative clamp count). Count exposed on `solver.stats`.
1.5 Fix the tautological `velocity_comparison.png`: plot actual node position `x[:,0]`
    and its numerical derivative against `trajectory.position/velocity`.
1.6 Persist diagnostics in every run's `config.json`: cumulative `mass_leaked`,
    clamp-event count, min dt encountered, wall-clock time.

**Gate 1:** golden runs (0.4a, 0.4b) reproduce previous outputs to machine precision
(clamp never bound in either, so the honest clock is a mathematical no-op). One commit
per item.

## Phase 2 — Boundary-cell absorption (the chronic fix)

2.1 **Grid/state support for cell retirement.** Add `LagrangianGrid.remove_boundary_cell(side)`
    and a matching FlowState shrink. Implementation choice: physically shrink the arrays
    (np.delete-style reallocation), NOT an active-window/offset scheme — absorption events
    are rare (O(100) per run vs 10^5 steps), so O(N) reallocation is negligible, and
    shrinking keeps every downstream consumer (`len`, slicing, integrator loops) correct
    without an indexing layer. Invalidate/refresh any cached array views.
2.2 **Conservative absorption operator** on `MovingPorousPistonBC`: when triggered, merge
    boundary cell fully into its neighbor using the existing conservative-remap math
    (mass, momentum, total energy exact — reuse `_conservative_merge_split` steps 1-7 with
    a one-cell final state instead of an equal split), then retire the boundary cell.
    The piston face becomes the left face of the former neighbor. `cell_index` bookkeeping
    unchanged (still 0 / n-1).
2.3 **Trigger policy:** absorb when boundary-cell mass < `f_absorb` × current *neighbor*
    mass (default f_absorb = 0.5, tunable). Rationale: mass-based (not width-based)
    so compression alone doesn't trigger it; neighbor-relative so it adapts as the grid
    coarsens near the face. Keep the existing split path only for the mass-*gain*
    direction (u_p < u_g, cell growing).
2.4 **Replace merge-split for drainage** entirely; delete the recursive inward
    propagation (piston.py:1917-1981) for the shrink direction.
2.5 **Drainage guard update:** `get_max_dt_constraint` becomes mass-based
    (≤ X% of boundary-cell *mass* per step) and, because absorption keeps the boundary
    cell within a factor of ~2 of its neighbor, the guard stays O(CFL dt) and should
    effectively never bind. Keep it as a safety net.
2.6 **Snapshot format decision:** with a shrinking grid, per-snapshot arrays change
    length. Recommended: left-pad each saved snapshot with NaN to the original size so
    stacked npz arrays stay rectangular and column j always means "original cell j";
    plotting scripts treat NaN as retired. Record `n_active` per snapshot. Update
    `merge_chunks.py` and the post-processing loaders accordingly.
2.7 Minimum-cell-count guard: refuse to drop below e.g. 10% of the initial cell count;
    raise a clear error (a run that drains 90% of its cells has a physics/setup problem).

**Gate 2:** new unit tests (Phase 4) pass; golden porous run (0.4b) — where absorption
never triggers at truncated t_end — still reproduces Phase 1 output exactly. A longer
porous run shows: dt no longer collapses, cell count decreases stepwise, conservation
errors (mass balance including `mass_leaked`, momentum, total energy) within tolerance.

## Phase 3 — dt architecture cleanup (removes the latent desync class)

3.1 Move the porous constraint into `compute_timestep()` so the solver's suggested dt
    already includes it; `step_forward(None)` then needs no post-hoc clamping.
3.2 Public `solver.suggest_dt()`; drivers stop duplicating the CFL formula
    (`run_experiment_reconstruction.py:762-764`, `run_pele_reconstruction.py:783`) and
    call `step_forward(None)` (or `suggest_dt()` when they need dt beforehand for
    recording logic).
3.3 Optional strict mode: `SolverConfig(strict_dt=True)` raises instead of warning when
    an explicit caller dt cannot be honored.

**Gate 3:** golden runs unchanged; test suite green.

## Phase 4 — Tests (written alongside phases 2–3)

4.1 Two-clock test: explicit-dt stepping with an active guard; assert warning raised,
    honest `TimeStepInfo.dt`, `solver.time` consistent with integrated dt.
4.2 Absorption conservation test: construct a draining boundary cell, trigger absorption,
    assert exact mass/momentum/total-energy conservation and valid EOS state.
4.3 dt-collapse regression: synthetic hard-drainage scenario; assert min dt over the run
    stays above a floor tied to the *interior* CFL (not the boundary cell) and that cell
    count decreases.
4.4 Piston-position fidelity test: porous run against a prescribed trajectory; assert
    node 0 tracks `trajectory.position(solver.time)` within tolerance at end time.

## Phase 5 — Validation on the real problem

5.1 Re-run the instrumented restart window (c4h10 φ=0.8 from the 2.95 ms snapshot,
    scratchpad `restart_repro.py`): expect zero clamp events, zero clock lag, dt at
    interior-CFL scale throughout.
5.2 Full c4h10 φ=0.8 case, new output dir (do NOT overwrite `porous_results_dt_10_ns`).
    Success criteria:
    - piston node tracks trajectory (deficit < 0.1%),
    - step count within ~1.5× of the solid run (~62k), wall time in the same ballpark —
      not 12+ hours,
    - mass/energy balance closes with `mass_leaked` accounted,
    - porous-vs-solid state differences are modest (genuine porous physics), not the
      current corrupted-magnitude gaps.
5.3 If 5.2 passes: re-run all 9 density-ratio cases into `porous_results_corrected/`;
    regenerate comparison plots with the fixed (non-tautological) validation plot.
5.4 Spot-re-check `pele_burning_vel_extended` (the one Pele run with 2.6 mm / 0.2%
    deficit) — expected to drop to ~0 with the guard no longer binding.

## Phase 6 — Merge & documentation

6.1 Merge `fix/piston-cell-collapse` → `main`; push (project policy).
6.2 Update `docs/CITATIONS.md`: GDTk L1D (cell absorption), Benson 1992 (remap),
    with file paths and line numbers.
6.3 Update `ENERGY_ERROR_ANALYSIS.md` / `PISTON_BC_RESEARCH.md` if their conclusions
    referenced the corrupted batch.
6.4 Mark the corrupted batch: drop a `README_CORRUPTED.md` into
    `porous_results_dt_10_ns/` explaining the defect and pointing at the corrected dir
    (keep the data — it is the reproduction evidence).

## Risk register

| Risk | Mitigation |
|---|---|
| Array shrinking breaks a consumer that cached `n_cells` | Phase 2.1 audits all `grid.n_cells` / array-length uses (grep); golden-run gates catch regressions |
| Absorption introduces a pressure blip at the piston face | Conservative remap is exact for mass/momentum/energy; test 4.2 + monitor p at face in 5.1 |
| Ragged snapshots break post-processing | Phase 2.6 NaN-padding keeps npz rectangular; update loaders in the same commit |
| dt still collapses from *interior* cells (flame compression, not the piston) | Diagnose via 1.6 min-dt/limiting-cell logging; that is a separate (real, physical) CFL cost — do not floor it; consider grid coarsening study instead |
| Absorption threshold too aggressive (visible smoothing near face) | f_absorb configurable; sweep 0.25/0.5 in 5.1 and compare face-region profiles |
