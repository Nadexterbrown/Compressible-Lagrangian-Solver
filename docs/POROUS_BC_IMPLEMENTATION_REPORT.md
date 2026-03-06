# Porous Piston Boundary Condition Implementation Report

## Executive Summary

This report documents the implementation of a porous piston boundary condition (`MovingPorousPistonBC`) that allows the gas velocity at the boundary to differ from the piston velocity. This enables simulation of scenarios where the flame/piston position tracks experimental data while the gas dynamics respond to a different (typically lower) gas velocity.

---

## 1. Files Modified

### 1.1 Core Changes

| File | Type | Description |
|------|------|-------------|
| `src/lagrangian_solver/boundary/piston.py` | **NEW CLASS** | Added `MovingPorousPistonBC` (~400 lines) |
| `src/lagrangian_solver/boundary/__init__.py` | Export | Added `MovingPorousPistonBC` to `__all__` |
| `src/lagrangian_solver/core/grid.py` | **NEW METHOD** | Added `add_boundary_mass()` (~40 lines) |
| `src/lagrangian_solver/core/solver.py` | **MODIFIED** | Added hooks in `_compute_rhs()` and `_update_porous_boundary_mass()` |

### 1.2 Change Isolation Analysis

**Question: Are changes isolated to porous BC only?**

**Answer: Partially.** The changes use duck-typing (`hasattr()` checks) to conditionally activate porous-specific behavior. However, there are modifications to shared components:

#### Isolated (Porous-Only):
- `MovingPorousPistonBC` class - only instantiated when user requests porous BC
- `add_boundary_mass()` in grid.py - only called by porous BCs

#### Shared Components Modified:
- `solver.py:_compute_rhs()` - adds `hasattr` checks for `apply_position_rate`
- `solver.py:_update_porous_boundary_mass()` - adds `hasattr` checks for `update_boundary_mass` and `check_merge_split`
- `grid.py` - new method `add_boundary_mass()` added to `LagrangianGrid`

---

## 2. Detailed Change Descriptions

### 2.1 `MovingPorousPistonBC` Class

**Location:** `src/lagrangian_solver/boundary/piston.py` (lines ~1470-2020)

**Purpose:** Implements a porous piston where:
- Boundary node position evolves at piston velocity `u_p(t)`
- Gas dynamics (Riemann solver) use gas velocity `u_g(t) = u_p(t) + offset`
- Mass flux through boundary when `u_p != u_g`

**Key Parameters:**
```python
MovingPorousPistonBC(
    side: BoundarySide,
    eos: EOSBase,
    trajectory: TrajectoryInterpolator,
    gas_velocity_offset: float = 0.0,      # u_g = u_p + offset
    gas_velocity_min: Optional[float] = None,  # Clamp u_g >= min
    ...
)
```

**Key Methods:**

| Method | Purpose |
|--------|---------|
| `get_piston_velocity(t)` | Returns piston velocity from trajectory |
| `get_gas_velocity(t)` | Returns gas velocity (piston + offset, clamped) |
| `apply_position_rate(d_x, ...)` | Overrides `d_x[boundary]` to use piston velocity |
| `update_boundary_mass(grid, state, dt)` | Updates cell mass due to porosity flux |
| `check_merge_split(grid, state)` | Triggers cell redistribution if needed |
| `_conservative_merge_split(...)` | Performs mass/volume redistribution |

### 2.2 Merge-Split Algorithm

**Trigger Conditions (THREE criteria):**
1. **Mass ratio:** `dm[boundary] / dm[neighbor]` outside [0.5, 2.0]
2. **Volume ratio:** `dx[boundary] / dx[neighbor]` outside [0.5, 2.0]
3. **Absolute minimum:** `dx[boundary] < 10%` of initial average cell size

**Algorithm:**
1. Merge two adjacent cells conceptually (total mass, total volume, total energy)
2. Split equally: each cell gets half the mass and half the volume
3. Move interface position to achieve equal volumes
4. Recalculate density: `rho = dm / dx`
5. Recalculate pressure, temperature, sound speed via EOS

**Recursion:** After merging cells 0+1, checks if cell 1 needs to merge with cell 2, etc. Propagates up to half the grid.

### 2.3 Grid Mass Update

**Location:** `src/lagrangian_solver/core/grid.py:369-406`

```python
def add_boundary_mass(self, side: BoundarySide, dm: float) -> None:
    """
    Adjust boundary cell mass for porous BC.

    Args:
        side: Which boundary (LEFT or RIGHT)
        dm: Mass change (positive = cell gains mass)
    """
```

**Implications:** This method modifies `grid._dm` and `grid._m` arrays, which are normally constant in a Lagrangian code. This breaks the "fixed cell mass" Lagrangian assumption for the boundary cell only.

### 2.4 Solver Modifications

**Location:** `src/lagrangian_solver/core/solver.py`

**`_compute_rhs()` (lines 331-335):**
```python
# For porous BCs, override d_x at boundaries
if hasattr(self._bc_left, 'apply_position_rate'):
    self._bc_left.apply_position_rate(d_x, state, grid, self._time)
if hasattr(self._bc_right, 'apply_position_rate'):
    self._bc_right.apply_position_rate(d_x, state, grid, self._time)
```

**`_update_porous_boundary_mass()` (lines 446-457):**
```python
if hasattr(self._bc_left, 'update_boundary_mass'):
    self._bc_left.update_boundary_mass(self._grid, self._state, dt)
    if hasattr(self._bc_left, 'check_merge_split'):
        self._bc_left.check_merge_split(self._grid, self._state)
```

---

## 3. Implications on Solver as a Whole

### 3.1 Performance Impact

| Aspect | Impact | Notes |
|--------|--------|-------|
| Non-porous simulations | **Negligible** | Only `hasattr()` checks (~ns overhead) |
| Porous simulations | **Moderate** | Merge-split involves EOS calls |
| Memory | **None** | No additional arrays allocated |

### 3.2 Conservation Properties

| Property | Non-Porous | Porous |
|----------|------------|--------|
| Mass | Conserved | **Not conserved** (intentional - mass flows through boundary) |
| Momentum | Conserved | Conserved (within numerical precision) |
| Total Energy | Conserved | **Approximate** (merge-split averages internal energy) |

### 3.3 Stability

**Non-porous BCs:** Unchanged. The `hasattr()` checks return `False`, so no new code paths are executed.

**Porous BC:**
- CFL timestep can decrease significantly due to cell compression
- Merge-split mitigates cell collapse but introduces thermodynamic discontinuities
- Pressure oscillations possible at merge-split boundaries

### 3.4 Code Coupling

The changes introduce coupling between:
- `LagrangianGrid` ↔ `MovingPorousPistonBC` (via `add_boundary_mass`)
- `CompatibleLagrangianSolver` ↔ `MovingPorousPistonBC` (via `apply_position_rate`, `update_boundary_mass`, `check_merge_split`)

This coupling uses **duck-typing** rather than inheritance, which is flexible but less explicit.

---

## 4. Implications on Porous Results

### 4.1 Physical Accuracy

**What the porous BC models:**
- Piston/flame position tracks experimental trajectory
- Gas at boundary has velocity = piston velocity + offset
- Mass flux: `dm/dt = -rho * (u_piston - u_gas)` flows through boundary

**Limitations:**
1. **Linearized Riemann:** Uses acoustic impedance relation `p* = p_0 + Z_0 * (u_g - u_0)` instead of full nonlinear Riemann solver
2. **Merge-split averaging:** When cells are redistributed, internal energy is averaged, which may not preserve entropy correctly
3. **EOS consistency:** After merge-split, pressure is recalculated from averaged (rho, e), which may not match the original thermodynamic path

### 4.2 Numerical Artifacts

| Artifact | Cause | Mitigation |
|----------|-------|------------|
| Pressure oscillations | Merge-split creates density discontinuities | Interface movement equalizes density |
| CFL reduction | Boundary cell compression | Volume-based merge-split trigger |
| Mass imbalance | Porous flux | Recursive merge-split propagates deficit |

### 4.3 Observed Behavior (from test runs)

| Metric | Without Merge-Split | With Merge-Split |
|--------|---------------------|------------------|
| Simulation time reached | ~60 μs (crash) | 500+ μs |
| Pressure range | 1.4 - 16 MPa (unstable) | 1.5 - 1.8 MPa (stable) |
| Timestep behavior | Collapse to ~0 | Gradual decrease |

---

## 5. Achieving Complete Isolation

To make the porous BC completely isolated (no modifications to shared components):

### Option A: Subclass the Solver

Create `PorousLagrangianSolver` that overrides `_compute_rhs()`:

```python
class PorousLagrangianSolver(CompatibleLagrangianSolver):
    def _compute_rhs(self, state, grid):
        d_tau, d_u, d_e, d_x = super()._compute_rhs(state, grid)

        # Porous-specific: override d_x at boundary
        if hasattr(self._bc_left, 'apply_position_rate'):
            self._bc_left.apply_position_rate(d_x, state, grid, self._time)
        ...
        return d_tau, d_u, d_e, d_x
```

**Pros:** No changes to base solver
**Cons:** Code duplication, user must choose correct solver class

### Option B: Solver Hooks/Callbacks

Add a hook system to the solver:

```python
class CompatibleLagrangianSolver:
    def __init__(self, ..., rhs_post_hooks=None):
        self._rhs_post_hooks = rhs_post_hooks or []

    def _compute_rhs(self, state, grid):
        d_tau, d_u, d_e, d_x = ...  # Original computation
        for hook in self._rhs_post_hooks:
            hook(d_tau, d_u, d_e, d_x, state, grid, self._time)
        return d_tau, d_u, d_e, d_x
```

**Pros:** Clean separation, extensible
**Cons:** Requires refactoring solver

### Option C: BC Protocol Extension (Current Approach)

The current implementation uses duck-typing:

```python
if hasattr(self._bc_left, 'apply_position_rate'):
    self._bc_left.apply_position_rate(...)
```

**Pros:** Minimal code changes, backward compatible
**Cons:** Implicit coupling, `hasattr` checks in hot path

### Recommendation

The **current approach (Option C)** is acceptable because:
1. `hasattr()` overhead is negligible (~10ns per call)
2. The checks only activate for porous BCs
3. Non-porous simulations are completely unaffected in behavior

For stricter isolation, **Option B** (hooks) would be the cleanest long-term solution.

---

## 6. Summary

### Changes Made
1. Added `MovingPorousPistonBC` class with gas velocity offset/minimum
2. Added recursive merge-split with mass, volume, and absolute minimum triggers
3. Added `add_boundary_mass()` to `LagrangianGrid`
4. Added porous-specific hooks in solver's `_compute_rhs()` and time-stepping

### Isolation Status
- **Functionally isolated:** Non-porous simulations behave identically
- **Code isolation:** Partial - shared components have `hasattr()` checks

### Key Parameters
```python
MovingPorousPistonBC(
    ...,
    gas_velocity_offset=-119.0,  # u_gas = u_piston - 119 m/s
    gas_velocity_min=0.0,         # Don't allow negative gas velocity
)
```

### Merge-Split Thresholds
- Mass/Volume ratio: [0.5, 2.0]
- Absolute minimum: 10% of initial average cell size

---

## 7. References

- Toro, E.F. "Riemann Solvers and Numerical Methods for Fluid Dynamics" (3rd ed., Springer, 2009), Section 6.3 - Boundary conditions via ghost cells
- Després, B. "Numerical Methods for Eulerian and Lagrangian Conservation Laws" (Birkhäuser, 2017), Chapter 3 - Lagrangian formulation
