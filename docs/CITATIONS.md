# Citations

This file documents all references, citations, and sources used in the 1D Compressible Lagrangian Solver.

## Format

Each citation should include:
- **Reference**: Full citation (author, title, publication, year, DOI/URL)
- **Used in**: File path and line numbers where the reference is applied
- **Description**: Brief explanation of what concept/algorithm is being referenced

## Citations

### Example Entry
```
**Reference**: Author, A. B. (Year). Title of the paper. Journal Name, Volume(Issue), Pages. DOI

**Used in**: `src/module.py:45-67`

**Description**: Implementation of [algorithm/concept name]
```

---

## References

### [Toro2009] - Primary Reference for Numerical Schemes

**Reference**: Toro, E.F. (2009). *Riemann Solvers and Numerical Methods for Fluid Dynamics: A Practical Introduction* (3rd ed.). Springer-Verlag Berlin Heidelberg. ISBN: 978-3-540-25202-3. DOI: 10.1007/b79761

**Used in**: Baseline reference for all numerical schemes

**Description**: Primary reference for Riemann solvers, Godunov methods, flux calculations, and numerical methods for compressible flow simulations. This text serves as the foundational reference for the solver's numerical implementation.

### [Despres2017] - Secondary Reference for Lagrangian Formulation

**Reference**: Després, B. (2017). *Numerical Methods for Eulerian and Lagrangian Conservation Laws*. Birkhäuser, Cham. ISBN: 978-3-319-50354-4. DOI: 10.1007/978-3-319-50355-1

**Used in**: Reference for 1D Lagrangian conservation equation formulation

**Description**: Secondary reference specifically for the 1D Lagrangian formulation of conservation laws. Provides theoretical foundation and numerical methods for Lagrangian hydrodynamics.

### [Cantera] - Thermodynamic Properties and EOS

**Reference**: Goodwin, D.G., Speth, R.L., Moffat, H.K., and Weber, B.W. (2023). *Cantera: An Object-oriented Software Toolkit for Chemical Kinetics, Thermodynamics, and Transport Processes*. Version 3.0.0. https://www.cantera.org. DOI: 10.5281/zenodo.8137090

**Used in**: All equation of state and thermodynamic property calculations

**Description**: Open-source chemical kinetics software used for all EOS calculations, species thermodynamic properties, and gas mixture handling throughout the solver.

---

## Implementation-Specific Citations

### Ideal Gas EOS

**Reference**: [Toro2009] Section 1.2, Equations (1.20)-(1.25)

**Used in**: `src/lagrangian_solver/equations/eos.py:108-165`

**Description**: Ideal gas equation of state implementation with relations p = (γ-1)ρe, c = √(γp/ρ)

### Exact Riemann Solver

**Reference**: [Toro2009] Chapter 4, Section 4.3, Algorithm 4.1

**Used in**: `src/lagrangian_solver/numerics/riemann.py:100-250`

**Description**: Newton-Raphson iterative exact Riemann solver with pressure function formulation. Initial guess methods from Section 4.3.1.

### HLLC Approximate Solver

**Reference**: [Toro2009] Chapter 10, Section 10.4

**Used in**: `src/lagrangian_solver/numerics/riemann.py:320-420`

**Description**: Harten-Lax-van Leer-Contact approximate Riemann solver with wave speed estimates from Section 10.5.1.

### Toro Riemann Test Problems

**Reference**: [Toro2009] Section 4.3.3, Table 4.1

**Used in**: `tests/verification/test_toro_riemann.py`

**Description**: Five canonical Riemann problems for solver validation: Sod shock tube, two rarefactions, left blast, right blast, and two-shock collision.

### Lagrangian Conservation Equations

**Reference**: [Despres2017] Chapter 3, Equations (3.7)-(3.9), (3.15)-(3.17)

**Used in**: `src/lagrangian_solver/equations/conservation.py`, `src/lagrangian_solver/core/state.py`

**Description**: Lagrangian form of 1D compressible Euler equations in mass coordinates: ∂τ/∂t - ∂u/∂m = 0, ∂u/∂t + ∂p/∂m = 0, ∂E/∂t + ∂(pu)/∂m = 0

### Heun's Method (Time Integration)

**Reference**: [Toro2009] Section 6.4.2

**Used in**: `src/lagrangian_solver/numerics/time_integration.py:85-140`

**Description**: Second-order explicit predictor-corrector time integration (improved Euler/RK2)

### CFL Condition

**Reference**: [Toro2009] Section 6.3

**Used in**: `src/lagrangian_solver/numerics/time_integration.py:60-80`, `src/lagrangian_solver/core/grid.py:175-195`

**Description**: Courant-Friedrichs-Lewy stability condition dt ≤ CFL × min(dx/(|u|+c))

### Characteristic Boundary Conditions

**Reference**: Poinsot, T.J., & Lele, S.K. (1992). Boundary conditions for direct simulations of compressible viscous flows. *Journal of Computational Physics*, 101(1), 104-129.

**Used in**: `src/lagrangian_solver/boundary/open.py`

**Description**: Non-reflecting characteristic boundary conditions for open boundaries based on NSCBC methodology

### Beavers-Joseph Porous Interface Condition

**Reference**: Beavers, G.S., & Joseph, D.D. (1967). Boundary conditions at a naturally permeable wall. *Journal of Fluid Mechanics*, 30(1), 197-207.

**Used in**: `src/lagrangian_solver/boundary/piston.py:90-130`

**Description**: Slip velocity condition for flow at porous boundaries: u_slip = (√K/α_BJ) × du/dn

### Von Neumann-Richtmyer Quadratic Artificial Viscosity

**Reference**: Von Neumann, J. & Richtmyer, R.D. (1950). A method for the numerical calculation of hydrodynamic shocks. *Journal of Applied Physics*, 21(3), 232-237. DOI: 10.1063/1.1699639

**Used in**: `src/lagrangian_solver/numerics/artificial_viscosity.py:48-55`

**Description**: Quadratic artificial viscosity for shock capturing in Lagrangian hydrodynamics. The viscous stress Q_quad = c₂ × ρ × dx² × (∂u/∂x)² controls shock width and overall entropy production in compression regions.

### Landshoff Linear Artificial Viscosity

**Reference**: Landshoff, R. (1955). A numerical method for treating fluid flow in the presence of shocks. *Los Alamos Scientific Laboratory Report LA-1930*.

**Used in**: `src/lagrangian_solver/numerics/artificial_viscosity.py:45-55`

**Description**: Linear artificial viscosity term Q_lin = c₁ × ρ × c × dx × |∂u/∂x| that supplements the quadratic VNR viscosity. The linear term damps high-frequency oscillations at the shock head that the quadratic term alone cannot eliminate. As noted in the original report: "Although quadratic viscosity successfully controlled unphysical post-shock oscillations, it did NOT totally eliminate them nor the troublesome overshoots that typically appeared at the head of the numerical shock front."

### Compatible Energy Discretization

**Reference**: Caramana, E.J., Burton, D.E., Shashkov, M.J., & Whalen, P.P. (1998). The construction of compatible hydrodynamics algorithms utilizing conservation of total energy. *Journal of Computational Physics*, 146(1), 227-262. DOI: 10.1006/jcph.1998.6029

**Used in**:
- `src/lagrangian_solver/equations/conservation.py` - CompatibleConservation class
- `src/lagrangian_solver/numerics/time_integration.py` - CompatibleHeunIntegrator class
- `src/lagrangian_solver/core/solver.py` - CompatibleLagrangianSolver class
- `src/lagrangian_solver/core/state.py:203-280` - from_internal_energy() method

**Description**: Compatible energy discretization for Lagrangian hydrodynamics. The key insight is using the SAME cell-centered stress σ = p + Q in both momentum and energy equations:
- Momentum: du_j/dt = -(σ_j - σ_{j-1}) / dm_face
- Energy: de_i/dt = -σ_i × dτ_i/dt = -σ_i × (u_{i+1} - u_i) / dm_i

This guarantees exact total energy conservation to machine precision (O(10^-15) relative error). Internal energy e is the PRIMARY evolved variable; total energy E = e + ½u² is derived for diagnostics only.

### Lagrangian Staggered-Grid Energy Conservation

**Reference**: Burton, D.E. (1992). Conservation of energy, momentum, and angular momentum in Lagrangian staggered-grid hydrodynamics. *Lawrence Livermore National Laboratory Report UCRL-JC-105926*.

**Used in**: `src/lagrangian_solver/equations/conservation.py`, `src/lagrangian_solver/core/solver.py`

**Description**: Theoretical foundation for exact conservation properties on staggered Lagrangian grids. The staggered arrangement (cell-centered thermodynamics, face-centered velocity) enables exact discrete conservation when the discretization is "compatible" - using consistent operators for momentum and energy.

### Rankine-Hugoniot Shock Relations for Piston BC

**Reference**: [Toro2009] Chapter 3, Sections 3.1-3.2

**Used in**: `src/lagrangian_solver/boundary/piston.py:140-220` - CompatiblePistonBC class

**Description**: Exact Rankine-Hugoniot jump conditions for computing wall pressure at a moving piston. For compression (shock), uses Newton iteration to solve:
- f(p*) = Δu - (p* - p) × A, where A = √[2/(ρ(γ+1)) × (p* + (γ-1)/(γ+1) × p)]

For expansion (rarefaction), uses isentropic relation:
- p*/p = [1 + (γ-1)/2 × Δu/c]^(2γ/(γ-1))

### GDTk L1D Reference Implementation

**Reference**: GDTk - Gas Dynamics Toolkit, L1D (1D Lagrangian Gas Dynamics). https://github.com/gdtk-uq/gdtk

**Used in**: Design reference for compatible Lagrangian solver architecture

**Description**: Open-source reference implementation of 1D Lagrangian gas dynamics with compatible energy discretization. Used as architectural reference for the solver design, particularly the treatment of staggered variables and boundary conditions.

### Artificial Heat Flux for Contact Discontinuity Spreading

**Reference**: Noh, W.F. (1987). "Errors for calculations of strong shocks using an artificial viscosity and an artificial heat flux." *Journal of Computational Physics*, 72(1), 78-120. DOI: 10.1016/0021-9991(87)90074-X

**Used in**: `src/lagrangian_solver/numerics/artificial_heat_conduction.py`

**Description**: Artificial heat conduction for spreading contact discontinuities in Lagrangian hydrodynamics. Analogous to artificial viscosity for shocks, artificial heat flux smooths contact discontinuities (density jumps with constant pressure) over several cells. The formulation uses density gradient as the activation switch:

- Linear term: κ_lin = κ₁ × ρ × c × dx × |d(ln ρ)/dx| (damps oscillations)
- Quadratic term: κ_quad = κ₂ × ρ × dx² × |d(ln ρ)/dx|² (controls contact width)

The conservative divergence form ensures exact energy conservation:
    de/dt = -σ × dτ/dt - dq/dm
where q = -κ × dT/dx is the heat flux at cell faces.

### Riemann-Based Ghost Cell Boundary Condition for Piston Problems

**Reference**: Toro, E.F. (2009). "Riemann Solvers and Numerical Methods for Fluid Dynamics", 3rd Ed., Springer, Section 6.3.

**Reference**: Burton, D.E. (1992). Conservation of energy, momentum, and angular momentum in Lagrangian staggered-grid hydrodynamics. *Lawrence Livermore National Laboratory Report UCRL-JC-105926*.

**Used in**:
- `src/lagrangian_solver/boundary/piston.py:571-831` - RiemannGhostPistonBC class
- `src/lagrangian_solver/equations/conservation.py:179-219` - Ghost cell stress in momentum equation
- `tests/verification/piston_shock_plots.py:431-436` - BC usage

**Description**: For piston-driven flows, the `RiemannGhostPistonBC` solves the boundary Riemann problem to obtain the correct wall pressure. This interface stress is used in the momentum equation for the first interior face, which then correctly propagates to the energy equation via the compatible formulation `d_e = -σ × dτ`.

Key insight: The compatible energy discretization works correctly **without explicit energy correction** when the momentum equation properly handles the boundary stress. Direct energy correction at the boundary was found to cause energy overshoot due to cumulative effects.

### Flame-Coupled Piston Velocity (Clavin-Tofaili Formulation)

**Reference**: Clavin, P., & Tofaili, H. (2021). Theory of flame-driven piston motion in confined combustion.

**Used in**:
- `scripts/flame_elongation_trajectory/flame_property_interpolator.py` - 2D P-T flame property interpolation
- `scripts/flame_elongation_trajectory/flame_elongation_trajectory.py` - Elongation functions σ(t)
- `scripts/flame_elongation_trajectory/flame_coupled_piston_bc.py` - Iterative BC implementation
- `scripts/flame_elongation_trajectory/test_flame_coupled_piston.py` - Verification tests

**Description**: Implements the Clavin-Tofaili flame elongation model for piston velocity:

    u_p = (σ(t) - 1) × (ρ_u / ρ_b) × S_L

where:
- σ(t) is an imposed elongation function (power law, exponential, linear)
- ρ_u / ρ_b is the unburned-to-burned density ratio
- S_L is the laminar flame speed

The challenge is that S_L and density ratio depend on local (P, T) at the piston face, but (P, T) depend on u_p through shock relations. This creates a nonlinear coupling requiring iteration at each timestep.

The implementation uses:
1. `FlamePropertyInterpolator` - 2D RegularGridInterpolator for flame data from Cantera
2. `FlameCoupledPistonBC` - Iterative boundary condition that solves the coupling via:
   - Initial guess from uncoupled velocity
   - Boundary Riemann solver for (P_face, T_face)
   - Update velocity from Clavin-Tofaili formula
   - Under-relaxation for stability
   - Convergence check

Reference: [Toro2009] Section 6.3 for boundary Riemann problem formulation.
