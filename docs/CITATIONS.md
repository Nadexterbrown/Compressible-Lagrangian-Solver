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
