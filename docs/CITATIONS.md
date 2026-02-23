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
