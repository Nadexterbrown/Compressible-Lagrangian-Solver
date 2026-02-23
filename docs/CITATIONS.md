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
