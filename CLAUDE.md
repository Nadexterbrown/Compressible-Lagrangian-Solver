# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

1D Compressible Lagrangian Solver - A robust and accurate Lagrangian fluid dynamics solver for evaluating compressible flows of various and evolving behaviors in one dimension.

## Technical Specifications

### Boundary Conditions
All boundary conditions must be generic and support:

**Solid Boundaries:**
- Adiabatic (no heat flux)
- Isothermal (fixed temperature)

**Moving Solid Boundaries:**
- Adiabatic
- Isothermal
- Porous

**Open Boundaries:**
- First order extrapolation from interior (respects compressibility communication direction)

### Time Integration
- Second order accurate: Heun's method (explicit predictor-corrector)

### Numerical Schemes
- Primary reference: Toro, E.F. "Riemann Solvers and Numerical Methods for Fluid Dynamics" (3rd ed., Springer, 2009)
- Secondary reference (1D Lagrangian formulation): Després, B. "Numerical Methods for Eulerian and Lagrangian Conservation Laws" (Birkhäuser, 2017)

### Equation of State
All EOS calculations handled via Cantera. Application scripts must use:
```python
INPUT_PARAMS_CONFIG = {
    'T': 503,                    # Temperature [K]
    'P': 10e5,                   # Pressure [Pa]
    'Phi': 1.0,                  # Equivalence ratio
    'Fuel': 'H2',                # Fuel species
    'Oxidizer': 'O2:1, N2:3.76', # Oxidizer (air)
    'mech': '../../chemical_mechanism/LiDryer.yaml',  # Cantera mechanism file
}
```

### Core Requirements
- Lagrangian formulation: mesh moves with fluid
- Compressible flow capability
- Support for evolving flow behaviors
- Cantera for thermodynamic property calculations

## Development Guidelines

### Scientific Standards
- Follow accepted scientific coding practices
- All research must be thorough with proper citations
- Every algorithm or method implementation must reference verified sources (academic papers, textbooks, or established codebases)
- Git commits are acceptable as citations for implementation decisions
- All citations must be recorded in `docs/CITATIONS.md` with file paths and line numbers

### Version Control
- Push to git after every change to the Python package
- Maintain a thorough history of all changes
- Commit messages should document what was changed and why
- Do not include "Co-Authored-By" lines in commits

### Language
- Python

### Dependencies
- Cantera (thermodynamic properties and EOS calculations)

## Expected Domain Knowledge

When contributing to this project, familiarity with:
- Lagrangian formulation for fluid dynamics (mesh moves with the fluid)
- Conservation laws: mass, momentum, energy
- Equation of state relationships
- Numerical methods: Riemann solvers, time integration schemes
- Stability and accuracy considerations for compressible flow simulations
