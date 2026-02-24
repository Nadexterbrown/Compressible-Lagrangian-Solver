# 1D Compressible Lagrangian Solver

A robust and accurate Lagrangian fluid dynamics solver for evaluating compressible flows in one dimension.

## Features

- **Lagrangian formulation**: Mesh moves with the fluid, naturally handling material interfaces
- **Compressible flow**: Full support for shock waves, rarefactions, and contact discontinuities
- **Cantera EOS integration**: Real gas thermodynamic properties via Cantera
- **Riemann solvers**: Exact and HLLC approximate Riemann solvers
- **Second-order time integration**: Heun's method (explicit predictor-corrector)
- **Artificial viscosity**: Von Neumann-Richtmyer + Landshoff shock capturing with Noh's wall heating fix

## Installation

```bash
# Clone the repository
git clone https://github.com/Nadexterbrown/Compressible-Lagrangian-Solver.git
cd Compressible-Lagrangian-Solver

# Install dependencies
pip install numpy cantera matplotlib scipy
```

## Quick Start

```python
from lagrangian_solver.core.grid import LagrangianGrid, GridConfig
from lagrangian_solver.core.solver import LagrangianSolver, SolverConfig
from lagrangian_solver.equations.eos import IdealGasEOS
from lagrangian_solver.numerics.artificial_viscosity import ArtificialViscosityConfig

# Create grid
grid = LagrangianGrid(GridConfig(n_cells=100, x_min=0.0, x_max=1.0))

# Create EOS
eos = IdealGasEOS(gamma=1.4, R=287.0)

# Configure solver with artificial viscosity
config = SolverConfig(
    cfl=0.5,
    t_end=0.25,
    artificial_viscosity=ArtificialViscosityConfig(
        c_quad=2.0,  # Quadratic (shock capturing)
        c_lin=0.5,   # Linear (oscillation damping)
        c_heat=0.1,  # Heat conduction (wall heating fix)
    ),
)

# Create and run solver
solver = LagrangianSolver(grid=grid, eos=eos, config=config)
solver.set_riemann_ic(x_disc=0.5, rho_L=1.0, u_L=0.0, p_L=1.0,
                                  rho_R=0.125, u_R=0.0, p_R=0.1)
stats = solver.run()
```

## Boundary Conditions

### Solid Boundaries
- **Adiabatic**: No heat flux through boundary
- **Isothermal**: Fixed temperature at boundary

### Moving Solid Boundaries (Piston)
- **Adiabatic piston**: Prescribed velocity, no heat transfer
- **Isothermal piston**: Prescribed velocity, fixed temperature
- **Porous piston**: Beavers-Joseph slip condition

### Open Boundaries
- First-order extrapolation respecting compressibility

## Artificial Viscosity

The solver includes artificial viscosity for robust shock capturing:

**Von Neumann-Richtmyer (quadratic)**:
```
Q = rho * c_Q * dx^2 * (du/dx)^2   for compression (du/dx < 0)
```

**Landshoff (linear)**:
```
Q_lin = rho * c_L * c * dx * |du/dx|
```

**Noh's artificial heat conduction** (wall heating fix):
```
q = c_H * rho * c * dT
```

This prevents the "wall heating" numerical artifact at reflecting boundaries.

## Verification Tests

### Toro's Five Riemann Problems
Run the standard test suite from [Toro2009] Section 4.3.3:
```bash
python tests/verification/toro_riemann_plots.py
```

### Piston-Driven Shock Waves
Test shock generation by moving pistons at Mach 1-4:
```bash
python tests/verification/piston_shock_plots.py
```

## Project Structure

```
src/lagrangian_solver/
├── core/
│   ├── grid.py          # Lagrangian grid management
│   ├── solver.py        # Main solver orchestrator
│   └── state.py         # Flow state data structures
├── equations/
│   ├── conservation.py  # Conservation law residuals
│   └── eos.py           # Equation of state (Ideal, Cantera)
├── numerics/
│   ├── riemann.py       # Exact and HLLC Riemann solvers
│   ├── time_integration.py  # Heun, Euler, SSPRK3
│   └── artificial_viscosity.py  # Shock capturing
├── boundary/
│   ├── wall.py          # Solid wall conditions
│   ├── piston.py        # Moving boundary conditions
│   └── open.py          # Open/non-reflecting conditions
└── io/
    ├── input.py         # Configuration parsing
    └── output.py        # Results output
```

## References

- Toro, E.F. (2009). *Riemann Solvers and Numerical Methods for Fluid Dynamics* (3rd ed.). Springer.
- Despres, B. (2017). *Numerical Methods for Eulerian and Lagrangian Conservation Laws*. Birkhauser.
- Noh, W.F. (2001). "Revisiting Wall Heating". *J. Comput. Physics*, 169(1), 405-407.
- Von Neumann, J. & Richtmyer, R.D. (1950). "A method for the numerical calculation of hydrodynamic shocks". *J. Applied Physics*, 21, 232-237.

See `docs/CITATIONS.md` for complete citation list with file locations.

## License

[Add your license here]
