"""
Input configuration and parsing for simulations.

This module provides configuration classes and utilities for setting up
Lagrangian solver simulations.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Any, Optional, List, Union
import json
from pathlib import Path


class RiemannSolverType(Enum):
    """Available Riemann solver types."""

    EXACT = auto()
    HLLC = auto()


class TimeIntegratorType(Enum):
    """Available time integration schemes."""

    FORWARD_EULER = auto()
    HEUN = auto()  # 2nd order, recommended
    SSPRK3 = auto()


class BoundaryConditionType(Enum):
    """Available boundary condition types."""

    SOLID_WALL = auto()
    MOVING_PISTON = auto()
    OPEN = auto()
    SYMMETRY = auto()


@dataclass
class GridParameters:
    """
    Grid configuration parameters.

    Attributes:
        n_cells: Number of computational cells
        x_min: Left boundary position [m]
        x_max: Right boundary position [m]
        stretch_factor: Grid stretching (1.0 = uniform)
    """

    n_cells: int = 100
    x_min: float = 0.0
    x_max: float = 1.0
    stretch_factor: float = 1.0

    def validate(self):
        """Validate grid parameters."""
        if self.n_cells < 1:
            raise ValueError(f"n_cells must be >= 1, got {self.n_cells}")
        if self.x_max <= self.x_min:
            raise ValueError(f"x_max must be > x_min")
        if self.stretch_factor <= 0:
            raise ValueError(f"stretch_factor must be > 0")


@dataclass
class TimeParameters:
    """
    Time integration parameters.

    Attributes:
        t_end: Final simulation time [s]
        cfl: CFL number for time step control
        dt_max: Maximum time step [s] (optional)
        dt_output: Time interval for output [s]
        integrator: Time integration scheme
    """

    t_end: float = 1.0
    cfl: float = 0.5
    dt_max: Optional[float] = None
    dt_output: float = 0.1
    integrator: TimeIntegratorType = TimeIntegratorType.HEUN

    def validate(self):
        """Validate time parameters."""
        if self.t_end <= 0:
            raise ValueError(f"t_end must be > 0, got {self.t_end}")
        if not (0 < self.cfl <= 1):
            raise ValueError(f"cfl must be in (0, 1], got {self.cfl}")
        if self.dt_max is not None and self.dt_max <= 0:
            raise ValueError(f"dt_max must be > 0, got {self.dt_max}")


@dataclass
class NumericsParameters:
    """
    Numerical method parameters.

    Attributes:
        riemann_solver: Type of Riemann solver
        riemann_tol: Tolerance for exact Riemann solver
        riemann_max_iter: Maximum iterations for exact Riemann solver
    """

    riemann_solver: RiemannSolverType = RiemannSolverType.EXACT
    riemann_tol: float = 1e-8
    riemann_max_iter: int = 100


@dataclass
class EOSParameters:
    """
    Equation of state parameters.

    For ideal gas:
        gamma: Ratio of specific heats
        R: Specific gas constant [J/(kg·K)]

    For Cantera:
        mechanism_file: Path to Cantera mechanism file
        fuel: Fuel species
        oxidizer: Oxidizer species string
        phi: Equivalence ratio
    """

    # Ideal gas parameters
    gamma: float = 1.4
    R: float = 287.05

    # Cantera parameters
    use_cantera: bool = False
    mechanism_file: Optional[str] = None
    fuel: Optional[str] = None
    oxidizer: Optional[str] = None
    phi: Optional[float] = None

    def validate(self):
        """Validate EOS parameters."""
        if self.use_cantera:
            if self.mechanism_file is None:
                raise ValueError("mechanism_file required for Cantera EOS")
        else:
            if self.gamma <= 1:
                raise ValueError(f"gamma must be > 1, got {self.gamma}")
            if self.R <= 0:
                raise ValueError(f"R must be > 0, got {self.R}")


@dataclass
class InitialCondition:
    """
    Initial condition specification.

    Supports uniform states and Riemann problems.
    """

    # Uniform initial condition
    rho: float = 1.0
    u: float = 0.0
    p: float = 1e5

    # Riemann problem (optional)
    is_riemann: bool = False
    x_discontinuity: float = 0.5
    rho_L: Optional[float] = None
    u_L: Optional[float] = None
    p_L: Optional[float] = None
    rho_R: Optional[float] = None
    u_R: Optional[float] = None
    p_R: Optional[float] = None

    def validate(self):
        """Validate initial condition."""
        if self.is_riemann:
            required = ["rho_L", "u_L", "p_L", "rho_R", "u_R", "p_R"]
            for attr in required:
                if getattr(self, attr) is None:
                    raise ValueError(f"{attr} required for Riemann problem")


@dataclass
class BoundaryConfig:
    """
    Boundary condition configuration for one boundary.

    Attributes:
        bc_type: Type of boundary condition
        thermal: Thermal BC type ('adiabatic' or 'isothermal')
        temperature: Wall/piston temperature [K] (for isothermal)
        velocity: Piston velocity [m/s] or profile name
        back_pressure: Back pressure [Pa] (for open BC)
        porous: Whether boundary is porous
        permeability: Darcy permeability [m²]
        slip_coefficient: Beavers-Joseph slip coefficient
    """

    bc_type: BoundaryConditionType = BoundaryConditionType.SOLID_WALL
    thermal: str = "adiabatic"
    temperature: Optional[float] = None
    velocity: Union[float, str, None] = None
    back_pressure: Optional[float] = None
    porous: bool = False
    permeability: float = 0.0
    slip_coefficient: float = 0.0


@dataclass
class OutputParameters:
    """
    Output configuration.

    Attributes:
        output_dir: Directory for output files
        format: Output format ('hdf5', 'csv', or 'both')
        save_every_n_steps: Save every N time steps (alternative to dt_output)
        fields: List of fields to output
    """

    output_dir: str = "./output"
    format: str = "csv"
    save_every_n_steps: Optional[int] = None
    fields: List[str] = field(
        default_factory=lambda: ["x", "rho", "u", "p", "T", "e"]
    )


@dataclass
class SimulationConfig:
    """
    Complete simulation configuration.

    Aggregates all configuration parameters for a simulation run.
    """

    grid: GridParameters = field(default_factory=GridParameters)
    time: TimeParameters = field(default_factory=TimeParameters)
    numerics: NumericsParameters = field(default_factory=NumericsParameters)
    eos: EOSParameters = field(default_factory=EOSParameters)
    initial: InitialCondition = field(default_factory=InitialCondition)
    bc_left: BoundaryConfig = field(default_factory=BoundaryConfig)
    bc_right: BoundaryConfig = field(default_factory=BoundaryConfig)
    output: OutputParameters = field(default_factory=OutputParameters)

    def validate(self):
        """Validate all configuration parameters."""
        self.grid.validate()
        self.time.validate()
        self.eos.validate()
        self.initial.validate()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SimulationConfig":
        """
        Create configuration from dictionary.

        Args:
            data: Configuration dictionary

        Returns:
            SimulationConfig instance
        """
        config = cls()

        if "grid" in data:
            for key, value in data["grid"].items():
                setattr(config.grid, key, value)

        if "time" in data:
            for key, value in data["time"].items():
                if key == "integrator":
                    value = TimeIntegratorType[value.upper()]
                setattr(config.time, key, value)

        if "numerics" in data:
            for key, value in data["numerics"].items():
                if key == "riemann_solver":
                    value = RiemannSolverType[value.upper()]
                setattr(config.numerics, key, value)

        if "eos" in data:
            for key, value in data["eos"].items():
                setattr(config.eos, key, value)

        if "initial" in data:
            for key, value in data["initial"].items():
                setattr(config.initial, key, value)

        if "bc_left" in data:
            for key, value in data["bc_left"].items():
                if key == "bc_type":
                    value = BoundaryConditionType[value.upper()]
                setattr(config.bc_left, key, value)

        if "bc_right" in data:
            for key, value in data["bc_right"].items():
                if key == "bc_type":
                    value = BoundaryConditionType[value.upper()]
                setattr(config.bc_right, key, value)

        if "output" in data:
            for key, value in data["output"].items():
                setattr(config.output, key, value)

        return config

    @classmethod
    def from_json(cls, filepath: Union[str, Path]) -> "SimulationConfig":
        """
        Load configuration from JSON file.

        Args:
            filepath: Path to JSON configuration file

        Returns:
            SimulationConfig instance
        """
        with open(filepath, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "grid": {
                "n_cells": self.grid.n_cells,
                "x_min": self.grid.x_min,
                "x_max": self.grid.x_max,
                "stretch_factor": self.grid.stretch_factor,
            },
            "time": {
                "t_end": self.time.t_end,
                "cfl": self.time.cfl,
                "dt_max": self.time.dt_max,
                "dt_output": self.time.dt_output,
                "integrator": self.time.integrator.name,
            },
            "numerics": {
                "riemann_solver": self.numerics.riemann_solver.name,
                "riemann_tol": self.numerics.riemann_tol,
                "riemann_max_iter": self.numerics.riemann_max_iter,
            },
            "eos": {
                "gamma": self.eos.gamma,
                "R": self.eos.R,
                "use_cantera": self.eos.use_cantera,
                "mechanism_file": self.eos.mechanism_file,
            },
            "initial": {
                "rho": self.initial.rho,
                "u": self.initial.u,
                "p": self.initial.p,
                "is_riemann": self.initial.is_riemann,
            },
            "bc_left": {
                "bc_type": self.bc_left.bc_type.name,
                "thermal": self.bc_left.thermal,
            },
            "bc_right": {
                "bc_type": self.bc_right.bc_type.name,
                "thermal": self.bc_right.thermal,
            },
            "output": {
                "output_dir": self.output.output_dir,
                "format": self.output.format,
            },
        }

    def to_json(self, filepath: Union[str, Path]) -> None:
        """
        Save configuration to JSON file.

        Args:
            filepath: Path to output JSON file
        """
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


# Standard input parameter configuration as specified in CLAUDE.md
INPUT_PARAMS_CONFIG = {
    "T": 503,  # Temperature [K]
    "P": 10e5,  # Pressure [Pa]
    "Phi": 1.0,  # Equivalence ratio
    "Fuel": "H2",  # Fuel species
    "Oxidizer": "O2:1, N2:3.76",  # Oxidizer (air)
    "mech": "../../chemical_mechanism/LiDryer.yaml",  # Cantera mechanism file
}
