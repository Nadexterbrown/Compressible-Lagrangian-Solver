"""
Porous piston boundary condition helpers for PeleC trajectory data.

This module provides factory functions and helpers for creating MovingPorousPistonBC
instances from PeleC trajectory data or synthetic trajectory data.

The core MovingPorousPistonBC class is in:
    lagrangian_solver.boundary.piston.MovingPorousPistonBC

This module adds:
    - GasVelocitySource enum for selecting velocity source
    - GasVelocityModifier for applying scale/offset to velocities
    - Factory functions for common use cases
"""

import sys
from pathlib import Path
from typing import Optional, Callable
from enum import Enum
from dataclasses import dataclass

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from lagrangian_solver.boundary import MovingPorousPistonBC, BoundarySide, ThermalBCType
from lagrangian_solver.equations.eos import EOSBase


class GasVelocitySource(Enum):
    """Source for gas velocity at porous piston surface."""
    TRAJECTORY_BURNED_GAS = "trajectory_burned_gas"  # Use burned_gas_velocity from trajectory
    TRAJECTORY_FLAME_GAS = "trajectory_flame_gas"    # Use flame_gas_velocity from trajectory
    TRAJECTORY_FLAME = "trajectory_flame"            # Use flame_velocity from trajectory
    CUSTOM_FUNCTION = "custom_function"              # Use custom function f(t)


@dataclass
class GasVelocityModifier:
    """
    Modifiers applied to gas velocity: v_final = scale * v_source + offset

    Common uses:
    - scale=1.0, offset=-100: Subtract 100 m/s from source velocity
    - scale=0.5, offset=0: Half the source velocity
    - scale=1.0, offset=0: Use source velocity unchanged
    """
    scale: float = 1.0
    offset: float = 0.0

    def apply(self, v: float) -> float:
        """Apply modification to velocity."""
        return self.scale * v + self.offset


def create_porous_piston_from_trajectory(
    side: BoundarySide,
    eos: EOSBase,
    trajectory,
    gas_velocity_source: GasVelocitySource = GasVelocitySource.TRAJECTORY_BURNED_GAS,
    gas_velocity_modifier: Optional[GasVelocityModifier] = None,
    custom_gas_velocity_func: Optional[Callable[[float], float]] = None,
    velocity_min: Optional[float] = None,
    velocity_max: Optional[float] = None,
    time_offset: float = 0.0,
    thermal_bc: ThermalBCType = ThermalBCType.ADIABATIC,
    piston_temperature: Optional[float] = None,
    tol: float = 1e-8,
) -> MovingPorousPistonBC:
    """
    Create a MovingPorousPistonBC from trajectory data.

    The piston position is taken from the trajectory's position() method.
    The gas velocity is taken from the specified source with optional modification.

    Parameters
    ----------
    side : BoundarySide
        Which side of domain (LEFT or RIGHT)
    eos : EOSBase
        Equation of state
    trajectory : PeleTrajectoryInterpolator or SyntheticTrajectoryInterpolator
        Trajectory data providing position() and velocity methods
    gas_velocity_source : GasVelocitySource
        Source for gas velocity (burned_gas, flame_gas, flame, or custom)
    gas_velocity_modifier : GasVelocityModifier, optional
        Scale and offset applied to gas velocity
    custom_gas_velocity_func : callable, optional
        Custom function f(t) -> v_gas if source is CUSTOM_FUNCTION
    velocity_min : float, optional
        Minimum gas velocity (clamp)
    velocity_max : float, optional
        Maximum gas velocity (clamp)
    time_offset : float
        Time offset for trajectory lookup (sim_time = data_time + offset)
    thermal_bc : ThermalBCType
        Thermal boundary condition type
    piston_temperature : float, optional
        Fixed temperature for isothermal BC
    tol : float
        Riemann solver tolerance

    Returns
    -------
    MovingPorousPistonBC
        Configured porous piston boundary condition
    """
    modifier = gas_velocity_modifier or GasVelocityModifier()

    # Validate custom function
    if gas_velocity_source == GasVelocitySource.CUSTOM_FUNCTION:
        if custom_gas_velocity_func is None:
            raise ValueError(
                "custom_gas_velocity_func required when source is CUSTOM_FUNCTION"
            )

    def piston_position(t: float) -> float:
        """Get piston position from trajectory."""
        t_data = t - time_offset
        return trajectory.position(t_data)

    def gas_velocity(t: float) -> float:
        """Get gas velocity from specified source with modifications."""
        t_data = t - time_offset

        # Get base velocity from source
        if gas_velocity_source == GasVelocitySource.TRAJECTORY_BURNED_GAS:
            v_base = trajectory.burned_gas_velocity(t_data)
            if v_base is None:
                raise ValueError(
                    "Trajectory does not have burned_gas_velocity data. "
                    "Use a different gas_velocity_source."
                )

        elif gas_velocity_source == GasVelocitySource.TRAJECTORY_FLAME_GAS:
            v_base = trajectory.gas_velocity(t_data)

        elif gas_velocity_source == GasVelocitySource.TRAJECTORY_FLAME:
            v_base = trajectory.velocity(t_data)

        elif gas_velocity_source == GasVelocitySource.CUSTOM_FUNCTION:
            v_base = custom_gas_velocity_func(t)

        else:
            raise ValueError(f"Unknown gas velocity source: {gas_velocity_source}")

        # Apply modifier
        v_modified = modifier.apply(v_base)

        # Apply clamps
        if velocity_min is not None:
            v_modified = max(velocity_min, v_modified)
        if velocity_max is not None:
            v_modified = min(velocity_max, v_modified)

        return v_modified

    return MovingPorousPistonBC(
        side=side,
        eos=eos,
        piston_position=piston_position,
        gas_velocity=gas_velocity,
        tol=tol,
        thermal_bc=thermal_bc,
        piston_temperature=piston_temperature,
    )


def create_porous_piston_with_burned_gas(
    side: BoundarySide,
    eos: EOSBase,
    trajectory,
    velocity_scale: float = 1.0,
    velocity_offset: float = 0.0,
    **kwargs,
) -> MovingPorousPistonBC:
    """
    Convenience function: Create porous piston using burned_gas_velocity.

    Parameters
    ----------
    side : BoundarySide
        Boundary side
    eos : EOSBase
        Equation of state
    trajectory : trajectory interpolator
        Trajectory data with burned_gas_velocity
    velocity_scale : float
        Scale factor for gas velocity
    velocity_offset : float
        Offset for gas velocity [m/s]
    **kwargs
        Additional arguments passed to create_porous_piston_from_trajectory

    Returns
    -------
    MovingPorousPistonBC
    """
    modifier = GasVelocityModifier(scale=velocity_scale, offset=velocity_offset)
    return create_porous_piston_from_trajectory(
        side=side,
        eos=eos,
        trajectory=trajectory,
        gas_velocity_source=GasVelocitySource.TRAJECTORY_BURNED_GAS,
        gas_velocity_modifier=modifier,
        **kwargs,
    )


def create_porous_piston_with_flame_gas(
    side: BoundarySide,
    eos: EOSBase,
    trajectory,
    velocity_scale: float = 1.0,
    velocity_offset: float = 0.0,
    **kwargs,
) -> MovingPorousPistonBC:
    """
    Convenience function: Create porous piston using flame_gas_velocity.

    Parameters
    ----------
    side : BoundarySide
        Boundary side
    eos : EOSBase
        Equation of state
    trajectory : trajectory interpolator
        Trajectory data with gas_velocity (flame_gas_velocity)
    velocity_scale : float
        Scale factor for gas velocity
    velocity_offset : float
        Offset for gas velocity [m/s]
    **kwargs
        Additional arguments passed to create_porous_piston_from_trajectory

    Returns
    -------
    MovingPorousPistonBC
    """
    modifier = GasVelocityModifier(scale=velocity_scale, offset=velocity_offset)
    return create_porous_piston_from_trajectory(
        side=side,
        eos=eos,
        trajectory=trajectory,
        gas_velocity_source=GasVelocitySource.TRAJECTORY_FLAME_GAS,
        gas_velocity_modifier=modifier,
        **kwargs,
    )


def create_porous_piston_with_custom_velocity(
    side: BoundarySide,
    eos: EOSBase,
    trajectory,
    gas_velocity_func: Callable[[float], float],
    **kwargs,
) -> MovingPorousPistonBC:
    """
    Create porous piston with custom gas velocity function.

    Parameters
    ----------
    side : BoundarySide
        Boundary side
    eos : EOSBase
        Equation of state
    trajectory : trajectory interpolator
        Trajectory data (used for piston position)
    gas_velocity_func : callable
        Function f(t) -> v_gas [m/s]
    **kwargs
        Additional arguments passed to create_porous_piston_from_trajectory

    Returns
    -------
    MovingPorousPistonBC
    """
    return create_porous_piston_from_trajectory(
        side=side,
        eos=eos,
        trajectory=trajectory,
        gas_velocity_source=GasVelocitySource.CUSTOM_FUNCTION,
        custom_gas_velocity_func=gas_velocity_func,
        **kwargs,
    )


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    from pele_data_loader import PeleDataLoader, PeleTrajectoryInterpolator
    from synthetic_data_loader import load_synthetic_trajectory

    print("=" * 70)
    print("POROUS PISTON BC HELPERS TEST")
    print("=" * 70)

    # Test with PeleC data
    pele_data_dir = Path(__file__).parent / "pele_data" / "truncated_raw_data"
    if pele_data_dir.exists():
        print("\n--- Testing with PeleC data ---")
        loader = PeleDataLoader(pele_data_dir)
        data = loader.load()
        trajectory = PeleTrajectoryInterpolator(data, extrapolate=False)

        # Create dummy EOS
        class DummyEOS:
            def get_gamma(self, rho, p):
                import numpy as np
                return np.array([1.4])
            def sound_speed(self, rho, p):
                import numpy as np
                return np.array([500.0])
            def internal_energy(self, rho, p):
                return p / (0.4 * rho)

        eos = DummyEOS()

        bc = create_porous_piston_with_burned_gas(
            side=BoundarySide.LEFT,
            eos=eos,
            trajectory=trajectory,
        )

        print(f"\nMovingPorousPistonBC created successfully")
        print(f"  At t=0: x_piston={bc.get_piston_position(0)*100:.4f} cm")
        print(f"          v_piston={bc.get_piston_velocity(0):.2f} m/s")
        print(f"          v_gas={bc.get_gas_velocity(0):.2f} m/s")
        print(f"          mass_flux_vel={bc.get_mass_flux_velocity(0):.2f} m/s")

    # Test with synthetic data
    synth_data_path = Path(__file__).parent / "pele_data" / "synthetic_data" / "pele_collective_data.csv"
    if synth_data_path.exists():
        print("\n--- Testing with synthetic data ---")
        trajectory = load_synthetic_trajectory(str(synth_data_path))

        bc = create_porous_piston_with_flame_gas(
            side=BoundarySide.LEFT,
            eos=eos,
            trajectory=trajectory,
            velocity_scale=0.8,
            velocity_offset=-20.0,
        )

        print(f"\nMovingPorousPistonBC with flame_gas created successfully")
        print(f"  At t=50µs: x_piston={bc.get_piston_position(50e-6)*100:.4f} cm")
        print(f"             v_piston={bc.get_piston_velocity(50e-6):.2f} m/s")
        print(f"             v_gas={bc.get_gas_velocity(50e-6):.2f} m/s")
        print(f"             (with scale=0.8, offset=-20)")

    print("\nTest complete.")
