"""
Reconstruct experimental flame simulations using 1D Lagrangian solver.

Uses experimental trajectory data to drive a piston boundary condition,
allowing comparison between experiment and 1D model.

Generates outputs:
- config.json, timeseries.npz, piston_history.npz
- final_state_x.png, final_state_m.png
- animation_x.mp4, animation_m.mp4
- X-T diagrams: rho_xt.png, u_xt.png, p_xt.png, T_xt.png, e_xt.png, ds_xt.png
- M-T diagrams: rho_mt.png, u_mt.png, p_mt.png, T_mt.png, e_mt.png, ds_mt.png
- snapshots/ directory

Usage:
    python run_experiment_reconstruction.py --fuel c2h6 --phi phi_0.8
    python run_experiment_reconstruction.py --data-file path/to/data.txt
"""

import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any
import cantera as ct


# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from lagrangian_solver import LagrangianGrid, FlowState, GridConfig
from lagrangian_solver.core.state import create_uniform_state
from lagrangian_solver.core.solver import CompatibleLagrangianSolver as LagrangianSolver, SolverConfig
from lagrangian_solver.equations.eos import CanteraEOS
from lagrangian_solver.boundary.base import BoundarySide, ThermalBCType
from lagrangian_solver.boundary.open import OpenBC
from lagrangian_solver.boundary import MovingDataDrivenPistonBC, MovingPorousPistonBC

from experiment_data_loader import (
    ExperimentalDataLoader,
    ExperimentalTrajectoryInterpolator,
    get_available_cases,
)


# Chemical mechanisms directory
MECH_DIR = Path(__file__).parent.parent.parent / 'src' / 'chemical_mechanisms'

# Initial conditions for experimental cases (0.3 atm, ~303 K ambient)
# These are approximate conditions for the experiments
EXPERIMENT_CONDITIONS = {
    'c2h6': {
        'T': 300,           # Temperature [K] (ambient)
        'P': 0.3 * ct.one_atm,         # Pressure [Pa] (0.3 atm)
        'Fuel': 'C2H6',     # Ethane
        'Oxidizer': 'O2:1', # Pure oxygen
        'mech': str(MECH_DIR / 'sandiego_mechCK.yaml'),
    },
    'c4h10': {
        'T': 300,           # Temperature [K] (ambient)
        'P': 0.3 * ct.one_atm,         # Pressure [Pa] (0.3 atm)
        'Fuel': 'C4H10',    # n-Butane
        'Oxidizer': 'O2:1', # Pure oxygen
        'mech': str(MECH_DIR / 'sandiego_mechCK.yaml'),
    },
}

# Equivalence ratios for each case
PHI_VALUES = {
    'phi_0.8': 0.8,
    'phi_1.0': 1.0,
    'phi_1.3': 1.3,
    'phi_1.49': 1.49,
    'phi_1.6': 1.6,
}


def create_initial_state(grid, eos, T, P):
    """Create initial uniform state."""
    eos.set_state_TP(T, P)
    gas = eos.gas
    rho_init = gas.density
    c_init = gas.sound_speed

    state = create_uniform_state(
        n_cells=grid.n_cells,
        x_left=grid.x[0],
        x_right=grid.x[-1],
        rho=rho_init,
        u=0.0,
        p=P,
        eos=eos,
    )

    return state, rho_init, c_init


# =============================================================================
# Output Generation Functions
# =============================================================================

def plot_final_state(saved_data: Dict, config: Dict, output_file: str, use_mass_coord: bool = False):
    """Create final state plot in physical or mass coordinates."""
    n_times = len(saved_data['t'])
    n_cells_initial = config['n_cells']

    # Get final state
    x_final = saved_data['x'][-1]
    x_centers = 0.5 * (x_final[:-1] + x_final[1:])
    rho = saved_data['rho'][-1]
    n_cells_final = len(rho)

    # Mass coordinate (use final cell count, not initial)
    rho_init = config['rho_init']
    L_init = config['domain_length']
    dm = rho_init * L_init / n_cells_initial
    m_centers = (np.arange(n_cells_final) + 0.5) * dm

    if use_mass_coord:
        x_plot = m_centers
        x_label = 'm [kg/m²]'
        coord_name = 'Mass'
    else:
        x_plot = x_centers * 100  # Convert to cm
        x_label = 'x [cm]'
        coord_name = 'Physical'

    u_nodes = saved_data['u'][-1]
    u = 0.5 * (u_nodes[:-1] + u_nodes[1:])
    p = saved_data['p'][-1]
    T = saved_data['T'][-1]
    e = saved_data['e'][-1]
    s = saved_data['s'][-1]
    t_final = saved_data['t'][-1]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Density
    ax = axes[0, 0]
    ax.plot(x_plot, rho, 'b-', lw=1.5)
    ax.set_xlabel(x_label)
    ax.set_ylabel('Density [kg/m³]')
    ax.set_title('Density')
    ax.grid(True, alpha=0.3)

    # Velocity
    ax = axes[0, 1]
    ax.plot(x_plot, u, 'g-', lw=1.5)
    ax.set_xlabel(x_label)
    ax.set_ylabel('Velocity [m/s]')
    ax.set_title('Velocity')
    ax.grid(True, alpha=0.3)

    # Pressure
    ax = axes[0, 2]
    ax.plot(x_plot, p / 1e6, 'r-', lw=1.5)
    ax.set_xlabel(x_label)
    ax.set_ylabel('Pressure [MPa]')
    ax.set_title('Pressure')
    ax.grid(True, alpha=0.3)

    # Temperature
    ax = axes[1, 0]
    ax.plot(x_plot, T, 'm-', lw=1.5)
    ax.set_xlabel(x_label)
    ax.set_ylabel('Temperature [K]')
    ax.set_title('Temperature')
    ax.grid(True, alpha=0.3)

    # Internal Energy
    ax = axes[1, 1]
    ax.plot(x_plot, e / 1e6, 'c-', lw=1.5)
    ax.set_xlabel(x_label)
    ax.set_ylabel('Internal Energy [MJ/kg]')
    ax.set_title('Internal Energy')
    ax.grid(True, alpha=0.3)

    # Entropy
    ax = axes[1, 2]
    s_init = saved_data['s'][0][0]
    ax.plot(x_plot, s - s_init, 'k-', lw=1.5)
    ax.set_xlabel(x_label)
    ax.set_ylabel('Δs [J/(kg·K)]')
    ax.set_title('Entropy Change')
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'Final State ({coord_name} Space) - t = {t_final*1e3:.2f} ms', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_xt_diagrams(saved_data: Dict, config: Dict, output_dir: str, use_mass_coord: bool = False,
                     max_time_points: int = 500):
    """Create X-T (space-time) diagrams for all variables."""
    output_path = Path(output_dir)

    times = saved_data['t']
    n_times_raw = len(times)
    n_cells_initial = len(saved_data['rho'][0])

    # Subsample if too many time points to avoid memory issues
    if n_times_raw > max_time_points:
        skip = n_times_raw // max_time_points
        indices = list(range(0, n_times_raw, skip))
        print(f"  Subsampling X-T data: {n_times_raw} -> {len(indices)} time points")
    else:
        indices = list(range(n_times_raw))

    n_times = len(indices)
    times = saved_data['t'][indices]

    # Compute coordinates using initial cell count (for mass coordinate)
    rho_init = config['rho_init']
    L_init = config['domain_length']
    dm = rho_init * L_init / n_cells_initial
    m_cell = (np.arange(n_cells_initial) + 0.5) * dm

    # Build 2D arrays for each variable (subsampled)
    rho_xt = np.array([saved_data['rho'][i] for i in indices])
    p_xt = np.array([saved_data['p'][i] for i in indices])
    T_xt = np.array([saved_data['T'][i] for i in indices])
    e_xt = np.array([saved_data['e'][i] for i in indices])

    # Velocity needs cell-centering
    u_xt = np.array([0.5 * (saved_data['u'][i][:-1] + saved_data['u'][i][1:])
                     for i in indices])

    # Entropy change
    s_init = saved_data['s'][0][0]
    ds_xt = np.array([saved_data['s'][i] - s_init for i in indices])

    # Physical x coordinates (time-varying)
    x_xt = np.array([0.5 * (saved_data['x'][i][:-1] + saved_data['x'][i][1:])
                     for i in indices])

    # Choose coordinate system
    if use_mass_coord:
        coord_2d = np.tile(m_cell, (n_times, 1))
        x_label = 'm [kg/m²]'
        coord_name = 'mass'
        suffix = '_mt'
    else:
        coord_2d = x_xt
        x_label = 'x [m]'
        coord_name = 'physical'
        suffix = '_xt'

    # Time in milliseconds
    t_ms = times * 1e3

    variables = [
        ('Density', rho_xt, 'kg/m³', 'rho'),
        ('Velocity', u_xt, 'm/s', 'u'),
        ('Pressure', p_xt, 'Pa', 'p'),
        ('Temperature', T_xt, 'K', 'T'),
        ('Internal Energy', e_xt, 'J/kg', 'e'),
        ('Entropy Change', ds_xt, 'J/(kg·K)', 'ds'),
    ]

    for name, data_2d, unit, var_name in variables:
        fig, ax = plt.subplots(figsize=(10, 8))

        T_mesh, _ = np.meshgrid(t_ms, np.arange(n_cells_initial), indexing='ij')
        X_plot = coord_2d

        pcm = ax.pcolormesh(X_plot, T_mesh, data_2d, shading='auto', cmap='viridis')
        fig.colorbar(pcm, ax=ax, label=f'{name} [{unit}]')

        ax.set_xlabel(x_label)
        ax.set_ylabel('Time [ms]')
        ax.set_title(f'{name} - {coord_name.title()} Space')

        plt.tight_layout()
        filename = output_path / f'{var_name}{suffix}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close(fig)

    print(f"  Saved X-T diagrams ({coord_name}): {output_path}")


def create_animation(saved_data: Dict, config: Dict, output_file: str,
                     use_mass_coord: bool = False, fps: int = 30):
    """Create MP4 animation from simulation results using OpenCV."""
    try:
        import cv2
    except ImportError:
        print(f"  Skipping animation (cv2 not available)")
        return

    n_times = len(saved_data['t'])
    n_cells = len(saved_data['rho'][0])

    # Mass coordinate
    rho_init = config['rho_init']
    L_init = config['domain_length']
    dm = rho_init * L_init / n_cells
    m_centers = (np.arange(n_cells) + 0.5) * dm

    # Determine axis limits
    rho_min = min(np.min(saved_data['rho'][i]) for i in range(n_times))
    rho_max = max(np.max(saved_data['rho'][i]) for i in range(n_times))
    p_min = min(np.min(saved_data['p'][i]) for i in range(n_times))
    p_max = max(np.max(saved_data['p'][i]) for i in range(n_times))
    u_min = min(np.min(0.5*(saved_data['u'][i][:-1]+saved_data['u'][i][1:])) for i in range(n_times))
    u_max = max(np.max(0.5*(saved_data['u'][i][:-1]+saved_data['u'][i][1:])) for i in range(n_times))
    T_min = min(np.min(saved_data['T'][i]) for i in range(n_times))
    T_max = max(np.max(saved_data['T'][i]) for i in range(n_times))

    # Add margins
    def add_margin(vmin, vmax, frac=0.05):
        margin = (vmax - vmin) * frac
        return vmin - margin, vmax + margin

    rho_lim = add_margin(rho_min, rho_max)
    p_lim = add_margin(p_min / 1e6, p_max / 1e6)
    u_lim = add_margin(u_min, u_max)
    T_lim = add_margin(T_min, T_max)

    if use_mass_coord:
        x_lim = (0, m_centers[-1] * 1.05)
        x_label = 'm [kg/m²]'
        coord_name = 'Mass Coordinate'
    else:
        x_max = max(np.max(saved_data['x'][i]) for i in range(n_times))
        x_lim = (0, x_max * 100 * 1.05)
        x_label = 'x [cm]'
        coord_name = 'Physical Space'

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    lines = {}
    for ax in axes.flat:
        ax.set_xlim(x_lim)

    axes[0, 0].set_ylim(rho_lim)
    axes[0, 0].set_ylabel('Density [kg/m³]')
    axes[0, 1].set_ylim(u_lim)
    axes[0, 1].set_ylabel('Velocity [m/s]')
    axes[1, 0].set_ylim(p_lim)
    axes[1, 0].set_ylabel('Pressure [MPa]')
    axes[1, 1].set_ylim(T_lim)
    axes[1, 1].set_ylabel('Temperature [K]')

    for ax in axes.flat:
        ax.set_xlabel(x_label)
        ax.grid(True, alpha=0.3)

    lines['rho'], = axes[0, 0].plot([], [], 'b-', lw=1.5)
    lines['u'], = axes[0, 1].plot([], [], 'g-', lw=1.5)
    lines['p'], = axes[1, 0].plot([], [], 'r-', lw=1.5)
    lines['T'], = axes[1, 1].plot([], [], 'm-', lw=1.5)

    title = fig.suptitle('', fontsize=14)

    def update(frame_idx):
        t = saved_data['t'][frame_idx]
        x_nodes = saved_data['x'][frame_idx]
        x_centers = 0.5 * (x_nodes[:-1] + x_nodes[1:])

        rho = saved_data['rho'][frame_idx]
        u_nodes = saved_data['u'][frame_idx]
        u = 0.5 * (u_nodes[:-1] + u_nodes[1:])
        p = saved_data['p'][frame_idx]
        T = saved_data['T'][frame_idx]

        if use_mass_coord:
            x_plot = m_centers
        else:
            x_plot = x_centers * 100

        lines['rho'].set_data(x_plot, rho)
        lines['u'].set_data(x_plot, u)
        lines['p'].set_data(x_plot, p / 1e6)
        lines['T'].set_data(x_plot, T)

        title.set_text(f'Experiment Reconstruction ({coord_name})\nt = {t*1e3:.2f} ms')

    # Subsample frames for reasonable file size
    frame_skip = max(1, n_times // 500)
    frames = list(range(0, n_times, frame_skip))
    n_frames = len(frames)

    # Initialize video writer
    fig.tight_layout()
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    # Generate frames
    for i, frame_idx in enumerate(frames):
        update(frame_idx)
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        video_writer.write(img_bgr)

    video_writer.release()
    plt.close(fig)
    print(f"  Saved animation: {output_file}")


def save_snapshots(saved_data: Dict, config: Dict, output_dir: str, snapshot_interval: int = 1):
    """Save simulation snapshots in format compatible with interactive_plotter.py."""
    output_path = Path(output_dir)
    snapshots_path = output_path / 'snapshots'
    snapshots_path.mkdir(parents=True, exist_ok=True)

    n_frames = len(saved_data['t'])
    n_cells_initial = config['n_cells']
    rho_init = config['rho_init']
    L_init = config['domain_length']

    # Mass per cell (fixed)
    dm = rho_init * L_init / n_cells_initial

    saved_count = 0
    for i in range(0, n_frames, snapshot_interval):
        t_snap = saved_data['t'][i]
        x_snap = saved_data['x'][i]
        x_centers = 0.5 * (x_snap[:-1] + x_snap[1:])

        u_snap = saved_data['u'][i]
        u_centers = 0.5 * (u_snap[:-1] + u_snap[1:])

        n_cells_frame = len(saved_data['rho'][i])
        m_centers = (np.arange(n_cells_frame) + 0.5) * dm

        save_dict = {
            't': t_snap,
            'step': i,
            'x_centers': x_centers,
            'x_interfaces': x_snap,
            'm_centers': m_centers,
            'rho': saved_data['rho'][i],
            'u': u_centers,
            'u_nodes': u_snap,
            'p': saved_data['p'][i],
            'e': saved_data['e'][i],
            'T': saved_data['T'][i],
            's': saved_data['s'][i],
            'piston_velocity': saved_data['u_piston'][i],
        }

        np.savez(snapshots_path / f'snapshot_{saved_count:06d}.npz', **save_dict)
        saved_count += 1

    # Save piston history
    np.savez(
        output_path / 'piston_history.npz',
        times=saved_data['t'],
        piston_velocity=saved_data['u_piston'],
    )

    print(f"  Saved snapshots: {saved_count} files in {snapshots_path}")


def plot_velocity_comparison(saved_data: Dict, traj_data, output_file: str):
    """Plot velocity comparison between 1D solver and trajectory data."""
    fig, ax = plt.subplots(figsize=(10, 6))

    times_ms = saved_data['t'] * 1e3

    # Plot piston velocity (grid motion) - should match flame velocity
    ax.plot(times_ms, saved_data['u_piston'], 'b-', lw=2, label='Piston velocity (1D solver)')

    # Plot trajectory flame velocity
    ax.plot(traj_data.time * 1e3, traj_data.flame_velocity, 'r--', lw=1.5, alpha=0.7, label='Experimental flame velocity')

    ax.set_xlabel('Time [ms]')
    ax.set_ylabel('Velocity [m/s]')
    ax.set_title('Velocity Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close(fig)


# =============================================================================
# Main Simulation
# =============================================================================

def run_reconstruction(
    fuel: str,
    phi_name: str,
    data_file: str = None,
    domain_length: float = 3.0,
    n_cells: int = 600,
    cfl: float = 0.4,
    av_linear: float = 0.3,
    av_quad: float = 2.0,
    velocity_scale: float = 1.0,
    velocity_offset: float = 0.0,
    velocity_min: float = None,
    output_dir: str = None,
    # Mixture configuration (optional overrides)
    T_init: float = None,
    P_init: float = None,
    fuel_species: str = None,
    oxidizer: str = None,
    mechanism: str = None,
    # Density ratio BC option
    use_density_ratio_bc: bool = False,
    # Minimum timestep (enforced, may violate CFL)
    dt_min: float = None,
):
    """
    Run experiment reconstruction simulation.

    Parameters
    ----------
    fuel : str
        Fuel type ('c2h6' or 'c4h10')
    phi_name : str
        Equivalence ratio name (e.g., 'phi_0.8', 'phi_1.0')
    data_file : str, optional
        Path to data file. If None, auto-detects from fuel/phi.
    domain_length : float
        Domain length [m]
    n_cells : int
        Number of cells
    cfl : float
        CFL number
    av_linear : float
        Linear artificial viscosity coefficient
    av_quad : float
        Quadratic artificial viscosity coefficient
    velocity_scale : float
        Scale factor for piston velocity (default 1.0)
    velocity_offset : float
        Value to add to scaled velocity [m/s] (default 0.0)
    velocity_min : float, optional
        Minimum allowed piston velocity [m/s]
    output_dir : str, optional
        Output directory
    T_init : float, optional
        Initial temperature [K]. If None, uses default for fuel.
    P_init : float, optional
        Initial pressure [Pa]. If None, uses default for fuel.
    fuel_species : str, optional
        Cantera fuel species name. If None, uses default for fuel.
    oxidizer : str, optional
        Cantera oxidizer string. If None, uses default for fuel.
    mechanism : str, optional
        Cantera mechanism file. If None, uses default for fuel.
    use_density_ratio_bc : bool
        If True, use porous BC with gas velocity computed from density ratio.
        Gas velocity = ((sigma-1)/sigma) * U_f, where sigma = rho_u/rho_b.
        rho_u from simulation state (cell 0), rho_b from Cantera equilibrate('HP').
    dt_min : float, optional
        Minimum timestep [s]. If CFL requires smaller dt, this value is used instead.
        WARNING: May violate CFL and cause numerical instability at high compressions.
    """
    print("=" * 70)
    print("EXPERIMENT RECONSTRUCTION SIMULATION")
    print("=" * 70)

    # Get default conditions for this fuel
    if fuel not in EXPERIMENT_CONDITIONS:
        raise ValueError(f"Unknown fuel: {fuel}. Use 'c2h6' or 'c4h10'.")

    default_conditions = EXPERIMENT_CONDITIONS[fuel]
    phi = PHI_VALUES.get(phi_name, 1.0)

    # Build conditions with optional overrides
    conditions = {
        'T': T_init if T_init is not None else default_conditions['T'],
        'P': P_init if P_init is not None else default_conditions['P'],
        'Fuel': fuel_species if fuel_species is not None else default_conditions['Fuel'],
        'Oxidizer': oxidizer if oxidizer is not None else default_conditions['Oxidizer'],
        'mech': mechanism if mechanism is not None else default_conditions['mech'],
    }

    print(f"\nFuel: {fuel.upper()}, Phi: {phi}")

    # Find data file
    if data_file is None:
        base_dir = Path(__file__).parent
        cases = get_available_cases(str(base_dir))
        key = (fuel, phi_name)
        if key not in cases:
            raise FileNotFoundError(
                f"No data file found for {fuel} {phi_name}. "
                f"Available: {list(cases.keys())}"
            )
        data_file = cases[key]
    else:
        data_file = Path(data_file)

    print(f"\nLoading trajectory data from: {data_file}")

    loader = ExperimentalDataLoader(str(data_file))
    data = loader.load()
    trajectory = ExperimentalTrajectoryInterpolator(data, extrapolate=False)

    print(f"  {trajectory}")
    print(f"  Flame velocity range: [{data.flame_velocity.min():.1f}, {data.flame_velocity.max():.1f}] m/s")

    t_end = trajectory.t_max

    # Setup output directory
    if output_dir is None:
        results_subdir = "porous_results" if use_density_ratio_bc else "results"
        output_dir = Path(__file__).parent / results_subdir / f"{fuel}_{phi_name}"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create EOS
    mech_path = conditions['mech']
    print(f"\nCreating EOS with mechanism: {mech_path}")
    try:
        eos = CanteraEOS(mech_path)
    except Exception as e:
        raise RuntimeError(f"Could not load mechanism {mech_path}: {e}")

    eos.set_mixture(conditions['Fuel'], conditions['Oxidizer'], phi)
    eos.set_state_TP(conditions['T'], conditions['P'])

    # Create grid
    grid_config = GridConfig(n_cells=n_cells, x_min=0.0, x_max=domain_length)
    grid = LagrangianGrid(grid_config)

    # Create initial state
    state, rho_init, c_init = create_initial_state(grid, eos, conditions['T'], conditions['P'])

    print(f"\nSimulation setup:")
    print(f"  Domain: {domain_length} m, {n_cells} cells")
    print(f"  t_end: {t_end*1e3:.3f} ms")
    print(f"  Initial: rho={rho_init:.6f} kg/m³, c={c_init:.1f} m/s")
    print(f"  Initial: T={conditions['T']} K, P={conditions['P']/1e5:.2f} bar")
    print(f"  AV: c_linear={av_linear}, c_quad={av_quad}")
    if dt_min is not None:
        print(f"  dt_min: {dt_min*1e9:.1f} ns (WARNING: may violate CFL)")
    if velocity_scale != 1.0:
        print(f"  Velocity scale: {velocity_scale}")
    if velocity_offset != 0.0:
        print(f"  Velocity offset: {velocity_offset:+.1f} m/s")
    if velocity_min is not None:
        print(f"  Velocity min: {velocity_min} m/s")

    # Create boundary conditions
    if use_density_ratio_bc:
        # Create single Cantera gas object for burned gas density calculation
        # This object is reused throughout the simulation - only state is updated
        eq_gas = ct.Solution(conditions['mech'])
        eq_gas.set_equivalence_ratio(phi, conditions['Fuel'], conditions['Oxidizer'])

        # Cache reactant mass fractions to restore quickly (avoids string parsing)
        reactant_Y = eq_gas.Y.copy()

        # Compute initial burned gas state
        eq_gas.TP = conditions['T'], conditions['P']
        eq_gas.equilibrate('HP')
        rho_b_init = eq_gas.density
        T_b_init = eq_gas.T
        sigma_init = rho_init / rho_b_init

        # State container for density ratio BC
        # sigma used for gas velocity calculation, T_b and rho_b stored for output/analysis
        cached_state = {'sigma': sigma_init, 'T_b': T_b_init, 'rho_b': rho_b_init}

        def gas_velocity_from_density_ratio(t: float) -> float:
            """Gas velocity from density ratio: u_g = ((sigma-1)/sigma) * U_f"""
            sigma = cached_state['sigma']
            U_f = trajectory.velocity(t)
            return ((sigma - 1.0) / sigma) * U_f

        left_bc = MovingPorousPistonBC(
            side=BoundarySide.LEFT, eos=eos,
            trajectory=trajectory,
            gas_velocity_func=gas_velocity_from_density_ratio,
            thermal_bc=ThermalBCType.ADIABATIC,
        )

        # Callback to update cached state after each step
        def update_cached_state(state, t, step):
            rho_u = state.rho[0]
            # Restore reactant composition and set current T, P
            eq_gas.Y = reactant_Y
            eq_gas.TP = state.T[0], state.p[0]
            # Equilibrate at constant H, P to get burned gas state
            eq_gas.equilibrate('HP')
            cached_state['sigma'] = rho_u / eq_gas.density
            cached_state['T_b'] = eq_gas.T
            cached_state['rho_b'] = eq_gas.density

    else:
        left_bc = MovingDataDrivenPistonBC(
            side=BoundarySide.LEFT, eos=eos, trajectory=trajectory,
            velocity_scale=velocity_scale,
            velocity_offset=velocity_offset,
            velocity_min=velocity_min,
            thermal_bc=ThermalBCType.ADIABATIC,
        )
        update_cached_state = None  # No callback needed

    right_bc = OpenBC(side=BoundarySide.RIGHT, eos=eos, p_external=conditions['P'])

    # Create solver
    solver_config = SolverConfig(cfl=cfl, av_linear=av_linear, av_quad=av_quad, av_enabled=True, dt_min=dt_min)
    solver = LagrangianSolver(grid=grid, eos=eos, bc_left=left_bc, bc_right=right_bc, config=solver_config)
    solver.set_initial_condition(state)

    # Register step callback if using density ratio BC
    if use_density_ratio_bc and update_cached_state is not None:
        solver.add_step_callback(update_cached_state)

    # Storage for timeseries (buffered, flushed periodically)
    saved_data = {
        't': [], 'x': [], 'rho': [], 'u': [], 'p': [], 'e': [], 'T': [], 's': [],
        'u_piston': [],  # Piston velocity (grid motion)
        'T_b': [],       # Burned gas equilibrium temperature (density ratio BC only)
        'rho_b': [],     # Burned gas equilibrium density (density ratio BC only)
    }

    # Recording interval - time-based to target ~1000 snapshots
    # Record every dt_record seconds of simulation time
    target_snapshots = 1000
    dt_record = t_end / target_snapshots
    t_next_record = 0.0

    # Incremental output: flush to disk every N steps
    flush_interval = 10000  # Write to disk every 10000 steps
    chunk_dir = output_dir / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    chunk_count = 0

    def flush_chunk():
        """Write current buffer to disk and clear it."""
        nonlocal chunk_count
        if len(saved_data['t']) == 0:
            return
        chunk_file = chunk_dir / f"chunk_{chunk_count:04d}.npz"
        chunk_arrays = {k: np.array(v) for k, v in saved_data.items()}
        np.savez(chunk_file, **chunk_arrays)
        print(f"  [Flush] Wrote {len(saved_data['t'])} snapshots to {chunk_file.name}")
        # Clear buffer
        for k in saved_data:
            saved_data[k] = []
        chunk_count += 1

    bc_type = "density ratio porous" if use_density_ratio_bc else "solid piston"
    print(f"\nRunning: {bc_type} reconstruction")
    print(f"  Record every {dt_record*1e6:.1f} μs (target ~{target_snapshots} snapshots)")
    print(f"  Flush to disk every {flush_interval} steps")

    step = 0
    t = 0.0

    while t < t_end:
        current_state = solver.state

        c = eos.sound_speed(current_state.rho, current_state.p)
        dt_cell = grid.dx / (c + np.abs(current_state.u[:-1] + current_state.u[1:]) / 2)
        dt = cfl * np.min(dt_cell)

        # Enforce minimum timestep
        if dt_min is not None and dt < dt_min:
            dt = dt_min

        if t + dt > t_end:
            dt = t_end - t

        # Record state (time-based)
        if t >= t_next_record:
            saved_data['t'].append(t)
            saved_data['x'].append(grid.x.copy())
            saved_data['rho'].append(current_state.rho.copy())
            saved_data['u'].append(current_state.u.copy())
            saved_data['p'].append(current_state.p.copy())
            saved_data['e'].append(current_state.e.copy())
            saved_data['T'].append(current_state.T.copy())
            saved_data['s'].append(current_state.s.copy())
            saved_data['u_piston'].append(left_bc.get_piston_velocity(t))
            # Burned gas equilibrium state (density ratio BC only)
            if use_density_ratio_bc:
                saved_data['T_b'].append(cached_state['T_b'])
                saved_data['rho_b'].append(cached_state['rho_b'])
            else:
                saved_data['T_b'].append(np.nan)
                saved_data['rho_b'].append(np.nan)
            t_next_record += dt_record

        try:
            solver.step_forward(dt)
        except Exception as e:
            print(f"  ERROR at step {step}, t={t*1e3:.3f} ms: {e}")
            flush_chunk()  # Save what we have before breaking
            break

        t += dt
        step += 1

        # Periodic flush to disk
        if step % flush_interval == 0:
            flush_chunk()

        if step % 500 == 0:
            print(f"  Step {step:6d}, t={t*1e3:8.3f} ms, dt={dt*1e9:.1f} ns, "
                  f"u_p={left_bc.get_piston_velocity(t):8.1f} m/s, "
                  f"p_max={current_state.p.max()/1e5:8.2f} bar")

    # Flush any remaining data
    flush_chunk()

    print(f"\nCompleted: {step} steps, {chunk_count} chunks written")

    # Consolidate chunks into single saved_data dict
    print(f"\nConsolidating {chunk_count} chunks...")
    chunk_files = sorted(chunk_dir.glob("chunk_*.npz"))
    consolidated = {k: [] for k in ['t', 'x', 'rho', 'u', 'p', 'e', 'T', 's', 'u_piston', 'T_b', 'rho_b']}
    for chunk_file in chunk_files:
        chunk = np.load(chunk_file, allow_pickle=True)
        for k in consolidated:
            if k in chunk:
                consolidated[k].append(chunk[k])
        chunk.close()

    # Concatenate arrays
    saved_data = {}
    for k, arrays in consolidated.items():
        if arrays:
            saved_data[k] = np.concatenate(arrays, axis=0)
        else:
            saved_data[k] = np.array([])

    print(f"  Total snapshots: {len(saved_data['t'])}")

    # Config dictionary
    config = {
        "case_name": f"experiment_reconstruction_{fuel}_{phi_name}",
        "fuel": fuel,
        "phi": phi,
        "phi_name": phi_name,
        "domain_length": domain_length,
        "n_cells": n_cells,
        "t_end": t_end,
        "cfl": cfl,
        "av_linear": av_linear,
        "av_quad": av_quad,
        "velocity_scale": velocity_scale,
        "velocity_offset": velocity_offset,
        "velocity_min": velocity_min,
        "data_source": str(data_file),
        "n_data_points": len(data.time),
        "data_time_range": [float(data.time[0]), float(data.time[-1])],
        "n_steps": step,
        "rho_init": rho_init,
        "c_init": c_init,
        "T_init": conditions['T'],
        "p_init": conditions['P'],
        "use_density_ratio_bc": use_density_ratio_bc,
    }

    # Save outputs
    print(f"\nSaving results to: {output_dir}")

    # Save timeseries
    np.savez(output_dir / "timeseries.npz", **saved_data, config=config)
    print(f"  Saved: timeseries.npz")

    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Saved: config.json")

    # Final state plots
    plot_final_state(saved_data, config, str(output_dir / "final_state_x.png"), use_mass_coord=False)
    plot_final_state(saved_data, config, str(output_dir / "final_state_m.png"), use_mass_coord=True)
    print(f"  Saved: final_state_x.png, final_state_m.png")

    # X-T diagrams
    plot_xt_diagrams(saved_data, config, str(output_dir), use_mass_coord=False)
    plot_xt_diagrams(saved_data, config, str(output_dir), use_mass_coord=True)

    # Animations
    create_animation(saved_data, config, str(output_dir / "animation_x.mp4"), use_mass_coord=False)
    create_animation(saved_data, config, str(output_dir / "animation_m.mp4"), use_mass_coord=True)

    # Snapshots
    save_snapshots(saved_data, config, str(output_dir), snapshot_interval=1)

    # Velocity comparison plot
    plot_velocity_comparison(saved_data, data, str(output_dir / "velocity_comparison.png"))
    print(f"  Saved: velocity_comparison.png")

    print(f"\nDone! Results saved to: {output_dir}")
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Run experiment reconstruction with 1D Lagrangian solver")
    parser.add_argument("--fuel", type=str, choices=['c2h6', 'c4h10'], default='c2h6',
                        help="Fuel type")
    parser.add_argument("--phi", type=str, default='phi_1.0',
                        help="Equivalence ratio name (e.g., phi_0.8, phi_1.0)")
    parser.add_argument("--data-file", type=str, default=None,
                        help="Path to data file (auto-detects from fuel/phi if not specified)")
    parser.add_argument("--domain", type=float, default=3.0, help="Domain length [m]")
    parser.add_argument("--cells", type=int, default=600, help="Number of cells")
    parser.add_argument("--cfl", type=float, default=0.4, help="CFL number")
    parser.add_argument("--velocity-scale", type=float, default=1.0,
                        help="Scale factor for piston velocity")
    parser.add_argument("--velocity-offset", type=float, default=0.0,
                        help="Value to add to scaled velocity [m/s]")
    parser.add_argument("--velocity-min", type=float, default=None,
                        help="Minimum allowed piston velocity [m/s]")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory")
    parser.add_argument("--use-density-ratio-bc", action="store_true",
                        help="Use porous BC with gas velocity from density ratio")
    parser.add_argument("--dt-min", type=float, default=None,
                        help="Minimum timestep [s] (e.g., 1e-9 for 1 ns). WARNING: May violate CFL.")

    args = parser.parse_args()

    run_reconstruction(
        fuel=args.fuel,
        phi_name=args.phi,
        data_file=args.data_file,
        domain_length=args.domain,
        n_cells=args.cells,
        cfl=args.cfl,
        velocity_scale=args.velocity_scale,
        velocity_offset=args.velocity_offset,
        velocity_min=args.velocity_min,
        output_dir=args.output_dir,
        use_density_ratio_bc=args.use_density_ratio_bc,
        dt_min=args.dt_min,
    )


if __name__ == "__main__":
    main()
