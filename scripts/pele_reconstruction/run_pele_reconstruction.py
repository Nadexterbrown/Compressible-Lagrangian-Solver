"""
Reconstruct PeleC flame simulations using 1D Lagrangian solver.

Uses PeleC trajectory data to drive a piston boundary condition,
allowing comparison between 3D direct simulation and 1D model.

Generates same outputs as piston_trajectory_study:
- config.json, timeseries.npz, piston_history.npz
- final_state_x.png, final_state_m.png
- animation_x.mp4, animation_m.mp4
- X-T diagrams: rho_xt.png, u_xt.png, p_xt.png, T_xt.png, e_xt.png, ds_xt.png
- M-T diagrams: rho_mt.png, u_mt.png, p_mt.png, T_mt.png, e_mt.png, ds_mt.png
- snapshots/ directory

Usage:
    python run_pele_reconstruction.py [--parts 1 2 3]
"""

import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, List


# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from lagrangian_solver import LagrangianGrid, FlowState, GridConfig
from lagrangian_solver.core.state import create_uniform_state
from lagrangian_solver.core.solver import LagrangianSolver, SolverConfig
from lagrangian_solver.equations.eos import CanteraEOS
from lagrangian_solver.boundary.base import BoundarySide, ThermalBCType
from lagrangian_solver.boundary.open import OpenBC
from lagrangian_solver.boundary import MovingDataDrivenPistonBC

from pele_data_loader import PeleDataLoader, PeleTrajectoryInterpolator
from synthetic_data_loader import SyntheticDataLoader, SyntheticTrajectoryInterpolator


class DataSource:
    """Data source types for trajectory loading."""
    PELE = "pele"
    SYNTHETIC = "synthetic"


# Default parameters matching PeleC H2/O2 deflagration
INPUT_PARAMS_CONFIG = {
    'T': 503,                    # Temperature [K]
    'P': 10e5,                   # Pressure [Pa] (10 bar)
    'Phi': 1.0,                  # Equivalence ratio
    'Fuel': 'H2',                # Fuel species
    'Oxidizer': 'O2:1', # Oxidizer (pure oxygen)
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
# Output Generation Functions (matching piston_trajectory_study)
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

    plt.suptitle(f'Final State ({coord_name} Space) - t = {t_final*1e6:.1f} µs', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_xt_diagrams(saved_data: Dict, config: Dict, output_dir: str, use_mass_coord: bool = False):
    """Create X-T (space-time) diagrams for all variables."""
    output_path = Path(output_dir)

    times = saved_data['t']
    n_times = len(times)
    n_cells_initial = len(saved_data['rho'][0])

    # Compute coordinates using initial cell count (for mass coordinate)
    rho_init = config['rho_init']
    L_init = config['domain_length']
    dm = rho_init * L_init / n_cells_initial
    m_cell = (np.arange(n_cells_initial) + 0.5) * dm

    # Build 2D arrays for each variable
    rho_xt = np.array([saved_data['rho'][i] for i in range(n_times)])
    p_xt = np.array([saved_data['p'][i] for i in range(n_times)])
    T_xt = np.array([saved_data['T'][i] for i in range(n_times)])
    e_xt = np.array([saved_data['e'][i] for i in range(n_times)])

    # Velocity needs cell-centering
    u_xt = np.array([0.5 * (saved_data['u'][i][:-1] + saved_data['u'][i][1:])
                     for i in range(n_times)])

    # Entropy change
    s_init = saved_data['s'][0][0]
    ds_xt = np.array([saved_data['s'][i] - s_init for i in range(n_times)])

    # Physical x coordinates (time-varying)
    x_xt = np.array([0.5 * (saved_data['x'][i][:-1] + saved_data['x'][i][1:])
                     for i in range(n_times)])

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

    # Time in microseconds
    t_us = times * 1e6

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

        T_mesh, _ = np.meshgrid(t_us, np.arange(n_cells_initial), indexing='ij')
        X_plot = coord_2d

        pcm = ax.pcolormesh(X_plot, T_mesh, data_2d, shading='auto', cmap='viridis')
        fig.colorbar(pcm, ax=ax, label=f'{name} [{unit}]')

        ax.set_xlabel(x_label)
        ax.set_ylabel('Time [µs]')
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

        title.set_text(f'PeleC Reconstruction ({coord_name})\nt = {t*1e6:.1f} µs')

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

    times_us = saved_data['t'] * 1e6

    # Plot piston velocity (grid motion) - should match flame velocity
    ax.plot(times_us, saved_data['u_piston'], 'b-', lw=2, label='Piston velocity (1D solver)')

    # Plot trajectory flame velocity
    ax.plot(traj_data.time * 1e6, traj_data.flame_velocity, 'r--', lw=1.5, alpha=0.7, label='Trajectory flame velocity')

    ax.set_xlabel('Time [µs]')
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
    data_source: str = DataSource.PELE,
    data_path: str = None,
    parts: list = None,
    domain_length: float = 2.0,
    n_cells: int = 400,
    cfl: float = 0.4,
    av_linear: float = 0.3,
    av_quad: float = 2.0,
    velocity_scale: float = 1.0,
    velocity_offset: float = 0.0,
    velocity_min: float = None,
    output_dir: str = None,
):
    """
    Run PeleC reconstruction simulation.

    Parameters
    ----------
    data_source : str
        "pele" for PeleC raw data, "synthetic" for CSV data
    data_path : str, optional
        Path to data. For "pele": directory with part_*.txt files.
        For "synthetic": path to CSV file.
        If None, uses default paths.
    parts : list, optional
        For PeleC data: which part files to load
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
        Value to subtract from scaled velocity [m/s] (default 0.0)
    velocity_min : float, optional
        Minimum allowed piston velocity [m/s] (default None = no clamping)
    output_dir : str, optional
        Output directory
    """
    print("=" * 70)
    print("PELE RECONSTRUCTION SIMULATION")
    print("=" * 70)

    # Load trajectory data based on source
    if data_source == DataSource.PELE:
        # Load PeleC raw data
        if data_path is None:
            data_dir = Path(__file__).parent / "pele_data" / "truncated_raw_data"
        else:
            data_dir = Path(data_path)

        print(f"\nLoading PeleC trajectory data from: {data_dir}")

        loader = PeleDataLoader(data_dir)
        data = loader.load(parts)
        trajectory = PeleTrajectoryInterpolator(data, extrapolate=False)

        print(f"  {trajectory}")
        print(f"  Flame velocity range: [{data.flame_velocity.min():.1f}, {data.flame_velocity.max():.1f}] m/s")

    elif data_source == DataSource.SYNTHETIC:
        # Load synthetic CSV data
        if data_path is None:
            data_file = Path(__file__).parent / "pele_data" / "synthetic_data" / "pele_collective_data.csv"
        else:
            data_file = Path(data_path)
        data_dir = data_file  # For config compatibility

        print(f"\nLoading synthetic trajectory data from: {data_file}")

        loader = SyntheticDataLoader(data_file)
        data = loader.load()
        trajectory = SyntheticTrajectoryInterpolator(data, extrapolate=False)

        print(f"  {trajectory}")
        print(f"  Flame velocity range: [{data.flame_velocity.min():.1f}, {data.flame_velocity.max():.1f}] m/s")

    else:
        raise ValueError(f"Unknown data source: {data_source}. Use 'pele' or 'synthetic'.")

    t_end = trajectory.t_max

    # Setup output directory
    if output_dir is None:
        output_dir = Path(__file__).parent / "results" / "solid"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create EOS
    mech_path = Path(__file__).parent.parent / "flame_elongation_trajectory" / "cantera_data" / "Li-Dryer-H2-mechanism.yaml"
    print(f"\nCreating EOS with mechanism: {mech_path}")
    eos = CanteraEOS(str(mech_path))
    eos.set_mixture(INPUT_PARAMS_CONFIG['Fuel'], INPUT_PARAMS_CONFIG['Oxidizer'], INPUT_PARAMS_CONFIG['Phi'])
    eos.set_state_TP(INPUT_PARAMS_CONFIG['T'], INPUT_PARAMS_CONFIG['P'])

    # Create grid
    grid_config = GridConfig(n_cells=n_cells, x_min=0.0, x_max=domain_length)
    grid = LagrangianGrid(grid_config)

    # Create initial state
    state, rho_init, c_init = create_initial_state(grid, eos, INPUT_PARAMS_CONFIG['T'], INPUT_PARAMS_CONFIG['P'])

    print(f"\nSimulation setup:")
    print(f"  Domain: {domain_length} m, {n_cells} cells")
    print(f"  t_end: {t_end*1e3:.3f} ms")
    print(f"  Initial: rho={rho_init:.4f} kg/m³, c={c_init:.1f} m/s")
    print(f"  AV: c_linear={av_linear}, c_quad={av_quad}")
    if velocity_scale != 1.0:
        print(f"  Velocity scale: {velocity_scale}")
    if velocity_offset != 0.0:
        print(f"  Velocity offset: -{velocity_offset} m/s")
    if velocity_min is not None:
        print(f"  Velocity min: {velocity_min} m/s")

    # Create boundary conditions
    left_bc = MovingDataDrivenPistonBC(
        side=BoundarySide.LEFT, eos=eos, trajectory=trajectory,
        velocity_scale=velocity_scale,
        velocity_offset=velocity_offset,
        velocity_min=velocity_min,
        thermal_bc=ThermalBCType.ADIABATIC,
    )
    right_bc = OpenBC(side=BoundarySide.RIGHT, eos=eos, p_external=INPUT_PARAMS_CONFIG['P'])

    # Create solver
    solver_config = SolverConfig(cfl=cfl, av_linear=av_linear, av_quad=av_quad, av_enabled=True)
    solver = LagrangianSolver(grid=grid, eos=eos, bc_left=left_bc, bc_right=right_bc, config=solver_config)
    solver.set_initial_condition(state)

    # Storage for timeseries (matching piston_trajectory_study format)
    saved_data = {
        't': [], 'x': [], 'rho': [], 'u': [], 'p': [], 'e': [], 'T': [], 's': [],
        'u_piston': [],  # Piston velocity (grid motion)
    }

    # Recording interval
    dx_min = np.min(grid.dx)
    dt_approx = cfl * dx_min / c_init
    estimated_steps = int(t_end / dt_approx)
    record_interval = max(1, estimated_steps // 1000)

    print(f"\nRunning: solid piston reconstruction")
    print(f"  Record every {record_interval} steps")

    step = 0
    t = 0.0

    while t < t_end:
        current_state = solver.state

        c = eos.sound_speed(current_state.rho, current_state.p)
        dt_cell = grid.dx / (c + np.abs(current_state.u[:-1] + current_state.u[1:]) / 2)
        dt = cfl * np.min(dt_cell)

        if t + dt > t_end:
            dt = t_end - t

        # Record state
        if step % record_interval == 0:
            saved_data['t'].append(t)
            saved_data['x'].append(grid.x.copy())
            saved_data['rho'].append(current_state.rho.copy())
            saved_data['u'].append(current_state.u.copy())
            saved_data['p'].append(current_state.p.copy())
            saved_data['e'].append(current_state.e.copy())
            saved_data['T'].append(current_state.T.copy())
            saved_data['s'].append(current_state.s.copy())
            saved_data['u_piston'].append(left_bc.get_piston_velocity(t))

        try:
            solver.step_forward(dt)
        except Exception as e:
            print(f"  ERROR at step {step}, t={t*1e6:.1f} us: {e}")
            break

        t += dt
        step += 1

        if step % 500 == 0:
            print(f"  Step {step:6d}, t={t*1e6:8.2f} us, "
                  f"u_p={left_bc.get_piston_velocity(t):8.1f} m/s, "
                  f"p_max={current_state.p.max()/1e6:8.2f} MPa")

    print(f"\nCompleted: {step} steps")

    # Convert to arrays
    for key in saved_data:
        saved_data[key] = np.array(saved_data[key])

    # Config dictionary
    config = {
        "case_name": "pele_reconstruction_solid",
        "domain_length": domain_length,
        "n_cells": n_cells,
        "t_end": t_end,
        "cfl": cfl,
        "av_linear": av_linear,
        "av_quad": av_quad,
        "velocity_scale": velocity_scale,
        "velocity_offset": velocity_offset,
        "velocity_min": velocity_min,
        "data_source": str(data_dir),
        "n_data_points": len(data.time),
        "data_time_range": [float(data.time[0]), float(data.time[-1])],
        "n_steps": step,
        "rho_init": rho_init,
        "c_init": c_init,
        "T_init": INPUT_PARAMS_CONFIG['T'],
        "p_init": INPUT_PARAMS_CONFIG['P'],
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
    parser = argparse.ArgumentParser(description="Run PeleC reconstruction with 1D Lagrangian solver")
    parser.add_argument("--data-source", choices=["pele", "synthetic"], default="pele",
                        help="Data source: pele (raw PeleC data) or synthetic (CSV from GUI)")
    parser.add_argument("--data-path", type=str, default=None,
                        help="Path to data (directory for pele, CSV file for synthetic)")
    parser.add_argument("--parts", type=int, nargs="+", default=None,
                        help="Data parts to load for PeleC data (default: all)")
    parser.add_argument("--domain", type=float, default=2.0, help="Domain length [m]")
    parser.add_argument("--cells", type=int, default=400, help="Number of cells")
    parser.add_argument("--cfl", type=float, default=0.4, help="CFL number")
    parser.add_argument("--velocity-scale", type=float, default=1.0,
                        help="Scale factor for piston velocity (default 1.0)")
    parser.add_argument("--velocity-offset", type=float, default=0.0,
                        help="Value to subtract from scaled velocity [m/s] (default 0.0)")
    parser.add_argument("--velocity-min", type=float, default=None,
                        help="Minimum allowed piston velocity [m/s] (default: None, no clamping)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: results/solid)")

    args = parser.parse_args()

    run_reconstruction(
        data_source=args.data_source,
        data_path=args.data_path,
        parts=args.parts,
        domain_length=args.domain,
        n_cells=args.cells,
        cfl=args.cfl,
        velocity_scale=args.velocity_scale,
        velocity_offset=args.velocity_offset,
        velocity_min=args.velocity_min,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
