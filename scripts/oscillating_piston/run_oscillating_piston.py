"""
Oscillating Piston Simulation - Recreation of Wang, McDonald & Radulescu (JFM 2025)
===================================================================================

Recreates Figures 6 and 9 from:
"Shock-induced ignition and transition to detonation in the presence of
mechanically induced nonlinear acoustic forcing"
DOI: 10.1017/jfm.2025.229

Piston velocity profile:
    u_p(t) = u_p0 + A * sin(2 * pi * f * t)

Where:
    u_p0 = mean piston velocity [m/s]
    A    = oscillation amplitude [m/s]
    f    = oscillation frequency [Hz]

Initial conditions (inert case for Figures 6 & 9):
    - Pre-shock: T0 = 300 K, p0 = 5900 Pa
    - Stoichiometric 2H2 + O2 mixture
    - Post-shock: T1 ~ 1100 K, p1 ~ 1 atm

Author: Generated with Claude AI assistance
"""

import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Callable

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from lagrangian_solver import LagrangianGrid, FlowState, GridConfig
from lagrangian_solver.core.state import create_uniform_state
from lagrangian_solver.core.solver import CompatibleLagrangianSolver as LagrangianSolver, SolverConfig
from lagrangian_solver.equations.eos import CanteraEOS
from lagrangian_solver.boundary.base import BoundarySide, ThermalBCType
from lagrangian_solver.boundary.open import OpenBC
from lagrangian_solver.boundary.piston import CompatiblePistonBC


# =============================================================================
# OSCILLATING PISTON CONFIGURATION
# =============================================================================

# Reference values from Wang et al. (2025) - H2/O2 case
# Figure 9: Post-shock conditions (T1=1100K, p1=1atm), oscillating piston
DEFAULT_CONFIG = {
    # Piston parameters (Figure 9: oscillations in post-shock gas)
    'u_p0': 0.0,            # Mean piston velocity [m/s] (zero in post-shock frame)
    'amplitude': 325.27,    # Oscillation amplitude [m/s] (from paper)
    'frequency': 45.4e3,    # Oscillation frequency [Hz] (45.4 kHz)

    # Initial conditions (POST-SHOCK for Figure 9)
    'T0': 1100.0,           # Post-shock temperature [K]
    'p0': 101325.0,         # Post-shock pressure [Pa] (1 atm)
    'fuel': 'H2',           # Fuel species
    'oxidizer': 'O2:1',     # Oxidizer
    'phi': 1.0,             # Equivalence ratio (stoichiometric)
    'mechanism': 'gri30.yaml',  # Cantera mechanism

    # Domain
    'domain_length': 0.175, # Domain length [m]
    'n_cells': 1000,        # Number of cells

    # Time integration
    't_end': 80e-6,         # End time [s] (80 microseconds)
    'cfl': 0.3,             # CFL number (from paper)

    # Artificial viscosity
    'av_linear': 0.3,
    'av_quad': 2.0,

    # Ramping - smooth startup to prevent numerical instability
    'ramp_time': 0e-6,      # Ramp up over 5 μs
}


def create_oscillating_velocity_profile(
    u_p0: float,
    amplitude: float,
    frequency: float,
    ramp_time: float = 50e-6,
) -> Callable[[float], float]:
    """
    Create oscillating piston velocity profile with startup ramp.

    u_p(t) = ramp(t) * [u_p0 + A * sin(2 * pi * f * t)]

    The ramp smoothly increases from 0 to 1 over ramp_time to prevent
    numerical instabilities from sudden supersonic piston motion.
    """
    omega = 2.0 * np.pi * frequency

    def velocity(t: float) -> float:
        # Smooth ramp using sin^2 profile
        if t < ramp_time:
            ramp = np.sin(0.5 * np.pi * t / ramp_time) ** 2
        else:
            ramp = 1.0

        return ramp * (u_p0 + amplitude * np.sin(omega * t))

    return velocity


# =============================================================================
# Output Generation Functions (same as experiment_reconstruction)
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

    # Mass coordinate
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

    plt.suptitle(f'Final State ({coord_name} Space) - t = {t_final*1e6:.1f} μs', fontsize=14)
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

    # Subsample if too many time points
    if n_times_raw > max_time_points:
        skip = n_times_raw // max_time_points
        indices = list(range(0, n_times_raw, skip))
        print(f"  Subsampling X-T data: {n_times_raw} -> {len(indices)} time points")
    else:
        indices = list(range(n_times_raw))

    n_times = len(indices)
    times = saved_data['t'][indices]

    # Mass coordinate
    rho_init = config['rho_init']
    L_init = config['domain_length']
    dm = rho_init * L_init / n_cells_initial
    m_cell = (np.arange(n_cells_initial) + 0.5) * dm

    # Build 2D arrays
    rho_xt = np.array([saved_data['rho'][i] for i in indices])
    p_xt = np.array([saved_data['p'][i] for i in indices])
    T_xt = np.array([saved_data['T'][i] for i in indices])
    e_xt = np.array([saved_data['e'][i] for i in indices])

    u_xt = np.array([0.5 * (saved_data['u'][i][:-1] + saved_data['u'][i][1:])
                     for i in indices])

    s_init = saved_data['s'][0][0]
    ds_xt = np.array([saved_data['s'][i] - s_init for i in indices])

    x_xt = np.array([0.5 * (saved_data['x'][i][:-1] + saved_data['x'][i][1:])
                     for i in indices])

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

    # Time in microseconds for this case
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

        # Set color bounds (Temperature has lower bound of 900 K)
        if var_name == 'T':
            vmin, vmax = 900, None
        else:
            vmin, vmax = None, None

        pcm = ax.pcolormesh(X_plot, T_mesh, data_2d, shading='auto', cmap='rainbow', vmin=vmin, vmax=vmax)
        fig.colorbar(pcm, ax=ax, label=f'{name} [{unit}]')

        ax.set_xlabel(x_label)
        ax.set_ylabel('Time [μs]')
        ax.set_title(f'{name} - {coord_name.title()} Space')

        # Set phi axis limits for mass coordinate plots
        if use_mass_coord:
            ax.set_xlim(0, 0.004)

        plt.tight_layout()
        filename = output_path / f'{var_name}{suffix}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close(fig)

    print(f"  Saved X-T diagrams ({coord_name})")


def create_animation(saved_data: Dict, config: Dict, output_file: str,
                     use_mass_coord: bool = False, fps: int = 30):
    """Create MP4 animation from simulation results."""
    try:
        import cv2
    except ImportError:
        print(f"  Skipping animation (cv2 not available)")
        return

    n_times = len(saved_data['t'])
    n_cells = len(saved_data['rho'][0])

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

        title.set_text(f'Oscillating Piston ({coord_name})\nt = {t*1e6:.1f} μs')

    frame_skip = max(1, n_times // 500)
    frames = list(range(0, n_times, frame_skip))

    fig.tight_layout()
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    for i, frame_idx in enumerate(frames):
        update(frame_idx)
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(height, width, 3)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        video_writer.write(img_bgr)

    video_writer.release()
    plt.close(fig)
    print(f"  Saved animation: {Path(output_file).name}")


def save_snapshots(saved_data: Dict, config: Dict, output_dir: str, snapshot_interval: int = 1):
    """Save simulation snapshots compatible with interactive_plotter.py."""
    output_path = Path(output_dir)
    snapshots_path = output_path / 'snapshots'
    snapshots_path.mkdir(parents=True, exist_ok=True)

    n_frames = len(saved_data['t'])
    n_cells_initial = config['n_cells']
    rho_init = config['rho_init']
    L_init = config['domain_length']
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

    print(f"  Saved {saved_count} snapshots + piston_history.npz")


def plot_piston_velocity(saved_data: Dict, config: Dict, output_file: str):
    """Plot piston velocity history."""
    fig, ax = plt.subplots(figsize=(10, 5))

    t_us = saved_data['t'] * 1e6
    u_piston = saved_data['u_piston']

    ax.plot(t_us, u_piston, 'b-', lw=1.5, label='Piston velocity')
    ax.axhline(config['u_p0'], color='r', ls='--', lw=1, alpha=0.7,
               label=f"Mean u_p0 = {config['u_p0']:.0f} m/s")

    # Mark oscillation bounds
    amplitude = config['amplitude_ratio'] * config['u_p0']
    ax.axhline(config['u_p0'] + amplitude, color='g', ls=':', lw=1, alpha=0.5)
    ax.axhline(config['u_p0'] - amplitude, color='g', ls=':', lw=1, alpha=0.5)

    ax.set_xlabel('Time [μs]')
    ax.set_ylabel('Piston Velocity [m/s]')
    ax.set_title(f"Oscillating Piston Velocity (f = {config['frequency']/1e3:.1f} kHz)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {Path(output_file).name}")


# =============================================================================
# Main Simulation
# =============================================================================

def run_oscillating_piston(config: Dict = None, output_dir: str = None):
    """Run oscillating piston simulation."""
    if config is None:
        config = DEFAULT_CONFIG.copy()

    print("=" * 70)
    print("OSCILLATING PISTON SIMULATION")
    print("Recreation of Wang, McDonald & Radulescu (JFM 2025) - Figures 6 & 9")
    print("=" * 70)

    # Setup output directory
    if output_dir is None:
        output_dir = Path(__file__).parent / "results"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract parameters
    u_p0 = config['u_p0']
    # Support both absolute amplitude and ratio
    if 'amplitude' in config:
        amplitude = config['amplitude']
    else:
        amplitude = config.get('amplitude_ratio', 0.2) * u_p0
    frequency = config['frequency']
    T0 = config['T0']
    p0 = config['p0']
    domain_length = config['domain_length']
    n_cells = config['n_cells']
    t_end = config['t_end']
    cfl = config['cfl']
    ramp_time = config['ramp_time']

    print(f"\nPiston parameters:")
    print(f"  Mean velocity u_p0 = {u_p0:.1f} m/s")
    if u_p0 != 0:
        print(f"  Amplitude A = {amplitude:.1f} m/s ({amplitude/u_p0*100:.0f}%)")
    else:
        print(f"  Amplitude A = {amplitude:.1f} m/s")
    print(f"  Frequency f = {frequency/1e3:.1f} kHz")
    print(f"  Period T = {1e6/frequency:.2f} μs")
    print(f"  Ramp time = {ramp_time*1e6:.1f} μs")

    # Create EOS
    print(f"\nCreating EOS with mechanism: {config['mechanism']}")
    eos = CanteraEOS(config['mechanism'])
    eos.set_mixture(config['fuel'], config['oxidizer'], config['phi'])
    eos.set_state_TP(T0, p0)

    rho0 = eos.gas.density
    c0 = eos.gas.sound_speed
    gamma = eos.gas.cp / eos.gas.cv

    print(f"\nInitial state:")
    print(f"  T0 = {T0:.1f} K")
    print(f"  p0 = {p0:.1f} Pa ({p0/101325:.4f} atm)")
    print(f"  rho0 = {rho0:.6f} kg/m³")
    print(f"  c0 = {c0:.1f} m/s")
    print(f"  gamma = {gamma:.3f}")
    print(f"  Mach (mean) = {u_p0/c0:.2f}")
    print(f"  Mach (amplitude) = {amplitude/c0:.2f}")

    # Create grid
    grid_config = GridConfig(n_cells=n_cells, x_min=0.0, x_max=domain_length)
    grid = LagrangianGrid(grid_config)

    # Create initial state
    state = create_uniform_state(
        n_cells=grid.n_cells,
        x_left=grid.x[0],
        x_right=grid.x[-1],
        rho=rho0,
        u=0.0,
        p=p0,
        eos=eos,
    )

    print(f"\nDomain:")
    print(f"  Length = {domain_length*100:.1f} cm")
    print(f"  Cells = {n_cells}")
    print(f"  dx = {domain_length/n_cells*1000:.3f} mm")

    # Create oscillating velocity profile
    velocity_profile = create_oscillating_velocity_profile(u_p0, amplitude, frequency, ramp_time)

    # Create boundary conditions
    left_bc = CompatiblePistonBC(
        side=BoundarySide.LEFT,
        eos=eos,
        velocity=velocity_profile,
        thermal_bc=ThermalBCType.ADIABATIC,
        ramp_time=0.0,  # Ramp is built into velocity_profile
    )
    right_bc = OpenBC(side=BoundarySide.RIGHT, eos=eos, p_external=p0)

    # Create solver
    solver_config = SolverConfig(
        cfl=cfl,
        av_linear=config['av_linear'],
        av_quad=config['av_quad'],
        av_enabled=True,
    )
    solver = LagrangianSolver(
        grid=grid,
        eos=eos,
        bc_left=left_bc,
        bc_right=right_bc,
        config=solver_config,
    )
    solver.set_initial_condition(state)

    print(f"\nSimulation:")
    print(f"  t_end = {t_end*1e6:.1f} μs")
    print(f"  CFL = {cfl}")
    print(f"  AV: linear={config['av_linear']}, quad={config['av_quad']}")

    # Storage - time-based recording for ~1000 snapshots
    target_snapshots = 1000
    dt_record = t_end / target_snapshots
    t_next_record = 0.0

    saved_data = {
        't': [],
        'x': [],
        'rho': [],
        'u': [],
        'p': [],
        'e': [],
        'T': [],
        's': [],
        'u_piston': [],
    }

    print(f"\nRunning simulation...")
    print(f"  Target ~{target_snapshots} snapshots")

    step = 0
    t = 0.0

    while t < t_end:
        current_state = solver.state

        # Compute timestep
        c = eos.sound_speed(current_state.rho, current_state.p)
        u_avg = 0.5 * (current_state.u[:-1] + current_state.u[1:])
        dt_cell = grid.dx / (c + np.abs(u_avg))
        dt = cfl * np.min(dt_cell)

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
            saved_data['u_piston'].append(velocity_profile(t))
            t_next_record += dt_record

        # Step forward
        try:
            solver.step_forward(dt)
        except Exception as e:
            print(f"  ERROR at step {step}, t={t*1e6:.3f} μs: {e}")
            break

        # Single source of truth: the solver's clock advances by the dt it
        # actually integrated (which may be smaller than requested if a BC
        # constraint clamped it). Never keep a parallel script clock.
        t = solver.time
        step += 1

        if step % 1000 == 0:
            print(f"  Step {step:6d}, t={t*1e6:8.2f} μs, u_p={velocity_profile(t):.1f} m/s")

    print(f"\nCompleted: {step} steps, {len(saved_data['t'])} snapshots")

    # Convert to arrays
    for key in saved_data:
        saved_data[key] = np.array(saved_data[key])

    # Build config for output
    config_out = {
        'case_name': 'oscillating_piston',
        'u_p0': u_p0,
        'amplitude': amplitude,
        'amplitude_ratio': amplitude / u_p0 if u_p0 != 0 else 0,
        'frequency': frequency,
        'ramp_time': ramp_time,
        'domain_length': domain_length,
        'n_cells': n_cells,
        't_end': t_end,
        'cfl': cfl,
        'av_linear': config['av_linear'],
        'av_quad': config['av_quad'],
        'T0': T0,
        'p0': p0,
        'rho_init': rho0,
        'c_init': c0,
        'gamma': gamma,
        'fuel': config['fuel'],
        'oxidizer': config['oxidizer'],
        'phi': config['phi'],
        'mechanism': config['mechanism'],
        'n_steps': step,
    }

    # Save outputs
    print(f"\nSaving results to: {output_dir}")

    # Save timeseries
    np.savez(output_dir / "timeseries.npz", **saved_data, config=config_out)
    print(f"  Saved: timeseries.npz")

    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump(config_out, f, indent=2)
    print(f"  Saved: config.json")

    # Final state plots
    plot_final_state(saved_data, config_out, str(output_dir / "final_state_x.png"), use_mass_coord=False)
    plot_final_state(saved_data, config_out, str(output_dir / "final_state_m.png"), use_mass_coord=True)
    print(f"  Saved: final_state_x.png, final_state_m.png")

    # X-T diagrams (physical and mass coordinates)
    plot_xt_diagrams(saved_data, config_out, str(output_dir), use_mass_coord=False)
    plot_xt_diagrams(saved_data, config_out, str(output_dir), use_mass_coord=True)

    # Animations
    create_animation(saved_data, config_out, str(output_dir / "animation_x.mp4"), use_mass_coord=False)
    create_animation(saved_data, config_out, str(output_dir / "animation_m.mp4"), use_mass_coord=True)

    # Snapshots
    save_snapshots(saved_data, config_out, str(output_dir), snapshot_interval=1)

    # Piston velocity plot
    plot_piston_velocity(saved_data, config_out, str(output_dir / "velocity_comparison.png"))

    print(f"\nDone! Results saved to: {output_dir}")
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Oscillating piston simulation (Wang et al. JFM 2025)"
    )
    parser.add_argument("--u_p0", type=float, default=None,
                        help="Mean piston velocity [m/s]")
    parser.add_argument("--amplitude", type=float, default=None,
                        help="Oscillation amplitude ratio (e.g., 0.2 for 20%%)")
    parser.add_argument("--frequency", type=float, default=None,
                        help="Oscillation frequency [Hz]")
    parser.add_argument("--t_end", type=float, default=None,
                        help="End time [s]")
    parser.add_argument("--n_cells", type=int, default=None,
                        help="Number of cells")
    parser.add_argument("--ramp_time", type=float, default=None,
                        help="Ramp time [s] (e.g., 5e-6 for 5 μs)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory")

    args = parser.parse_args()

    # Build config with overrides
    config = DEFAULT_CONFIG.copy()
    if args.u_p0 is not None:
        config['u_p0'] = args.u_p0
    if args.amplitude is not None:
        config['amplitude_ratio'] = args.amplitude
    if args.frequency is not None:
        config['frequency'] = args.frequency
    if args.t_end is not None:
        config['t_end'] = args.t_end
    if args.n_cells is not None:
        config['n_cells'] = args.n_cells
    if args.ramp_time is not None:
        config['ramp_time'] = args.ramp_time

    run_oscillating_piston(config, args.output)
