#!/usr/bin/env python3
"""
Interactive X-Y Plotter for Experimental Reconstruction Results
================================================================

Tkinter-based GUI for creating customizable X-Y plots from simulation results.
Supports independent selection of x and y variables with data trimming per case.

Features:
- Independent X and Y variable selection
- Ethane (solid) and butane (dashed) line styles
- Per-case data trimming via sliders
- Unit conversions: pressure (MPa), time (ms), position (cm)
- Length to DDT calculation

Author: Nolan Dyck with Claude AI assistance


Usage:
    python interactive_xy_plotter.py                    # Use default results directory
    python interactive_xy_plotter.py --data 10ns       # Use porous_results_dt_10_ns
    python interactive_xy_plotter.py --data 1ns        # Use porous_results_dt_1_ns
    python interactive_xy_plotter.py --data porous     # Use porous_results
    python interactive_xy_plotter.py --data /path/to/dir  # Use custom path
"""

import argparse
import json
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

# =============================================================================
# CONFIGURATION
# =============================================================================

# Base directory (script location)
SCRIPT_DIR = Path(__file__).parent.parent

# Data directory shortcuts
DATA_DIR_SHORTCUTS = {
    'results': SCRIPT_DIR / 'results',
    'default': SCRIPT_DIR / 'results',
    '10ns': SCRIPT_DIR / 'porous_results_dt_10_ns',
    '1ns': SCRIPT_DIR / 'porous_results_dt_1_ns',
    'porous': SCRIPT_DIR / 'porous_results',
}

# Default results directory
DEFAULT_RESULTS_DIR = DATA_DIR_SHORTCUTS['results']

# Color palette for phi values (matches existing color scheme)
PHI_COLORS = {
    0.8: 'black',
    1.0: '#c1272d',   # dark red
    1.3: '#eecc16',   # gold/yellow
    1.49: '#008176',  # teal (C4H10 only)
    1.6: 'blue',
}

# Line styles: ethane = solid, butane = dashed
FUEL_LINE_STYLES = {
    'c2h6': '-',    # Solid for ethane
    'c4h10': '--',  # Dashed for butane
}

# Variable definitions with units and conversion factors
# All data is stored in SI units, these define display conversions
VARIABLE_DEFS = {
    't': {
        'label': 'Time',
        'unit_si': 's',
        'unit_display': 'ms',
        'scale': 1e3,  # multiply SI by this to get display units
    },
    'x_piston': {
        'label': 'Piston Position',
        'unit_si': 'm',
        'unit_display': 'cm',
        'scale': 1e2,
    },
    'x_ddt': {
        'label': 'Distance from Ignition',
        'unit_si': 'm',
        'unit_display': 'cm',
        'scale': 1e2,
    },
    'u_piston': {
        'label': 'Piston Velocity',
        'unit_si': 'm/s',
        'unit_display': 'm/s',
        'scale': 1.0,
    },
    'p_piston': {
        'label': 'Pressure Gain',
        'unit_si': 'Pa',
        'unit_display': 'MPa',
        'scale': 1e-6,
    },
    'T_piston': {
        'label': 'Piston Temperature',
        'unit_si': 'K',
        'unit_display': 'K',
        'scale': 1.0,
    },
    'rho_piston': {
        'label': 'Piston Density',
        'unit_si': 'kg/m^3',
        'unit_display': 'kg/m^3',
        'scale': 1.0,
    },
    'p_max': {
        'label': 'Max Pressure',
        'unit_si': 'Pa',
        'unit_display': 'MPa',
        'scale': 1e-6,
    },
    'T_max': {
        'label': 'Max Temperature',
        'unit_si': 'K',
        'unit_display': 'K',
        'scale': 1.0,
    },
    'rho_max': {
        'label': 'Max Density',
        'unit_si': 'kg/m^3',
        'unit_display': 'kg/m^3',
        'scale': 1.0,
    },
    'u_max': {
        'label': 'Max Velocity',
        'unit_si': 'm/s',
        'unit_display': 'm/s',
        'scale': 1.0,
    },
    'phi': {
        'label': 'Equivalence Ratio',
        'unit_si': '-',
        'unit_display': '-',
        'scale': 1.0,
    },
}

# Matplotlib settings for publication-quality plots
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 24,
    'axes.labelsize': 32,
    'axes.titlesize': 32,
    'xtick.labelsize': 28,
    'ytick.labelsize': 28,
    'legend.fontsize': 24,
    'figure.titlesize': 32,
    'mathtext.fontset': 'stix',
})

# Default font size for axis labels
DEFAULT_AXIS_FONT_SIZE = 32


# =============================================================================
# DATA LOADING
# =============================================================================

def load_case_data(case_dir: Path) -> Optional[Dict]:
    """Load timeseries and configuration data for a case."""
    if not case_dir.exists():
        return None

    ts_path = case_dir / "timeseries.npz"
    config_path = case_dir / "config.json"

    if not ts_path.exists() or not config_path.exists():
        return None

    try:
        ts = np.load(ts_path, allow_pickle=True)
        with open(config_path) as f:
            config = json.load(f)

        return {
            't': ts['t'],
            'x': ts['x'],
            'rho': ts['rho'],
            'u': ts['u'],
            'p': ts['p'],
            'e': ts['e'],
            'T': ts['T'],
            's': ts['s'],
            'u_piston': ts['u_piston'],
            'config': config,
        }
    except Exception as e:
        print(f"Error loading {case_dir}: {e}")
        return None


def extract_case_variables(data: Dict) -> Dict[str, np.ndarray]:
    """Extract all plottable variables from case data."""
    t = data['t']

    # Piston face data (first cell/node)
    x_piston = data['x'][:, 0]
    u_piston = data['u_piston']
    p_piston = data['p'][:, 0]
    T_piston = data['T'][:, 0]
    rho_piston = data['rho'][:, 0]

    # Max domain values
    p_max = np.max(data['p'], axis=1)
    T_max = np.max(data['T'], axis=1)
    rho_max = np.max(data['rho'], axis=1)
    u_max = np.max(np.abs(data['u']), axis=1)

    # Length to DDT: use piston position as the current "run-up distance"
    # This represents how far the flame has traveled
    x_ddt = x_piston.copy()

    # Equivalence ratio (constant for this case)
    phi = np.full_like(t, data['config'].get('phi', 1.0))

    return {
        't': t,
        'x_piston': x_piston,
        'x_ddt': x_ddt,
        'u_piston': u_piston,
        'p_piston': p_piston,
        'T_piston': T_piston,
        'rho_piston': rho_piston,
        'p_max': p_max,
        'T_max': T_max,
        'rho_max': rho_max,
        'u_max': u_max,
        'phi': phi,
    }


def discover_cases(results_dir: Path = None) -> List[Dict]:
    """Discover all available cases in the results directory."""
    if results_dir is None:
        results_dir = DEFAULT_RESULTS_DIR

    cases = []

    if not results_dir.exists():
        return cases

    for case_dir in sorted(results_dir.iterdir()):
        if not case_dir.is_dir():
            continue

        # Skip non-case directories
        if case_dir.name in ['comparison_plots', 'comparison_org']:
            continue

        # Parse case name (e.g., "c2h6_phi_1.0")
        parts = case_dir.name.split('_phi_')
        if len(parts) != 2:
            continue

        fuel = parts[0]  # c2h6 or c4h10
        try:
            phi = float(parts[1])
        except ValueError:
            continue

        # Load case data
        data = load_case_data(case_dir)
        if data is None:
            continue

        # Extract variables
        variables = extract_case_variables(data)

        cases.append({
            'name': case_dir.name,
            'fuel': fuel,
            'phi': phi,
            'dir': case_dir,
            'config': data['config'],
            'variables': variables,
            'color': PHI_COLORS.get(phi, 'gray'),
            'linestyle': FUEL_LINE_STYLES.get(fuel, '-'),
        })

    return cases


# =============================================================================
# INTERACTIVE PLOTTER GUI
# =============================================================================

class InteractiveXYPlotter:
    """Tkinter-based interactive plotter for X-Y variable selection."""

    def __init__(self, root: tk.Tk, results_dir: Path = None):
        self.root = root
        self.results_dir = results_dir if results_dir else DEFAULT_RESULTS_DIR
        self.root.title(f"Interactive X-Y Plotter - {self.results_dir.name}")
        self.root.geometry("1400x900")

        # Load case data
        self.cases = discover_cases(self.results_dir)
        if not self.cases:
            messagebox.showerror("Error", f"No cases found in {self.results_dir}")
            self.root.destroy()
            return

        # Group cases by fuel
        self.ethane_cases = [c for c in self.cases if c['fuel'] == 'c2h6']
        self.butane_cases = [c for c in self.cases if c['fuel'] == 'c4h10']

        # Trim state: (start_frac, end_frac) for each case
        self.trim_state: Dict[str, Tuple[float, float]] = {}
        for case in self.cases:
            self.trim_state[case['name']] = (0.0, 1.0)

        # Case enable state
        self.case_enabled: Dict[str, tk.BooleanVar] = {}
        for case in self.cases:
            var = tk.BooleanVar(value=True)
            self.case_enabled[case['name']] = var

        # Create UI
        self._create_widgets()

        # Initial plot
        self._update_plot()

    def _create_widgets(self):
        """Create the GUI layout."""
        # Main container with horizontal panes
        self.paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Left panel: controls
        self.control_frame = ttk.Frame(self.paned, width=350)
        self.paned.add(self.control_frame, weight=0)

        # Right panel: plot
        self.plot_frame = ttk.Frame(self.paned)
        self.paned.add(self.plot_frame, weight=1)

        self._create_control_panel()
        self._create_plot_panel()

    def _create_control_panel(self):
        """Create the control panel with variable selection and trim controls."""
        # Create scrollable canvas for controls
        canvas = tk.Canvas(self.control_frame, width=340)
        scrollbar = ttk.Scrollbar(self.control_frame, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Bind mousewheel
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        row = 0

        # === Variable Selection ===
        var_frame = ttk.LabelFrame(self.scrollable_frame, text="Variable Selection", padding=10)
        var_frame.grid(row=row, column=0, sticky="ew", padx=5, pady=5)
        row += 1

        # Get available variables
        var_names = list(VARIABLE_DEFS.keys())
        var_labels = [f"{VARIABLE_DEFS[v]['label']} [{VARIABLE_DEFS[v]['unit_display']}]"
                      for v in var_names]

        # X variable
        ttk.Label(var_frame, text="X Variable:").grid(row=0, column=0, sticky="w", pady=2)
        self.x_var = ttk.Combobox(var_frame, values=var_labels, state="readonly", width=30)
        self.x_var.current(0)  # Default: time
        self.x_var.grid(row=0, column=1, pady=2, padx=5)
        self.x_var.bind("<<ComboboxSelected>>", lambda e: self._update_plot())

        # Y variable
        ttk.Label(var_frame, text="Y Variable:").grid(row=1, column=0, sticky="w", pady=2)
        self.y_var = ttk.Combobox(var_frame, values=var_labels, state="readonly", width=30)
        self.y_var.current(4)  # Default: pressure
        self.y_var.grid(row=1, column=1, pady=2, padx=5)
        self.y_var.bind("<<ComboboxSelected>>", lambda e: self._update_plot())

        # === Case Selection ===
        case_frame = ttk.LabelFrame(self.scrollable_frame, text="Case Selection", padding=10)
        case_frame.grid(row=row, column=0, sticky="ew", padx=5, pady=5)
        row += 1

        # Ethane cases
        if self.ethane_cases:
            ttk.Label(case_frame, text="Ethane (C2H6) - Solid Lines:",
                     font=('TkDefaultFont', 9, 'bold')).grid(row=0, column=0, columnspan=2, sticky="w")
            for i, case in enumerate(self.ethane_cases):
                cb = ttk.Checkbutton(
                    case_frame,
                    text=f"phi = {case['phi']}",
                    variable=self.case_enabled[case['name']],
                    command=self._update_plot
                )
                cb.grid(row=i+1, column=0, sticky="w", padx=10)

        # Butane cases
        if self.butane_cases:
            butane_start = len(self.ethane_cases) + 2
            ttk.Label(case_frame, text="Butane (C4H10) - Dashed Lines:",
                     font=('TkDefaultFont', 9, 'bold')).grid(row=butane_start, column=0, columnspan=2, sticky="w")
            for i, case in enumerate(self.butane_cases):
                cb = ttk.Checkbutton(
                    case_frame,
                    text=f"phi = {case['phi']}",
                    variable=self.case_enabled[case['name']],
                    command=self._update_plot
                )
                cb.grid(row=butane_start+i+1, column=0, sticky="w", padx=10)

        # Select/Deselect all buttons
        btn_frame = ttk.Frame(case_frame)
        btn_frame.grid(row=100, column=0, columnspan=2, pady=5)
        ttk.Button(btn_frame, text="Select All", command=self._select_all).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Deselect All", command=self._deselect_all).pack(side=tk.LEFT, padx=2)

        # === Trim Controls ===
        trim_frame = ttk.LabelFrame(self.scrollable_frame, text="Data Trimming", padding=10)
        trim_frame.grid(row=row, column=0, sticky="ew", padx=5, pady=5)
        row += 1

        self.trim_sliders = {}

        # Create trim controls for each case
        trim_row = 0
        for fuel_name, fuel_cases in [("Ethane (C2H6)", self.ethane_cases),
                                       ("Butane (C4H10)", self.butane_cases)]:
            if not fuel_cases:
                continue

            ttk.Label(trim_frame, text=fuel_name,
                     font=('TkDefaultFont', 9, 'bold')).grid(row=trim_row, column=0, columnspan=3, sticky="w")
            trim_row += 1

            for case in fuel_cases:
                # Case label
                ttk.Label(trim_frame, text=f"phi = {case['phi']}:", width=10).grid(
                    row=trim_row, column=0, sticky="w", padx=5)

                # Start slider
                start_var = tk.DoubleVar(value=0.0)
                start_slider = ttk.Scale(
                    trim_frame, from_=0.0, to=1.0, variable=start_var,
                    orient=tk.HORIZONTAL, length=100,
                    command=lambda v, name=case['name'], which='start': self._on_trim_change(name, which, v)
                )
                start_slider.grid(row=trim_row, column=1, padx=2)

                # End slider
                end_var = tk.DoubleVar(value=1.0)
                end_slider = ttk.Scale(
                    trim_frame, from_=0.0, to=1.0, variable=end_var,
                    orient=tk.HORIZONTAL, length=100,
                    command=lambda v, name=case['name'], which='end': self._on_trim_change(name, which, v)
                )
                end_slider.grid(row=trim_row, column=2, padx=2)

                self.trim_sliders[case['name']] = (start_var, end_var)
                trim_row += 1

            trim_row += 1  # Extra spacing between fuel types

        # Trim labels
        ttk.Label(trim_frame, text="Start").grid(row=0, column=1)
        ttk.Label(trim_frame, text="End").grid(row=0, column=2)

        # Reset trim button
        ttk.Button(trim_frame, text="Reset All Trims",
                  command=self._reset_trims).grid(row=trim_row, column=0, columnspan=3, pady=5)

        # === Plot Options ===
        opt_frame = ttk.LabelFrame(self.scrollable_frame, text="Plot Options", padding=10)
        opt_frame.grid(row=row, column=0, sticky="ew", padx=5, pady=5)
        row += 1

        # Log scale options
        self.log_x = tk.BooleanVar(value=False)
        self.log_y = tk.BooleanVar(value=False)
        ttk.Checkbutton(opt_frame, text="Log X scale", variable=self.log_x,
                       command=self._update_plot).grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(opt_frame, text="Log Y scale", variable=self.log_y,
                       command=self._update_plot).grid(row=0, column=1, sticky="w")

        # Grid option
        self.show_grid = tk.BooleanVar(value=True)
        ttk.Checkbutton(opt_frame, text="Show grid", variable=self.show_grid,
                       command=self._update_plot).grid(row=1, column=0, sticky="w")

        # Legend option
        self.show_legend = tk.BooleanVar(value=True)
        ttk.Checkbutton(opt_frame, text="Show legend", variable=self.show_legend,
                       command=self._update_plot).grid(row=1, column=1, sticky="w")

        # Last point only option
        self.last_point_only = tk.BooleanVar(value=False)
        ttk.Checkbutton(opt_frame, text="Last point only", variable=self.last_point_only,
                       command=self._update_plot).grid(row=2, column=0, sticky="w")

        # Marker size (for last point only mode)
        ttk.Label(opt_frame, text="Marker size:").grid(row=2, column=1, sticky="w")
        self.marker_size = tk.DoubleVar(value=10.0)
        ms_spin = ttk.Spinbox(opt_frame, from_=2.0, to=20.0, increment=1.0,
                             textvariable=self.marker_size, width=5,
                             command=self._update_plot)
        ms_spin.grid(row=2, column=2, sticky="w")
        ms_spin.bind("<Return>", lambda e: self._update_plot())

        # Line width
        ttk.Label(opt_frame, text="Line width:").grid(row=3, column=0, sticky="w")
        self.line_width = tk.DoubleVar(value=2.0)
        lw_spin = ttk.Spinbox(opt_frame, from_=0.5, to=5.0, increment=0.5,
                             textvariable=self.line_width, width=5,
                             command=self._update_plot)
        lw_spin.grid(row=3, column=1, sticky="w")
        lw_spin.bind("<Return>", lambda e: self._update_plot())

        # Axis font size
        ttk.Label(opt_frame, text="Axis font size:").grid(row=4, column=0, sticky="w")
        self.axis_font_size = tk.IntVar(value=DEFAULT_AXIS_FONT_SIZE)
        font_slider = ttk.Scale(
            opt_frame, from_=10, to=40, variable=self.axis_font_size,
            orient=tk.HORIZONTAL, length=120,
            command=lambda v: self._update_plot()
        )
        font_slider.grid(row=4, column=1, columnspan=2, sticky="w")

        # Font size label showing current value
        self.font_size_label = ttk.Label(opt_frame, text=f"{DEFAULT_AXIS_FONT_SIZE}")
        self.font_size_label.grid(row=4, column=3, sticky="w", padx=5)

        # === Buttons ===
        btn_frame = ttk.Frame(self.scrollable_frame)
        btn_frame.grid(row=row, column=0, sticky="ew", padx=5, pady=10)
        row += 1

        ttk.Button(btn_frame, text="Update Plot", command=self._update_plot).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Save Figure", command=self._save_figure).pack(side=tk.LEFT, padx=5)

    def _create_plot_panel(self):
        """Create the matplotlib plot panel."""
        self.fig = Figure(figsize=(10, 7), dpi=100)
        self.ax = self.fig.add_subplot(111)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Navigation toolbar
        toolbar = NavigationToolbar2Tk(self.canvas, self.plot_frame)
        toolbar.update()

    def _get_selected_var_key(self, combo: ttk.Combobox) -> str:
        """Get the variable key from combobox selection."""
        idx = combo.current()
        return list(VARIABLE_DEFS.keys())[idx]

    def _on_trim_change(self, case_name: str, which: str, value: str):
        """Handle trim slider changes."""
        try:
            val = float(value)
        except ValueError:
            return

        start, end = self.trim_state[case_name]
        if which == 'start':
            # Ensure start < end
            start = min(val, end - 0.01)
            self.trim_sliders[case_name][0].set(start)
        else:
            # Ensure end > start
            end = max(val, start + 0.01)
            self.trim_sliders[case_name][1].set(end)

        self.trim_state[case_name] = (start, end)
        self._update_plot()

    def _select_all(self):
        """Select all cases."""
        for var in self.case_enabled.values():
            var.set(True)
        self._update_plot()

    def _deselect_all(self):
        """Deselect all cases."""
        for var in self.case_enabled.values():
            var.set(False)
        self._update_plot()

    def _reset_trims(self):
        """Reset all trim sliders to full range."""
        for case_name in self.trim_state:
            self.trim_state[case_name] = (0.0, 1.0)
            if case_name in self.trim_sliders:
                self.trim_sliders[case_name][0].set(0.0)
                self.trim_sliders[case_name][1].set(1.0)
        self._update_plot()

    def _update_plot(self):
        """Redraw the plot with current settings."""
        self.ax.clear()

        # Get selected variables
        x_key = self._get_selected_var_key(self.x_var)
        y_key = self._get_selected_var_key(self.y_var)

        x_def = VARIABLE_DEFS[x_key]
        y_def = VARIABLE_DEFS[y_key]

        # Plot each enabled case
        for case in self.cases:
            if not self.case_enabled[case['name']].get():
                continue

            variables = case['variables']

            # Get data
            if x_key not in variables or y_key not in variables:
                continue

            x_data = variables[x_key] * x_def['scale']
            y_data = variables[y_key] * y_def['scale']

            # Apply trim
            start_frac, end_frac = self.trim_state[case['name']]
            n = len(x_data)
            start_idx = int(start_frac * n)
            end_idx = int(end_frac * n)

            if end_idx <= start_idx:
                continue

            x_trimmed = x_data[start_idx:end_idx]
            y_trimmed = y_data[start_idx:end_idx]

            # Create label with Greek phi symbol and subscript chemical formulas
            fuel_name = r"$\mathrm{C_2H_6}$" if case['fuel'] == 'c2h6' else r"$\mathrm{C_4H_{10}}$"
            label = f"{fuel_name}, φ={case['phi']}"

            # Plot - either full line or last point only
            if self.last_point_only.get():
                # Plot only the last point as a marker
                marker = 'o' if case['fuel'] == 'c2h6' else 's'  # circle for ethane, square for butane
                self.ax.plot(
                    x_trimmed[-1], y_trimmed[-1],
                    color=case['color'],
                    marker=marker,
                    markersize=self.marker_size.get(),
                    linestyle='none',
                    label=label
                )
            else:
                # Plot full line
                self.ax.plot(
                    x_trimmed, y_trimmed,
                    color=case['color'],
                    linestyle=case['linestyle'],
                    linewidth=self.line_width.get(),
                    label=label
                )

        # Get font size and update label
        font_size = self.axis_font_size.get()
        self.font_size_label.config(text=f"{font_size}")

        # Set labels with font size
        self.ax.set_xlabel(f"{x_def['label']} [{x_def['unit_display']}]", fontsize=font_size)
        self.ax.set_ylabel(f"{y_def['label']} [{y_def['unit_display']}]", fontsize=font_size)

        # Set tick label font size
        self.ax.tick_params(axis='both', labelsize=font_size - 4)

        # Log scale
        if self.log_x.get():
            self.ax.set_xscale('log')
        if self.log_y.get():
            self.ax.set_yscale('log')

        # Grid
        if self.show_grid.get():
            self.ax.grid(True, alpha=0.3)

        # Legend
        if self.show_legend.get():
            self.ax.legend(loc='best', framealpha=0.9, fontsize=font_size - 6)

        self.fig.tight_layout()
        self.canvas.draw()

    def _save_figure(self):
        """Save the current figure to file."""
        from tkinter import filedialog

        filepath = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[
                ("PNG files", "*.png"),
                ("PDF files", "*.pdf"),
                ("SVG files", "*.svg"),
                ("All files", "*.*")
            ],
            initialdir=str(self.results_dir / 'comparison_plots')
        )

        if filepath:
            self.fig.savefig(filepath, dpi=300, bbox_inches='tight')
            messagebox.showinfo("Saved", f"Figure saved to:\n{filepath}")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Interactive X-Y Plotter for Experimental Reconstruction Results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Data directory shortcuts:
  results   - Default results directory (scripts/experiment_reconstruction/results)
  10ns      - Porous results with 10 ns timestep (porous_results_dt_10_ns)
  1ns       - Porous results with 1 ns timestep (porous_results_dt_1_ns)
  porous    - Porous results (porous_results)

Examples:
  python interactive_xy_plotter.py                  # Use default results
  python interactive_xy_plotter.py --data 10ns     # Use 10 ns porous results
  python interactive_xy_plotter.py --data 1ns      # Use 1 ns porous results
  python interactive_xy_plotter.py --data /path/to/custom/dir
        """
    )
    parser.add_argument(
        '--data', '-d',
        type=str,
        default='results',
        help='Data directory to use. Can be a shortcut (results, 10ns, 1ns, porous) or a path.'
    )
    return parser.parse_args()


def resolve_data_dir(data_arg: str) -> Path:
    """Resolve data directory from argument (shortcut or path)."""
    # Check if it's a shortcut
    if data_arg.lower() in DATA_DIR_SHORTCUTS:
        return DATA_DIR_SHORTCUTS[data_arg.lower()]

    # Otherwise treat as a path
    path = Path(data_arg)
    if path.is_absolute():
        return path
    else:
        # Relative to script directory
        return SCRIPT_DIR / path


def main():
    """Main entry point."""
    args = parse_args()
    results_dir = resolve_data_dir(args.data)

    print(f"Using data directory: {results_dir}")

    if not results_dir.exists():
        print(f"Error: Directory does not exist: {results_dir}")
        print("\nAvailable shortcuts:")
        for name, path in DATA_DIR_SHORTCUTS.items():
            exists = "  [exists]" if path.exists() else "  [not found]"
            print(f"  {name}: {path}{exists}")
        return

    root = tk.Tk()
    app = InteractiveXYPlotter(root, results_dir)
    root.mainloop()


if __name__ == "__main__":
    main()
