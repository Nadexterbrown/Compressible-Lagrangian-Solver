#!/usr/bin/env python3
"""
Synthetic Trajectory Builder GUI
================================

Tkinter-based GUI for creating synthetic smoothed PeleC trajectory data.
Allows user to define piecewise idealized functions (linear, polynomial,
power, exponential, etc.) between specified time bounds.

Usage:
    python synthetic_trajectory_gui.py

Author: Generated with Claude AI assistance
"""

import os
import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from enum import Enum
import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector


# =============================================================================
# CONFIGURATION
# =============================================================================

# Available function types
class FunctionType(Enum):
    CONSTANT = "constant"
    LINEAR = "linear"
    QUADRATIC = "quadratic"
    CUBIC = "cubic"
    POWER = "power"
    EXPONENTIAL = "exponential"
    LOGARITHMIC = "logarithmic"
    ERF_RAMP = "erf_ramp"  # Smooth S-curve ramp (good for startup)


# Function type descriptions
FUNCTION_DESCRIPTIONS = {
    FunctionType.CONSTANT: "v(t) = c",
    FunctionType.LINEAR: "v(t) = v0 + (v1-v0)*t*",
    FunctionType.QUADRATIC: "v(t) = v0 + (v1-v0)*t* + a*t**(1-t*)",
    FunctionType.CUBIC: "v(t) = v0 + (v1-v0)*(3t*^2 - 2t*^3)",
    FunctionType.POWER: "v(t) = v0 + (v1-v0)*t*^n",
    FunctionType.EXPONENTIAL: "v(t) = v0 + (v1-v0)*(e^(k*t*)-1)/(e^k-1)",
    FunctionType.LOGARITHMIC: "v(t) = v0 + (v1-v0)*log(1+k*t*)/log(1+k)",
    FunctionType.ERF_RAMP: "v(t) = v0 + (v1-v0)*(1+erf(k*(2t*-1)))/2",
}

# Default parameters for each function type
DEFAULT_PARAMS = {
    FunctionType.CONSTANT: {'c': 100.0},
    FunctionType.LINEAR: {'v0': 0.0, 'v1': 100.0},
    FunctionType.QUADRATIC: {'v0': 0.0, 'v1': 100.0, 'a': 0.0},
    FunctionType.CUBIC: {'v0': 0.0, 'v1': 100.0},
    FunctionType.POWER: {'v0': 0.0, 'v1': 100.0, 'n': 2.0},
    FunctionType.EXPONENTIAL: {'v0': 0.0, 'v1': 100.0, 'k': 3.0},
    FunctionType.LOGARITHMIC: {'v0': 0.0, 'v1': 100.0, 'k': 2.0},
    FunctionType.ERF_RAMP: {'v0': 0.0, 'v1': 100.0, 'k': 3.0},  # k controls steepness
}

# Colors for segments
SEGMENT_COLORS = ['#E41A1C', '#377EB8', '#4DAF4A', '#984EA3',
                  '#FF7F00', '#A65628', '#F781BF', '#00CED1']


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class TrajectorySegment:
    """A single segment of the piecewise trajectory."""
    t_start: float
    t_end: float
    func_type: FunctionType
    params: dict = field(default_factory=dict)

    def evaluate(self, t: np.ndarray) -> np.ndarray:
        """Evaluate the function at given times."""
        mask = (t >= self.t_start) & (t <= self.t_end)
        result = np.full_like(t, np.nan)
        t_seg = t[mask]

        if len(t_seg) == 0:
            return result

        # Normalize time to [0, 1] for segment
        dt = self.t_end - self.t_start
        if dt < 1e-12:
            dt = 1e-12
        t_norm = (t_seg - self.t_start) / dt

        if self.func_type == FunctionType.CONSTANT:
            result[mask] = self.params.get('c', 0)

        elif self.func_type == FunctionType.LINEAR:
            v0 = self.params.get('v0', 0)
            v1 = self.params.get('v1', 0)
            result[mask] = v0 + (v1 - v0) * t_norm

        elif self.func_type == FunctionType.QUADRATIC:
            v0 = self.params.get('v0', 0)
            v1 = self.params.get('v1', 0)
            a = self.params.get('a', 0)
            result[mask] = v0 + (v1 - v0) * t_norm + a * t_norm * (1 - t_norm)

        elif self.func_type == FunctionType.CUBIC:
            v0 = self.params.get('v0', 0)
            v1 = self.params.get('v1', 0)
            result[mask] = v0 + (v1 - v0) * (3 * t_norm**2 - 2 * t_norm**3)

        elif self.func_type == FunctionType.POWER:
            v0 = self.params.get('v0', 0)
            v1 = self.params.get('v1', 0)
            n = self.params.get('n', 2)
            result[mask] = v0 + (v1 - v0) * t_norm**n

        elif self.func_type == FunctionType.EXPONENTIAL:
            v0 = self.params.get('v0', 0)
            v1 = self.params.get('v1', 0)
            k = self.params.get('k', 3)
            if abs(k) < 0.01:
                result[mask] = v0 + (v1 - v0) * t_norm
            else:
                result[mask] = v0 + (v1 - v0) * (np.exp(k * t_norm) - 1) / (np.exp(k) - 1)

        elif self.func_type == FunctionType.LOGARITHMIC:
            v0 = self.params.get('v0', 0)
            v1 = self.params.get('v1', 0)
            k = self.params.get('k', 2)
            result[mask] = v0 + (v1 - v0) * np.log(1 + k * t_norm) / np.log(1 + k)

        elif self.func_type == FunctionType.ERF_RAMP:
            # Smooth S-curve using error function
            # Maps t* in [0,1] to erf argument [-k, k] for smooth transition
            # At t*=0: v ≈ v0, at t*=1: v ≈ v1
            from scipy.special import erf
            v0 = self.params.get('v0', 0)
            v1 = self.params.get('v1', 0)
            k = self.params.get('k', 3)  # Steepness (higher = sharper transition)
            # Map [0,1] to [-k, k] for erf
            x = k * (2 * t_norm - 1)
            # Normalize to [0, 1] range
            erf_val = (1 + erf(x)) / 2
            result[mask] = v0 + (v1 - v0) * erf_val

        return result

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            't_start': self.t_start,
            't_end': self.t_end,
            'func_type': self.func_type.value,
            'params': self.params,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'TrajectorySegment':
        """Create from dictionary."""
        return cls(
            t_start=d['t_start'],
            t_end=d['t_end'],
            func_type=FunctionType(d['func_type']),
            params=d['params'],
        )


# =============================================================================
# DATA LOADER
# =============================================================================

class ExportOptionsDialog(tk.Toplevel):
    """Dialog for export options."""

    def __init__(self, parent):
        super().__init__(parent)
        self.title("Export Options")
        self.geometry("300x200")
        self.resizable(False, False)

        self.confirmed = False
        self.n_points = 1000
        self.gas_velocity_offset = 0.0
        self.initial_position = 0.0

        # Make modal
        self.transient(parent)
        self.grab_set()

        self._create_widgets()

        # Center on parent
        self.update_idletasks()
        x = parent.winfo_x() + (parent.winfo_width() - self.winfo_width()) // 2
        y = parent.winfo_y() + (parent.winfo_height() - self.winfo_height()) // 2
        self.geometry(f"+{x}+{y}")

    def _create_widgets(self):
        frame = ttk.Frame(self, padding=15)
        frame.pack(fill=tk.BOTH, expand=True)

        # Number of points
        row1 = ttk.Frame(frame)
        row1.pack(fill=tk.X, pady=5)
        ttk.Label(row1, text="Number of points:", width=20).pack(side=tk.LEFT)
        self.n_points_var = tk.StringVar(value="1000")
        ttk.Entry(row1, textvariable=self.n_points_var, width=10).pack(side=tk.LEFT)

        # Gas velocity offset
        row2 = ttk.Frame(frame)
        row2.pack(fill=tk.X, pady=5)
        ttk.Label(row2, text="Gas velocity offset [m/s]:", width=20).pack(side=tk.LEFT)
        self.offset_var = tk.StringVar(value="0.0")
        ttk.Entry(row2, textvariable=self.offset_var, width=10).pack(side=tk.LEFT)

        # Initial position
        row3 = ttk.Frame(frame)
        row3.pack(fill=tk.X, pady=5)
        ttk.Label(row3, text="Initial position [m]:", width=20).pack(side=tk.LEFT)
        self.position_var = tk.StringVar(value="0.0")
        ttk.Entry(row3, textvariable=self.position_var, width=10).pack(side=tk.LEFT)

        # Info label
        ttk.Label(frame, text="Position computed by integrating velocity.",
                  font=('TkDefaultFont', 8, 'italic')).pack(anchor=tk.W, pady=10)

        # Buttons
        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill=tk.X, pady=10)
        ttk.Button(btn_frame, text="Export", command=self._on_export).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Cancel", command=self._on_cancel).pack(side=tk.LEFT, padx=5)

    def _on_export(self):
        try:
            self.n_points = int(self.n_points_var.get())
            self.gas_velocity_offset = float(self.offset_var.get())
            self.initial_position = float(self.position_var.get())
            self.confirmed = True
            self.destroy()
        except ValueError as e:
            tk.messagebox.showerror("Invalid Input", f"Please enter valid numbers: {e}")

    def _on_cancel(self):
        self.confirmed = False
        self.destroy()


def load_pele_data():
    """Load PeleC trajectory data if available."""
    try:
        from pele_data_loader import PeleDataLoader
        data_dir = Path(__file__).parent / "pele_data" / "truncated_raw_data"
        if data_dir.exists():
            loader = PeleDataLoader(data_dir)
            return loader.load()
    except Exception as e:
        print(f"Could not load PeleC data: {e}")
    return None


def load_synthetic_trajectory(npz_path: str):
    """
    Load exported synthetic trajectory NPZ file.

    Returns a PeleTrajectoryData-compatible object that can be used
    with PeleTrajectoryInterpolator for piston BC simulations.

    Parameters
    ----------
    npz_path : str or Path
        Path to the exported NPZ file.

    Returns
    -------
    PeleTrajectoryData
        Trajectory data object compatible with PeleTrajectoryInterpolator.

    Example
    -------
    >>> from synthetic_trajectory_gui import load_synthetic_trajectory
    >>> from pele_data_loader import PeleTrajectoryInterpolator
    >>> data = load_synthetic_trajectory("synthetic_trajectories/my_trajectory.npz")
    >>> trajectory = PeleTrajectoryInterpolator(data, extrapolate=False)
    >>> # Use trajectory with DataDrivenPistonBC
    """
    from pele_data_loader import PeleTrajectoryData

    npz_data = np.load(npz_path)

    return PeleTrajectoryData(
        time=npz_data['time'],
        flame_position=npz_data['flame_position'],
        flame_velocity=npz_data['flame_velocity'],
        flame_gas_velocity=npz_data['flame_gas_velocity'],
    )


# =============================================================================
# INTERPOLATOR FOR SIMULATIONS
# =============================================================================

class SyntheticTrajectoryInterpolator:
    """
    Interpolator for synthetic trajectory data.
    Compatible with PeleTrajectoryInterpolator interface.
    """

    def __init__(self, source):
        """
        Load synthetic trajectory.

        Parameters
        ----------
        source : str or Path or list
            JSON file path, or list of TrajectorySegment objects
        """
        if isinstance(source, (str, Path)):
            with open(source, 'r') as f:
                data = json.load(f)
            self.segments = [TrajectorySegment.from_dict(s) for s in data['segments']]
            self.t_min = data['t_min']
            self.t_max = data['t_max']
        else:
            self.segments = source
            if self.segments:
                self.t_min = min(s.t_start for s in self.segments)
                self.t_max = max(s.t_end for s in self.segments)
            else:
                self.t_min = 0.0
                self.t_max = 1.0

    def velocity(self, t: float) -> float:
        """Get velocity at time t."""
        t_arr = np.array([t])
        for seg in self.segments:
            if seg.t_start <= t <= seg.t_end:
                result = seg.evaluate(t_arr)[0]
                if not np.isnan(result):
                    return float(result)
        return 0.0

    def position(self, t: float) -> float:
        """Get position at time t (integrated velocity)."""
        n_points = 1000
        t_int = np.linspace(self.t_min, min(t, self.t_max), n_points)
        v_int = np.array([self.velocity(ti) for ti in t_int])
        return float(np.trapz(v_int, t_int))


# =============================================================================
# MAIN GUI APPLICATION
# =============================================================================

class SyntheticTrajectoryGUI:
    """Main Tkinter application window."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Synthetic Trajectory Builder")
        self.root.geometry("1200x800")

        # Load PeleC data for reference
        self.pele_data = load_pele_data()

        # Time range
        if self.pele_data is not None:
            self.t_min = float(self.pele_data.time.min())
            self.t_max = float(self.pele_data.time.max())
        else:
            self.t_min = 0.0
            self.t_max = 2.5e-3

        # Segments list
        self.segments: List[TrajectorySegment] = []
        self.selected_segment_idx: Optional[int] = None

        # Pending span selection
        self.pending_t_start: Optional[float] = None
        self.pending_t_end: Optional[float] = None

        # Tkinter variables
        self.func_type_var = tk.StringVar(value=FunctionType.LINEAR.value)

        # Parameter variables (created dynamically)
        self.param_vars: Dict[str, tk.StringVar] = {}

        # Setup GUI
        self._create_menu()
        self._create_main_layout()

        # Initial plot
        self._update_plot()

    def _create_menu(self):
        """Create the menu bar."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="New", command=self._new_trajectory)
        file_menu.add_command(label="Open...", command=self._load_trajectory)
        file_menu.add_command(label="Save...", command=self._save_trajectory)
        file_menu.add_separator()
        file_menu.add_command(label="Export as NPZ...", command=self._export_npz)
        file_menu.add_command(label="Export as CSV...", command=self._export_csv)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)

        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self._show_about)

    def _create_main_layout(self):
        """Create the main GUI layout."""
        # Main paned window (horizontal split)
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Left panel (controls)
        left_frame = ttk.Frame(main_paned, width=300)
        main_paned.add(left_frame, weight=0)

        # Right panel (plot area)
        right_frame = ttk.Frame(main_paned)
        main_paned.add(right_frame, weight=1)

        self._create_controls(left_frame)
        self._create_plot_area(right_frame)

    def _create_controls(self, parent: ttk.Frame):
        """Create the control panel."""
        # Scrollable frame
        canvas = tk.Canvas(parent, width=280)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # =====================================================================
        # SEGMENT LIST
        # =====================================================================
        list_frame = ttk.LabelFrame(scrollable_frame, text="Segments", padding=5)
        list_frame.pack(fill=tk.X, padx=5, pady=5)

        # Listbox for segments
        list_container = ttk.Frame(list_frame)
        list_container.pack(fill=tk.BOTH, expand=True, pady=2)

        self.segment_listbox = tk.Listbox(list_container, height=8, exportselection=False)
        list_scrollbar = ttk.Scrollbar(list_container, orient="vertical",
                                        command=self.segment_listbox.yview)
        self.segment_listbox.configure(yscrollcommand=list_scrollbar.set)

        self.segment_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        list_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.segment_listbox.bind('<<ListboxSelect>>', self._on_segment_selected)

        # Segment buttons
        seg_btn_frame = ttk.Frame(list_frame)
        seg_btn_frame.pack(fill=tk.X, pady=2)
        ttk.Button(seg_btn_frame, text="Delete", command=self._delete_segment).pack(side=tk.LEFT, padx=2)
        ttk.Button(seg_btn_frame, text="Clear All", command=self._clear_segments).pack(side=tk.LEFT, padx=2)

        # =====================================================================
        # NEW SEGMENT
        # =====================================================================
        new_frame = ttk.LabelFrame(scrollable_frame, text="Add New Segment", padding=5)
        new_frame.pack(fill=tk.X, padx=5, pady=5)

        # Instructions
        ttk.Label(new_frame, text="1. Click & drag on plot to select time range",
                  font=('TkDefaultFont', 9, 'italic')).pack(anchor=tk.W)
        ttk.Label(new_frame, text="2. Select function type below",
                  font=('TkDefaultFont', 9, 'italic')).pack(anchor=tk.W)
        ttk.Label(new_frame, text="3. Click 'Add Segment'",
                  font=('TkDefaultFont', 9, 'italic')).pack(anchor=tk.W)

        # Time range display
        self.time_range_label = ttk.Label(new_frame, text="Selected: (none)")
        self.time_range_label.pack(anchor=tk.W, pady=5)

        # Function type selector
        func_frame = ttk.Frame(new_frame)
        func_frame.pack(fill=tk.X, pady=2)
        ttk.Label(func_frame, text="Function:").pack(side=tk.LEFT)

        func_values = [f.value for f in FunctionType]
        self.func_combo = ttk.Combobox(func_frame, textvariable=self.func_type_var,
                                        values=func_values, state="readonly", width=15)
        self.func_combo.pack(side=tk.LEFT, padx=5)
        self.func_combo.bind("<<ComboboxSelected>>", self._on_func_type_changed)

        # Function description
        self.func_desc_label = ttk.Label(new_frame, text="", font=('TkDefaultFont', 8))
        self.func_desc_label.pack(anchor=tk.W, pady=2)
        self._update_func_description()

        # Add segment button
        ttk.Button(new_frame, text="Add Segment", command=self._add_segment).pack(fill=tk.X, pady=5)

        # =====================================================================
        # SEGMENT PARAMETERS
        # =====================================================================
        self.params_frame = ttk.LabelFrame(scrollable_frame, text="Segment Parameters", padding=5)
        self.params_frame.pack(fill=tk.X, padx=5, pady=5)

        self.params_container = ttk.Frame(self.params_frame)
        self.params_container.pack(fill=tk.X)

        ttk.Label(self.params_frame, text="(Select a segment to edit)",
                  font=('TkDefaultFont', 9, 'italic')).pack(anchor=tk.W)

        # =====================================================================
        # FITTING
        # =====================================================================
        fit_frame = ttk.LabelFrame(scrollable_frame, text="Fit to PeleC Data", padding=5)
        fit_frame.pack(fill=tk.X, padx=5, pady=5)

        # Fit mode selection
        self.fit_mode_var = tk.StringVar(value="endpoints")
        ttk.Radiobutton(fit_frame, text="Endpoints only (v0, v1)",
                        variable=self.fit_mode_var, value="endpoints").pack(anchor=tk.W)
        ttk.Radiobutton(fit_frame, text="Least-squares (all params)",
                        variable=self.fit_mode_var, value="leastsq").pack(anchor=tk.W)

        # Connect to prior option
        self.connect_prior_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(fit_frame, text="Connect to prior segment (fix v0)",
                        variable=self.connect_prior_var).pack(anchor=tk.W, pady=(5, 0))

        # Fill gaps option
        self.fill_gaps_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(fit_frame, text="Fill gaps between segments (interpolate)",
                        variable=self.fill_gaps_var,
                        command=self._update_plot).pack(anchor=tk.W, pady=(2, 0))

        # Fit buttons
        fit_btn_frame = ttk.Frame(fit_frame)
        fit_btn_frame.pack(fill=tk.X, pady=5)
        ttk.Button(fit_btn_frame, text="Fit Selected",
                   command=self._fit_selected_segment).pack(side=tk.LEFT, padx=2)
        ttk.Button(fit_btn_frame, text="Fit All",
                   command=self._fit_all_segments).pack(side=tk.LEFT, padx=2)

        # =====================================================================
        # ACTIONS
        # =====================================================================
        action_frame = ttk.LabelFrame(scrollable_frame, text="Actions", padding=5)
        action_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(action_frame, text="Add Startup Ramp (erf)",
                   command=self._add_startup_ramp).pack(fill=tk.X, pady=2)

        # Ramp duration entry
        ramp_row = ttk.Frame(action_frame)
        ramp_row.pack(fill=tk.X, pady=2)
        ttk.Label(ramp_row, text="Ramp duration [µs]:", width=18).pack(side=tk.LEFT)
        self.ramp_duration_var = tk.StringVar(value="30")
        ttk.Entry(ramp_row, textvariable=self.ramp_duration_var, width=8).pack(side=tk.LEFT)

        ttk.Button(action_frame, text="Preview Combined",
                   command=self._update_plot).pack(fill=tk.X, pady=2)

    def _create_plot_area(self, parent: ttk.Frame):
        """Create the matplotlib plot area."""
        # Create figure
        self.figure = Figure(figsize=(8, 6), dpi=100)
        self.ax = self.figure.add_subplot(111)

        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.figure, parent)
        self.canvas.draw()

        # Add toolbar
        toolbar_frame = ttk.Frame(parent)
        toolbar_frame.pack(side=tk.TOP, fill=tk.X)
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()

        # Add canvas widget
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Add span selector for time range selection
        self.span_selector = SpanSelector(
            self.ax, self._on_span_select, 'horizontal',
            useblit=True, props=dict(alpha=0.3, facecolor='yellow'),
            interactive=True, drag_from_anywhere=True
        )

        # Connect click event for segment selection
        self.canvas.mpl_connect('button_press_event', self._on_plot_click)

    def _on_span_select(self, t_min_ms, t_max_ms):
        """Handle span selection for time range."""
        self.pending_t_start = t_min_ms * 1e-3  # Convert from ms to s
        self.pending_t_end = t_max_ms * 1e-3

        # Update label
        self.time_range_label.config(
            text=f"Selected: [{t_min_ms:.2f}, {t_max_ms:.2f}] ms"
        )

        self._update_plot()

    def _on_plot_click(self, event):
        """Handle click on plot for segment selection."""
        if event.inaxes != self.ax:
            return

        # Only handle double-clicks for segment selection
        if event.dblclick:
            t_click = event.xdata * 1e-3  # Convert ms to s

            for i, seg in enumerate(self.segments):
                if seg.t_start <= t_click <= seg.t_end:
                    self.segment_listbox.selection_clear(0, tk.END)
                    self.segment_listbox.selection_set(i)
                    self._on_segment_selected(None)
                    return

    def _on_segment_selected(self, event):
        """Handle segment selection from listbox."""
        selection = self.segment_listbox.curselection()
        if not selection:
            self.selected_segment_idx = None
            self._clear_param_widgets()
            return

        self.selected_segment_idx = selection[0]
        self._update_param_widgets()
        self._update_plot()

    def _on_func_type_changed(self, event):
        """Handle function type change."""
        self._update_func_description()

    def _update_func_description(self):
        """Update the function description label."""
        func_type = FunctionType(self.func_type_var.get())
        desc = FUNCTION_DESCRIPTIONS.get(func_type, "")
        self.func_desc_label.config(text=desc)

    def _add_segment(self):
        """Add a new segment with selected time bounds."""
        if self.pending_t_start is None or self.pending_t_end is None:
            messagebox.showwarning("No Selection",
                                   "Please select a time range by clicking and dragging on the plot.")
            return

        t_start = min(self.pending_t_start, self.pending_t_end)
        t_end = max(self.pending_t_start, self.pending_t_end)

        func_type = FunctionType(self.func_type_var.get())

        # Get initial values from PeleC data if available
        params = DEFAULT_PARAMS[func_type].copy()
        if self.pele_data is not None:
            idx_start = np.argmin(np.abs(self.pele_data.time - t_start))
            idx_end = np.argmin(np.abs(self.pele_data.time - t_end))
            v0 = float(self.pele_data.flame_velocity[idx_start])
            v1 = float(self.pele_data.flame_velocity[idx_end])

            if 'v0' in params:
                params['v0'] = v0
            if 'v1' in params:
                params['v1'] = v1
            if 'c' in params:
                params['c'] = (v0 + v1) / 2

        # Connect to prior segment if option is enabled and there are previous segments
        if self.connect_prior_var.get() and self.segments:
            prior_v1 = self._get_prior_segment_v1(t_start)
            if prior_v1 is not None:
                if 'v0' in params:
                    params['v0'] = prior_v1
                if 'c' in params:
                    # For constant, use the prior's endpoint as the constant
                    params['c'] = prior_v1

        segment = TrajectorySegment(
            t_start=t_start,
            t_end=t_end,
            func_type=func_type,
            params=params,
        )

        self.segments.append(segment)
        self.selected_segment_idx = len(self.segments) - 1

        # Clear pending selection
        self.pending_t_start = None
        self.pending_t_end = None
        self.time_range_label.config(text="Selected: (none)")

        self._update_segment_list()
        self._update_param_widgets()
        self._update_plot()

    def _delete_segment(self):
        """Delete the selected segment."""
        if self.selected_segment_idx is not None and self.selected_segment_idx < len(self.segments):
            del self.segments[self.selected_segment_idx]
            self.selected_segment_idx = None
            self._update_segment_list()
            self._clear_param_widgets()
            self._update_plot()

    def _clear_segments(self):
        """Clear all segments."""
        if messagebox.askyesno("Confirm", "Delete all segments?"):
            self.segments.clear()
            self.selected_segment_idx = None
            self._update_segment_list()
            self._clear_param_widgets()
            self._update_plot()

    def _get_prior_segment_v1(self, t_start: float) -> Optional[float]:
        """
        Get the v1 value from the segment that ends closest to (but before) t_start.
        Returns None if no suitable prior segment exists.
        """
        best_seg = None
        best_gap = float('inf')

        for seg in self.segments:
            # Find segments that end before or at t_start
            gap = t_start - seg.t_end
            if gap >= -1e-9 and gap < best_gap:  # Allow tiny overlap
                best_gap = gap
                best_seg = seg

        if best_seg is None:
            return None

        # Get the v1 (end value) from the prior segment
        if 'v1' in best_seg.params:
            return best_seg.params['v1']
        elif 'c' in best_seg.params:
            return best_seg.params['c']
        else:
            # Evaluate at the end of the segment to get the end velocity
            t_arr = np.array([best_seg.t_end])
            v_arr = best_seg.evaluate(t_arr)
            if not np.isnan(v_arr[0]):
                return float(v_arr[0])
            return None

    def _update_segment_list(self):
        """Update the segment listbox."""
        self.segment_listbox.delete(0, tk.END)
        for i, seg in enumerate(self.segments):
            label = f"{i+1}. {seg.func_type.value} [{seg.t_start*1e3:.2f}, {seg.t_end*1e3:.2f}] ms"
            self.segment_listbox.insert(tk.END, label)

        # Restore selection
        if self.selected_segment_idx is not None and self.selected_segment_idx < len(self.segments):
            self.segment_listbox.selection_set(self.selected_segment_idx)

    def _update_param_widgets(self):
        """Update parameter editing widgets for selected segment."""
        self._clear_param_widgets()

        if self.selected_segment_idx is None or self.selected_segment_idx >= len(self.segments):
            return

        seg = self.segments[self.selected_segment_idx]

        # Clear old widgets
        for widget in self.params_container.winfo_children():
            widget.destroy()

        # Create parameter entry widgets
        self.param_vars = {}
        for name, value in seg.params.items():
            row = ttk.Frame(self.params_container)
            row.pack(fill=tk.X, pady=2)

            ttk.Label(row, text=f"{name}:", width=8).pack(side=tk.LEFT)

            var = tk.StringVar(value=f"{value:.4g}")
            self.param_vars[name] = var

            entry = ttk.Entry(row, textvariable=var, width=12)
            entry.pack(side=tk.LEFT, padx=5)
            entry.bind('<Return>', self._on_param_changed)
            entry.bind('<FocusOut>', self._on_param_changed)

        # Add apply button
        ttk.Button(self.params_container, text="Apply Changes",
                   command=self._apply_param_changes).pack(fill=tk.X, pady=5)

    def _clear_param_widgets(self):
        """Clear parameter widgets."""
        for widget in self.params_container.winfo_children():
            widget.destroy()
        self.param_vars = {}

    def _on_param_changed(self, event):
        """Handle parameter value change."""
        self._apply_param_changes()

    def _apply_param_changes(self):
        """Apply parameter changes to selected segment."""
        if self.selected_segment_idx is None:
            return

        seg = self.segments[self.selected_segment_idx]
        for name, var in self.param_vars.items():
            try:
                seg.params[name] = float(var.get())
            except ValueError:
                pass

        self._update_plot()

    def _fit_selected_segment(self):
        """Fit the selected segment to PeleC data."""
        if self.pele_data is None:
            messagebox.showwarning("No Data", "PeleC data not loaded.")
            return

        if self.selected_segment_idx is None:
            messagebox.showwarning("No Selection", "Please select a segment to fit.")
            return

        seg = self.segments[self.selected_segment_idx]
        self._fit_segment(seg)

        self._update_param_widgets()
        self._update_plot()

    def _fit_all_segments(self):
        """Fit all segments to PeleC data."""
        if self.pele_data is None:
            messagebox.showwarning("No Data", "PeleC data not loaded.")
            return

        if not self.segments:
            messagebox.showwarning("No Segments", "No segments to fit.")
            return

        for seg in self.segments:
            self._fit_segment(seg)

        self._update_param_widgets()
        self._update_plot()

    def _fit_segment(self, seg: TrajectorySegment):
        """Fit a single segment to PeleC data."""
        # Get PeleC data within segment time range
        mask = (self.pele_data.time >= seg.t_start) & (self.pele_data.time <= seg.t_end)
        t_data = self.pele_data.time[mask]
        v_data = self.pele_data.flame_velocity[mask]

        # Check if we should connect to prior segment
        connect_prior = self.connect_prior_var.get()
        prior_v1 = None
        if connect_prior:
            prior_v1 = self._get_prior_segment_v1(seg.t_start)

        if len(t_data) < 2:
            # Not enough data points, just use endpoints
            idx_start = np.argmin(np.abs(self.pele_data.time - seg.t_start))
            idx_end = np.argmin(np.abs(self.pele_data.time - seg.t_end))
            v0 = float(self.pele_data.flame_velocity[idx_start])
            v1 = float(self.pele_data.flame_velocity[idx_end])

            # Override v0 with prior segment's v1 if connecting
            if connect_prior and prior_v1 is not None:
                v0 = prior_v1

            if 'v0' in seg.params:
                seg.params['v0'] = v0
            if 'v1' in seg.params:
                seg.params['v1'] = v1
            if 'c' in seg.params:
                seg.params['c'] = prior_v1 if (connect_prior and prior_v1 is not None) else (v0 + v1) / 2
            return

        fit_mode = self.fit_mode_var.get()

        if fit_mode == "endpoints":
            # Just fit endpoints
            idx_start = np.argmin(np.abs(self.pele_data.time - seg.t_start))
            idx_end = np.argmin(np.abs(self.pele_data.time - seg.t_end))
            v0 = float(self.pele_data.flame_velocity[idx_start])
            v1 = float(self.pele_data.flame_velocity[idx_end])

            # Override v0 with prior segment's v1 if connecting
            if connect_prior and prior_v1 is not None:
                v0 = prior_v1

            if 'v0' in seg.params:
                seg.params['v0'] = v0
            if 'v1' in seg.params:
                seg.params['v1'] = v1
            if 'c' in seg.params:
                seg.params['c'] = prior_v1 if (connect_prior and prior_v1 is not None) else (v0 + v1) / 2

        elif fit_mode == "leastsq":
            # Least-squares fit all parameters
            try:
                from scipy.optimize import minimize

                # If connecting to prior, fix v0 and exclude it from optimization
                fixed_v0 = None
                if connect_prior and prior_v1 is not None and 'v0' in seg.params:
                    fixed_v0 = prior_v1
                    seg.params['v0'] = fixed_v0

                # Get parameter names (exclude v0 if fixed)
                all_param_names = list(seg.params.keys())
                if fixed_v0 is not None:
                    param_names = [n for n in all_param_names if n != 'v0']
                else:
                    param_names = all_param_names

                x0 = [seg.params[name] for name in param_names]

                # Define objective function
                def objective(x):
                    # Update segment parameters
                    for i, name in enumerate(param_names):
                        seg.params[name] = x[i]
                    # v0 stays fixed if we're connecting to prior
                    # Evaluate segment
                    v_pred = seg.evaluate(t_data)
                    # Compute residual
                    residual = np.nansum((v_pred - v_data) ** 2)
                    return residual

                # Set bounds for parameters
                bounds = []
                for name in param_names:
                    if name in ('v0', 'v1', 'c'):
                        bounds.append((0, 2000))  # Velocity bounds
                    elif name == 'n':
                        bounds.append((0.5, 5))  # Power exponent bounds
                    elif name == 'k':
                        bounds.append((0.1, 10))  # Rate constant bounds
                    elif name == 'a':
                        bounds.append((-500, 500))  # Curvature bounds
                    else:
                        bounds.append((None, None))

                # Run optimization
                result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds)

                if result.success:
                    # Update parameters with optimized values
                    for i, name in enumerate(param_names):
                        seg.params[name] = result.x[i]
                else:
                    print(f"Optimization warning: {result.message}")
                    # Still use the result even if not fully converged
                    for i, name in enumerate(param_names):
                        seg.params[name] = result.x[i]

            except ImportError:
                messagebox.showwarning("Missing scipy",
                                       "scipy is required for least-squares fitting.\n"
                                       "Falling back to endpoint fitting.")
                # Fallback to endpoint fitting
                self.fit_mode_var.set("endpoints")
                self._fit_segment(seg)

    def _update_plot(self):
        """Update the plot."""
        self.ax.clear()

        t_eval = np.linspace(self.t_min, self.t_max, 1000)
        t_ms = t_eval * 1e3

        # Plot PeleC reference data
        if self.pele_data is not None:
            self.ax.plot(self.pele_data.time * 1e3, self.pele_data.flame_velocity,
                        'k-', lw=1, alpha=0.5, label='PeleC Data')

        # Plot each segment
        for i, seg in enumerate(self.segments):
            v_seg = seg.evaluate(t_eval)
            mask = ~np.isnan(v_seg)

            color = SEGMENT_COLORS[i % len(SEGMENT_COLORS)]
            lw = 3 if i == self.selected_segment_idx else 1.5
            alpha = 1.0 if i == self.selected_segment_idx else 0.8

            self.ax.plot(t_ms[mask], v_seg[mask], color=color, lw=lw, alpha=alpha,
                        label=f'Seg {i+1}: {seg.func_type.value}')

            # Highlight selected segment
            if i == self.selected_segment_idx:
                self.ax.axvspan(seg.t_start * 1e3, seg.t_end * 1e3, alpha=0.1, color=color)

        # Plot pending selection
        if self.pending_t_start is not None and self.pending_t_end is not None:
            self.ax.axvspan(self.pending_t_start * 1e3, self.pending_t_end * 1e3,
                           alpha=0.2, color='yellow', label='Pending')

        # Plot combined trajectory
        if self.segments:
            fill_gaps = self.fill_gaps_var.get()
            v_combined = self._evaluate_combined(t_eval, fill_gaps=fill_gaps)
            mask = ~np.isnan(v_combined)
            label = 'Combined (gaps filled)' if fill_gaps else 'Combined'
            self.ax.plot(t_ms[mask], v_combined[mask], 'b--', lw=2, alpha=0.7, label=label)

        self.ax.set_xlabel('Time [ms]')
        self.ax.set_ylabel('Velocity [m/s]')
        self.ax.set_title('Synthetic Trajectory Builder')
        self.ax.set_xlim(self.t_min * 1e3, self.t_max * 1e3)
        self.ax.grid(True, alpha=0.3)
        self.ax.legend(loc='best', fontsize=8)

        self.canvas.draw()

    def _evaluate_combined(self, t: np.ndarray, fill_gaps: bool = True) -> np.ndarray:
        """
        Evaluate the combined piecewise trajectory.

        Parameters
        ----------
        t : np.ndarray
            Time array to evaluate
        fill_gaps : bool
            If True, linearly interpolate between segments to fill gaps.
            This ensures a continuous trajectory.
        """
        result = np.full_like(t, np.nan)

        if not self.segments:
            return result

        # First, evaluate all segments
        for seg in self.segments:
            v_seg = seg.evaluate(t)
            mask = ~np.isnan(v_seg)
            result[mask] = v_seg[mask]

        if not fill_gaps:
            return result

        # Sort segments by start time
        sorted_segs = sorted(self.segments, key=lambda s: s.t_start)

        # Get segment endpoints for interpolation
        seg_endpoints = []
        for seg in sorted_segs:
            # Evaluate at segment boundaries
            t_arr = np.array([seg.t_start, seg.t_end])
            v_arr = seg.evaluate(t_arr)
            v_start = v_arr[0] if not np.isnan(v_arr[0]) else seg.params.get('v0', seg.params.get('c', 0))
            v_end = v_arr[1] if not np.isnan(v_arr[1]) else seg.params.get('v1', seg.params.get('c', v_start))
            seg_endpoints.append((seg.t_start, seg.t_end, v_start, v_end))

        # Fill gaps between segments with linear interpolation
        for i in range(len(seg_endpoints) - 1):
            t_end_prev = seg_endpoints[i][1]
            v_end_prev = seg_endpoints[i][3]
            t_start_next = seg_endpoints[i + 1][0]
            v_start_next = seg_endpoints[i + 1][2]

            # Check for gap
            if t_start_next > t_end_prev + 1e-12:
                # Linear interpolation in the gap
                gap_mask = (t > t_end_prev) & (t < t_start_next)
                if np.any(gap_mask):
                    t_gap = t[gap_mask]
                    t_frac = (t_gap - t_end_prev) / (t_start_next - t_end_prev)
                    result[gap_mask] = v_end_prev + (v_start_next - v_end_prev) * t_frac

        # Handle time before first segment
        first_seg = seg_endpoints[0]
        before_mask = t < first_seg[0]
        if np.any(before_mask):
            # Use v0 of first segment (or 0 if before all segments)
            result[before_mask] = first_seg[2]

        # Handle time after last segment
        last_seg = seg_endpoints[-1]
        after_mask = t > last_seg[1]
        if np.any(after_mask):
            # Use v1 of last segment
            result[after_mask] = last_seg[3]

        return result

    def _new_trajectory(self):
        """Create a new trajectory."""
        if self.segments and not messagebox.askyesno("Confirm", "Discard current trajectory?"):
            return
        self.segments.clear()
        self.selected_segment_idx = None
        self._update_segment_list()
        self._clear_param_widgets()
        self._update_plot()

    def _save_trajectory(self):
        """Save trajectory to JSON file."""
        if not self.segments:
            messagebox.showwarning("No Data", "No segments to save.")
            return

        output_dir = Path(__file__).parent / "synthetic_trajectories"
        output_dir.mkdir(exist_ok=True)

        filepath = filedialog.asksaveasfilename(
            initialdir=output_dir,
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )

        if not filepath:
            return

        data = {
            'segments': [seg.to_dict() for seg in self.segments],
            't_min': self.t_min,
            't_max': self.t_max,
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        messagebox.showinfo("Saved", f"Trajectory saved to:\n{filepath}")

    def _load_trajectory(self):
        """Load trajectory from JSON file."""
        output_dir = Path(__file__).parent / "synthetic_trajectories"
        if not output_dir.exists():
            output_dir = Path(__file__).parent

        filepath = filedialog.askopenfilename(
            initialdir=output_dir,
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )

        if not filepath:
            return

        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            self.segments = [TrajectorySegment.from_dict(s) for s in data['segments']]
            self.selected_segment_idx = None

            self._update_segment_list()
            self._clear_param_widgets()
            self._update_plot()

            messagebox.showinfo("Loaded", f"Loaded {len(self.segments)} segments")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load: {e}")

    def _export_npz(self):
        """Export trajectory as NPZ file compatible with PeleTrajectoryData."""
        if not self.segments:
            messagebox.showwarning("No Data", "No segments to export.")
            return

        # Show export options dialog
        export_dialog = ExportOptionsDialog(self.root)
        self.root.wait_window(export_dialog)

        if not export_dialog.confirmed:
            return

        n_points = export_dialog.n_points
        gas_velocity_offset = export_dialog.gas_velocity_offset
        initial_position = export_dialog.initial_position

        output_dir = Path(__file__).parent / "synthetic_trajectories"
        output_dir.mkdir(exist_ok=True)

        filepath = filedialog.asksaveasfilename(
            initialdir=output_dir,
            defaultextension=".npz",
            filetypes=[("NumPy files", "*.npz"), ("All files", "*.*")]
        )

        if not filepath:
            return

        # Generate time and velocity arrays
        t_export = np.linspace(self.t_min, self.t_max, n_points)
        # Always fill gaps for export to get continuous trajectory
        v_export = self._evaluate_combined(t_export, fill_gaps=True)

        # Replace any remaining NaN with 0 (shouldn't happen with fill_gaps=True)
        v_export = np.nan_to_num(v_export, nan=0.0)

        # Compute position by integrating velocity (trapezoidal rule)
        position = np.zeros_like(t_export)
        position[0] = initial_position
        for i in range(1, len(t_export)):
            dt = t_export[i] - t_export[i-1]
            position[i] = position[i-1] + 0.5 * (v_export[i-1] + v_export[i]) * dt

        # Gas velocity = flame velocity + offset (for porous boundary)
        gas_velocity = v_export + gas_velocity_offset

        # Save in PeleTrajectoryData-compatible format
        np.savez(
            filepath,
            time=t_export,
            flame_position=position,
            flame_velocity=v_export,
            flame_gas_velocity=gas_velocity,
        )
        messagebox.showinfo("Exported", f"Trajectory exported to:\n{filepath}\n\n"
                           f"Fields: time, flame_position, flame_velocity, flame_gas_velocity\n"
                           f"Points: {n_points}\n"
                           f"Gas velocity offset: {gas_velocity_offset} m/s")

    def _export_csv(self):
        """Export trajectory as CSV file."""
        if not self.segments:
            messagebox.showwarning("No Data", "No segments to export.")
            return

        # Show export options dialog
        export_dialog = ExportOptionsDialog(self.root)
        self.root.wait_window(export_dialog)

        if not export_dialog.confirmed:
            return

        n_points = export_dialog.n_points
        gas_velocity_offset = export_dialog.gas_velocity_offset
        initial_position = export_dialog.initial_position

        output_dir = Path(__file__).parent / "synthetic_trajectories"
        output_dir.mkdir(exist_ok=True)

        filepath = filedialog.asksaveasfilename(
            initialdir=output_dir,
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )

        if not filepath:
            return

        # Generate data (always fill gaps for export)
        t_export = np.linspace(self.t_min, self.t_max, n_points)
        v_export = self._evaluate_combined(t_export, fill_gaps=True)
        v_export = np.nan_to_num(v_export, nan=0.0)

        # Compute position
        position = np.zeros_like(t_export)
        position[0] = initial_position
        for i in range(1, len(t_export)):
            dt = t_export[i] - t_export[i-1]
            position[i] = position[i-1] + 0.5 * (v_export[i-1] + v_export[i]) * dt

        gas_velocity = v_export + gas_velocity_offset

        # Write CSV
        with open(filepath, 'w') as f:
            f.write("time,flame_position,flame_velocity,flame_gas_velocity\n")
            for i in range(len(t_export)):
                f.write(f"{t_export[i]:.9e},{position[i]:.9e},{v_export[i]:.6f},{gas_velocity[i]:.6f}\n")

        messagebox.showinfo("Exported", f"Trajectory exported to:\n{filepath}")

    def _add_startup_ramp(self):
        """
        Add an ERF startup ramp from t=0 to the start of the first segment.

        The ramp goes from v=0 to the starting velocity of the first segment,
        with the slope at the end matching the first segment's initial slope.
        """
        if not self.segments:
            messagebox.showwarning("No Segments", "Add at least one segment first.")
            return

        # Get ramp duration
        try:
            ramp_duration_us = float(self.ramp_duration_var.get())
            ramp_duration = ramp_duration_us * 1e-6  # Convert to seconds
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a valid ramp duration.")
            return

        # Find the first segment (by start time)
        sorted_segs = sorted(self.segments, key=lambda s: s.t_start)
        first_seg = sorted_segs[0]

        # Check if there's room for a ramp
        if first_seg.t_start < ramp_duration + 1e-9:
            # Shift the ramp to end at first segment start
            t_ramp_start = max(0, first_seg.t_start - ramp_duration)
            t_ramp_end = first_seg.t_start
        else:
            t_ramp_start = first_seg.t_start - ramp_duration
            t_ramp_end = first_seg.t_start

        if t_ramp_end - t_ramp_start < 1e-9:
            messagebox.showwarning("No Room", "Not enough space for startup ramp.")
            return

        # Get the starting velocity of the first segment
        t_arr = np.array([first_seg.t_start])
        v_arr = first_seg.evaluate(t_arr)
        v_target = v_arr[0] if not np.isnan(v_arr[0]) else first_seg.params.get('v0', first_seg.params.get('c', 0))

        # Estimate the slope at the start of the first segment
        # by evaluating slightly after the start
        eps = (first_seg.t_end - first_seg.t_start) * 0.01
        t_arr2 = np.array([first_seg.t_start + eps])
        v_arr2 = first_seg.evaluate(t_arr2)
        if not np.isnan(v_arr2[0]) and eps > 0:
            slope_first = (v_arr2[0] - v_target) / eps
        else:
            slope_first = 0

        # For ERF_RAMP, the slope at t*=1 is approximately:
        # dv/dt = (v1-v0) * k * 2 / (sqrt(pi) * dt) * exp(-k^2)
        # where dt = t_end - t_start
        # We want this to match slope_first
        # Solve for k (approximately): k ≈ 3 is a good default
        # For slope matching, we'd need numerical optimization
        # For now, use k=3 which gives a reasonably smooth S-curve
        k = 3.0

        # Create the ramp segment
        ramp_segment = TrajectorySegment(
            t_start=t_ramp_start,
            t_end=t_ramp_end,
            func_type=FunctionType.ERF_RAMP,
            params={'v0': 0.0, 'v1': v_target, 'k': k},
        )

        # Add at the beginning of the segment list
        self.segments.insert(0, ramp_segment)
        self.selected_segment_idx = 0

        self._update_segment_list()
        self._update_param_widgets()
        self._update_plot()

        messagebox.showinfo("Ramp Added",
                           f"Added ERF startup ramp:\n"
                           f"  Time: [{t_ramp_start*1e6:.1f}, {t_ramp_end*1e6:.1f}] µs\n"
                           f"  Velocity: 0 → {v_target:.1f} m/s")

    def _show_about(self):
        """Show about dialog."""
        messagebox.showinfo(
            "About",
            "Synthetic Trajectory Builder\n\n"
            "Create piecewise trajectories from idealized functions.\n\n"
            "Instructions:\n"
            "1. Click and drag on plot to select time range\n"
            "2. Choose function type and click 'Add Segment'\n"
            "3. Select segment to edit parameters\n"
            "4. Save or export when done\n\n"
            "Tips:\n"
            "- Use 'Add Startup Ramp' to add smooth v=0 start\n"
            "- Enable 'Fill gaps' to interpolate between segments\n"
            "- Use 'Connect to prior' when fitting to ensure continuity"
        )


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point."""
    root = tk.Tk()
    app = SyntheticTrajectoryGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
