"""
Output writers for simulation data.

Provides HDF5 and CSV output formats for saving simulation results.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np

from lagrangian_solver.core.state import FlowState
from lagrangian_solver.core.grid import LagrangianGrid


@dataclass
class OutputFrame:
    """
    A single output frame (snapshot in time).

    Attributes:
        time: Simulation time [s]
        step: Time step number
        x: Face positions [m]
        x_cell: Cell center positions [m]
        rho: Density [kg/m³]
        u: Velocity at faces [m/s]
        p: Pressure [Pa]
        T: Temperature [K]
        e: Specific internal energy [J/kg]
        E: Total specific energy [J/kg]
        c: Sound speed [m/s]
        s: Specific entropy [J/(kg·K)]
    """

    time: float
    step: int
    x: np.ndarray
    x_cell: np.ndarray
    rho: np.ndarray
    u: np.ndarray
    p: np.ndarray
    T: np.ndarray
    e: np.ndarray
    E: np.ndarray
    c: np.ndarray
    s: np.ndarray

    @classmethod
    def from_state(
        cls, state: FlowState, grid: LagrangianGrid, time: float, step: int
    ) -> "OutputFrame":
        """Create output frame from flow state."""
        return cls(
            time=time,
            step=step,
            x=state.x.copy(),
            x_cell=grid.x_cell.copy(),
            rho=state.rho.copy(),
            u=state.u.copy(),
            p=state.p.copy(),
            T=state.T.copy(),
            e=state.e.copy(),
            E=state.E.copy(),
            c=state.c.copy(),
            s=state.s.copy(),
        )


class OutputWriter(ABC):
    """Abstract base class for output writers."""

    def __init__(self, output_dir: str, base_name: str = "solution"):
        """
        Initialize output writer.

        Args:
            output_dir: Directory for output files
            base_name: Base name for output files
        """
        self._output_dir = Path(output_dir)
        self._base_name = base_name
        self._output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def output_dir(self) -> Path:
        """Output directory."""
        return self._output_dir

    @abstractmethod
    def write_frame(self, frame: OutputFrame) -> None:
        """
        Write a single output frame.

        Args:
            frame: Output frame to write
        """
        pass

    @abstractmethod
    def write_metadata(self, metadata: Dict[str, Any]) -> None:
        """
        Write simulation metadata.

        Args:
            metadata: Dictionary of metadata (config, etc.)
        """
        pass

    @abstractmethod
    def finalize(self) -> None:
        """Finalize output (close files, etc.)."""
        pass


class CSVWriter(OutputWriter):
    """
    CSV output writer.

    Writes each frame to a separate CSV file.
    Cell-centered quantities and face positions are stored.
    """

    def __init__(
        self,
        output_dir: str,
        base_name: str = "solution",
        delimiter: str = ",",
    ):
        """
        Initialize CSV writer.

        Args:
            output_dir: Directory for output files
            base_name: Base name for output files
            delimiter: CSV delimiter (default comma)
        """
        super().__init__(output_dir, base_name)
        self._delimiter = delimiter
        self._frame_count = 0

    def write_frame(self, frame: OutputFrame) -> None:
        """Write frame to CSV file."""
        filename = self._output_dir / f"{self._base_name}_{self._frame_count:06d}.csv"

        # Cell-centered data
        n_cells = len(frame.rho)
        u_cell = 0.5 * (frame.u[:-1] + frame.u[1:])

        # Create header and data
        header = "x,rho,u,p,T,e,E,c,s"
        data = np.column_stack(
            [
                frame.x_cell,
                frame.rho,
                u_cell,
                frame.p,
                frame.T,
                frame.e,
                frame.E,
                frame.c,
                frame.s,
            ]
        )

        # Write with metadata comment
        with open(filename, "w") as f:
            f.write(f"# time = {frame.time}\n")
            f.write(f"# step = {frame.step}\n")
            f.write(f"# n_cells = {n_cells}\n")
            f.write(header + "\n")
            np.savetxt(f, data, delimiter=self._delimiter, fmt="%.10e")

        self._frame_count += 1

    def write_metadata(self, metadata: Dict[str, Any]) -> None:
        """Write metadata to JSON file."""
        import json

        filename = self._output_dir / f"{self._base_name}_metadata.json"
        with open(filename, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

    def finalize(self) -> None:
        """No finalization needed for CSV."""
        pass

    def write_faces(self, frame: OutputFrame) -> None:
        """
        Write face-centered data to separate CSV file.

        Useful for analyzing velocity at boundaries.
        """
        filename = self._output_dir / f"{self._base_name}_faces_{self._frame_count - 1:06d}.csv"

        header = "x,u"
        data = np.column_stack([frame.x, frame.u])

        with open(filename, "w") as f:
            f.write(f"# time = {frame.time}\n")
            f.write(header + "\n")
            np.savetxt(f, data, delimiter=self._delimiter, fmt="%.10e")


class HDF5Writer(OutputWriter):
    """
    HDF5 output writer.

    Writes all frames to a single HDF5 file with efficient storage.
    Supports compression and chunking for large datasets.
    """

    def __init__(
        self,
        output_dir: str,
        base_name: str = "solution",
        compression: str = "gzip",
        compression_level: int = 4,
    ):
        """
        Initialize HDF5 writer.

        Args:
            output_dir: Directory for output files
            base_name: Base name for output file
            compression: Compression algorithm ('gzip', 'lzf', or None)
            compression_level: Compression level (1-9 for gzip)
        """
        super().__init__(output_dir, base_name)
        self._compression = compression
        self._compression_level = compression_level
        self._file = None
        self._frame_count = 0
        self._initialized = False

    def _ensure_h5py(self):
        """Ensure h5py is available."""
        try:
            import h5py

            self._h5py = h5py
        except ImportError:
            raise ImportError("h5py required for HDF5 output. Install with: pip install h5py")

    def _initialize(self, frame: OutputFrame) -> None:
        """Initialize HDF5 file with first frame."""
        self._ensure_h5py()

        filename = self._output_dir / f"{self._base_name}.h5"
        self._file = self._h5py.File(filename, "w")

        # Create groups
        self._file.create_group("frames")
        self._file.create_group("metadata")

        # Store grid info
        n_cells = len(frame.rho)
        n_faces = len(frame.x)
        self._file.attrs["n_cells"] = n_cells
        self._file.attrs["n_faces"] = n_faces

        self._initialized = True

    def write_frame(self, frame: OutputFrame) -> None:
        """Write frame to HDF5 file."""
        if not self._initialized:
            self._initialize(frame)

        # Create group for this frame
        grp_name = f"frames/frame_{self._frame_count:06d}"
        grp = self._file.create_group(grp_name)

        # Store time and step
        grp.attrs["time"] = frame.time
        grp.attrs["step"] = frame.step

        # Store datasets with compression
        compression_opts = {}
        if self._compression:
            compression_opts["compression"] = self._compression
            if self._compression == "gzip":
                compression_opts["compression_opts"] = self._compression_level

        # Cell-centered data
        grp.create_dataset("x_cell", data=frame.x_cell, **compression_opts)
        grp.create_dataset("rho", data=frame.rho, **compression_opts)
        grp.create_dataset("p", data=frame.p, **compression_opts)
        grp.create_dataset("T", data=frame.T, **compression_opts)
        grp.create_dataset("e", data=frame.e, **compression_opts)
        grp.create_dataset("E", data=frame.E, **compression_opts)
        grp.create_dataset("c", data=frame.c, **compression_opts)

        # Face-centered data
        grp.create_dataset("x", data=frame.x, **compression_opts)
        grp.create_dataset("u", data=frame.u, **compression_opts)

        self._frame_count += 1

    def write_metadata(self, metadata: Dict[str, Any]) -> None:
        """Write metadata to HDF5 file."""
        if self._file is None:
            self._ensure_h5py()
            filename = self._output_dir / f"{self._base_name}.h5"
            self._file = self._h5py.File(filename, "w")
            self._file.create_group("frames")
            self._file.create_group("metadata")

        # Store metadata as attributes
        meta_grp = self._file["metadata"]
        for key, value in metadata.items():
            if isinstance(value, dict):
                sub_grp = meta_grp.create_group(key)
                for k, v in value.items():
                    if v is not None:
                        sub_grp.attrs[k] = v
            elif value is not None:
                meta_grp.attrs[key] = value

    def finalize(self) -> None:
        """Close HDF5 file."""
        if self._file is not None:
            self._file.attrs["n_frames"] = self._frame_count
            self._file.close()
            self._file = None


class MultiWriter(OutputWriter):
    """
    Multi-format writer that writes to multiple formats simultaneously.
    """

    def __init__(self, writers: List[OutputWriter]):
        """
        Initialize multi-writer.

        Args:
            writers: List of output writers to use
        """
        self._writers = writers

    def write_frame(self, frame: OutputFrame) -> None:
        """Write frame to all writers."""
        for writer in self._writers:
            writer.write_frame(frame)

    def write_metadata(self, metadata: Dict[str, Any]) -> None:
        """Write metadata to all writers."""
        for writer in self._writers:
            writer.write_metadata(metadata)

    def finalize(self) -> None:
        """Finalize all writers."""
        for writer in self._writers:
            writer.finalize()


def create_writer(
    output_dir: str,
    format: str = "csv",
    base_name: str = "solution",
) -> OutputWriter:
    """
    Factory function to create appropriate output writer.

    Args:
        output_dir: Directory for output files
        format: Output format ('csv', 'hdf5', or 'both')
        base_name: Base name for output files

    Returns:
        OutputWriter instance
    """
    if format.lower() == "csv":
        return CSVWriter(output_dir, base_name)
    elif format.lower() == "hdf5":
        return HDF5Writer(output_dir, base_name)
    elif format.lower() == "both":
        return MultiWriter(
            [
                CSVWriter(output_dir, base_name),
                HDF5Writer(output_dir, base_name),
            ]
        )
    else:
        raise ValueError(f"Unknown output format: {format}")


def load_csv_frame(filepath: str) -> OutputFrame:
    """
    Load a single CSV frame.

    Args:
        filepath: Path to CSV file

    Returns:
        OutputFrame with loaded data
    """
    # Read metadata from comments
    time = 0.0
    step = 0

    with open(filepath, "r") as f:
        for line in f:
            if line.startswith("# time"):
                time = float(line.split("=")[1])
            elif line.startswith("# step"):
                step = int(line.split("=")[1])
            elif not line.startswith("#"):
                break

    # Read data
    data = np.loadtxt(filepath, delimiter=",", comments="#", skiprows=1)

    x_cell = data[:, 0]
    rho = data[:, 1]
    u_cell = data[:, 2]
    p = data[:, 3]
    T = data[:, 4]
    e = data[:, 5]
    E = data[:, 6]
    c = data[:, 7]

    # Reconstruct face positions (approximate)
    dx = np.diff(x_cell)
    x = np.zeros(len(x_cell) + 1)
    x[0] = x_cell[0] - dx[0] / 2 if len(dx) > 0 else x_cell[0]
    x[1:-1] = 0.5 * (x_cell[:-1] + x_cell[1:])
    x[-1] = x_cell[-1] + dx[-1] / 2 if len(dx) > 0 else x_cell[-1]

    # Reconstruct face velocities (approximate)
    u = np.zeros(len(x_cell) + 1)
    u[0] = u_cell[0]
    u[1:-1] = 0.5 * (u_cell[:-1] + u_cell[1:])
    u[-1] = u_cell[-1]

    return OutputFrame(
        time=time,
        step=step,
        x=x,
        x_cell=x_cell,
        rho=rho,
        u=u,
        p=p,
        T=T,
        e=e,
        E=E,
        c=c,
    )
