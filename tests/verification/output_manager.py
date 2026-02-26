"""
Test Output Manager for Verification Tests

Provides a standardized way to save test outputs with:
- Timestamped subdirectories (YYYYMMDD_HHMMSS_description)
- Configuration files (config.json)
- Results files (results.json)
- Automatic organization by test type (toro, piston_shock, etc.)

This enables comparison of simulation results across different configurations
and code versions.
"""

import json
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np


@dataclass
class TestConfig:
    """
    Configuration for a verification test run.

    Attributes:
        test_type: Type of test (e.g., "toro", "piston_shock")
        n_cells: Number of grid cells
        cfl: CFL number
        gamma: Ratio of specific heats (for ideal gas tests)
        description: Human-readable description of this run
        dt_min: Minimum time step floor (optional)
        av_enabled: Whether artificial viscosity is enabled
        av_c_linear: AV linear coefficient (if enabled)
        av_c_quad: AV quadratic coefficient (if enabled)
        use_compatible_energy: Whether compatible energy discretization is used
        extra: Additional configuration parameters
    """
    test_type: str
    n_cells: int
    cfl: float
    gamma: float = 1.4
    description: str = ""
    dt_min: Optional[float] = None
    av_enabled: bool = False
    av_c_linear: float = 0.0
    av_c_quad: float = 0.0
    use_compatible_energy: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)

    # Filled in automatically
    timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        # Remove extra if empty
        if not d["extra"]:
            del d["extra"]
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TestConfig":
        """Create from dictionary."""
        return cls(**d)


@dataclass
class TestResult:
    """
    Results from a single test case.

    Attributes:
        test_id: Test identifier (e.g., 1, 2, 3 for Toro tests)
        steps: Number of time steps taken
        wall_time: Wall clock time in seconds
        final_time: Simulation time reached
        failed: Whether the test failed
        error_message: Error message if failed
        metrics: Additional metrics (e.g., L2 error, max error)
    """
    test_id: Any  # Can be int, str, etc.
    steps: int
    wall_time: float
    final_time: float
    failed: bool = False
    error_message: str = ""
    metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        d = {
            "steps": self.steps,
            "wall_time": self.wall_time,
            "final_time": self.final_time,
            "failed": self.failed,
        }
        if self.error_message:
            d["error_message"] = self.error_message
        if self.metrics:
            d["metrics"] = self.metrics
        return d


class OutputManager:
    """
    Manages test output directories and files.

    Creates timestamped output directories with:
    - config.json: Test configuration
    - results.json: Test results (updated as tests complete)
    - Individual test plots/data files

    Directory structure:
        output/
            {test_type}/
                {timestamp}_{description}/
                    config.json
                    results.json
                    plot_1.png
                    plot_2.png
                    ...
    """

    def __init__(
        self,
        base_dir: Path,
        config: TestConfig,
    ):
        """
        Initialize output manager.

        Args:
            base_dir: Base output directory (e.g., tests/verification/output)
            config: Test configuration
        """
        self.base_dir = Path(base_dir)
        self.config = config

        # Generate timestamp
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.config.timestamp = self.timestamp

        # Create output directory name
        self.dir_name = self._create_dir_name()
        self.output_dir = self.base_dir / config.test_type / self.dir_name

        # Results storage
        self.results: Dict[Any, TestResult] = {}

        # Create directory and save initial config
        self._initialize()

    def _create_dir_name(self) -> str:
        """Create directory name from timestamp and description."""
        # Sanitize description for use in directory name
        desc = self.config.description.lower()
        desc = desc.replace(" ", "_")
        desc = "".join(c for c in desc if c.isalnum() or c == "_")

        # Truncate if too long
        if len(desc) > 50:
            desc = desc[:50]

        return f"{self.timestamp}_{desc}" if desc else self.timestamp

    def _initialize(self):
        """Create output directory and save config."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._save_config()

    def _save_config(self):
        """Save configuration to config.json."""
        config_path = self.output_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

    def _save_results(self):
        """Save current results to results.json."""
        results_path = self.output_dir / "results.json"

        # Convert results to serializable format
        results_dict = {}
        for test_id, result in self.results.items():
            # Convert test_id to string for JSON keys
            key = str(test_id)
            results_dict[key] = result.to_dict()

        with open(results_path, "w") as f:
            json.dump(results_dict, f, indent=2)

    def add_result(self, result: TestResult):
        """
        Add a test result and update results.json.

        Args:
            result: Test result to add
        """
        self.results[result.test_id] = result
        self._save_results()

    def get_plot_path(self, filename: str) -> Path:
        """
        Get full path for a plot file.

        Args:
            filename: Plot filename (e.g., "toro_test_1.png")

        Returns:
            Full path for the plot file
        """
        return self.output_dir / filename

    def save_figure(self, fig, filename: str, **kwargs):
        """
        Save a matplotlib figure to the output directory.

        Args:
            fig: Matplotlib figure
            filename: Output filename
            **kwargs: Additional arguments for savefig (e.g., dpi, bbox_inches)
        """
        defaults = {"dpi": 150, "bbox_inches": "tight"}
        defaults.update(kwargs)

        path = self.get_plot_path(filename)
        fig.savefig(path, **defaults)
        print(f"  Saved: {filename}")

    def save_data(self, data: Dict[str, Any], filename: str):
        """
        Save data dictionary to a JSON file.

        Args:
            data: Data to save
            filename: Output filename (should end in .json)
        """
        path = self.output_dir / filename

        # Convert numpy arrays to lists for JSON serialization
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            return obj

        serializable = {k: convert(v) for k, v in data.items()}

        with open(path, "w") as f:
            json.dump(serializable, f, indent=2)

    def print_summary(self):
        """Print summary of test results."""
        print(f"\nResults saved to: {self.output_dir}")
        print(f"Configuration: {self.config.description}")

        n_passed = sum(1 for r in self.results.values() if not r.failed)
        n_total = len(self.results)
        print(f"Tests: {n_passed}/{n_total} passed")

        for test_id, result in sorted(self.results.items(), key=lambda x: str(x[0])):
            status = "FAILED" if result.failed else "PASSED"
            print(f"  Test {test_id}: {status} ({result.steps} steps, {result.wall_time:.2f}s)")


def create_output_manager(
    test_type: str,
    n_cells: int,
    cfl: float,
    description: str = "",
    gamma: float = 1.4,
    dt_min: float = None,
    av_enabled: bool = False,
    av_c_linear: float = 0.0,
    av_c_quad: float = 0.0,
    use_compatible_energy: bool = False,
    base_dir: Path = None,
    **extra
) -> OutputManager:
    """
    Create an OutputManager with the given configuration.

    Args:
        test_type: Type of test (e.g., "toro", "piston_shock")
        n_cells: Number of grid cells
        cfl: CFL number
        description: Human-readable description
        gamma: Ratio of specific heats
        dt_min: Minimum time step floor
        av_enabled: Whether artificial viscosity is enabled
        av_c_linear: AV linear coefficient
        av_c_quad: AV quadratic coefficient
        use_compatible_energy: Whether compatible energy is used
        base_dir: Base output directory (default: tests/verification/output)
        **extra: Additional configuration parameters

    Returns:
        Configured OutputManager
    """
    if base_dir is None:
        base_dir = Path(__file__).parent / "output"

    config = TestConfig(
        test_type=test_type,
        n_cells=n_cells,
        cfl=cfl,
        gamma=gamma,
        description=description,
        dt_min=dt_min,
        av_enabled=av_enabled,
        av_c_linear=av_c_linear,
        av_c_quad=av_c_quad,
        use_compatible_energy=use_compatible_energy,
        extra=extra if extra else {},
    )

    return OutputManager(base_dir, config)
