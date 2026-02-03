"""Centralized path management."""

from pathlib import Path
from typing import Optional


class Paths:
    """Centralized path management for the project.

    This class provides consistent access to project directories
    regardless of where code is executed from.

    Attributes:
        root: Project root directory.

    Example:
        paths = Paths()
        df = pd.read_csv(paths.data_dir / "stable_params.csv")
    """

    def __init__(self, root: Optional[Path] = None):
        """Initialize with project root.

        Args:
            root: Project root directory. If None, auto-detects by looking
                  for pyproject.toml or neural_field package.
        """
        if root is not None:
            self.root = Path(root)
        else:
            self.root = self._find_project_root()

    @staticmethod
    def _find_project_root() -> Path:
        """Find project root by walking up from current file."""
        current = Path(__file__).resolve().parent

        # Walk up looking for markers
        while current.parent != current:
            # Check for pyproject.toml
            if (current / "pyproject.toml").exists():
                return current
            # Check for neural_field package at this level
            if (current / "neural_field").is_dir() and (
                current / "neural_field" / "__init__.py"
            ).exists():
                return current
            # Check for data directory
            if (current / "data").is_dir():
                return current
            current = current.parent

        # Fallback to current working directory
        return Path.cwd()

    @property
    def data_dir(self) -> Path:
        """Data directory path."""
        return self.root / "data"

    @property
    def plots_dir(self) -> Path:
        """Plots directory path."""
        return self.root / "plots"

    @property
    def papers_dir(self) -> Path:
        """Papers directory path."""
        return self.root / "papers"

    @property
    def config_dir(self) -> Path:
        """Config directory path."""
        return self.root / "config"

    @property
    def notebooks_dir(self) -> Path:
        """Notebooks directory path."""
        return self.root / "notebooks"

    @property
    def code_dir(self) -> Path:
        """Legacy code directory path (for backwards compatibility)."""
        return self.root / "code"

    def ensure_dirs(self) -> None:
        """Create all standard directories if they don't exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.notebooks_dir.mkdir(parents=True, exist_ok=True)

    def data_file(self, name: str) -> Path:
        """Get path to a data file.

        Args:
            name: Filename (with extension).

        Returns:
            Full path to the data file.
        """
        return self.data_dir / name

    def plot_file(self, name: str) -> Path:
        """Get path to a plot file.

        Args:
            name: Filename (with extension).

        Returns:
            Full path to the plot file.
        """
        return self.plots_dir / name

    def config_file(self, name: str = "default.yaml") -> Path:
        """Get path to a config file.

        Args:
            name: Filename (default: default.yaml).

        Returns:
            Full path to the config file.
        """
        return self.config_dir / name


# Global singleton instance
_paths: Optional[Paths] = None


def get_paths() -> Paths:
    """Get the global Paths instance.

    Returns:
        Paths singleton.
    """
    global _paths
    if _paths is None:
        _paths = Paths()
    return _paths
