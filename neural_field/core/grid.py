"""Spatial grid implementation for neural field simulations."""

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from typing import Optional

from neural_field.config.params import GridParams


@dataclass
class SpatialGrid:
    """Represents the spatial discretization of a ring domain.

    The domain spans [-L, L] degrees with periodic boundary conditions.

    Attributes:
        L: Half-domain size in degrees.
        dx: Spatial resolution in degrees.
        n_points: Number of grid points.
        x: Array of grid positions.

    Example:
        grid = SpatialGrid.from_params(GridParams(L=180.0, dx=0.1))
        print(grid.x)  # Array from -180 to ~180
    """

    L: float
    dx: float
    n_points: int
    x: NDArray[np.floating]

    @classmethod
    def from_params(cls, params: GridParams) -> "SpatialGrid":
        """Create a SpatialGrid from GridParams.

        Args:
            params: GridParams object with L and dx values.

        Returns:
            SpatialGrid instance.
        """
        n_points = params.n_points
        dx = params.adjusted_dx
        x = np.linspace(-params.L, params.L, n_points, endpoint=False)
        return cls(L=params.L, dx=dx, n_points=n_points, x=x)

    @classmethod
    def create(cls, L: float = 180.0, dx: float = 0.05) -> "SpatialGrid":
        """Create a SpatialGrid directly from L and dx.

        Args:
            L: Half-domain size in degrees.
            dx: Spatial resolution in degrees.

        Returns:
            SpatialGrid instance.
        """
        return cls.from_params(GridParams(L=L, dx=dx))

    @property
    def center_idx(self) -> int:
        """Index of the center (x=0) point."""
        return self.n_points // 2

    @property
    def half_x(self) -> NDArray[np.floating]:
        """Positive half of the domain (x >= 0)."""
        return self.x[self.center_idx:]

    def index_to_angle(self, idx: int) -> float:
        """Convert grid index to angle in degrees.

        Args:
            idx: Grid index.

        Returns:
            Angle at that index in degrees.
        """
        return self.x[idx]

    def angle_to_index(self, angle: float) -> int:
        """Find the grid index closest to a given angle.

        Args:
            angle: Angle in degrees.

        Returns:
            Index of the closest grid point.
        """
        # Wrap angle to domain
        angle = (angle + self.L) % (2 * self.L) - self.L
        # Find closest index
        return int(np.argmin(np.abs(self.x - angle)))

    def zeros(self) -> NDArray[np.floating]:
        """Create an array of zeros with the grid shape."""
        return np.zeros(self.n_points)

    def ones(self) -> NDArray[np.floating]:
        """Create an array of ones with the grid shape."""
        return np.ones(self.n_points)
