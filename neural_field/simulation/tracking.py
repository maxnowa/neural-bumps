"""Bump tracking functionality."""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d

from neural_field.core.grid import SpatialGrid
from neural_field.core.math import wrap_deg, circ_dist, ring_interval_indices
from neural_field.core.results import BumpTrackingResult

# TODO rewrite and make more robust!
@dataclass
class BumpTracker:
    """Tracks bumps (activity peaks) in neural field simulations.

    The tracker identifies active regions (above threshold) and computes
    their centroids using circular mean weighting.

    Attributes:
        grid: SpatialGrid instance.
        theta: Activation threshold.
        max_bumps: Maximum number of bumps to track.
        smooth_sigma: Gaussian smoothing sigma for peak detection.
        min_width_deg: Minimum bump width to consider valid.
        min_mass: Minimum integrated mass to consider valid.
    """

    grid: SpatialGrid
    theta: float
    max_bumps: int = 2
    smooth_sigma: float = 1.0
    min_width_deg: float = 1.0
    min_mass: float = 0.05

    def track(
        self,
        u: NDArray[np.floating],
        prev_centers: Optional[NDArray[np.floating]] = None,
    ) -> BumpTrackingResult:
        """Track bumps in a neural field state.

        Args:
            u: Neural field activity array (shape: [n_points]).
            prev_centers: Previous bump centers for assignment continuity.

        Returns:
            BumpTrackingResult with centers, edges, and count.
        """
        N = len(u)
        dx = self.grid.dx
        x = self.grid.x

        # Smooth and threshold
        u_s = gaussian_filter1d(u, sigma=self.smooth_sigma, mode="wrap")
        active = (u_s > self.theta).astype(int)

        # Find connected components (transitions in/out of active)
        da = np.diff(np.r_[active, active[0]])
        left_edges = np.where(da == 1)[0]
        right_edges = np.where(da == -1)[0]

        # Extract components
        components = []
        for iL, iR in zip(left_edges, right_edges):
            idx = ring_interval_indices(iL, iR, N)

            # Mass calculation
            w = u[idx] - self.theta
            w[w < 0] = 0.0

            # Filter by width and mass
            if len(idx) * dx < self.min_width_deg:
                continue
            if np.sum(w) * dx < self.min_mass:
                continue

            # Compute circular mean center
            angles = np.deg2rad(x[idx])
            C = np.sum(w * np.cos(angles))
            S = np.sum(w * np.sin(angles))
            center = wrap_deg(np.rad2deg(np.arctan2(S, C)))

            components.append({
                "center": center,
                "mass": np.sum(w) * dx,
                "iL": iL,
                "iR": iR,
            })

        # Sort by mass and limit - we only keep max_bumps bumps
        components.sort(key=lambda c: c["mass"], reverse=True)
        components = components[: self.max_bumps]

        # Initialize output arrays
        centers = np.full(self.max_bumps, np.nan)
        xl = np.full(self.max_bumps, np.nan)
        xr = np.full(self.max_bumps, np.nan)

        # Assign bumps to slots
        if prev_centers is None:
            # First frame: assign by mass order
            for k, c in enumerate(components):
                centers[k] = c["center"]
                xl[k] = self.grid.index_to_angle(c["iL"])
                xr[k] = self.grid.index_to_angle(c["iR"])
        else:
            # Subsequent frames: match to previous positions
            centers, xl, xr = self._match_to_previous(
                components, prev_centers, centers, xl, xr
            )

        return BumpTrackingResult(
            centers=centers,
            xl=xl,
            xr=xr,
            n_detected=len(components),
        )

    def _match_to_previous(
        self,
        components: list,
        prev_centers: NDArray,
        centers: NDArray,
        xl: NDArray,
        xr: NDArray,
    ) -> Tuple[NDArray, NDArray, NDArray]:
        """Match detected components to previous bump positions."""
        used = set()

        # Match existing bumps
        for k in range(self.max_bumps):
            if np.isnan(prev_centers[k]):
                continue

            best_j, best_d = None, np.inf
            for j, c in enumerate(components):
                if j in used:
                    continue
                d = circ_dist(c["center"], prev_centers[k])
                if d < 90.0 and d < best_d:
                    best_d, best_j = d, j

            if best_j is not None:
                used.add(best_j)
                centers[k] = components[best_j]["center"]
                xl[k] = self.grid.index_to_angle(components[best_j]["iL"])
                xr[k] = self.grid.index_to_angle(components[best_j]["iR"])

        # Assign remaining components to free slots
        free = np.where(np.isnan(centers))[0]
        unused = [j for j in range(len(components)) if j not in used]

        for k, j in zip(free, unused):
            centers[k] = components[j]["center"]
            xl[k] = self.grid.index_to_angle(components[j]["iL"])
            xr[k] = self.grid.index_to_angle(components[j]["iR"])

        return centers, xl, xr
