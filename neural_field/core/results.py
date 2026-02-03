"""Result dataclasses for neural field simulations."""

from dataclasses import dataclass, field
from typing import Optional, Tuple
import numpy as np
from numpy.typing import NDArray


@dataclass
class BumpTrackingResult:
    """Result of bump tracking at a single time step.

    Attributes:
        centers: Array of bump center positions (shape: [max_bumps]).
        xl: Array of left edge positions (shape: [max_bumps]).
        xr: Array of right edge positions (shape: [max_bumps]).
        n_detected: Number of bumps actually detected.
    """

    centers: NDArray[np.floating]
    xl: NDArray[np.floating]
    xr: NDArray[np.floating]
    n_detected: int


@dataclass
class TrialResult:
    """Complete result from a simulation trial.

    This dataclass holds all outputs from running a single trial,
    including the bias measurement, trajectories, and neural field states.

    Attributes:
        bias: Final displacement from target location in degrees.
        trajectory: Time series of target bump center (shape: [n_steps]).
        trajectory_full: Time series of all bump centers (shape: [n_steps, n_bumps]).
        interfaces: Tuple of (xl_trajectory, xr_trajectory) arrays.
        initial_u: Neural field state at t=0 (shape: [n_points]).
        final_u: Neural field state at t=T_total (shape: [n_points]).
        target_loc: Initial target location in degrees.
        dist_loc: Initial distractor location in degrees.

    Example:
        result = simulator.run_trial(target_loc=0.0, dist_loc=40.0)
        print(f"Bias: {result.bias:.2f} degrees")
    """

    bias: float
    trajectory: NDArray[np.floating]
    trajectory_full: NDArray[np.floating]
    interfaces: Tuple[NDArray[np.floating], NDArray[np.floating]]
    initial_u: NDArray[np.floating]
    final_u: NDArray[np.floating]
    target_loc: float = 0.0
    dist_loc: float = 0.0

    @classmethod
    def from_dict(cls, d: dict, target_loc: float = 0.0, dist_loc: float = 0.0) -> "TrialResult":
        """Create TrialResult from a dictionary (for backwards compatibility).

        Args:
            d: Dictionary with keys: bias, trajectory, trajectory_full,
               interfaces, initial_u, final_u.
            target_loc: Target location used in the trial.
            dist_loc: Distractor location used in the trial.

        Returns:
            TrialResult instance.
        """
        return cls(
            bias=d["bias"],
            trajectory=d["trajectory"],
            trajectory_full=d["trajectory_full"],
            interfaces=d["interfaces"],
            initial_u=d["initial_u"],
            final_u=d["final_u"],
            target_loc=target_loc,
            dist_loc=dist_loc,
        )

    def to_dict(self) -> dict:
        """Convert to dictionary (for backwards compatibility).

        Returns:
            Dictionary with trial data.
        """
        return {
            "bias": self.bias,
            "trajectory": self.trajectory,
            "trajectory_full": self.trajectory_full,
            "interfaces": self.interfaces,
            "initial_u": self.initial_u,
            "final_u": self.final_u,
        }

    @property
    def xl_trajectory(self) -> NDArray[np.floating]:
        """Left edge trajectory (convenience accessor)."""
        return self.interfaces[0]

    @property
    def xr_trajectory(self) -> NDArray[np.floating]:
        """Right edge trajectory (convenience accessor)."""
        return self.interfaces[1]

    @property
    def final_target_pos(self) -> float:
        """Final position of the target bump."""
        valid_idx = np.where(~np.isnan(self.trajectory))[0]
        if len(valid_idx) > 0:
            return self.trajectory[valid_idx[-1]]
        return np.nan

    @property
    def final_dist_pos(self) -> Optional[float]:
        """Final position of the distractor bump (if present)."""
        if self.trajectory_full.shape[1] > 1:
            dist_traj = self.trajectory_full[:, 1]
            valid_idx = np.where(~np.isnan(dist_traj))[0]
            if len(valid_idx) > 0:
                return dist_traj[valid_idx[-1]]
        return None
