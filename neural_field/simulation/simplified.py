"""Simplified ODE simulation on bump centroids."""

from dataclasses import dataclass
from typing import Optional
import numpy as np
from numpy.typing import NDArray

from neural_field.config.params import SimulationConfig
from neural_field.core.grid import SpatialGrid
from neural_field.core.math import wrap_deg
from neural_field.core.results import TrialResult
from neural_field.kernels.physics import BumpPhysics


@dataclass
class SimplifiedSimulator:
    """Simplified simulator using ODE on bump centroids.

    Instead of solving the full PDE, this simulator tracks only
    the centroid positions using the interaction force derived
    from the kernel.

    The dynamics are:
        dx/dt = (1/alpha) * J(delta) * sign(delta)

    where alpha is the bump stiffness and J is the interaction force.

    Attributes:
        config: SimulationConfig instance.
        grid: SpatialGrid instance.
        physics: BumpPhysics with h, alpha, and J_force.
    """

    config: SimulationConfig
    grid: SpatialGrid
    physics: BumpPhysics

    def run_trial(
        self,
        target_loc: float = 0.0,
        dist_loc: float = 40.0,
        experiment_type: str = "simultaneous",
        interaction: str = "symmetric",
        with_noise: bool = True,
        **kwargs,
    ) -> TrialResult:
        """Run a simplified ODE simulation trial.

        Args:
            target_loc: Target position in degrees.
            dist_loc: Distractor position in degrees.
            experiment_type: "simultaneous" or "delayed".
            with_noise: Whether to include noise.
            **kwargs: Additional parameters (dist_dur).

        Returns:
            TrialResult with simulation outputs.
        """
        p = self.config
        steps = p.time.n_steps
        noise_mag = np.sqrt(p.dt) * p.eps if with_noise else 0.0

        # Parse experiment parameters
        if experiment_type == "simultaneous":
            dist_onset = 0
            dist_duration = 0
            init_distractor = True
        else:
            dist_onset = kwargs.get("dist_onset", 200)
            dist_duration = kwargs.get("dist_dur", 500)
            init_distractor = False
        print(f"Running Experiment in mode: {experiment_type}")
        # Initialize positions
        pos_1 = target_loc
        pos_2 = dist_loc if init_distractor else np.nan

        traj_centers = np.full((steps, 2), np.nan)

        # Time loop
        for t in range(steps):
            time = t * p.dt

            # 1. Distractor visibility logic
            if (time >= dist_onset) and (time < dist_onset + dist_duration):
                pos_2 = dist_loc
            elif experiment_type != "simultaneous": 
                # Only clear pos_2 if not in simultaneous mode
                pos_2 = np.nan
            
            # 2. Reset drifts for this step
            drift_1, drift_2 = 0.0, 0.0

            # 3. Calculate interaction if both exist
            if not np.isnan(pos_1) and not np.isnan(pos_2):
                delta = wrap_deg(pos_1 - pos_2)
                force = self.physics.get_interaction_force(delta)
                drift_1 = (1.0 / self.physics.alpha * p.physics.m1) * force * -np.sign(delta)
                if interaction == "symmetric":
                    drift_2 = (1.0 / self.physics.alpha * p.physics.m2) * force * np.sign(delta)

            # 4. ALWAYS update positions (Moved out of the 'else')
            if not np.isnan(pos_1):
                diff = noise_mag * np.random.normal()
                pos_1 = wrap_deg(pos_1 + drift_1 * p.dt + diff)

            if not np.isnan(pos_2):
                diff = noise_mag * np.random.normal()
                pos_2 = wrap_deg(pos_2 + drift_2 * p.dt + diff)

            traj_centers[t] = [pos_1, pos_2]

        # Create synthetic interfaces (centroid +/- half-width)
        h = self.physics.h
        traj_xl = traj_centers - h
        traj_xr = traj_centers + h

        # Create placeholder field (not computed in simplified mode)
        nan_field = np.full_like(self.grid.x, np.nan)

        return TrialResult(
            bias=wrap_deg(pos_1 - target_loc),
            trajectory=traj_centers[:, 0],
            trajectory_full=traj_centers,
            interfaces=(traj_xl, traj_xr),
            initial_u=nan_field,
            final_u=nan_field,
            target_loc=target_loc,
            dist_loc=dist_loc,
        )
