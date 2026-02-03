"""Full PDE simulation with FFT convolution."""

from dataclasses import dataclass
from typing import Optional
import numpy as np
from numpy.typing import NDArray

from neural_field.config.params import SimulationConfig
from neural_field.config.enums import RingMode
from neural_field.core.grid import SpatialGrid
from neural_field.core.math import wrap_deg
from neural_field.core.results import TrialResult
from neural_field.kernels.base import KernelFunction
from neural_field.kernels.physics import BumpPhysics
from neural_field.simulation.tracking import BumpTracker


@dataclass
class FullSimulator:
    """Full PDE simulator using FFT convolution.

    This simulator integrates the neural field equation:
        tau * du/dt = -u + w * f(u) + I_ext

    using Euler-Maruyama method with FFT-based convolution.

    Attributes:
        config: SimulationConfig instance.
        grid: SpatialGrid instance.
        lateral_kernel: Lateral kernel function.
        cross_kernel: Cross-coupling kernel function (optional).
        physics: BumpPhysics for interface tracking.
    """

    config: SimulationConfig
    grid: SpatialGrid
    lateral_kernel: KernelFunction
    cross_kernel: Optional[KernelFunction]
    physics: BumpPhysics

    def __post_init__(self):
        """Pre-compute FFTs of kernels."""
        x = self.grid.x
        dx = self.grid.dx

        # Compute kernel values
        w_lat = self.lateral_kernel.compute(x)
        self.W_lat_fft = np.fft.fft(np.fft.fftshift(w_lat)) * dx

        if self.cross_kernel is not None:
            w_cross = self.cross_kernel.compute(x)
            self.W_cross_fft = np.fft.fft(np.fft.fftshift(w_cross)) * dx
        else:
            self.W_cross_fft = None

        # Create tracker
        self.tracker = BumpTracker(
            grid=self.grid,
            theta=self.config.theta,
            max_bumps=2,
        )

    def run_trial(
        self,
        target_loc: float = 0.0,
        dist_loc: float = 40.0,
        experiment_type: str = "simultaneous",
        with_noise: bool = True,
        **kwargs,
    ) -> TrialResult:
        """Run a full PDE simulation trial.

        Args:
            target_loc: Target position in degrees.
            dist_loc: Distractor position in degrees.
            experiment_type: "simultaneous" or "delayed".
            with_noise: Whether to include noise.
            **kwargs: Additional parameters (dist_amp, dist_dur).

        Returns:
            TrialResult with simulation outputs.
        """
        p = self.config
        x = self.grid.x
        steps = p.time.n_steps

        # Parse experiment parameters
        dist_amplitude = kwargs.get("dist_amp", p.input.amp_dist)

        if experiment_type == "simultaneous":
            dist_onset = 0
            dist_duration = 0
            init_distractor_by_ic = True
        else:
            dist_onset = 200
            dist_duration = kwargs.get("dist_dur", 500)
            init_distractor_by_ic = False

        # Initialize fields
        u1 = np.zeros_like(x) - 0.2
        u2 = np.zeros_like(x) - 0.2

        # Create input masks
        half_width = p.input.width_input / 2.0
        dist_from_target = np.abs(wrap_deg(x - target_loc))
        dist_from_dist = np.abs(wrap_deg(x - dist_loc))
        target_mask = dist_from_target < half_width
        dist_mask = dist_from_dist < half_width

        # Apply initial conditions
        u1[target_mask] += p.input.amp_target

        if init_distractor_by_ic:
            if p.ring_mode == RingMode.SINGLE:
                u1[dist_mask] += dist_amplitude
            else:
                u2[dist_mask] += dist_amplitude

        initial_u_capture = u1.copy()

        # Storage
        traj_centers = np.full((steps, 2), np.nan)
        traj_xl = np.full((steps, 2), np.nan)
        traj_xr = np.full((steps, 2), np.nan)
        prev_centers = np.array([target_loc, dist_loc])
        noise_mag = np.sqrt(p.dt) * p.eps if with_noise else 0.0

        # Time loop
        for t in range(steps):
            time = t * p.dt

            # External input (for delayed mode)
            current_input_distractor = 0.0
            if (time >= dist_onset) and (time < dist_onset + dist_duration):
                current_input_distractor = np.zeros_like(x)
                current_input_distractor[dist_mask] = dist_amplitude

            # # Firing rates (soft threshold)
            # beta = 2.0
            # f1 = self._sigmoid(u1, p.theta, beta)
            f1 = (u1 > p.theta).astype(float)
            f2 = (u2 > p.theta).astype(float)

            # Convolutions
            lat_1 = np.fft.ifft(np.fft.fft(f1) * self.W_lat_fft).real

            # Noise
            Z1 = np.random.normal(scale=noise_mag, size=p.n_points) if with_noise else 0.0

            if p.ring_mode == RingMode.SINGLE:
                drift = (-u1 + lat_1 + current_input_distractor) / p.tau
                u1 += drift * p.dt + Z1
            else:
                lat_2 = np.fft.ifft(np.fft.fft(f2) * self.W_lat_fft).real
                cross_1 = np.fft.ifft(np.fft.fft(f2) * self.W_cross_fft).real
                Z2 = np.random.normal(scale=noise_mag, size=p.n_points) if with_noise else 0.0

                drift = (-u1 + lat_1 + cross_1) / p.tau
                u1 += drift * p.dt + Z1

            # Track bumps
            result = self.tracker.track(u1, prev_centers=prev_centers)
            prev_centers = result.centers
            traj_centers[t] = result.centers
            traj_xl[t] = result.xl
            traj_xr[t] = result.xr

        # Compute bias
        last_valid_idx = np.where(~np.isnan(traj_centers[:, 0]))[0]
        bias = np.nan
        if len(last_valid_idx) > 0:
            final_pos = traj_centers[last_valid_idx[-1], 0]
            bias = wrap_deg(final_pos - target_loc)

        return TrialResult(
            bias=bias,
            trajectory=traj_centers[:, 0],
            trajectory_full=traj_centers,
            interfaces=(traj_xl, traj_xr),
            initial_u=initial_u_capture,
            final_u=u1,
            target_loc=target_loc,
            dist_loc=dist_loc,
        )

    @staticmethod
    def _sigmoid(u: NDArray, theta: float, beta: float) -> NDArray:
        """Soft threshold (sigmoid) activation function."""
        return 1.0 / (1.0 + np.exp(-beta * (u - theta)))
