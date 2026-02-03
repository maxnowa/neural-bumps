"""Bump physics calculations from kernel properties."""

from dataclasses import dataclass
from typing import Callable, Optional
import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import interp1d
from scipy.optimize import brentq

from neural_field.config.params import SimulationConfig, PhysicsParams
from neural_field.config.enums import RingMode
from neural_field.core.grid import SpatialGrid
from neural_field.kernels.base import KernelFunction


@dataclass
class BumpPhysics:
    """Computed bump physics properties.

    These properties are derived from the kernel shape and threshold,
    and are used in the simplified (ODE) model.

    Attributes:
        h: Bump half-width in degrees.
        width_2h: Bump full width (2h) in degrees.
        alpha: Bump stiffness (|w(0) - w(2h)|).
        get_interaction_force: Callable that computes force as function of separation.
    """

    h: float
    width_2h: float
    alpha: float
    get_interaction_force: Callable[[float], float]


def compute_bump_physics(
    grid: SpatialGrid,
    lateral_kernel: KernelFunction,
    cross_kernel: Optional[KernelFunction],
    theta: float,
    ring_mode: RingMode,
    stable_width: Optional[float] = None,
    verbose: bool = False,
) -> BumpPhysics:
    """Compute bump physics from kernel properties.

    This function calculates:
    1. h (half-width): From the Amari existence condition
    2. alpha (stiffness): From the kernel values at bump edges
    3. J_force: Interaction force function from kernel integral

    Args:
        grid: SpatialGrid instance.
        lateral_kernel: Lateral kernel instance.
        cross_kernel: Cross-coupling kernel instance (can be None for single ring).
        theta: Activation threshold.
        ring_mode: Ring mode (single or dual).
        stable_width: Pre-calculated stable width (if known).
        verbose: Whether to print diagnostic messages.

    Returns:
        BumpPhysics dataclass with computed properties.
    """
    x = grid.x
    dx = grid.dx
    center_idx = grid.center_idx
    half_x = grid.half_x

    # Compute kernel values
    w_lat = lateral_kernel.compute(x)
    half_w_lat = w_lat[center_idx:]

    # Interpolators for lateral kernel
    w_lat_func = interp1d(half_x, half_w_lat, kind="linear", fill_value="extrapolate")
    W_lat_curve = np.cumsum(half_w_lat) * dx
    W_lat_func = interp1d(half_x, W_lat_curve, kind="cubic", fill_value="extrapolate")

    # --- Determine Bump Width (h) ---
    if stable_width is not None and stable_width > 0:
        if verbose:
            print(f"Using pre-calculated stable width: {stable_width:.2f}")
        width_2h = stable_width
        h = width_2h / 2.0
    else:
        if verbose:
            print("Solving for stable width...")
        width_2h, h = _solve_stable_width(
            half_x, W_lat_func, w_lat_func, theta, verbose
        )

    # --- Calculate Alpha (Stiffness) ---
    w_0 = w_lat_func(0.0)
    w_edge = w_lat_func(width_2h)
    alpha = np.abs(w_0 - w_edge)

    # --- Define Interaction Force J(delta) ---
    # Choose interaction kernel based on ring mode
    if ring_mode == RingMode.SINGLE or cross_kernel is None:
        target_w = w_lat
    else:
        target_w = cross_kernel.compute(x)

    half_w_inter = target_w[center_idx:]
    W_inter_curve = np.cumsum(half_w_inter) * dx
    W_inter_func = interp1d(
        half_x, W_inter_curve, kind="cubic", fill_value="extrapolate"
    )

    def J_force(delta_deg: float) -> float:
        """Calculate interaction force between two bumps.

        The force is computed from the convolution of two pulse functions
        with the interaction kernel.

        Args:
            delta_deg: Separation distance in degrees.

        Returns:
            Interaction force (positive = attractive).
        """
        d = np.abs(delta_deg)
        term1 = 2 * W_inter_func(d)
        term2 = W_inter_func(np.abs(d - 2 * h))
        term3 = W_inter_func(d + 2 * h)
        return 0.5 * (term1 - term2 - term3)

    return BumpPhysics(
        h=h,
        width_2h=width_2h,
        alpha=alpha,
        get_interaction_force=J_force,
    )


def _solve_stable_width(
    half_x: NDArray,
    W_lat_func: Callable,
    w_lat_func: Callable,
    theta: float,
    verbose: bool = False,
) -> tuple[float, float]:
    """Solve for stable bump width using Amari existence condition.

    Returns:
        Tuple of (full_width, half_width).
    """
    err_func = lambda d: W_lat_func(d) - theta

    test_d = np.linspace(0.5, 90.0, 500)
    test_vals = W_lat_func(test_d) - theta
    sign_flips = np.where(np.diff(np.sign(test_vals)))[0]

    for idx in sign_flips:
        d_low, d_high = test_d[idx], test_d[idx + 1]
        try:
            full_width = brentq(err_func, d_low, d_high)

            # Stability check: w(edge) must be negative
            if w_lat_func(full_width) < 0:
                return full_width, full_width / 2.0
        except ValueError:
            continue

    if verbose:
        print("Warning: No stable bump width found. Using defaults.")
    return 20.0, 10.0


def compute_bump_physics_from_config(
    config: SimulationConfig,
    lateral_kernel: KernelFunction,
    cross_kernel: Optional[KernelFunction] = None,
    verbose: bool = False,
) -> BumpPhysics:
    """Convenience function to compute bump physics from a SimulationConfig.

    Args:
        config: SimulationConfig instance.
        lateral_kernel: Lateral kernel instance.
        cross_kernel: Cross-coupling kernel (optional).
        verbose: Whether to print diagnostic messages.

    Returns:
        BumpPhysics dataclass.
    """
    from neural_field.config.params import GridParams

    grid = SpatialGrid.from_params(config.grid)

    return compute_bump_physics(
        grid=grid,
        lateral_kernel=lateral_kernel,
        cross_kernel=cross_kernel,
        theta=config.theta,
        ring_mode=config.ring_mode,
        stable_width=config.physics.stable_width,
        verbose=verbose,
    )
