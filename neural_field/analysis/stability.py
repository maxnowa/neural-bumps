"""Kernel stability analysis functions."""

from typing import Tuple, Optional, Any, Union
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from scipy.interpolate import interp1d

from neural_field.kernels.base import KernelFunction


def check_stability(
    kernels_or_x: Union[NDArray[np.floating], Any],
    theta_or_dx: Union[float, None] = None,
    w: NDArray[np.floating] = None,
    theta: float = None,
    min_width: float = 1.0,
    safety_margin: float = 1.1,
    verbose: bool = True,
    kernel_name: str = "Kernel",
    plot: bool = True,
    kernel_type: str = "lateral",
) -> Tuple[bool, Optional[float]]:
    """Check if a kernel supports stable bumps.

    Supports both new API (x, dx, w, theta) and legacy API (kernels, theta).

    This function checks the Amari existence condition:
    1. The integral of w from 0 to D must equal theta for some D
    2. The kernel value at D must be negative (stability)
    3. The peak input must exceed theta by a safety margin

    Args:
        kernels_or_x: Spatial grid positions OR legacy Kernels object.
        theta_or_dx: Spatial resolution (new API) or theta (legacy API).
        w: Kernel values at grid positions (new API only).
        theta: Activation threshold.
        min_width: Minimum bump width to consider valid.
        safety_margin: Multiplier for peak input robustness check.
        verbose: Whether to print diagnostic messages (or use legacy 'plot').
        kernel_name: Name for diagnostic output.
        plot: Legacy parameter (same as verbose).
        kernel_type: "lateral" or "cross" for legacy API.

    Returns:
        Tuple of (is_stable, stable_width) where stable_width is None if unstable.

    Example (new API):
        w_lat = kernel.compute(grid.x)
        is_stable, width = check_stability(grid.x, grid.dx, w_lat, theta=0.25)

    Example (legacy API):
        is_stable, width = check_stability(kernels, theta=0.25)
    """
    # Handle legacy API: check_stability(kernels, theta, ...)
    if hasattr(kernels_or_x, 'x') and hasattr(kernels_or_x, 'dx') and hasattr(kernels_or_x, 'w_lat'):
        kernels = kernels_or_x
        x = kernels.x
        dx = kernels.dx
        if theta is None:
            theta = theta_or_dx
        if kernel_type == "lateral":
            w = kernels.w_lat
            kernel_name = "Lateral Kernel"
        else:
            w = kernels.w_cross
            kernel_name = "Cross Kernel"
        verbose = plot  # Legacy uses 'plot' for verbose
    else:
        # New API
        x = kernels_or_x
        dx = theta_or_dx
        # w and theta passed as keyword arguments

    if theta is None:
        theta = 0.25  # Default

    # Slice positive half
    center_idx = len(x) // 2
    x_half = x[center_idx:]
    w_half = w[center_idx:]

    # Amari integral curve
    amari_curve = np.cumsum(w_half) * dx

    # Interpolator for finding roots
    amari_func = interp1d(x_half, amari_curve, kind="cubic")

    # Find sign changes (potential bump widths)
    diff = amari_curve - theta
    sign_changes = np.where(np.diff(np.sign(diff)))[0]

    for idx in sign_changes:
        x1, x2 = x_half[idx], x_half[idx + 1]
        y1, y2 = diff[idx], diff[idx + 1]

        # Linear interpolation to find root
        D_root = x1 - y1 * (x2 - x1) / (y2 - y1)

        # Filter 1: Minimum width
        if D_root < min_width:
            continue

        # Filter 2: Slope stability (w(edge) < 0)
        w_at_edge = w_half[idx] + (w_half[idx + 1] - w_half[idx]) * ((D_root - x1) / dx)
        if w_at_edge >= 0:
            continue

        # Filter 3: Peak robustness
        half_D = D_root / 2.0
        if half_D < x_half[-1]:
            center_input = 2 * amari_func(half_D)
        else:
            center_input = 0

        if center_input < (theta * safety_margin):
            continue

        # Passed all filters
        if verbose:
            print(f"--- {kernel_name} Stability ---")
            print(f"Found Stable Bump: Width = {D_root:.2f}°")

        return True, D_root

    if verbose:
        print(f"--- {kernel_name} Stability ---")
        print(f"No stable bumps found for theta={theta}.")

    return False, None


def search_stable_kernels(
    n_samples: int = 50000,
    theta: float = 0.25,
    save_to: Optional[str] = None,
    verbose: bool = True,
) -> Optional[pd.DataFrame]:
    """Search for stable triple-Gaussian kernel parameters via Monte Carlo.

    This function randomly samples kernel parameters and checks each
    for stability using the Amari existence condition.

    Args:
        n_samples: Number of random parameter sets to test.
        theta: Activation threshold.
        save_to: Path to save results CSV (optional).
        verbose: Whether to print progress.

    Returns:
        DataFrame of stable parameters, or None if none found.

    Example:
        df = search_stable_kernels(n_samples=10000)
        print(f"Found {len(df)} stable kernels")
    """
    if verbose:
        print(f"Generating {n_samples} random parameter sets...")

    # Parameter ranges
    ranges = {
        "A_ex": [0.5, 4.0],
        "s_ex": [1.0, 8.0],
        "A_inh": [0.1, 2.0],
        "s_inh": [4.0, 30.0],
        "A_att": [0.01, 1.0],
        "s_att": [30.0, 100.0],
    }

    # Create high-resolution grid for stability checking
    L = 180.0
    dx = 0.005
    n_points = int(round((2 * L) / dx))
    x_grid = np.linspace(-L, L, n_points, endpoint=False)

    # Generate random samples
    samples = np.random.uniform(0, 1, (n_samples, 6))
    keys = list(ranges.keys())
    for i, k in enumerate(keys):
        low, high = ranges[k]
        samples[:, i] = samples[:, i] * (high - low) + low

    valid_rows = []
    if verbose:
        print("Screening kernels...")

    for i, row in enumerate(samples):
        if verbose and i % 5000 == 0:
            print(f"  Processed {i}/{n_samples}...")

        A_ex, s_ex, A_inh, s_inh, A_att, s_att = row

        # Quick filters before full stability check
        # 1. Geometry: sigmas must be ordered
        if s_ex >= s_inh or s_inh >= s_att:
            continue

        # 2. Epilepsy check: net drive must be negative
        net_drive = (A_ex * s_ex) - (A_inh * s_inh) + (A_att * s_att)
        if net_drive >= 0:
            continue

        # 3. Full stability check
        w_lat = (
            A_ex * np.exp(-(x_grid**2) / (2 * s_ex**2))
            - A_inh * np.exp(-(x_grid**2) / (2 * s_inh**2))
            + A_att * np.exp(-(x_grid**2) / (2 * s_att**2))
        )

        is_stable, width = check_stability(
            x_grid, dx, w_lat, theta=theta, min_width=2.0, verbose=False
        )

        if is_stable:
            valid_rows.append(list(row) + [net_drive, width])

    # Process results
    if not valid_rows:
        if verbose:
            print("\nNO STABLE PARAMETERS FOUND.")
        return None

    columns = keys + ["net_drive", "stable_width"]
    df = pd.DataFrame(valid_rows, columns=columns)

    if verbose:
        print(f"\n--- RESULTS ---")
        print(f"Found {len(df)} stable kernels.")
        print(f"Average Stable Width: {df['stable_width'].mean():.2f}°")

        for col in keys:
            low = np.percentile(df[col], 5)
            high = np.percentile(df[col], 95)
            print(f"  {col}: {low:.2f} - {high:.2f}")

    if save_to:
        import os

        os.makedirs(os.path.dirname(save_to), exist_ok=True)
        df.to_csv(save_to, index=False)
        if verbose:
            print(f"Saved to {save_to}")

    return df
