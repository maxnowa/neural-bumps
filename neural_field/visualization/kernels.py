"""Kernel visualization functions."""

from typing import Optional, Tuple
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from neural_field.core.grid import SpatialGrid
from neural_field.kernels.base import KernelFunction
from neural_field.kernels.physics import BumpPhysics
from neural_field.config.enums import RingMode


def plot_kernel(
    grid: SpatialGrid,
    lateral_kernel: KernelFunction,
    cross_kernel: Optional[KernelFunction] = None,
    physics: Optional[BumpPhysics] = None,
    theta: float = 0.25,
    ring_mode: RingMode = RingMode.SINGLE,
    show_integral: bool = True,
    ax: Optional[Axes] = None,
) -> Axes:
    """Plot kernel profiles with optional integral overlay.

    Args:
        grid: SpatialGrid instance.
        lateral_kernel: Lateral kernel instance.
        cross_kernel: Cross kernel instance (optional).
        physics: BumpPhysics for marking width (optional).
        theta: Activation threshold.
        ring_mode: Ring mode (affects cross kernel display).
        show_integral: Whether to show the integral curve.
        ax: Matplotlib axes to plot on (creates new if None).

    Returns:
        Matplotlib axes with the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    x = grid.x
    dx = grid.dx

    # Compute kernel values
    w_lat = lateral_kernel.compute(x)

    # Plot lateral kernel
    (line1,) = ax.plot(
        x, w_lat, label=r"Kernel $w_{\mathrm{lat}}$", color="blue", linewidth=2
    )

    # Plot cross kernel if dual mode
    if ring_mode != RingMode.SINGLE and cross_kernel is not None:
        w_cross = cross_kernel.compute(x)
        ax.plot(
            x,
            w_cross,
            label=r"Kernel $w_{\mathrm{cross}}$",
            color="orange",
            linestyle="--",
        )

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Distance ($^\\circ$)")
    ax.set_ylabel("Interaction")
    ax.set_title("Kernel Profile")
    ax.grid(True, alpha=0.3)

    # Show integral and bump width markers
    if show_integral and physics is not None and physics.h > 0:
        ax2 = ax.twinx()
        center_idx = grid.center_idx
        half_x = grid.half_x
        W_int = np.cumsum(w_lat[center_idx:]) * dx

        ax2.plot(half_x, W_int, color="green", linestyle=":", label="Integral (Lat)")
        ax2.axhline(theta, color="red", linestyle=":", label="Theta")

        # Mark full width on integral plot
        ax2.scatter([physics.width_2h], [theta], color="red", zorder=5)

        # Mark half width on kernel plot
        ax.axvline(
            physics.h, color="k", linestyle="-.", alpha=0.5, label="Bump Edge $h$"
        )
        ax.axvline(-physics.h, color="k", linestyle="-.", alpha=0.5)

        ax2.set_ylabel("Integrated Input")

    ax.legend(loc="upper right")
    return ax


def plot_potential_and_force(
    x: NDArray[np.floating],
    force: NDArray[np.floating],
    potential: NDArray[np.floating],
    timescale: float = 2.5,
    save_path: Optional[str] = None,
) -> Tuple[Figure, Tuple[Axes, Axes]]:
    """Plot drift velocity and potential landscape side-by-side.

    Args:
        x: Spatial grid positions (separation distances).
        force: Force array (same shape as x).
        potential: Potential energy array (same shape as x).
        timescale: Time constant for converting force to velocity.
        save_path: Path to save figure (optional).

    Returns:
        Tuple of (figure, (ax1, ax2)).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    # Convert force to velocity
    velocity = force / timescale

    # Panel 1: Velocity / Force
    ax1.plot(x, velocity, color="#0072B2", linewidth=2.5)
    ax1.set_title("Drift Velocity", fontsize=14, fontweight="bold", pad=10)
    ax1.set_xlabel("Separation Distance ($^\\circ$)", fontsize=12)
    ax1.set_ylabel("Velocity ($^\\circ$/s)", fontsize=12)
    ax1.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.set_xlim(left=0)

    # Panel 2: Potential Landscape
    ax2.plot(x, potential, color="#D55E00", linewidth=2.5, label="Analytical Potential")
    ax2.set_title("Potential Landscape $U(x)$", fontsize=14, fontweight="bold", pad=10)
    ax2.set_xlabel("Separation Distance ($^\\circ$)", fontsize=12)
    ax2.set_ylabel("Potential Energy (a.u.)", fontsize=12)
    ax2.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(frameon=False, fontsize=11)
    ax2.set_xlim(left=0)

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Figure saved to {save_path}")

    plt.show()
    return fig, (ax1, ax2)
