"""Simulation result visualization functions."""

from typing import Optional, Tuple, Union, Any
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from neural_field.core.grid import SpatialGrid
from neural_field.core.results import TrialResult
from neural_field.core.math import wrap_deg


def _extract_result_data(result: Union[TrialResult, dict]) -> Tuple:
    """Extract data from either TrialResult or legacy dict format."""
    if isinstance(result, dict):
        return (
            result.get("bias", np.nan),
            result.get("trajectory", np.array([])),
            result.get("trajectory_full", np.array([[]])),
            result.get("interfaces", (np.array([[]]), np.array([[]]))),
            result.get("initial_u", np.array([])),
            result.get("final_u", np.array([])),
        )
    return (
        result.bias,
        result.trajectory,
        result.trajectory_full,
        result.interfaces,
        result.initial_u,
        result.final_u,
    )


def plot_bump_evolution(
    sim_or_grid: Union[SpatialGrid, Any],
    result: Union[TrialResult, dict],
    dt_or_title: Union[float, str] = None,
    title: str = "Bump Evolution",
) -> Figure:
    """Visualize bump trajectory and neural field profiles.

    Supports both new API (SpatialGrid, TrialResult) and legacy API (Simulation, dict).

    Args:
        sim_or_grid: SpatialGrid instance or legacy Simulation object.
        result: TrialResult from simulation or legacy dict.
        dt_or_title: Time step (float) or title (str) for legacy compatibility.
        title: Plot title.

    Returns:
        Matplotlib figure.
    """
    # Handle legacy API (sim object has .x and .p.dt attributes)
    if hasattr(sim_or_grid, 'p') and hasattr(sim_or_grid, 'x'):
        x = sim_or_grid.x
        dt = sim_or_grid.p.dt
        if isinstance(dt_or_title, str):
            title = dt_or_title
    else:
        grid = sim_or_grid
        x = grid.x
        dt = dt_or_title if isinstance(dt_or_title, (int, float)) else 0.01

    # Extract result data (supports both TrialResult and dict)
    bias, trajectory, trajectory_full, interfaces, initial_u, final_u = _extract_result_data(result)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    # Plot 1: Trajectory
    time = np.arange(len(trajectory)) * dt
    ax[0].plot(time, trajectory, "k-", lw=2, label="Target Bump")

    if trajectory_full.shape[1] > 1:
        ax[0].plot(
            time,
            trajectory_full[:, 1],
            "r--",
            alpha=0.5,
            label="Distractor",
        )

    ax[0].set_title(f"{title}\nDrift: {bias:.2f}°")
    ax[0].set_xlabel("Time (ms)")
    ax[0].set_ylabel("Centroid Position (deg)")
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)

    # Plot 2: Spatial profile
    u_start = np.zeros_like(x) - 1.0
    dist_vec = np.abs(wrap_deg(x - 0.0))
    u_start += np.exp(-(dist_vec**2) / (2 * 5.0**2))

    ax[1].plot(x, u_start, color="gray", linestyle="--", alpha=0.6, label="Start (t=0)")

    if not np.all(np.isnan(final_u)):
        ax[1].plot(x, final_u, color="blue", lw=2, label="End (t=T)")

    ax[1].set_title("Neural Activity Profile u(x)")
    ax[1].set_xlabel("Position (deg)")
    ax[1].set_ylabel("Activity u")
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    return fig


def plot_results(
    sim_or_grid: Union[SpatialGrid, Any],
    result: Union[TrialResult, dict],
    target_loc: float = 0.0,
    dist_loc: float = 40.0,
    condition: Optional[str] = None,
    dt: float = None,
    theta: float = None,
) -> Figure:
    """Plot simulation results with initial/final states and trajectories.

    Supports both new API (SpatialGrid, TrialResult) and legacy API (Simulation, dict).

    Args:
        sim_or_grid: SpatialGrid instance or legacy Simulation object.
        result: TrialResult from simulation or legacy dict.
        target_loc: Target location for reference lines.
        dist_loc: Distractor location for reference lines.
        condition: Optional label for experiment condition.
        dt: Time step (for converting to real time). Auto-detected for legacy objects.
        theta: Activation threshold. Auto-detected for legacy objects.

    Returns:
        Matplotlib figure.
    """
    # Handle legacy API
    if hasattr(sim_or_grid, 'p') and hasattr(sim_or_grid, 'x'):
        x = sim_or_grid.x
        if dt is None:
            dt = sim_or_grid.p.dt
        if theta is None:
            theta = sim_or_grid.p.theta
    else:
        grid = sim_or_grid
        x = grid.x
        if dt is None:
            dt = 0.01
        if theta is None:
            theta = 0.25

    # Extract result data
    bias, trajectory, trajectory_full, interfaces, initial_u, final_u = _extract_result_data(result)
    u_start = initial_u
    centers = trajectory_full

    # Setup figure
    cond_label = f" ({condition})" if condition else ""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), gridspec_kw=dict(wspace=0.3))

    # Colors
    c_init, c_final, c_thr = "#4D4D4D", "#2E2E2E", "#E7298A"
    c_bump = ["#1B9E77", "#D95F02"]

    # Calculate y-limits
    all_u = np.concatenate([u_start, final_u])
    valid_u = all_u[~np.isnan(all_u)]
    if len(valid_u) > 0:
        y_min, y_max = np.min(valid_u), np.max(valid_u)
        y_range = y_max - y_min
        if y_range < 0.1:
            y_range = 1.0
        pad = y_range * 0.1
        ylim_u = (y_min - pad, y_max + pad)
    else:
        ylim_u = (-1, 1)

    # Panel 1: Initial state
    axes[0].plot(x, u_start, lw=2.5, color=c_init, label="u(t=0)")
    axes[0].axhline(theta, ls="--", lw=1.5, color=c_thr, label="Theta")
    axes[0].set_title(f"Initial State{cond_label}", fontweight="bold")
    axes[0].set_ylabel("Activity u")
    axes[0].set_ylim(ylim_u)
    axes[0].set_xlim(-30, 30)

    # Panel 2: Final state
    axes[1].plot(x, final_u, lw=2.5, color=c_final, label="u(t=end)")
    axes[1].axhline(theta, ls="--", lw=1.5, color=c_thr)
    axes[1].set_title(f"Final State{cond_label}", fontweight="bold")
    axes[1].set_ylim(ylim_u)
    axes[1].set_xlim(-30, 30)

    # Panel 3: Trajectories
    t = np.arange(len(centers)) * dt / 1000.0

    axes[2].plot(t, centers[:, 0], lw=2.5, color=c_bump[0], label="Target")

    if centers.shape[1] > 1 and not np.all(np.isnan(centers[:, 1])):
        axes[2].plot(
            t, centers[:, 1], lw=2.5, color=c_bump[1], label="Distractor", alpha=0.8
        )

    axes[2].axhline(target_loc, color=c_bump[0], ls=":", alpha=0.4)
    axes[2].axhline(dist_loc, color=c_bump[1], ls=":", alpha=0.4)
    axes[2].set_title("Drift Trajectories", fontweight="bold")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Position (deg)")
    axes[2].legend(frameon=False, loc="best")

    # Final styling
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, alpha=0.15)
        ax.set_xlabel("Position (deg)" if ax != axes[2] else "Time (s)")

    plt.tight_layout()
    plt.show()


def plot_space_time(
    sim_or_result: Union[TrialResult, dict, Any],
    result_or_xlim: Union[TrialResult, dict, Tuple[float, float], None] = None,
    x_lim: Optional[Tuple[float, float]] = None,
    y_lim: Optional[Tuple[float, float]] = None,
    title: str = "Bump Evolution (Space-Time)",
) -> Figure:
    """Plot space-time diagram of bump evolution.

    Supports both new API (TrialResult, dt) and legacy API (Simulation, dict).

    Time is on the Y-axis, position on X-axis.
    Bumps are shown as white regions on black background.

    Args:
        sim_or_result: TrialResult, dict result, or legacy Simulation object.
        result_or_xlim: Result dict (legacy) or x_lim tuple (new API).
        x_lim: Optional x-axis limits (position).
        y_lim: Optional y-axis limits (time).
        title: Plot title.

    Returns:
        Matplotlib figure.
    """
    # Handle legacy API: plot_space_time(sim, result)
    if hasattr(sim_or_result, 'p') and hasattr(sim_or_result, 'x'):
        sim = sim_or_result
        result = result_or_xlim
        dt = sim.p.dt
        bias, trajectory, trajectory_full, interfaces, initial_u, final_u = _extract_result_data(result)
    # Handle new API: plot_space_time(result, dt, x_lim, y_lim)
    elif isinstance(sim_or_result, (TrialResult, dict)):
        result = sim_or_result
        dt = result_or_xlim if isinstance(result_or_xlim, (int, float)) else 0.01
        if isinstance(result_or_xlim, tuple):
            x_lim = result_or_xlim
        bias, trajectory, trajectory_full, interfaces, initial_u, final_u = _extract_result_data(result)
    else:
        raise TypeError(f"Unexpected type for first argument: {type(sim_or_result)}")

    xl_traj, xr_traj = interfaces
    centers = trajectory_full

    # Time vector (seconds)
    t = np.arange(len(centers)) * dt / 1000.0

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.set_facecolor("black")

    # Colors
    colors_edge = ["#4DA6FF", "#FF6B6B"]
    colors_center = ["#0055A4", "#A40000"]

    n_bumps = centers.shape[1]

    for k in range(n_bumps):
        xl = xl_traj[:, k]
        xr = xr_traj[:, k]
        c = centers[:, k]

        valid = ~np.isnan(xl) & ~np.isnan(xr)

        if np.any(valid):
            t_valid = t[valid]
            xl_valid = xl[valid]
            xr_valid = xr[valid]
            c_valid = c[valid]

            # Fill bump body
            ax.fill_betweenx(
                t_valid,
                xl_valid,
                xr_valid,
                color="white",
                alpha=1.0,
                edgecolor="none",
                zorder=1,
            )

            # Plot edges
            ax.plot(
                xl_valid,
                t_valid,
                color=colors_edge[k % 2],
                lw=2,
                label=f"Bump {k+1} Edge",
                zorder=2,
            )
            ax.plot(xr_valid, t_valid, color=colors_edge[k % 2], lw=2, zorder=2)

            # Plot centroid
            ax.plot(
                c_valid,
                t_valid,
                color=colors_center[k % 2],
                lw=1.5,
                ls="--",
                alpha=0.8,
                zorder=3,
            )

    ax.set_xlabel("Position $x$ (deg)", fontsize=12)
    ax.set_ylabel("Time $t$ (s)", fontsize=12)
    ax.set_title(title, fontsize=13)

    # Set limits
    if x_lim is not None:
        ax.set_xlim(x_lim)
    else:
        valid_coords = centers[~np.isnan(centers)]
        if len(valid_coords) > 0:
            limit = np.max(np.abs(valid_coords)) + 10.0
            ax.set_xlim(-limit, limit)
        else:
            ax.set_xlim(-180, 180)

    if y_lim is not None:
        ax.set_ylim(y_lim)
    else:
        ax.set_ylim(0, t[-1])

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.show()
