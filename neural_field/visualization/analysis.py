"""Analysis visualization functions."""

from typing import List, Optional
import numpy as np
from numpy.typing import NDArray
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def plot_candidates(
    candidates: pd.DataFrame,
    distances: NDArray[np.floating],
    save_path: Optional[str] = None,
    max_lines: int = 20,
) -> Optional[Figure]:
    """Plot velocity profiles of candidate kernels.

    Args:
        candidates: DataFrame with velocity columns (v_10, v_20, etc.).
        distances: Array of distances corresponding to velocity columns.
        save_path: Path to save candidates CSV (optional).
        max_lines: Maximum number of individual curves to plot.

    Returns:
        Matplotlib figure, or None if no candidates.
    """
    if len(candidates) == 0:
        print("No kernels matched the filter criteria.")
        return None

    # Sort by velocity at 50 degrees
    if "v_50" in candidates.columns:
        candidates = candidates.sort_values("v_50", ascending=False)

    fig = plt.figure(figsize=(10, 6))

    # Extract velocity columns
    v_cols = [f"v_{d}" for d in distances if f"v_{d}" in candidates.columns]

    if not v_cols:
        print("No velocity columns found in candidates DataFrame.")
        return None

    # Plot individual curves
    for i in range(min(max_lines, len(candidates))):
        row = candidates.iloc[i]
        velocities = row[v_cols].values
        plt.plot(distances, velocities, alpha=0.5, lw=1.5)

    # Plot mean curve
    mean_velocity = candidates[v_cols].mean(axis=0).values
    plt.plot(distances, mean_velocity, "k--", lw=3, label="Mean Velocity Profile")

    plt.axhline(0, color="k", lw=1)
    if 50 in distances:
        plt.axvline(50, color="r", linestyle=":", label="Target Peak (50°)")

    plt.title(f"Velocity Profiles of Top {len(candidates)} Candidates")
    plt.xlabel("Separation Distance (deg)")
    plt.ylabel("Est. Drift Velocity (deg/s)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # Save if requested
    if save_path:
        candidates.to_csv(save_path, index=False)
        print(f"Saved filtered candidates to '{save_path}'")

    return fig


def plot_stable_params_correlations(
    df: pd.DataFrame,
    param_cols: Optional[List[str]] = None,
    max_samples: int = 1000,
) -> Figure:
    """Plot pairwise correlations of stable kernel parameters.

    Args:
        df: DataFrame with stable kernel parameters.
        param_cols: Columns to include (default: A_ex, s_ex, A_inh, s_inh, A_att, s_att).
        max_samples: Maximum samples to plot (for performance).

    Returns:
        Matplotlib figure.
    """
    try:
        import seaborn as sns
    except ImportError:
        print("seaborn required for correlation plots. Install with: pip install seaborn")
        return None

    if param_cols is None:
        param_cols = ["A_ex", "s_ex", "A_inh", "s_inh", "A_att", "s_att"]
        param_cols = [c for c in param_cols if c in df.columns]

    # Subsample for performance
    plot_df = df.sample(min(max_samples, len(df))) if len(df) > max_samples else df

    g = sns.PairGrid(plot_df[param_cols], diag_sharey=False, corner=True)
    g.map_lower(sns.kdeplot, fill=True, cmap="Blues")
    g.map_diag(sns.histplot, kde=True, color="green")
    g.fig.suptitle("Stable & Safe Parameter Space", y=1.02)
    plt.show()

    return g.fig


def plot_bias_heatmap(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    bias_col: str = "bias_50",
    n_bins: int = 20,
) -> Figure:
    """Plot heatmap of bias as function of two parameters.

    Args:
        df: DataFrame with parameters and bias values.
        x_col: Column name for x-axis parameter.
        y_col: Column name for y-axis parameter.
        bias_col: Column name for bias values.
        n_bins: Number of bins in each dimension.

    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Create 2D histogram
    x = df[x_col].values
    y = df[y_col].values
    z = df[bias_col].values

    # Bin the data
    x_bins = np.linspace(x.min(), x.max(), n_bins + 1)
    y_bins = np.linspace(y.min(), y.max(), n_bins + 1)

    # Compute mean bias in each bin
    z_binned = np.zeros((n_bins, n_bins))
    z_counts = np.zeros((n_bins, n_bins))

    for xi, yi, zi in zip(x, y, z):
        ix = min(int((xi - x.min()) / (x.max() - x.min()) * n_bins), n_bins - 1)
        iy = min(int((yi - y.min()) / (y.max() - y.min()) * n_bins), n_bins - 1)
        z_binned[iy, ix] += zi
        z_counts[iy, ix] += 1

    z_mean = np.where(z_counts > 0, z_binned / z_counts, np.nan)

    # Plot heatmap
    im = ax.imshow(
        z_mean,
        origin="lower",
        extent=[x.min(), x.max(), y.min(), y.max()],
        aspect="auto",
        cmap="RdBu_r",
    )

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(f"Mean {bias_col}")

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f"Bias Dependence on {x_col} vs {y_col}")

    plt.tight_layout()
    plt.show()
    return fig
