"""Empirical validation utilities."""

import os
import time
from typing import Optional, TYPE_CHECKING
import numpy as np
from numpy.typing import NDArray
import pandas as pd

if TYPE_CHECKING:
    from neural_field.config.params import SimulationConfig
    from neural_field.simulation.base import Simulator


def calculate_force_at_dist(df: pd.DataFrame, x: float) -> pd.Series:
    """Calculate kernel value w(x) for all rows in a DataFrame.

    Assumes triple-Gaussian kernel with columns:
    A_ex, s_ex, A_inh, s_inh, A_att, s_att

    Args:
        df: DataFrame with kernel parameters.
        x: Distance at which to evaluate.

    Returns:
        Series of kernel values for each row.
    """
    term_ex = df["A_ex"] * np.exp(-(x**2) / (2 * df["s_ex"] ** 2))
    term_inh = df["A_inh"] * np.exp(-(x**2) / (2 * df["s_inh"] ** 2))
    term_att = df["A_att"] * np.exp(-(x**2) / (2 * df["s_att"] ** 2))
    return term_ex - term_inh + term_att


def run_empirical_validation(
    candidates_df: pd.DataFrame,
    create_config_fn,
    create_simulator_fn,
    distances: NDArray = None,
    output_path: Optional[str] = None,
    plot_results: bool = True,
    verbose: bool = True,
) -> Optional[pd.DataFrame]:
    """Run empirical validation of candidate kernels.

    For each candidate kernel, runs simulations at multiple distances
    and records the measured bias.

    Args:
        candidates_df: DataFrame with kernel parameters.
        create_config_fn: Function that takes a row and returns SimulationConfig.
        create_simulator_fn: Function that takes a config and returns a Simulator.
        distances: Array of distances to test (default: 10 to 80 in steps of 5).
        output_path: Path to save results CSV.
        plot_results: Whether to generate plots.
        verbose: Whether to print progress.

    Returns:
        DataFrame with bias curves, or None if no valid results.

    Example:
        def make_config(row):
            config = SimulationConfig()
            config.lateral_kernel.A_ex = row['A_ex']
            # ... set other params
            return config

        results = run_empirical_validation(
            candidates_df,
            create_config_fn=make_config,
            create_simulator_fn=create_simulator,
        )
    """
    if distances is None:
        distances = np.arange(10, 85, 5)

    if verbose:
        print(f"Starting empirical validation for {len(candidates_df)} kernels...")
        print(f"Testing {len(distances)} distances per kernel.")

    all_bias_curves = []
    valid_indices = []

    start_time = time.time()

    try:
        from tqdm.auto import tqdm

        iterator = tqdm(
            candidates_df.iterrows(),
            total=len(candidates_df),
            desc="Simulating Kernels",
        )
        use_tqdm = True
    except ImportError:
        iterator = candidates_df.iterrows()
        use_tqdm = False

    for idx, row in iterator:
        try:
            # Create config from row
            config = create_config_fn(row)
            sim = create_simulator_fn(config)

        except Exception as e:
            if use_tqdm:
                from tqdm.auto import tqdm

                tqdm.write(f"  [Skipping ID {idx}] Setup failed: {e}")
            elif verbose:
                print(f"  [Skipping ID {idx}] Setup failed: {e}")
            continue

        # Run bias curve
        current_curve = []
        try:
            for d in distances:
                res = sim.run_trial(
                    target_loc=0.0,
                    dist_loc=float(d),
                    experiment_type="simultaneous",
                    with_noise=False,
                )
                current_curve.append(res.bias)

            all_bias_curves.append(current_curve)
            valid_indices.append(idx)

        except Exception as e:
            if use_tqdm:
                from tqdm.auto import tqdm

                tqdm.write(f"  [Skipping ID {idx}] Simulation failed: {e}")
            elif verbose:
                print(f"  [Skipping ID {idx}] Simulation failed: {e}")
            continue

    elapsed = time.time() - start_time
    if verbose:
        print(f"Finished in {elapsed:.1f} seconds.")

    # Process results
    if not all_bias_curves:
        if verbose:
            print("No valid results generated.")
        return None

    all_bias_curves = np.array(all_bias_curves)

    # Create result DataFrame
    df_results = candidates_df.loc[valid_indices].copy()

    # Add bias columns
    for i, d in enumerate(distances):
        df_results[f"bias_{d}"] = all_bias_curves[:, i]

    # Save
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_results.to_csv(output_path, index=False)
        if verbose:
            print(f"Saved results to: {output_path}")

    # Plot
    if plot_results:
        _plot_validation_results(all_bias_curves, distances, len(df_results))

    return df_results


def _plot_validation_results(
    all_bias_curves: NDArray, distances: NDArray, n_kernels: int
):
    """Generate validation result plots."""
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    for curve in all_bias_curves:
        plt.plot(distances, curve, color="blue", alpha=0.15, lw=1)

    mean_curve = np.mean(all_bias_curves, axis=0)
    plt.plot(distances, mean_curve, "k--", lw=2.5, label="Mean Empirical Bias")

    plt.axhline(0, color="k", lw=1)
    if 50 in distances or (50 > distances.min() and 50 < distances.max()):
        plt.axvline(50, color="r", ls=":", alpha=0.6, label="50° Target")

    plt.xlabel("Separation Distance (deg)")
    plt.ylabel("Measured Bias (deg)")
    plt.title(f"Empirical Validation: {n_kernels} Kernels")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def particle_model(
    m1: float = 2.0,
    m2: float = 1.0,
    sigma: float = 20.0,
    strength: float = 1.0,
    rise_power: float = 2.0,
    decay_power: float = 1.0,
    plot: bool = True,
):
    """Run simple particle model for intuition building.

    Two particles interact via a tunable force kernel.

    Args:
        m1: Mass of particle 1 (heavy).
        m2: Mass of particle 2 (light).
        sigma: Force kernel width.
        strength: Force strength.
        rise_power: Force kernel rise power.
        decay_power: Force kernel decay power.
        plot: Whether to generate plots.

    Returns:
        Tuple of (distances, biases_1, biases_2).
    """

    def tunable_force(distance, sigma, strength, rise_power, decay_power):
        """Custom force kernel."""
        if distance < 1e-9:
            return 0.0

        x = distance / sigma

        # Peak normalization
        peak_x = (rise_power / decay_power) ** (1 / decay_power)
        peak_val = (peak_x**rise_power) * np.exp(-(peak_x**decay_power))

        raw_val = (x**rise_power) * np.exp(-(x**decay_power))
        normalized_val = raw_val / peak_val

        return -strength * normalized_val

    def run_simulation(initial_dist, m1, m2, sigma, strength, steps=1000, dt=0.01):
        x1, x2 = 0.0, initial_dist
        for _ in range(steps):
            dist = x2 - x1
            f = tunable_force(dist, sigma, strength, rise_power, decay_power)
            v1 = -f / m1
            v2 = f / m2
            x1 += v1 * dt
            x2 += v2 * dt
        return x1 - 0.0, x2 - initial_dist

    # Bias sweep
    distances = np.linspace(0, 100, 50)
    biases_1 = []
    biases_2 = []

    for d in distances:
        b1, b2 = run_simulation(d, m1, m2, sigma, strength)
        biases_1.append(b1)
        biases_2.append(b2)

    if plot:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 5))
        plt.plot(distances, biases_1, label=f"Heavy (m={m1})", color="blue", lw=3)
        plt.plot(
            distances, biases_2, label=f"Light (m={m2})", color="orange", ls="--", lw=2
        )
        plt.title("Particle Model: Bias vs. Initial Distance")
        plt.xlabel("Initial Distance (Degrees)")
        plt.ylabel("Displacement (Bias)")
        plt.axhline(0, color="black", lw=0.5)
        plt.axvline(sigma, color="gray", ls="--", label="Sigma")
        plt.grid(True)
        plt.legend()
        plt.show()

    return distances, biases_1, biases_2
