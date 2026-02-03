"""Factory function for creating simulators."""

from typing import Union

from neural_field.config.params import SimulationConfig
from neural_field.config.enums import SimulationMode, RingMode
from neural_field.core.grid import SpatialGrid
from neural_field.kernels.factory import create_lateral_kernel, create_cross_kernel
from neural_field.kernels.physics import compute_bump_physics
from neural_field.simulation.base import Simulator
from neural_field.simulation.full import FullSimulator
from neural_field.simulation.simplified import SimplifiedSimulator


def create_simulator(
    config: SimulationConfig,
    verbose: bool = False,
) -> Union[FullSimulator, SimplifiedSimulator]:
    """Create a simulator from configuration.

    This factory function:
    1. Creates the spatial grid
    2. Creates lateral and cross kernels
    3. Computes bump physics
    4. Instantiates the appropriate simulator class

    Args:
        config: SimulationConfig instance.
        verbose: Whether to print diagnostic messages.

    Returns:
        FullSimulator or SimplifiedSimulator based on config.sim_mode.

    Example:
        config = SimulationConfig(sim_mode=SimulationMode.FULL)
        simulator = create_simulator(config)
        result = simulator.run_trial(target_loc=0.0, dist_loc=40.0)
    """
    # Create grid
    grid = SpatialGrid.from_params(config.grid)

    # Create kernels
    lateral_kernel = create_lateral_kernel(config.lateral_kernel)

    cross_kernel = None
    if config.ring_mode == RingMode.DUAL:
        cross_kernel = create_cross_kernel(config.cross_kernel)

    # Compute physics
    physics = compute_bump_physics(
        grid=grid,
        lateral_kernel=lateral_kernel,
        cross_kernel=cross_kernel,
        theta=config.theta,
        ring_mode=config.ring_mode,
        stable_width=config.physics.stable_width,
        verbose=verbose,
    )

    # Create simulator
    if config.sim_mode == SimulationMode.FULL:
        return FullSimulator(
            config=config,
            grid=grid,
            lateral_kernel=lateral_kernel,
            cross_kernel=cross_kernel,
            physics=physics,
        )
    else:
        return SimplifiedSimulator(
            config=config,
            grid=grid,
            physics=physics,
        )
