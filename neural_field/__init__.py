"""
Neural Field Simulation Framework

A framework for studying attentional interactions in working memory through
neural field models on ring topology.

Example usage:
    from neural_field import SimulationConfig, create_simulator
    from neural_field.config import SimulationMode, RingMode, LateralKernelType
    from neural_field.visualization import plot_results, plot_space_time

    config = SimulationConfig()
    simulator = create_simulator(config)
    results = simulator.run_trial(target_loc=0.0, dist_loc=40.0)
"""

from neural_field.config import (
    SimulationConfig,
    GridParams,
    TimeParams,
    LateralKernelParams,
    CrossKernelParams,
    InputParams,
    PhysicsParams,
    SimulationMode,
    RingMode,
    LateralKernelType,
    CrossKernelType,
)
from neural_field.core import SpatialGrid, TrialResult, BumpTrackingResult
from neural_field.kernels import (
    create_lateral_kernel,
    create_cross_kernel,
    BumpPhysics,
)
from neural_field.simulation import create_simulator

__version__ = "0.1.0"

__all__ = [
    # Config
    "SimulationConfig",
    "GridParams",
    "TimeParams",
    "LateralKernelParams",
    "CrossKernelParams",
    "InputParams",
    "PhysicsParams",
    "SimulationMode",
    "RingMode",
    "LateralKernelType",
    "CrossKernelType",
    # Core
    "SpatialGrid",
    "TrialResult",
    "BumpTrackingResult",
    # Kernels
    "create_lateral_kernel",
    "create_cross_kernel",
    "BumpPhysics",
    # Simulation
    "create_simulator",
]
