"""Configuration module for neural field simulations."""

from neural_field.config.enums import (
    SimulationMode,
    RingMode,
    LateralKernelType,
    CrossKernelType,
)
from neural_field.config.params import (
    GridParams,
    TimeParams,
    LateralKernelParams,
    CrossKernelParams,
    InputParams,
    PhysicsParams,
    SimulationConfig,
)
from neural_field.config.loader import load_config, merge_configs

__all__ = [
    "SimulationMode",
    "RingMode",
    "LateralKernelType",
    "CrossKernelType",
    "GridParams",
    "TimeParams",
    "LateralKernelParams",
    "CrossKernelParams",
    "InputParams",
    "PhysicsParams",
    "SimulationConfig",
    "load_config",
    "merge_configs",
]
