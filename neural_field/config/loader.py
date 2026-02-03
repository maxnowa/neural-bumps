"""YAML configuration loading utilities."""

from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

from neural_field.config.params import (
    SimulationConfig,
    GridParams,
    TimeParams,
    LateralKernelParams,
    CrossKernelParams,
    InputParams,
    PhysicsParams,
)
from neural_field.config.enums import (
    SimulationMode,
    RingMode,
    LateralKernelType,
    CrossKernelType,
)


def load_config(path: Union[str, Path]) -> SimulationConfig:
    """Load configuration from a YAML file.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        SimulationConfig object with loaded values.

    Example:
        config = load_config("config/default.yaml")
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        data = yaml.safe_load(f)

    return _dict_to_config(data)


def merge_configs(
    base: SimulationConfig, overrides: Dict[str, Any]
) -> SimulationConfig:
    """Merge override values into a base configuration.

    Args:
        base: Base SimulationConfig object.
        overrides: Dictionary of values to override.

    Returns:
        New SimulationConfig with merged values.

    Example:
        config = merge_configs(config, {"time": {"T_total": 500.0}})
    """
    # Convert base to dict
    base_dict = _config_to_dict(base)

    # Deep merge
    merged = _deep_merge(base_dict, overrides)

    # Convert back to config
    return _dict_to_config(merged)


def _dict_to_config(data: Dict[str, Any]) -> SimulationConfig:
    """Convert a dictionary to SimulationConfig."""
    # Handle top-level mode fields
    sim_mode = data.get("simulation", {}).get("mode", "full")
    ring_mode = data.get("simulation", {}).get("ring_mode", "dual")

    # Build nested params
    grid_data = data.get("grid", {})
    time_data = data.get("time", {})
    lateral_data = data.get("lateral_kernel", {})
    cross_data = data.get("cross_kernel", {})
    input_data = data.get("input", {})
    physics_data = data.get("physics", {})

    # Handle kernel type conversion
    if "type" in lateral_data:
        lateral_data["kernel_type"] = lateral_data.pop("type")
    if "type" in cross_data:
        cross_data["kernel_type"] = cross_data.pop("type")

    return SimulationConfig(
        sim_mode=SimulationMode(sim_mode),
        ring_mode=RingMode(ring_mode),
        grid=GridParams(**grid_data),
        time=TimeParams(**time_data),
        lateral_kernel=LateralKernelParams(**lateral_data),
        cross_kernel=CrossKernelParams(**cross_data),
        input=InputParams(**input_data),
        physics=PhysicsParams(**physics_data),
    )


def _config_to_dict(config: SimulationConfig) -> Dict[str, Any]:
    """Convert SimulationConfig to a dictionary."""
    return {
        "simulation": {
            "mode": str(config.sim_mode),
            "ring_mode": str(config.ring_mode),
        },
        "grid": {
            "L": config.grid.L,
            "dx": config.grid.dx,
        },
        "time": {
            "dt": config.time.dt,
            "tau": config.time.tau,
            "T_total": config.time.T_total,
            "eps": config.time.eps,
        },
        "lateral_kernel": {
            "kernel_type": str(config.lateral_kernel.kernel_type),
            "A_ex": config.lateral_kernel.A_ex,
            "sig_ex": config.lateral_kernel.sig_ex,
            "A_inh": config.lateral_kernel.A_inh,
            "sig_inh": config.lateral_kernel.sig_inh,
            "A_attr": config.lateral_kernel.A_attr,
            "sig_attr": config.lateral_kernel.sig_attr,
            "A": config.lateral_kernel.A,
            "sig_custom": config.lateral_kernel.sig_custom,
            "A_custom": config.lateral_kernel.A_custom,
            "rise_power": config.lateral_kernel.rise_power,
            "decay_power": config.lateral_kernel.decay_power,
        },
        "cross_kernel": {
            "kernel_type": str(config.cross_kernel.kernel_type),
            "J_cross": config.cross_kernel.J_cross,
            "sig_cross": config.cross_kernel.sig_cross,
            "rise_power": config.cross_kernel.rise_power,
            "decay_power": config.cross_kernel.decay_power,
        },
        "input": {
            "amp_target": config.input.amp_target,
            "amp_dist": config.input.amp_dist,
            "width_input": config.input.width_input,
            "theta": config.input.theta,
        },
        "physics": {
            "m1": config.physics.m1,
            "m2": config.physics.m2,
            "stable_width": config.physics.stable_width,
        },
    }


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """Deep merge two dictionaries."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def get_default_config_path() -> Path:
    """Get the path to the default configuration file."""
    # Walk up from this file to find config/default.yaml
    current = Path(__file__).parent
    while current.parent != current:
        config_path = current / "config" / "default.yaml"
        if config_path.exists():
            return config_path
        config_path = current.parent / "config" / "default.yaml"
        if config_path.exists():
            return config_path
        current = current.parent
    raise FileNotFoundError("Could not find default.yaml configuration file")


def load_default_config() -> SimulationConfig:
    """Load the default configuration."""
    try:
        return load_config(get_default_config_path())
    except FileNotFoundError:
        # Return a default config if file not found
        return SimulationConfig()
