"""Typed dataclass configuration for neural field simulations."""

from dataclasses import dataclass, field
from typing import Optional

from neural_field.config.enums import (
    SimulationMode,
    RingMode,
    LateralKernelType,
    CrossKernelType,
)


@dataclass
class GridParams:
    """Spatial grid parameters."""

    L: float = 180.0
    """Half-domain size in degrees. Full domain is [-L, L]."""

    dx: float = 0.05
    """Spatial resolution in degrees."""

    @property
    def n_points(self) -> int:
        """Number of grid points (computed from L and dx)."""
        return int(round((2 * self.L) / self.dx))

    @property
    def adjusted_dx(self) -> float:
        """Adjusted dx to exactly fit the domain."""
        return (2 * self.L) / self.n_points


@dataclass
class TimeParams:
    """Time integration parameters."""

    dt: float = 0.01
    """Time step in ms."""

    tau: float = 1.0
    """Time constant in ms."""

    T_total: float = 1000.0
    """Total simulation time in ms."""

    eps: float = 0.05
    """Noise strength (epsilon)."""

    @property
    def n_steps(self) -> int:
        """Number of time steps."""
        return int(self.T_total / self.dt)


@dataclass
class LateralKernelParams:
    """Parameters for lateral (within-ring) kernels."""

    kernel_type: LateralKernelType = LateralKernelType.MEXICAN_HAT
    """Type of lateral kernel to use."""

    # Triple Gaussian / Mexican Hat parameters
    A_ex: float = 2.2
    """Excitatory amplitude."""

    sig_ex: float = 4.0
    """Excitatory width (sigma)."""

    A_inh: float = 1.4
    """Inhibitory amplitude."""

    sig_inh: float = 19.0
    """Inhibitory width (sigma)."""

    A_attr: float = 0.05
    """Attraction amplitude (triple_gaussian only)."""

    sig_attr: float = 40.0
    """Attraction width (triple_gaussian only)."""

    # Kilpatrick parameter
    A: float = 1.0
    """Amplitude for Kilpatrick kernel."""

    # Quad exp parameters
    sig_custom: float = 20.0
    """Custom sigma for quad_exp kernel."""

    A_custom: float = 1.0
    """Custom amplitude for quad_exp kernel."""

    rise_power: float = 2.0
    """Rise power for quad_exp kernel."""

    decay_power: float = 1.0
    """Decay power for quad_exp kernel."""


@dataclass
class CrossKernelParams:
    """Parameters for cross-ring coupling kernels."""

    kernel_type: CrossKernelType = CrossKernelType.GAUSSIAN
    """Type of cross kernel to use."""

    J_cross: float = 1.2
    """Cross-coupling strength."""

    sig_cross: float = 40.0
    """Cross-coupling width (sigma)."""

    # Shared with lateral for quad_exp
    rise_power: float = 2.0
    """Rise power for quad_exp kernel."""

    decay_power: float = 1.0
    """Decay power for quad_exp kernel."""

    # Double bump parameters
    c1: float = 10.0
    """Center offset for first bump (double_bump)."""

    center: float = 0.0
    """Overall center offset (double_bump)."""

    A_ex1: float = 1.0
    """Amplitude of first bump (double_bump)."""

    sig_ex1: float = 5.0
    """Width of first bump (double_bump)."""

    A_ex2: float = 1.0
    """Amplitude of second bump (double_bump)."""

    sig_ex2: float = 5.0
    """Width of second bump (double_bump)."""

    # Mexican hat cross parameters
    A_ex_cross: float = 1.0
    """Excitatory amplitude (mexican_hat cross)."""

    sig_ex_cross: float = 5.0
    """Excitatory width (mexican_hat cross)."""

    A_inh_cross: float = 0.5
    """Inhibitory amplitude (mexican_hat cross)."""

    sig_inh_cross: float = 15.0
    """Inhibitory width (mexican_hat cross)."""


@dataclass
class InputParams:
    """Input stimulus parameters."""

    amp_target: float = 1.0
    """Target stimulus amplitude."""

    amp_dist: float = 1.0
    """Distractor stimulus amplitude."""

    width_input: float = 5.0
    """Input width in degrees."""

    theta: float = 0.25
    """Activation threshold."""


@dataclass
class PhysicsParams:
    """Bump physics parameters (for simplified model)."""

    m1: float = 1.0
    """Mass coefficient for bump 1."""

    m2: float = 1.0
    """Mass coefficient for bump 2."""

    stable_width: Optional[float] = None
    """Pre-calculated stable bump width (if known)."""


@dataclass
class SimulationConfig:
    """Complete configuration for a neural field simulation.

    This is the main configuration object that combines all parameter groups.
    It can be created directly or loaded from a YAML file.

    Example:
        config = SimulationConfig(
            sim_mode=SimulationMode.FULL,
            ring_mode=RingMode.SINGLE,
            time=TimeParams(T_total=500.0),
        )
    """

    sim_mode: SimulationMode = SimulationMode.FULL
    """Simulation mode (full PDE or simplified ODE)."""

    ring_mode: RingMode = RingMode.DUAL
    """Ring topology (single or dual population)."""

    grid: GridParams = field(default_factory=GridParams)
    """Spatial grid parameters."""

    time: TimeParams = field(default_factory=TimeParams)
    """Time integration parameters."""

    lateral_kernel: LateralKernelParams = field(default_factory=LateralKernelParams)
    """Lateral kernel parameters."""

    cross_kernel: CrossKernelParams = field(default_factory=CrossKernelParams)
    """Cross-coupling kernel parameters."""

    input: InputParams = field(default_factory=InputParams)
    """Input stimulus parameters."""

    physics: PhysicsParams = field(default_factory=PhysicsParams)
    """Bump physics parameters."""

    def __post_init__(self):
        """Convert string values to enums if needed."""
        if isinstance(self.sim_mode, str):
            self.sim_mode = SimulationMode(self.sim_mode)
        if isinstance(self.ring_mode, str):
            self.ring_mode = RingMode(self.ring_mode)
        if isinstance(self.lateral_kernel.kernel_type, str):
            self.lateral_kernel.kernel_type = LateralKernelType(
                self.lateral_kernel.kernel_type
            )
        if isinstance(self.cross_kernel.kernel_type, str):
            self.cross_kernel.kernel_type = CrossKernelType(
                self.cross_kernel.kernel_type
            )

    @property
    def dx(self) -> float:
        """Spatial resolution (convenience accessor)."""
        return self.grid.adjusted_dx

    @property
    def n_points(self) -> int:
        """Number of grid points (convenience accessor)."""
        return self.grid.n_points

    @property
    def dt(self) -> float:
        """Time step (convenience accessor)."""
        return self.time.dt

    @property
    def tau(self) -> float:
        """Time constant (convenience accessor)."""
        return self.time.tau

    @property
    def T_total(self) -> float:
        """Total time (convenience accessor)."""
        return self.time.T_total

    @property
    def eps(self) -> float:
        """Noise strength (convenience accessor)."""
        return self.time.eps

    @property
    def theta(self) -> float:
        """Activation threshold (convenience accessor)."""
        return self.input.theta

    @property
    def L(self) -> float:
        """Half-domain size (convenience accessor)."""
        return self.grid.L


# Backwards compatibility alias
ModelParams = SimulationConfig
