"""Enums for neural field simulation configuration."""

from enum import Enum, auto


class SimulationMode(str, Enum):
    """Simulation mode selection."""

    FULL = "full"
    """Full PDE simulation with FFT convolution."""

    SIMPLIFIED = "simplified"
    """Reduced ODE simulation on bump centroids."""

    def __str__(self) -> str:
        return self.value


class RingMode(str, Enum):
    """Ring topology configuration."""

    SINGLE = "single"
    """Single population ring."""

    DUAL = "dual"
    """Two populations with cross-coupling."""

    def __str__(self) -> str:
        return self.value


class LateralKernelType(str, Enum):
    """Lateral (within-ring) kernel types."""

    TRIPLE_GAUSSIAN = "triple_gaussian"
    """Excitatory + inhibitory + attraction Gaussians."""

    MEXICAN_HAT = "mexican_hat"
    """Classic neural field kernel (excitatory - inhibitory)."""

    KILPATRICK = "kilpatrick"
    """Exponential decay form from Kilpatrick & Ermentrout."""

    QUAD_EXP = "quad_exp"
    """Tunable rise/decay power kernel."""

    def __str__(self) -> str:
        return self.value


class CrossKernelType(str, Enum):
    """Cross-ring coupling kernel types."""

    GAUSSIAN = "gaussian"
    """Simple Gaussian coupling."""

    TENT = "tent"
    """Split tent kernel (zero in middle, linear decay tails)."""

    QUAD_EXP = "quad_exp"
    """Tunable rise/decay power kernel."""

    DOUBLE_BUMP = "double_bump"
    """Two symmetric Gaussian bumps."""

    MEXICAN_HAT = "mexican_hat"
    """Mexican hat cross-coupling."""

    def __str__(self) -> str:
        return self.value


class ExperimentType(str, Enum):
    """Experiment protocol types."""

    SIMULTANEOUS = "simultaneous"
    """Target and distractor both appear at t=0."""

    DELAYED = "delayed"
    """Target at t=0, distractor appears later."""

    def __str__(self) -> str:
        return self.value
