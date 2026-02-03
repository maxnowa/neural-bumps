"""Cross-ring coupling kernel implementations."""

from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

from neural_field.config.params import CrossKernelParams


@dataclass
class GaussianCrossKernel:
    """Simple Gaussian cross-coupling kernel.

    The kernel is defined as:
        w(x) = J_cross * exp(-x^2 / (2 * sig_cross^2))
    """

    J_cross: float
    sig_cross: float

    @classmethod
    def from_params(cls, params: CrossKernelParams) -> "GaussianCrossKernel":
        """Create kernel from CrossKernelParams."""
        return cls(J_cross=params.J_cross, sig_cross=params.sig_cross)

    def compute(self, x: NDArray[np.floating]) -> NDArray[np.floating]:
        """Compute kernel values at positions x."""
        return self.J_cross * np.exp(-(x**2) / (2 * self.sig_cross**2))


@dataclass
class TentCrossKernel:
    """Split tent kernel with dead zone in the middle.

    The kernel has:
    - Zero for |x| < gap/2 (dead zone)
    - Linear decay from J_cross to 0 for |x| in [gap/2, gap/2 + width]
    """

    J_cross: float
    width: float
    gap: float = 4.0

    @classmethod
    def from_params(cls, params: CrossKernelParams) -> "TentCrossKernel":
        """Create kernel from CrossKernelParams."""
        return cls(J_cross=params.J_cross, width=params.sig_cross)

    def compute(self, x: NDArray[np.floating]) -> NDArray[np.floating]:
        """Compute kernel values at positions x."""
        dist = np.abs(x)
        half_gap = self.gap / 2.0

        result = np.zeros_like(x)
        # Active region: outside gap but inside tail width
        mask = (dist >= half_gap) & (dist < (half_gap + self.width))
        # Linear decay from J_cross to 0
        result[mask] = self.J_cross * (1.0 - (dist[mask] - half_gap) / self.width)

        return result


@dataclass
class QuadExpCrossKernel:
    """Tunable rise/decay power cross-coupling kernel.

    Same functional form as QuadExpKernel for lateral kernels.
    """

    sigma: float
    strength: float
    rise_power: float
    decay_power: float

    @classmethod
    def from_params(cls, params: CrossKernelParams) -> "QuadExpCrossKernel":
        """Create kernel from CrossKernelParams."""
        return cls(
            sigma=params.sig_cross,
            strength=params.J_cross,
            rise_power=params.rise_power,
            decay_power=params.decay_power,
        )

    def compute(self, x: NDArray[np.floating]) -> NDArray[np.floating]:
        """Compute kernel values at positions x."""
        dist = np.abs(x)
        x_scaled = dist / self.sigma

        a, b = self.rise_power, self.decay_power
        if b > 0 and a > 0:
            peak_loc = (a / b) ** (1 / b)
            peak_val = (peak_loc**a) * np.exp(-(peak_loc**b))
        else:
            peak_val = 1.0

        raw_val = np.power(x_scaled, a) * np.exp(-np.power(x_scaled, b))
        return self.strength * (raw_val / peak_val)


@dataclass
class DoubleBumpCrossKernel:
    """Double bump cross-coupling kernel.

    Two symmetric Gaussian bumps offset from center.
    """

    A_ex1: float
    sig_ex1: float
    A_ex2: float
    sig_ex2: float
    c1: float
    center: float

    @classmethod
    def from_params(cls, params: CrossKernelParams) -> "DoubleBumpCrossKernel":
        """Create kernel from CrossKernelParams."""
        return cls(
            A_ex1=params.A_ex1,
            sig_ex1=params.sig_ex1,
            A_ex2=params.A_ex2,
            sig_ex2=params.sig_ex2,
            c1=params.c1,
            center=params.center,
        )

    def compute(self, x: NDArray[np.floating]) -> NDArray[np.floating]:
        """Compute kernel values at positions x."""
        c2 = -self.c1  # Symmetric offset
        g1 = self.A_ex1 * np.exp(
            -((x - self.c1 - self.center) ** 2) / (2 * self.sig_ex1**2)
        )
        g2 = self.A_ex2 * np.exp(
            -((x - c2 - self.center) ** 2) / (2 * self.sig_ex2**2)
        )
        return g1 + g2


@dataclass
class MexicanHatCrossKernel:
    """Mexican hat cross-coupling kernel.

    Same form as lateral Mexican hat but for cross-coupling.
    """

    A_ex: float
    sig_ex: float
    A_inh: float
    sig_inh: float

    @classmethod
    def from_params(cls, params: CrossKernelParams) -> "MexicanHatCrossKernel":
        """Create kernel from CrossKernelParams."""
        return cls(
            A_ex=params.A_ex_cross,
            sig_ex=params.sig_ex_cross,
            A_inh=params.A_inh_cross,
            sig_inh=params.sig_inh_cross,
        )

    def compute(self, x: NDArray[np.floating]) -> NDArray[np.floating]:
        """Compute kernel values at positions x."""
        excitatory = self.A_ex * np.exp(-(x**2) / (2 * self.sig_ex**2))
        inhibitory = self.A_inh * np.exp(-(x**2) / (2 * self.sig_inh**2))
        return excitatory - inhibitory
