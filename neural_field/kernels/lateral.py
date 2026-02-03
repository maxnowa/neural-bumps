"""Lateral (within-ring) kernel implementations."""

from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

from neural_field.config.params import LateralKernelParams


@dataclass
class MexicanHatKernel:
    """Classic Mexican hat kernel (difference of Gaussians).

    The kernel is defined as:
        w(x) = A_ex * exp(-x^2 / (2 * sig_ex^2)) - A_inh * exp(-x^2 / (2 * sig_inh^2))

    This creates local excitation with broader lateral inhibition.
    """

    A_ex: float
    sig_ex: float
    A_inh: float
    sig_inh: float

    @classmethod
    def from_params(cls, params: LateralKernelParams) -> "MexicanHatKernel":
        """Create kernel from LateralKernelParams."""
        return cls(
            A_ex=params.A_ex,
            sig_ex=params.sig_ex,
            A_inh=params.A_inh,
            sig_inh=params.sig_inh,
        )

    def compute(self, x: NDArray[np.floating]) -> NDArray[np.floating]:
        """Compute kernel values at positions x."""
        excitatory = self.A_ex * np.exp(-(x**2) / (2 * self.sig_ex**2))
        inhibitory = self.A_inh * np.exp(-(x**2) / (2 * self.sig_inh**2))
        return excitatory - inhibitory


@dataclass
class TripleGaussianKernel:
    """Triple Gaussian kernel with attraction term.

    The kernel is defined as:
        w(x) = A_ex * exp(...) - A_inh * exp(...) + A_attr * exp(...)

    The attraction term (third Gaussian) allows for long-range attraction.
    """

    A_ex: float
    sig_ex: float
    A_inh: float
    sig_inh: float
    A_attr: float
    sig_attr: float

    @classmethod
    def from_params(cls, params: LateralKernelParams) -> "TripleGaussianKernel":
        """Create kernel from LateralKernelParams."""
        return cls(
            A_ex=params.A_ex,
            sig_ex=params.sig_ex,
            A_inh=params.A_inh,
            sig_inh=params.sig_inh,
            A_attr=params.A_attr,
            sig_attr=params.sig_attr,
        )

    def compute(self, x: NDArray[np.floating]) -> NDArray[np.floating]:
        """Compute kernel values at positions x."""
        g1 = self.A_ex * np.exp(-(x**2) / (2 * self.sig_ex**2))
        g2 = self.A_inh * np.exp(-(x**2) / (2 * self.sig_inh**2))
        g3 = self.A_attr * np.exp(-(x**2) / (2 * self.sig_attr**2))
        return g1 - g2 + g3


@dataclass
class KilpatrickKernel:
    """Kilpatrick exponential decay kernel.

    The kernel is defined as:
        w(x) = A * (1 - |x|) * exp(-|x|)

    This form comes from Kilpatrick & Ermentrout (2013).
    """

    A: float

    @classmethod
    def from_params(cls, params: LateralKernelParams) -> "KilpatrickKernel":
        """Create kernel from LateralKernelParams."""
        return cls(A=params.A)

    def compute(self, x: NDArray[np.floating]) -> NDArray[np.floating]:
        """Compute kernel values at positions x."""
        dist = np.abs(x)
        return self.A * (1.0 - dist) * np.exp(-dist)


@dataclass
class QuadExpKernel:
    """Tunable rise/decay power kernel.

    The kernel is defined as:
        w(x) = A * (|x|/sigma)^a * exp(-(|x|/sigma)^b) / peak_normalization

    Parameters a (rise_power) and b (decay_power) control the shape:
    - a=2, b=1: Quadratic rise, exponential decay
    - a=2, b=2: Gaussian-like profile
    """

    sigma: float
    strength: float
    rise_power: float
    decay_power: float

    @classmethod
    def from_params(cls, params: LateralKernelParams) -> "QuadExpKernel":
        """Create kernel from LateralKernelParams."""
        return cls(
            sigma=params.sig_custom,
            strength=params.A_custom,
            rise_power=params.rise_power,
            decay_power=params.decay_power,
        )

    def compute(self, x: NDArray[np.floating]) -> NDArray[np.floating]:
        """Compute kernel values at positions x."""
        dist = np.abs(x)
        x_scaled = dist / self.sigma

        # Analytical peak normalization
        a, b = self.rise_power, self.decay_power
        if b > 0 and a > 0:
            peak_loc = (a / b) ** (1 / b)
            peak_val = (peak_loc**a) * np.exp(-(peak_loc**b))
        else:
            peak_val = 1.0

        raw_val = np.power(x_scaled, a) * np.exp(-np.power(x_scaled, b))
        return self.strength * (raw_val / peak_val)
