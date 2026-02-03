"""Kernel implementations for neural field simulations."""

from neural_field.kernels.base import KernelFunction
from neural_field.kernels.lateral import (
    MexicanHatKernel,
    TripleGaussianKernel,
    KilpatrickKernel,
    QuadExpKernel,
)
from neural_field.kernels.cross import (
    GaussianCrossKernel,
    TentCrossKernel,
    QuadExpCrossKernel,
    DoubleBumpCrossKernel,
    MexicanHatCrossKernel,
)
from neural_field.kernels.factory import create_lateral_kernel, create_cross_kernel
from neural_field.kernels.physics import BumpPhysics, compute_bump_physics

__all__ = [
    "KernelFunction",
    "MexicanHatKernel",
    "TripleGaussianKernel",
    "KilpatrickKernel",
    "QuadExpKernel",
    "GaussianCrossKernel",
    "TentCrossKernel",
    "QuadExpCrossKernel",
    "DoubleBumpCrossKernel",
    "MexicanHatCrossKernel",
    "create_lateral_kernel",
    "create_cross_kernel",
    "BumpPhysics",
    "compute_bump_physics",
]
