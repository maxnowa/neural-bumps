"""Factory functions for creating kernel instances."""

from typing import Union

from neural_field.config.params import (
    LateralKernelParams,
    CrossKernelParams,
    SimulationConfig,
)
from neural_field.config.enums import LateralKernelType, CrossKernelType
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


# Type alias for any lateral kernel
LateralKernel = Union[
    MexicanHatKernel, TripleGaussianKernel, KilpatrickKernel, QuadExpKernel
]

# Type alias for any cross kernel
CrossKernel = Union[
    GaussianCrossKernel,
    TentCrossKernel,
    QuadExpCrossKernel,
    DoubleBumpCrossKernel,
    MexicanHatCrossKernel,
]


def create_lateral_kernel(
    params: Union[LateralKernelParams, SimulationConfig],
    kernel_type: LateralKernelType = None,
) -> LateralKernel:
    """Create a lateral kernel from parameters.

    Args:
        params: LateralKernelParams or SimulationConfig containing parameters.
        kernel_type: Optional override for kernel type. If None, uses type from params.

    Returns:
        A kernel instance implementing the KernelFunction protocol.

    Raises:
        ValueError: If kernel type is unknown.

    Example:
        kernel = create_lateral_kernel(config.lateral_kernel)
        values = kernel.compute(grid.x)
    """
    if isinstance(params, SimulationConfig):
        params = params.lateral_kernel

    if kernel_type is None:
        kernel_type = params.kernel_type

    if isinstance(kernel_type, str):
        kernel_type = LateralKernelType(kernel_type)

    if kernel_type == LateralKernelType.MEXICAN_HAT:
        return MexicanHatKernel.from_params(params)
    elif kernel_type == LateralKernelType.TRIPLE_GAUSSIAN:
        return TripleGaussianKernel.from_params(params)
    elif kernel_type == LateralKernelType.KILPATRICK:
        return KilpatrickKernel.from_params(params)
    elif kernel_type == LateralKernelType.QUAD_EXP:
        return QuadExpKernel.from_params(params)
    else:
        raise ValueError(f"Unknown lateral kernel type: {kernel_type}")


def create_cross_kernel(
    params: Union[CrossKernelParams, SimulationConfig],
    kernel_type: CrossKernelType = None,
) -> CrossKernel:
    """Create a cross-coupling kernel from parameters.

    Args:
        params: CrossKernelParams or SimulationConfig containing parameters.
        kernel_type: Optional override for kernel type. If None, uses type from params.

    Returns:
        A kernel instance implementing the KernelFunction protocol.

    Raises:
        ValueError: If kernel type is unknown.

    Example:
        kernel = create_cross_kernel(config.cross_kernel)
        values = kernel.compute(grid.x)
    """
    if isinstance(params, SimulationConfig):
        params = params.cross_kernel

    if kernel_type is None:
        kernel_type = params.kernel_type

    if isinstance(kernel_type, str):
        kernel_type = CrossKernelType(kernel_type)

    if kernel_type == CrossKernelType.GAUSSIAN:
        return GaussianCrossKernel.from_params(params)
    elif kernel_type == CrossKernelType.TENT:
        return TentCrossKernel.from_params(params)
    elif kernel_type == CrossKernelType.QUAD_EXP:
        return QuadExpCrossKernel.from_params(params)
    elif kernel_type == CrossKernelType.DOUBLE_BUMP:
        return DoubleBumpCrossKernel.from_params(params)
    elif kernel_type == CrossKernelType.MEXICAN_HAT:
        return MexicanHatCrossKernel.from_params(params)
    else:
        raise ValueError(f"Unknown cross kernel type: {kernel_type}")
