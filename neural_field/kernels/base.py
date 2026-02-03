"""Base protocol for kernel functions."""

from typing import Protocol, runtime_checkable
import numpy as np
from numpy.typing import NDArray


@runtime_checkable
class KernelFunction(Protocol):
    """Protocol for spatial kernel functions.

    All kernel implementations must provide a compute method that takes
    spatial positions and returns kernel values.

    Example:
        class MyKernel:
            def compute(self, x: NDArray) -> NDArray:
                return np.exp(-x**2)

        kernel = MyKernel()
        values = kernel.compute(grid.x)
    """

    def compute(self, x: NDArray[np.floating]) -> NDArray[np.floating]:
        """Compute kernel values at given positions.

        Args:
            x: Array of spatial positions (typically grid.x).

        Returns:
            Array of kernel values at each position.
        """
        ...
