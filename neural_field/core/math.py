"""Mathematical utilities for circular/periodic computations."""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def wrap_deg(a: ArrayLike) -> NDArray[np.floating]:
    """Wrap angle(s) to [-180, 180) degrees.

    Args:
        a: Angle or array of angles in degrees.

    Returns:
        Wrapped angle(s) in the range [-180, 180).

    Example:
        >>> wrap_deg(270)
        -90.0
        >>> wrap_deg(-200)
        160.0
    """
    a = np.asarray(a)
    return (a + 180.0) % 360.0 - 180.0


def circ_dist(a: ArrayLike, b: ArrayLike) -> NDArray[np.floating]:
    """Compute circular distance between angles.

    Args:
        a: First angle or array of angles in degrees.
        b: Second angle or array of angles in degrees.

    Returns:
        Absolute circular distance in degrees (always positive, max 180).

    Example:
        >>> circ_dist(10, 350)
        20.0
        >>> circ_dist(-170, 170)
        20.0
    """
    return np.abs(wrap_deg(np.asarray(a) - np.asarray(b)))


def circ_mean(angles: ArrayLike, weights: ArrayLike = None) -> float:
    """Compute circular mean of angles.

    Args:
        angles: Array of angles in degrees.
        weights: Optional weights for each angle.

    Returns:
        Circular mean angle in degrees.

    Example:
        >>> circ_mean([350, 10])  # Should be close to 0
        0.0
    """
    angles = np.asarray(angles)
    rad = np.deg2rad(angles)

    if weights is not None:
        weights = np.asarray(weights)
        C = np.sum(weights * np.cos(rad))
        S = np.sum(weights * np.sin(rad))
    else:
        C = np.sum(np.cos(rad))
        S = np.sum(np.sin(rad))

    return wrap_deg(np.rad2deg(np.arctan2(S, C))).item()


def ring_interval_indices(iL: int, iR: int, N: int) -> NDArray[np.intp]:
    """Get indices for a circular interval on a ring.

    Handles wrap-around correctly when iL > iR.

    Args:
        iL: Left index (inclusive).
        iR: Right index (inclusive).
        N: Total number of points on the ring.

    Returns:
        Array of indices from iL to iR (wrapping if necessary).

    Example:
        >>> ring_interval_indices(8, 2, 10)  # Wraps around
        array([8, 9, 0, 1, 2])
    """
    if iL <= iR:
        return np.arange(iL, iR + 1)
    else:
        return np.r_[np.arange(iL, N), np.arange(0, iR + 1)]
