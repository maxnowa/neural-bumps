"""Core abstractions for neural field simulations."""

from neural_field.core.grid import SpatialGrid
from neural_field.core.results import TrialResult, BumpTrackingResult
from neural_field.core.math import wrap_deg, circ_dist

__all__ = [
    "SpatialGrid",
    "TrialResult",
    "BumpTrackingResult",
    "wrap_deg",
    "circ_dist",
]
