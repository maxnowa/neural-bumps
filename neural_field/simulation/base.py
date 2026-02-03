"""Base protocol for simulation engines."""

from typing import Protocol, runtime_checkable

from neural_field.core.results import TrialResult


@runtime_checkable
class Simulator(Protocol):
    """Protocol for simulation engines.

    All simulator implementations must provide a run_trial method
    that executes a single simulation trial.

    Example:
        simulator = create_simulator(config)
        result = simulator.run_trial(target_loc=0.0, dist_loc=40.0)
    """

    def run_trial(
        self,
        target_loc: float = 0.0,
        dist_loc: float = 40.0,
        experiment_type: str = "simultaneous",
        with_noise: bool = True,
        **kwargs,
    ) -> TrialResult:
        """Run a single simulation trial.

        Args:
            target_loc: Target position in degrees.
            dist_loc: Distractor position in degrees.
            experiment_type: "simultaneous" or "delayed".
            with_noise: Whether to include noise in the simulation.
            **kwargs: Additional experiment-specific parameters.

        Returns:
            TrialResult containing bias, trajectories, and field states.
        """
        ...
