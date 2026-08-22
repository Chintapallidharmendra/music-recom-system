"""Shared per-arm lazy-state helper.

Arms are track_ids (arbitrary strings), not sequential indices, and the candidate arm set
can vary per call -- so every policy keeps a dict keyed by arm_id, initialized lazily the
first time an arm is seen, rather than a fixed-size array sized to a known arm count.
"""

import numpy as np


def zero_int() -> int:
    return 0


def zero_float() -> float:
    return 0.0


def one_float() -> float:
    return 1.0


class ArmStateDict(dict):
    """dict[arm_id -> state] that lazily creates state via a factory on first access.

    The factory must be a named module-level function, not a lambda/closure -- policies
    get pickled for the MLflow Model Registry (see mlops/tracking.py), and pickle can't
    serialize a lambda bound to an instance attribute.
    """

    def __init__(self, factory):
        super().__init__()
        self._factory = factory

    def get_or_create(self, arm_id):
        if arm_id not in self:
            self[arm_id] = self._factory()
        return self[arm_id]


def ridge_inverse(matrix: np.ndarray, ridge: float = 1e-6) -> np.ndarray:
    """Matrix inverse guarded against numerical singularity from repeated updates."""
    d = matrix.shape[0]
    return np.linalg.inv(matrix + ridge * np.eye(d))
