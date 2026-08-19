"""Random policy -- baseline every other policy must beat on cumulative regret."""
import numpy as np


class RandomPolicy:
    def __init__(self, seed: int = None):
        self._rng = np.random.default_rng(seed)

    def select_action(self, context: np.ndarray, arms: list) -> str:
        return self._rng.choice(arms)

    def update(self, arm_id: str, context: np.ndarray, reward: float) -> None:
        pass  # no learning
