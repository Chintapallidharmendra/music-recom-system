"""Epsilon-greedy: explore uniformly at random with prob epsilon, else exploit the
arm with the highest running-average reward seen so far (unseen arms default to 0.0)."""

import numpy as np

from bandit.policies._common import ArmStateDict, zero_float, zero_int


class EpsilonGreedyPolicy:
    def __init__(self, epsilon: float = 0.1, seed: int = None):
        self.epsilon = epsilon
        self._rng = np.random.default_rng(seed)
        self._counts = ArmStateDict(zero_int)
        self._values = ArmStateDict(zero_float)

    def select_action(self, context: np.ndarray, arms: list) -> str:
        if self._rng.random() < self.epsilon:
            return self._rng.choice(arms)
        values = [self._values.get_or_create(a) for a in arms]
        return arms[int(np.argmax(values))]

    def update(self, arm_id: str, context: np.ndarray, reward: float) -> None:
        self._counts[arm_id] = self._counts.get_or_create(arm_id) + 1
        n = self._counts[arm_id]
        value = self._values.get_or_create(arm_id)
        self._values[arm_id] = value + (reward - value) / n
