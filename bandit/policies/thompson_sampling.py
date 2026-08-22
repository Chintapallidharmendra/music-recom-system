"""Thompson Sampling with Beta priors (non-contextual). Reward is binarized to a
success signal (reward > 0) since our reward map isn't already Bernoulli."""

import numpy as np

from bandit.policies._common import ArmStateDict, one_float


class ThompsonSamplingPolicy:
    def __init__(self, seed: int = None):
        self._rng = np.random.default_rng(seed)
        self._alpha = ArmStateDict(one_float)
        self._beta = ArmStateDict(one_float)

    def select_action(self, context: np.ndarray, arms: list) -> str:
        samples = [
            self._rng.beta(self._alpha.get_or_create(a), self._beta.get_or_create(a)) for a in arms
        ]
        return arms[int(np.argmax(samples))]

    def update(self, arm_id: str, context: np.ndarray, reward: float) -> None:
        success = 1.0 if reward > 0 else 0.0
        self._alpha[arm_id] = self._alpha.get_or_create(arm_id) + success
        self._beta[arm_id] = self._beta.get_or_create(arm_id) + (1.0 - success)
