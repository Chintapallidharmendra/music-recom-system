"""UCB1: pick every unseen arm at least once, then the arm maximizing
value + sqrt(2 * ln(total_pulls) / arm_pulls)."""

import numpy as np

from bandit.policies._common import ArmStateDict, zero_float, zero_int


class UCB1Policy:
    def __init__(self):
        self._counts = ArmStateDict(zero_int)
        self._values = ArmStateDict(zero_float)
        self._t = 0

    def select_action(self, context: np.ndarray, arms: list) -> str:
        unseen = [a for a in arms if self._counts.get_or_create(a) == 0]
        if unseen:
            return unseen[0]

        t = max(self._t, 1)
        scores = [self._values[a] + np.sqrt(2 * np.log(t) / self._counts[a]) for a in arms]
        return arms[int(np.argmax(scores))]

    def update(self, arm_id: str, context: np.ndarray, reward: float) -> None:
        self._t += 1
        self._counts[arm_id] = self._counts.get_or_create(arm_id) + 1
        n = self._counts[arm_id]
        value = self._values.get_or_create(arm_id)
        self._values[arm_id] = value + (reward - value) / n
