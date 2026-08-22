"""Linear Thompson Sampling: Bayesian linear regression per arm over the context
vector. Contextual counterpart to plain Thompson Sampling -- pairs with LinUCB as a
Bayesian-vs-frequentist comparison of contextual policies (see PROJECT_PLAN.md)."""

import numpy as np

from bandit.policies._common import ridge_inverse


class LinearThompsonSamplingPolicy:
    def __init__(self, v: float = 0.3, seed: int = None, context_dim: int = None):
        """context_dim: see linucb.py's docstring -- same rationale, same convention
        (leading dims of context, i.e. genre affinity when context_dim=8)."""
        self.v = v  # exploration variance scale
        self.context_dim = context_dim
        self._rng = np.random.default_rng(seed)
        # Plain dicts, not ArmStateDict -- see linucb.py's comment: no factory needed,
        # and policies must stay picklable for the MLflow Model Registry.
        self._A: dict = {}
        self._b: dict = {}

    def _ensure_arm(self, arm_id: str, dim: int) -> None:
        if self._A.get(arm_id) is None:
            self._A[arm_id] = np.eye(dim)
            self._b[arm_id] = np.zeros(dim)

    def select_action(self, context: np.ndarray, arms: list) -> str:
        context = np.asarray(context, dtype=np.float64)
        if self.context_dim:
            context = context[: self.context_dim]
        dim = len(context)
        scores = []
        for arm_id in arms:
            self._ensure_arm(arm_id, dim)
            A_inv = ridge_inverse(self._A[arm_id])
            A_inv = (A_inv + A_inv.T) / 2  # guard against asymmetry from float error
            theta_hat = A_inv @ self._b[arm_id]
            theta_tilde = self._rng.multivariate_normal(theta_hat, self.v**2 * A_inv)
            scores.append(float(theta_tilde @ context))
        return arms[int(np.argmax(scores))]

    def update(self, arm_id: str, context: np.ndarray, reward: float) -> None:
        context = np.asarray(context, dtype=np.float64)
        if self.context_dim:
            context = context[: self.context_dim]
        self._ensure_arm(arm_id, len(context))
        self._A[arm_id] += np.outer(context, context)
        self._b[arm_id] += reward * context
