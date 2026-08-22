"""LinUCB: ridge regression per arm with a UCB exploration bonus. Frequentist
counterpart to linear_thompson_sampling.py. Matrix inversion is ridge-guarded so
1000+ updates never produce NaNs even if an arm's design matrix gets ill-conditioned."""

import numpy as np

from bandit.policies._common import ridge_inverse


class LinUCBPolicy:
    def __init__(self, alpha: float = 1.0, context_dim: int = None):
        """context_dim: if set, only the first `context_dim` entries of the context
        vector are used. Ridge regression needs roughly O(dim) samples per arm to
        converge; with a high-dimensional context and few samples per arm, restricting
        to the leading (highest-signal) dimensions keeps the regression well-posed.
        build_user_context.py orders context as [genre_affinity(8), audio_features(60)],
        so context_dim=8 uses genre affinity only -- the dominant, lower-noise signal."""
        self.alpha = alpha
        self.context_dim = context_dim
        # Plain dicts (not ArmStateDict): _ensure_arm below handles lazy init directly,
        # no factory needed -- and policies get pickled for the MLflow Model Registry
        # (mlops/tracking.py), where a lambda factory wouldn't survive serialization.
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
            theta = A_inv @ self._b[arm_id]
            ucb = theta @ context + self.alpha * np.sqrt(max(context @ A_inv @ context, 0.0))
            scores.append(float(ucb))
        return arms[int(np.argmax(scores))]

    def update(self, arm_id: str, context: np.ndarray, reward: float) -> None:
        context = np.asarray(context, dtype=np.float64)
        if self.context_dim:
            context = context[: self.context_dim]
        self._ensure_arm(arm_id, len(context))
        self._A[arm_id] += np.outer(context, context)
        self._b[arm_id] += reward * context
