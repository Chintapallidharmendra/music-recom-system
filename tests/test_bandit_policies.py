"""Interface-conformance tests for all bandit policies. Every policy must satisfy
select_action(context, arms) -> arm in arms and update(arm_id, context, reward) without
raising -- this is the contract service/main.py relies on to swap policies via a config
flag with zero code changes.
"""

import numpy as np
import pytest

from bandit.policies.epsilon_greedy import EpsilonGreedyPolicy
from bandit.policies.linear_thompson_sampling import LinearThompsonSamplingPolicy
from bandit.policies.linucb import LinUCBPolicy
from bandit.policies.random_policy import RandomPolicy
from bandit.policies.thompson_sampling import ThompsonSamplingPolicy
from bandit.policies.ucb1 import UCB1Policy

CONTEXT_DIM = 8
ARMS = ["track_a", "track_b", "track_c"]


def make_context(rng):
    return rng.normal(size=CONTEXT_DIM)


NON_CONTEXTUAL_FACTORIES = {
    "random": lambda: RandomPolicy(seed=0),
    "epsilon_greedy": lambda: EpsilonGreedyPolicy(epsilon=0.2, seed=0),
    "ucb1": lambda: UCB1Policy(),
    "thompson": lambda: ThompsonSamplingPolicy(seed=0),
}

CONTEXTUAL_FACTORIES = {
    "linucb": lambda: LinUCBPolicy(alpha=1.0),
    "linear_thompson": lambda: LinearThompsonSamplingPolicy(v=0.3, seed=0),
}

ALL_FACTORIES = {**NON_CONTEXTUAL_FACTORIES, **CONTEXTUAL_FACTORIES}


@pytest.mark.parametrize("name", ALL_FACTORIES.keys())
def test_select_action_returns_valid_arm(name):
    rng = np.random.default_rng(1)
    policy = ALL_FACTORIES[name]()
    context = make_context(rng)
    arm = policy.select_action(context, ARMS)
    assert arm in ARMS


@pytest.mark.parametrize("name", ALL_FACTORIES.keys())
def test_update_does_not_raise(name):
    rng = np.random.default_rng(2)
    policy = ALL_FACTORIES[name]()
    for _ in range(20):
        context = make_context(rng)
        arm = policy.select_action(context, ARMS)
        reward = float(rng.choice([-1.0, 1.0, 2.0, 3.0]))
        policy.update(arm, context, reward)  # must not raise


@pytest.mark.parametrize("name", ALL_FACTORIES.keys())
def test_handles_growing_arm_set(name):
    """Arm-count handling: policies must work whether they've seen an arm before or not."""
    rng = np.random.default_rng(3)
    policy = ALL_FACTORIES[name]()
    context = make_context(rng)
    policy.update("track_a", context, 1.0)
    arm = policy.select_action(context, ["track_a", "track_new"])
    assert arm in ["track_a", "track_new"]


@pytest.mark.parametrize("name", ALL_FACTORIES.keys())
def test_policy_is_picklable_after_updates(name):
    """Policies get pickled for the MLflow Model Registry (mlops/tracking.py) -- a
    lambda factory on an ArmStateDict broke this once (see bandit/policies/_common.py's
    zero_int/zero_float/one_float); this guards against that regression."""
    import pickle

    rng = np.random.default_rng(5)
    policy = ALL_FACTORIES[name]()
    for _ in range(10):
        context = make_context(rng)
        arm = policy.select_action(context, ARMS)
        policy.update(arm, context, 1.0)

    restored = pickle.loads(pickle.dumps(policy))
    assert restored.select_action(make_context(rng), ARMS) in ARMS


@pytest.mark.parametrize("name", ["linucb", "linear_thompson"])
def test_contextual_policy_no_nans_after_many_updates(name):
    rng = np.random.default_rng(4)
    policy = CONTEXTUAL_FACTORIES[name]()
    for _ in range(1000):
        context = make_context(rng)
        arm = policy.select_action(context, ARMS)
        reward = float(rng.normal())
        policy.update(arm, context, reward)
    final_context = make_context(rng)
    for arm_id, A in policy._A.items():
        assert np.all(np.isfinite(A)), f"NaN/Inf in A matrix for arm {arm_id}"
    assert policy.select_action(final_context, ARMS) in ARMS
