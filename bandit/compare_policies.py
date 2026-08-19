"""Run all six bandit policies through the same offline replay log and compare
CTR, average reward, and cumulative regret. Acceptance criterion: LinUCB (and
Linear Thompson Sampling) must beat Random on cumulative regret.
"""
import argparse

import pandas as pd

from bandit.policies.epsilon_greedy import EpsilonGreedyPolicy
from bandit.policies.linear_thompson_sampling import LinearThompsonSamplingPolicy
from bandit.policies.linucb import LinUCBPolicy
from bandit.policies.random_policy import RandomPolicy
from bandit.policies.thompson_sampling import ThompsonSamplingPolicy
from bandit.policies.ucb1 import UCB1Policy
from bandit.replay_evaluator import generate_replay_log, replay_evaluate
from bandit.reward_simulator import RewardSimulator
from mlops.tracking import log_replay_evaluation

POLICY_FACTORIES = {
    "random": lambda seed: RandomPolicy(seed=seed),
    "epsilon_greedy": lambda seed: EpsilonGreedyPolicy(epsilon=0.1, seed=seed),
    "ucb1": lambda seed: UCB1Policy(),
    "thompson_sampling": lambda seed: ThompsonSamplingPolicy(seed=seed),
    "linear_thompson_sampling": lambda seed: LinearThompsonSamplingPolicy(
        v=0.3, seed=seed, context_dim=8
    ),
    "linucb": lambda seed: LinUCBPolicy(alpha=1.0, context_dim=8),
}


def compare(n_events: int, pool_size: int, seed: int, log_to_mlflow: bool = True) -> pd.DataFrame:
    log = generate_replay_log(n_events, pool_size, seed)
    sim = RewardSimulator()

    rows = []
    for name, factory in POLICY_FACTORIES.items():
        policy = factory(seed)
        metrics = replay_evaluate(policy, log, sim)
        rows.append({"policy": name, **metrics})
        if log_to_mlflow:
            params = {"n_events": n_events, "pool_size": pool_size, "seed": seed}
            log_replay_evaluation(name, params, metrics)

    return pd.DataFrame(rows).set_index("policy")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-events", type=int, default=8000)
    parser.add_argument("--pool-size", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = compare(args.n_events, args.pool_size, args.seed)
    print(df.to_string(float_format=lambda x: f"{x:.4f}"))

    random_regret = df.loc["random", "cumulative_regret"]
    linucb_regret = df.loc["linucb", "cumulative_regret"]
    lints_regret = df.loc["linear_thompson_sampling", "cumulative_regret"]
    print(f"\nLinUCB beats Random on regret: {linucb_regret < random_regret}")
    print(f"Linear TS beats Random on regret: {lints_regret < random_regret}")


if __name__ == "__main__":
    main()
