"""Offline bandit replay evaluation (Li et al. 2011 replay method).

Builds one fixed logged dataset of (context, logged_action, logged_reward) triples
collected under a uniform-random logging policy over a random candidate set per event
-- required for the replay method's unbiasedness. Every policy is replayed against the
SAME log: a policy only learns from (and is scored on) events where its own chosen
action happens to match what the log recorded, per contracts/... acceptance criteria
in PROJECT_PLAN.md. Since this is a synthetic environment, ground-truth
reward_simulator.expected_reward() is also used to compute regret against the true
optimal arm for matched events -- a luxury real-world replay evaluation doesn't have.
"""
import argparse

import numpy as np
import pandas as pd

from bandit.reward_simulator import REWARD_MAP, RewardSimulator
from data.build_user_context import build_user_context


def generate_replay_log(
    n_events: int,
    pool_size: int,
    seed: int,
    profiles_path: str = "data/user_profiles.parquet",
    features_path: str = "data/features.parquet",
    logs_path: str = "data/synthetic_logs.parquet",
) -> list[dict]:
    """A FIXED candidate pool of `pool_size` tracks is drawn once and reused as the arm
    set for every event -- mirroring the architecture's "Candidate Generator retrieves
    Top-N" step. This is required for arm-level learning (UCB1, Thompson, LinUCB, ...):
    with a fresh random candidate set per event, a given track_id would appear only a
    handful of times across the whole log and every policy would stay stuck in its
    cold-start "arm never seen before" branch, unable to differentiate itself."""
    rng = np.random.default_rng(seed)
    sim = RewardSimulator(profiles_path, features_path)
    profiles = pd.read_parquet(profiles_path)
    features = pd.read_parquet(features_path)
    plays = pd.read_parquet(logs_path)
    user_ids = profiles["user_id"].to_numpy()

    all_track_ids = features["track_id"].to_numpy()
    candidate_pool = list(rng.choice(all_track_ids, size=pool_size, replace=False))

    # Contexts are static per user for the duration of this offline evaluation --
    # a standard simplification (build_user_context is deterministic given fixed history).
    context_cache: dict = {}

    events = []
    for _ in range(n_events):
        user_id = rng.choice(user_ids)
        if user_id not in context_cache:
            context_cache[user_id] = build_user_context(user_id, plays, features)
        context = context_cache[user_id]

        logged_arm = rng.choice(candidate_pool)  # uniform-random logging policy
        action = sim.sample_action(user_id, logged_arm, rng)
        reward = REWARD_MAP[action]

        events.append({
            "user_id": user_id,
            "context": context,
            "candidates": candidate_pool,
            "logged_arm": logged_arm,
            "reward": reward,
        })
    return events


def replay_evaluate(policy, log: list[dict], reward_sim: RewardSimulator) -> dict:
    matched = 0
    cumulative_reward = 0.0
    cumulative_optimal = 0.0
    positive_outcomes = 0
    optimal_cache: dict = {}  # candidate pool is fixed, so optimal only varies by user

    for event in log:
        chosen = policy.select_action(event["context"], event["candidates"])
        if chosen != event["logged_arm"]:
            continue  # standard replay: skip events where the policy disagrees with the log

        matched += 1
        reward = event["reward"]
        policy.update(chosen, event["context"], reward)

        cumulative_reward += reward
        positive_outcomes += 1 if reward > 0 else 0
        user_id = event["user_id"]
        if user_id not in optimal_cache:
            optimal_cache[user_id] = max(
                reward_sim.expected_reward(user_id, c) for c in event["candidates"]
            )
        cumulative_optimal += optimal_cache[user_id]

    return {
        "matched_events": matched,
        "match_rate": matched / len(log) if log else 0.0,
        "ctr": positive_outcomes / matched if matched else 0.0,
        "avg_reward": cumulative_reward / matched if matched else 0.0,
        "cumulative_reward": cumulative_reward,
        "cumulative_regret": cumulative_optimal - cumulative_reward,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-events", type=int, default=5000)
    parser.add_argument("--pool-size", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from bandit.policies.random_policy import RandomPolicy

    log = generate_replay_log(args.n_events, args.pool_size, args.seed)
    sim = RewardSimulator()
    policy = RandomPolicy(seed=args.seed)
    metrics = replay_evaluate(policy, log, sim)
    print(f"Random policy replay metrics (n_events={len(log)}): {metrics}")


if __name__ == "__main__":
    main()
