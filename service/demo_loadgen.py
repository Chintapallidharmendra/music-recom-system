"""Drives synthetic /recommend -> reward_simulator -> /feedback traffic against a
running service instance. There are no real users to click /feedback in a solo academic
build, so this script plays that role for the end-to-end demo (see PROJECT_PLAN.md's
end-to-end smoke sequence).
"""

import argparse

import httpx
import numpy as np
import pandas as pd

from bandit.reward_simulator import RewardSimulator


def run(host: str, n: int, seed: int) -> None:
    profiles = pd.read_parquet("data/user_profiles.parquet")
    user_ids = profiles["user_id"].to_numpy()
    sim = RewardSimulator()
    rng = np.random.default_rng(seed)

    with httpx.Client(base_url=host, timeout=10.0) as client:
        for i in range(n):
            user_id = rng.choice(user_ids)

            resp = client.post("/recommend", json={"user_id": user_id})
            resp.raise_for_status()
            track_id = resp.json()["track_id"]

            action = sim.sample_action(user_id, track_id, rng)
            client.post(
                "/feedback",
                json={"user_id": user_id, "track_id": track_id, "action": action},
            ).raise_for_status()

            if (i + 1) % 50 == 0:
                print(f"{i + 1}/{n} interactions sent")

        metrics = client.get("/metrics").json()
        print(f"final /metrics: {metrics}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="http://localhost:8000")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    run(args.host, args.n, args.seed)


if __name__ == "__main__":
    main()
