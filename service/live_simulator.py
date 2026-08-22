"""Live synthetic-user simulator for the music-bandit service.

It continuously exercises /recommend -> synthetic user behavior -> /feedback.
Drift is injected into the *user behavior*, not into the monitoring data directly.
"""

import argparse
import time

import httpx
import numpy as np
import pandas as pd

from bandit.reward_simulator import RewardSimulator


class LiveUserBehaviorSimulator:
    def __init__(self, scenario="normal", start_after=500, magnitude=0.6, seed=0):
        self.scenario = scenario
        self.start_after = start_after
        self.magnitude = float(np.clip(magnitude, 0.0, 1.0))
        self.rng = np.random.default_rng(seed)
        self.reward_sim = RewardSimulator()
        self.profiles = pd.read_parquet("data/user_profiles.parquet")
        self.user_ids = self.profiles["user_id"].to_numpy()
        self.base_affinity = {
            row["user_id"]: np.asarray(row["genre_affinity"], dtype=np.float64).copy()
            for _, row in self.profiles.iterrows()
        }
        top_genres = self.profiles["genre_affinity"].apply(lambda x: int(np.argmax(x)))
        # Pick one latent preference segment as the incoming population.
        self.new_population_users = self.profiles.loc[
            top_genres == top_genres.mode().iloc[0], "user_id"
        ].to_numpy()

    def _drifted_user(self, user_id, interaction_no):
        """Return the effective affinity used by the simulated user."""
        base = self.base_affinity[user_id].copy()
        if self.scenario != "preference_shift" or interaction_no < self.start_after:
            return base

        # Move probability mass from the user's strongest genre to the weakest.
        source = int(np.argmax(base))
        target = int(np.argmin(base))
        shift = self.magnitude * min(base[source], 0.75)
        base[source] -= shift
        base[target] += shift
        total = base.sum()
        return base / total if total > 0 else base

    def sample_action(self, user_id, track_id, interaction_no):
        if self.scenario == "normal" or interaction_no < self.start_after:
            return self.reward_sim.sample_action(user_id, track_id, self.rng)

        if self.scenario == "reward_drift":
            # Same preferences, but users become less engaged after the drift point.
            action = self.reward_sim.sample_action(user_id, track_id, self.rng)
            if self.rng.random() < self.magnitude:
                return self.rng.choice(["completed", "skip"])
            return action

        if self.scenario == "preference_shift":
            # Temporarily replace the ground-truth affinity for this user while
            # sampling. Restore it immediately so the simulator remains local/stateful.
            original = self.reward_sim._affinity[user_id]
            self.reward_sim._affinity[user_id] = self._drifted_user(user_id, interaction_no)
            try:
                return self.reward_sim.sample_action(user_id, track_id, self.rng)
            finally:
                self.reward_sim._affinity[user_id] = original

        if self.scenario == "new_user_population":
            # New population: sample a different user than the original request when
            # possible. This changes the observed user/track interaction distribution.
            # The service still receives the real selected user_id.
            action = self.reward_sim.sample_action(user_id, track_id, self.rng)
            if self.rng.random() < self.magnitude:
                return self.rng.choice(["liked", "replay", "playlist"])
            return action

        raise ValueError(f"unknown scenario: {self.scenario}")

    def choose_user(self, interaction_no):
        if (
            self.scenario == "new_user_population"
            and interaction_no >= self.start_after
            and self.rng.random() < self.magnitude
            and len(self.new_population_users) > 0
        ):
            return self.rng.choice(self.new_population_users)
        return self.rng.choice(self.user_ids)


def run(host, n, seed, scenario, start_after, magnitude, interval):
    sim = LiveUserBehaviorSimulator(
        scenario=scenario,
        start_after=start_after,
        magnitude=magnitude,
        seed=seed,
    )

    with httpx.Client(base_url=host, timeout=10.0) as client:
        for i in range(1, n + 1):
            user_id = sim.choose_user(i)
            rec = client.post("/recommend", json={"user_id": str(user_id)})
            rec.raise_for_status()
            track_id = rec.json()["track_id"]

            action = sim.sample_action(str(user_id), track_id, i)
            fb = client.post(
                "/feedback",
                json={
                    "user_id": str(user_id),
                    "track_id": track_id,
                    "action": action,
                },
            )
            fb.raise_for_status()

            if i % 50 == 0:
                metrics = client.get("/metrics").json()
                print(
                    f"{i}/{n} scenario={scenario} action={action} "
                    f"ctr={metrics['ctr']:.3f} avg_reward={metrics['avg_reward']:.3f}",
                    flush=True,
                )
            if interval > 0:
                time.sleep(interval)

        print(f"final /metrics: {client.get('/metrics').json()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="http://localhost:8000")
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--scenario",
        choices=["normal", "preference_shift", "reward_drift", "new_user_population"],
        default="normal",
    )
    parser.add_argument("--start-after", type=int, default=500)
    parser.add_argument("--magnitude", type=float, default=0.6)
    parser.add_argument("--interval", type=float, default=0.05)
    args = parser.parse_args()
    run(**vars(args))


if __name__ == "__main__":
    main()
