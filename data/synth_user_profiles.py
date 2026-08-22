"""Generate synthetic user genre-affinity profiles.

The single source of ground truth for "what a synthetic user likes" — see
contracts/synthetic_data.md. Both generate_synthetic_logs.py and
bandit/reward_simulator.py read data/user_profiles.parquet produced here;
neither should recompute its own affinity model.
"""

import argparse

import numpy as np
import pandas as pd

GENRES = [
    "Electronic",
    "Experimental",
    "Folk",
    "Hip-Hop",
    "Instrumental",
    "International",
    "Pop",
    "Rock",
]


def generate_user_profiles(n_users: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # Dirichlet alpha < 1 -> peaked preferences (most users favor a few genres),
    # rather than uniform(1) which would make every user near-uniform across genres.
    affinities = rng.dirichlet(alpha=np.full(len(GENRES), 0.5), size=n_users)
    novelty_bias = rng.beta(a=2, b=8, size=n_users)  # mostly low, occasional exploratory users
    return pd.DataFrame(
        {
            "user_id": [f"user_{i:06d}" for i in range(n_users)],
            "genre_affinity": list(affinities.astype(np.float32)),
            "novelty_bias": novelty_bias.astype(np.float32),
        }
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-users", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="data/user_profiles.parquet")
    args = parser.parse_args()

    df = generate_user_profiles(args.n_users, args.seed)
    df.to_parquet(args.out, index=False)
    print(f"wrote {len(df)} user profiles to {args.out}")
    print(f"genre order: {GENRES}")


if __name__ == "__main__":
    main()
