# Synthetic interaction/reward layer — schema & interfaces (FROZEN)

Added per `dataset_reconciliation.md`'s NO-GO decision on the real Last.fm↔FMA join. This is
the one shared ground-truth model that both Track A's historical-log generator and Track B's
reward simulator read from — neither should define its own affinity logic.

## `data/user_profiles.parquet` — schema

| Column           | Type        | Notes                                                        |
|-------------------|-------------|----------------------------------------------------------------|
| `user_id`         | string      | `user_%06d`, primary key                                       |
| `genre_affinity`  | float32[8]  | Dirichlet-sampled over the 8 FMA-small genres (see `parquet_schema.md` for canonical order), sums to 1.0 |
| `novelty_bias`    | float32     | in [0, 1]; probability weight given to off-affinity exploration when sampling plays/rewards |

Generated once by `data/synth_user_profiles.py`. This file is the single source of truth for
"what a synthetic user likes" — `generate_synthetic_logs.py` and `reward_simulator.py` both
read it and must not recompute their own affinity vectors.

## `bandit/reward_simulator.py` — interface

```python
def sample_action(user_id: str, track_id: str) -> str:
    """Returns one of: completed, liked, playlist, skip, replay (see kafka_topics.md reward map).
    Combines the user's genre_affinity (from user_profiles.parquet) for the track's genre with
    a content-similarity term against the track's audio features, plus novelty_bias-weighted
    noise so the same (user, track) pair isn't perfectly deterministic."""

def expected_reward(user_id: str, track_id: str) -> float:
    """Deterministic ground-truth expected reward for a (user, track) pair, used by
    replay_evaluator.py for counterfactual/offline evaluation without resampling noise."""
```

## Why this exists

`build_user_context.py` (Track A) needs historical plays to compute recency-weighted genre
affinity. `replay_evaluator.py` (Track B) needs ground-truth reward for offline replay. Since
the real Last.fm join is NO-GO, both need synthetic data — and if each track invented its own
affinity model independently, a policy could look good against Track B's ground truth while
building context from a Track A history that implies a *different* ground truth. This contract
prevents that by making `user_profiles.parquet` the one place affinity is defined.
