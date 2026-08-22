"""Ground-truth reward/preference model for synthetic users.

Implements the interface frozen in contracts/synthetic_data.md. Imports the SAME
genre_affinity ground truth as data/generate_synthetic_logs.py (from user_profiles.parquet) --
does not recompute its own -- so history and reward stay consistent. Adds a content-similarity
term against the track's audio features via a per-user latent taste vector (deterministic
function of user_id) so two tracks in the same genre aren't equally rewarding.

Used by bandit/replay_evaluator.py for offline replay AND imported live by service/main.py /
service/demo_loadgen.py, since there are no real users to click /feedback.
"""

import numpy as np
import pandas as pd

from data.synth_user_profiles import GENRES

# Reward mapping frozen in contracts/kafka_topics.md -- single canonical definition,
# service/main.py imports this rather than redefining it.
REWARD_MAP = {
    "completed": 1.0,
    "liked": 2.0,
    "playlist": 3.0,
    "skip": -1.0,
    "replay": 2.0,
}

_GENRE_SCORE_WEIGHT = 0.7
_CONTENT_SCORE_WEIGHT = 0.3
_ACTION_NOISE_STD = 0.15


class RewardSimulator:
    def __init__(
        self,
        profiles_path: str = "data/user_profiles.parquet",
        features_path: str = "data/features.parquet",
    ):
        profiles = pd.read_parquet(profiles_path)
        self._affinity = {
            row["user_id"]: np.asarray(row["genre_affinity"], dtype=np.float64)
            for _, row in profiles.iterrows()
        }
        self._novelty = dict(zip(profiles["user_id"], profiles["novelty_bias"]))
        self._genre_idx = {g: i for i, g in enumerate(GENRES)}

        features = pd.read_parquet(features_path)
        self._genre = dict(zip(features["track_id"], features["genre"]))
        self._audio = {
            row["track_id"]: self._normalized_audio_vector(row) for _, row in features.iterrows()
        }
        audio_dim = len(next(iter(self._audio.values())))
        self._audio_dim = audio_dim
        self._user_taste_cache: dict = {}

    @staticmethod
    def _normalized_audio_vector(row: pd.Series) -> np.ndarray:
        # A compact, roughly-comparable-scale content signature -- not the full 60-dim
        # feature vector, just enough to differentiate tracks within a genre.
        mfcc_mean = np.asarray(row["mfcc_mean"], dtype=np.float64)
        chroma_mean = np.asarray(row["chroma_mean"], dtype=np.float64)
        tempo = np.array([float(row["tempo"]) / 200.0])  # rough normalization
        vec = np.concatenate([mfcc_mean[:5] / 100.0, chroma_mean, tempo])
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def _user_taste_vector(self, user_id: str) -> np.ndarray:
        if user_id not in self._user_taste_cache:
            rng = np.random.default_rng(abs(hash(user_id)) % (2**32))
            vec = rng.normal(size=self._audio_dim)
            self._user_taste_cache[user_id] = vec / np.linalg.norm(vec)
        return self._user_taste_cache[user_id]

    def _combined_score(self, user_id: str, track_id: str) -> float:
        """Deterministic score in [0, 1], no sampling noise."""
        affinity = self._affinity.get(user_id)
        genre = self._genre.get(track_id)
        audio = self._audio.get(track_id)
        if affinity is None or genre is None or audio is None:
            return 0.5  # unknown user/track -- neutral score

        genre_score = float(affinity[self._genre_idx[genre]])  # already in [0, 1]-ish
        cos_sim = float(np.dot(self._user_taste_vector(user_id), audio))
        content_score = (cos_sim + 1.0) / 2.0  # rescale [-1, 1] -> [0, 1]

        return _GENRE_SCORE_WEIGHT * genre_score + _CONTENT_SCORE_WEIGHT * content_score

    def expected_reward(self, user_id: str, track_id: str) -> float:
        """Deterministic ground-truth expected reward, used for offline counterfactual
        evaluation without resampling noise."""
        combined = self._combined_score(user_id, track_id)
        return -1.0 + 4.0 * combined  # combined in [0,1] -> reward in [-1, 3]

    def sample_action(self, user_id: str, track_id: str, rng: np.random.Generator = None) -> str:
        """Returns one of: completed, liked, playlist, skip, replay."""
        rng = rng or np.random.default_rng()
        profiles_novelty = self._novelty_bias(user_id)

        if rng.random() < profiles_novelty:
            return rng.choice(list(REWARD_MAP.keys()))

        combined = self._combined_score(user_id, track_id)
        noisy = np.clip(combined + rng.normal(0, _ACTION_NOISE_STD), 0.0, 1.0)

        if noisy > 0.85:
            return "playlist"
        if noisy > 0.65:
            return rng.choice(["liked", "replay"])
        if noisy > 0.35:
            return "completed"
        return "skip"

    def _novelty_bias(self, user_id: str) -> float:
        affinity = self._affinity.get(user_id)
        return 0.1 if affinity is None else self._novelty_lookup(user_id)

    def _novelty_lookup(self, user_id: str) -> float:
        if not hasattr(self, "_novelty"):
            profiles = pd.read_parquet("data/user_profiles.parquet")
            self._novelty = dict(zip(profiles["user_id"], profiles["novelty_bias"]))
        return float(self._novelty.get(user_id, 0.1))


if __name__ == "__main__":
    sim = RewardSimulator()
    profiles = pd.read_parquet("data/user_profiles.parquet")
    features = pd.read_parquet("data/features.parquet")
    user_id = profiles["user_id"].iloc[0]

    from collections import Counter

    rng = np.random.default_rng(0)
    sample_tids = features["track_id"].sample(200, random_state=0)
    actions = [sim.sample_action(user_id, tid, rng) for tid in sample_tids]
    print(f"user {user_id} action distribution over 200 tracks: {Counter(actions)}")

    first_genre = features["genre"].iloc[0]
    same_genre = features[features["genre"] == first_genre]["track_id"].tolist()[:5]
    rewards = [sim.expected_reward(user_id, tid) for tid in same_genre]
    print(f"expected_reward varies within same genre (content term works): {rewards}")
