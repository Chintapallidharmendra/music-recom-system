"""get_features(track_id) read API backed by features.parquet.

Loads the whole parquet into an in-memory dict at startup so get_features() is a plain
dict lookup (<5ms warm, no disk I/O per call) -- this is what service/main.py imports.
"""
import logging
import time

import numpy as np
import pandas as pd

from data.build_user_context import _audio_vector

try:
    # Optional: standalone scripts that only import this module for a quick check
    # (see __main__ below) still work even if prometheus_client isn't installed.
    from monitoring.metrics import track_parquet_load
except ImportError:  # pragma: no cover
    from contextlib import contextmanager

    @contextmanager
    def track_parquet_load(_file_label: str):
        yield lambda _rows: None


logger = logging.getLogger(__name__)


class FeatureStore:
    def __init__(self, path: str = "data/features.parquet"):
        with track_parquet_load(path.rsplit("/", 1)[-1]) as record_rows:
            df = pd.read_parquet(path)
            record_rows(len(df))
        self._features = {
            row["track_id"]: _audio_vector(row).astype(np.float32)
            for _, row in df.iterrows()
        }
        self._genres = dict(zip(df["track_id"], df["genre"]))
        self.n_features = len(next(iter(self._features.values())))
        self.track_ids = list(self._features.keys())
        logger.info(
            "FeatureStore loaded track_count=%d feature_dim=%d source=%s",
            len(self._features), self.n_features, path,
        )

    def get_features(self, track_id: str) -> np.ndarray:
        return self._features[track_id]

    def get_genre(self, track_id: str) -> str:
        return self._genres[track_id]

    def __contains__(self, track_id: str) -> bool:
        return track_id in self._features

    def __len__(self) -> int:
        return len(self._features)


if __name__ == "__main__":
    store = FeatureStore()
    print(f"loaded {len(store)} tracks, {store.n_features}-dim audio vectors")

    tid = store.track_ids[0]
    n = 10_000
    start = time.perf_counter()
    for _ in range(n):
        store.get_features(tid)
    elapsed_ms = (time.perf_counter() - start) * 1000 / n
    print(f"get_features() avg latency over {n} calls: {elapsed_ms:.5f} ms")
    assert elapsed_ms < 5, "must be under 5ms warm-start per contracts/parquet_schema.md"
