"""Batch librosa feature extraction over FMA-small.

Writes data/features.parquet matching contracts/parquet_schema.md exactly:
track_id, genre, mfcc_mean[20], mfcc_var[20], chroma_mean[12], tempo, contrast_mean[7].

Run once; cache the output. Re-run only if extraction logic changes.
"""
import argparse
import os
import subprocess
# Must be set before numpy/librosa import spin up BLAS threads, otherwise
# multiprocessing.Pool workers oversubscribe cores against each other.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import multiprocessing as mp

import librosa
import numpy as np
import pandas as pd

FMA_METADATA_DIR = "datasets/archive/fma_metadata/fma_metadata"
FMA_AUDIO_DIR = "datasets/archive/fma_small/fma_small"


def load_small_subset() -> pd.DataFrame:
    tracks = pd.read_csv(f"{FMA_METADATA_DIR}/tracks.csv", index_col=0, header=[0, 1, 2])
    tracks.columns = tracks.columns.droplevel(2)
    small = tracks[tracks[("set", "subset")] == "small"]
    return small[[("track", "genre_top")]].droplevel(0, axis=1).rename(
        columns={"genre_top": "genre"}
    )

def load_audio_ffmpeg(path, sr=22050):
    command = [
        "ffmpeg",
        "-v", "error",
        "-i", str(path),
        "-f", "f32le",
        "-acodec", "pcm_f32le",
        "-ac", "1",
        "-ar", str(sr),
        "pipe:1",
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    if result.returncode != 0:
        error = result.stderr.decode(
            "utf-8",
            errors="replace",
        )

        raise RuntimeError(
            f"FFmpeg failed: {error}"
        )

    y = np.frombuffer(
        result.stdout,
        dtype=np.float32,
    )

    if y.size == 0:
        raise RuntimeError(
            "FFmpeg returned empty audio"
        )

    return y, sr

def track_audio_path(track_id: int) -> str:
    tid_str = f"{track_id:06d}"
    return os.path.join(FMA_AUDIO_DIR, tid_str[:3], f"{tid_str}.mp3")


def extract_features(path: str) -> dict:
    y, sr = load_audio_ffmpeg(path=path)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    # tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    onset_env = librosa.onset.onset_strength(
    y=y,
    sr=sr
    )

    tempo = librosa.feature.tempo(
    onset_envelope=onset_env,
    sr=sr
    )[0]
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    return {
        "mfcc_mean": mfcc.mean(axis=1).astype(np.float32),
        "mfcc_var": mfcc.var(axis=1).astype(np.float32),
        "chroma_mean": chroma.mean(axis=1).astype(np.float32),
        # "tempo": float(np.asarray(tempo).reshape(-1)[0]),
        "tempo": float(tempo),
        "contrast_mean": contrast.mean(axis=1).astype(np.float32),
    }


def _process_one(args):
    track_id, genre = args
    path = track_audio_path(track_id)
    try:
        feats = extract_features(path)
    except Exception as exc:  # noqa: BLE001 - report and skip, don't kill the whole batch
        return {"track_id": str(track_id), "error": str(exc)}
    return {
        "track_id": str(track_id),
        "genre": genre,
        **feats,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="process only N tracks (testing)")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--out", default="data/features.parquet")
    args = parser.parse_args()

    subset = load_small_subset()
    items = list(subset["genre"].items())
    if args.limit:
        items = items[: args.limit]

    print(f"extracting features for {len(items)} tracks with {args.workers} workers...")
    with mp.Pool(args.workers) as pool:
        results = pool.map(_process_one, items, chunksize=8)

    errors = [r for r in results if "error" in r]
    ok = [r for r in results if "error" not in r]
    if errors:
        print(f"WARNING: {len(errors)} tracks failed extraction, e.g. {errors[0]}")

    df = pd.DataFrame(ok)
    df.to_parquet(args.out, index=False)
    print(f"wrote {len(df)} rows to {args.out}")


if __name__ == "__main__":
    main()
