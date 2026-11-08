"""Opt-in real-model test; never runs during unittest discovery.

Supply a <=10-second clip with an identifiable sound, a matching text query,
and an unrelated query. The checkpoint is downloaded on first use.
"""

import argparse
from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import time

import numpy as np
import soundfile as sf

from ensemble_atlas.audio import inspect_audio, read_window
from ensemble_atlas.config import Config, SAMPLE_RATE
from ensemble_atlas.coordinator import LocalCoordinator
from ensemble_atlas.embeddings import ClapEmbedder
from ensemble_atlas.search import search
from ensemble_atlas.storage import Storage
from ensemble_atlas.worker import run_worker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--query", required=True)
    parser.add_argument("--distractor", required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Model cache location; index is temporary.")
    args = parser.parse_args()
    frames, rate = inspect_audio(args.fixture)
    duration = frames / rate
    if duration > 10:
        parser.error("Use a fixture no longer than 10 seconds.")
    config = Config(data_dir=args.data_dir)
    started = time.monotonic()
    embedder = ClapEmbedder(config)
    model_seconds = time.monotonic() - started
    with TemporaryDirectory(prefix="atlas-smoke-") as temporary:
        root = Path(temporary)
        # Put a known event after ten seconds of silence to test timestamp retrieval.
        clip = read_window(str(args.fixture), 0, duration)[:round(duration * SAMPLE_RATE)]
        recording = root / "silence-then-event.wav"
        sf.write(recording, np.concatenate([np.zeros(10 * SAMPLE_RATE, dtype=np.float32), clip]), SAMPLE_RATE)
        local_config = replace(config, data_dir=root / "index")
        storage = Storage(local_config)
        coordinator = LocalCoordinator(storage)
        ingested = coordinator.ingest(recording)
        summary = run_worker(coordinator, local_config, "smoke-worker", embedder)
        # Reopen SQLite, proving search does not rely on the worker's in-memory state.
        reopened = Storage(local_config)
        hits = search(reopened, embedder, args.query, top_k=10)
        distractors = search(reopened, embedder, args.distractor, top_k=10)
        audio_hits = search(reopened, embedder, audio=args.fixture, top_k=10)
        assert summary["completed"] == ingested["chunks"], summary
        assert len(hits) == 1, hits
        matches = hits[0]["matches"]
        assert matches[0]["end"] > 10 and matches[0]["start"] >= 5, hits
        assert hits[0]["score"] > distractors[0]["score"], (hits, distractors)
        assert hits[0]["score"] > next(match["score"] for match in matches if match["start"] == 0), hits
        audio_matches = audio_hits[0]["matches"]
        assert audio_matches[0]["end"] > 10 and audio_matches[0]["start"] >= 5, audio_hits
        assert audio_hits[0]["score"] > next(match["score"] for match in audio_matches if match["start"] == 0), audio_hits
        print(json.dumps({"model_load_seconds": model_seconds, "worker": summary,
                          "query": args.query, "distractor": args.distractor,
                          "recording_score": hits[0]["score"], "matches": matches,
                          "distractor_best_score": distractors[0]["score"],
                          "audio_query_matches": audio_matches}, indent=2))
        print("PASS: real projected embeddings, persistence, and timestamp retrieval.")
        print("Inspect the fixture and results manually to assess semantic relevance.")


if __name__ == "__main__":
    main()
