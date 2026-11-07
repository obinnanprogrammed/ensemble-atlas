import heapq

import numpy as np

from .embeddings import Embedder, normalize
from .storage import Storage


def search(storage: Storage, embedder: Embedder, query: str, top_k: int = 5,
           windows_per_recording: int = 3) -> list[dict]:
    """Rank recordings by their best completed window, with ranked timestamps."""
    if not query.strip():
        raise ValueError("Search text cannot be blank.")
    if top_k < 1:
        raise ValueError("top-k must be at least one.")
    if windows_per_recording < 1:
        raise ValueError("windows-per-recording must be at least one.")
    rows = storage.completed()
    if not rows:
        return []
    for row in rows:
        if any(row[key] != value for key, value in storage.config.contract.items()):
            raise ValueError("Index contains incompatible model/revision or preprocessing metadata.")
    query_vector = normalize(embedder.text(query), storage.config.dimension)
    recordings = {}
    for row in rows:
        try:
            raw = np.load(row["artifact_path"], allow_pickle=False)
            vector = normalize(raw, row["dimension"])
        except (OSError, ValueError) as error:
            raise ValueError(f"Cannot read embedding for {row['chunk_id']}: {error}") from error
        recording = recordings.setdefault(row["recording_id"], {
            "recording_id": row["recording_id"], "original_name": row["original_name"],
            "audio_path": row["audio_path"], "matches": [],
        })
        recording["matches"].append({
            "chunk_id": row["chunk_id"], "start": row["start"], "end": row["end"],
            "score": float(np.dot(query_vector, vector)),
        })
    hits = []
    for recording in recordings.values():
        recording["matches"] = heapq.nlargest(windows_per_recording, recording["matches"],
                                             key=lambda match: match["score"])
        recording["score"] = recording["matches"][0]["score"]
        hits.append(recording)
    return heapq.nlargest(top_k, hits, key=lambda row: row["score"])
