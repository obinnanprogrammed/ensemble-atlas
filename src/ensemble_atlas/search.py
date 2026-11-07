import heapq

import numpy as np

from .embeddings import Embedder, normalize
from .storage import Storage


def search(storage: Storage, embedder: Embedder, query: str, top_k: int = 5) -> list[dict]:
    if not query.strip():
        raise ValueError("Search text cannot be blank.")
    if top_k < 1:
        raise ValueError("top-k must be at least one.")
    rows = storage.completed()
    if not rows:
        return []
    for row in rows:
        if any(row[key] != value for key, value in storage.config.contract.items()):
            raise ValueError("Index contains incompatible model/revision or preprocessing metadata.")
    query_vector = normalize(embedder.text(query), storage.config.dimension)
    hits = []
    for row in rows:
        try:
            raw = np.load(row["artifact_path"], allow_pickle=False)
            vector = normalize(raw, row["dimension"])
        except (OSError, ValueError) as error:
            raise ValueError(f"Cannot read embedding for {row['chunk_id']}: {error}") from error
        row["score"] = float(np.dot(query_vector, vector))
        hits.append(row)
    return heapq.nlargest(top_k, hits, key=lambda row: row["score"])
