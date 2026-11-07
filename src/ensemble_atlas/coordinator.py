"""Serializable coordinator boundary; workers never issue SQL."""

from pathlib import Path
import time
from typing import Protocol
import uuid

from .embeddings import normalize
from .storage import Storage


class Coordinator(Protocol):
    def register(self, worker_id: str) -> None: ...
    def claim(self, worker_id: str) -> dict | None: ...
    def heartbeat(self, job_id: str, attempt_token: str) -> bool: ...
    def complete(self, job_id: str, attempt_token: str, embedding: list[float], elapsed_seconds: float) -> bool: ...
    def fail(self, job_id: str, attempt_token: str, error: str) -> bool: ...
    def status(self) -> dict: ...


class LocalCoordinator:
    def __init__(self, storage: Storage, clock=time.time):
        self.storage = storage
        self.config = storage.config
        self.clock = clock

    def ingest(self, source: Path) -> dict:
        return self.storage.ingest(source)

    def register(self, worker_id: str) -> None:
        if not worker_id.strip():
            raise ValueError("Worker ID cannot be blank.")
        with self.storage.connection(write=True) as db:
            db.execute("INSERT INTO workers VALUES (?, ?) ON CONFLICT(id) DO UPDATE SET last_seen=excluded.last_seen",
                       (worker_id, self.clock()))

    def claim(self, worker_id: str) -> dict | None:
        with self.storage.connection(write=True) as db:
            now = self.clock()
            db.execute("""UPDATE jobs SET state=CASE WHEN attempt_count >= ? THEN 'failed' ELSE 'pending' END,
                          lease_expiration=NULL, last_error='Lease expired'
                          WHERE state='leased' AND lease_expiration <= ?""", (self.config.max_attempts, now))
            if db.execute("SELECT 1 FROM workers WHERE id=?", (worker_id,)).fetchone() is None:
                raise ValueError("Register the worker before claiming work.")
            db.execute("UPDATE workers SET last_seen=? WHERE id=?", (now, worker_id))
            if db.execute("SELECT 1 FROM jobs WHERE worker_id=? AND state='leased'", (worker_id,)).fetchone():
                raise ValueError(f"Worker ID {worker_id!r} already has an active chunk; use a distinct ID.")
            row = db.execute("""SELECT j.*, r.audio_path FROM jobs j JOIN recordings r ON r.id=j.recording_id
                                WHERE j.state='pending' AND j.attempt_count < ?
                                ORDER BY j.recording_id, j.start LIMIT 1""", (self.config.max_attempts,)).fetchone()
            if row is None:
                return None
            token = uuid.uuid4().hex
            expiration = now + self.config.lease_seconds
            db.execute("""UPDATE jobs SET state='leased', attempt_count=attempt_count+1,
                          attempt_token=?, worker_id=?, lease_expiration=? WHERE id=?""",
                       (token, worker_id, expiration, row["id"]))
            return {"job_id": row["id"], "audio_path": row["audio_path"], "start": row["start"],
                    "end": row["end"], "attempt_token": token, "lease_expiration": expiration}

    def _active(self, db, job_id, token):
        return db.execute("""SELECT * FROM jobs WHERE id=? AND state='leased'
                             AND attempt_token=? AND lease_expiration > ?""",
                          (job_id, token, self.clock())).fetchone()

    def heartbeat(self, job_id: str, attempt_token: str) -> bool:
        with self.storage.connection(write=True) as db:
            row = self._active(db, job_id, attempt_token)
            if row is None:
                return False
            now = self.clock()
            db.execute("UPDATE jobs SET lease_expiration=? WHERE id=?", (now + self.config.lease_seconds, job_id))
            db.execute("UPDATE workers SET last_seen=? WHERE id=?", (now, row["worker_id"]))
            return True

    def complete(self, job_id: str, attempt_token: str, embedding: list[float], elapsed_seconds: float = 0) -> bool:
        vector = normalize(embedding, self.config.dimension)
        # Reject stale calls before writing an artifact. Recheck after the disk write.
        with self.storage.connection() as db:
            row = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
            if row is not None and row["state"] == "completed" and row["attempt_token"] == attempt_token:
                return True
            if self._active(db, job_id, attempt_token) is None:
                return False
        # Unique per submission too, so a duplicate call cannot replace accepted bytes.
        path = self.storage.persist_embedding(job_id, f"{attempt_token}-{uuid.uuid4().hex}", vector)
        with self.storage.connection(write=True) as db:
            row = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
            if row["state"] == "completed" and row["attempt_token"] == attempt_token:
                return True
            if self._active(db, job_id, attempt_token) is None:
                return False
            db.execute("INSERT INTO embeddings VALUES (?, ?, ?, ?, ?, ?)",
                       (job_id, str(path), self.config.dimension, self.config.model_id,
                        self.config.model_revision, self.config.preprocessing_version))
            db.execute("UPDATE jobs SET state='completed', lease_expiration=NULL, elapsed_seconds=? WHERE id=?",
                       (elapsed_seconds, job_id))
            return True

    def fail(self, job_id: str, attempt_token: str, error: str) -> bool:
        with self.storage.connection(write=True) as db:
            row = self._active(db, job_id, attempt_token)
            if row is None:
                return False
            state = "failed" if row["attempt_count"] >= self.config.max_attempts else "pending"
            db.execute("UPDATE jobs SET state=?, lease_expiration=NULL, last_error=? WHERE id=?",
                       (state, error[:2000], job_id))
            return True

    def status(self) -> dict:
        with self.storage.connection() as db:
            # A single read snapshot keeps the counts and worker view consistent.
            db.execute("BEGIN")
            counts = dict.fromkeys(("pending", "leased", "completed", "failed"), 0)
            counts.update({row[0]: row[1] for row in db.execute("SELECT state, COUNT(*) FROM jobs GROUP BY state")})
            workers = [dict(row) for row in db.execute("""SELECT w.id, w.last_seen, j.id AS active_job,
                       j.lease_expiration FROM workers w LEFT JOIN jobs j
                       ON j.worker_id=w.id AND j.state='leased' ORDER BY w.id""")]
            return {"counts": counts, "recordings": db.execute("SELECT COUNT(*) FROM recordings").fetchone()[0],
                    "workers": workers,
                    "retries": db.execute("SELECT COALESCE(SUM(MAX(attempt_count-1, 0)), 0) FROM jobs").fetchone()[0],
                    "chunk_seconds": db.execute("SELECT COALESCE(SUM(elapsed_seconds), 0) FROM jobs WHERE state='completed'").fetchone()[0],
                    "errors": [dict(row) for row in db.execute("SELECT id, state, last_error FROM jobs WHERE last_error IS NOT NULL")]}
