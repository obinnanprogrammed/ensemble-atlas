from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import sqlite3
import uuid

import numpy as np

from .audio import inspect_audio, windows
from .config import Config


SCHEMA = """
CREATE TABLE IF NOT EXISTS index_contract (
    id INTEGER PRIMARY KEY CHECK (id = 1), value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS recordings (
    id TEXT PRIMARY KEY, original_name TEXT NOT NULL, audio_path TEXT NOT NULL,
    duration REAL NOT NULL CHECK (duration > 0), ingestion_state TEXT NOT NULL CHECK (ingestion_state = 'ready')
);
CREATE TABLE IF NOT EXISTS workers (
    id TEXT PRIMARY KEY, last_seen REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY, recording_id TEXT NOT NULL REFERENCES recordings(id),
    start REAL NOT NULL CHECK (start >= 0), end REAL NOT NULL CHECK (end > start),
    state TEXT NOT NULL DEFAULT 'pending' CHECK (state IN ('pending', 'leased', 'completed', 'failed')),
    attempt_count INTEGER NOT NULL DEFAULT 0 CHECK (attempt_count >= 0),
    attempt_token TEXT, worker_id TEXT REFERENCES workers(id), lease_expiration REAL,
    last_error TEXT, elapsed_seconds REAL,
    UNIQUE (recording_id, start)
);
CREATE INDEX IF NOT EXISTS jobs_scheduling ON jobs(state, lease_expiration);
CREATE TABLE IF NOT EXISTS embeddings (
    chunk_id TEXT PRIMARY KEY REFERENCES jobs(id), artifact_path TEXT NOT NULL UNIQUE,
    dimension INTEGER NOT NULL CHECK (dimension > 0), model_id TEXT NOT NULL,
    model_revision TEXT NOT NULL, preprocessing_version TEXT NOT NULL
);
"""


def sync_directory(directory: Path) -> None:
    """Make a renamed artifact durable before committing its DB reference."""
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class Storage:
    def __init__(self, config: Config):
        self.config = config
        config.data_dir.mkdir(parents=True, exist_ok=True)
        self.database = config.data_dir / "atlas.sqlite3"
        with self.connection() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript(SCHEMA)
        with self.connection(write=True) as db:
            expected = json.dumps(config.contract, sort_keys=True)
            db.execute("INSERT OR IGNORE INTO index_contract VALUES (1, ?)", (expected,))
            if db.execute("SELECT value FROM index_contract WHERE id=1").fetchone()[0] != expected:
                raise ValueError("Incompatible model/revision or preprocessing version. Use a separate --data-dir.")

    @contextmanager
    def connection(self, *, write=False):
        # Connections are created and closed in their owning process/thread.
        db = sqlite3.connect(self.database, timeout=10, isolation_level=None)
        db.row_factory = sqlite3.Row
        try:
            db.execute("PRAGMA foreign_keys=ON")
            db.execute("PRAGMA busy_timeout=10000")
            if write:
                db.execute("BEGIN IMMEDIATE")
            yield db
            if write:
                db.commit()
        except BaseException:
            if write:
                db.rollback()
            raise
        finally:
            db.close()

    def ingest(self, source: Path) -> dict:
        if not source.is_file():
            raise ValueError(f"Recording does not exist or is not a file: {source}")
        recording_id = uuid.uuid4().hex
        directory = self.config.data_dir / "audio"
        directory.mkdir(exist_ok=True)
        target = directory / f"{recording_id}{source.suffix.lower()}"
        temporary = directory / f"{recording_id}.tmp"
        try:
            with source.open("rb") as reader, temporary.open("xb") as writer:
                shutil.copyfileobj(reader, writer)
                writer.flush()
                os.fsync(writer.fileno())
            frames, rate = inspect_audio(temporary)
            os.replace(temporary, target)
            sync_directory(directory)
            count = 0
            with self.connection(write=True) as db:
                db.execute("INSERT INTO recordings VALUES (?, ?, ?, ?, 'ready')",
                           (recording_id, source.name, str(target), frames / rate))
                for start, end in windows(frames, rate):
                    db.execute("INSERT INTO jobs (id, recording_id, start, end) VALUES (?, ?, ?, ?)",
                               (f"{recording_id}-{count}", recording_id, start, end))
                    count += 1
            return {"recording_id": recording_id, "chunks": count, "duration": frames / rate}
        except BaseException:
            temporary.unlink(missing_ok=True)
            target.unlink(missing_ok=True)
            raise

    def persist_embedding(self, job_id: str, token: str, vector: np.ndarray) -> Path:
        directory = self.config.data_dir / "embeddings"
        directory.mkdir(exist_ok=True)
        # Only internal IDs/tokens reach here; each claim owns a different path.
        target = directory / f"{job_id}-{token}.npy"
        temporary = directory / f"{job_id}-{token}-{uuid.uuid4().hex}.tmp"
        try:
            with temporary.open("xb") as output:
                np.save(output, vector, allow_pickle=False)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, target)
            sync_directory(directory)
        finally:
            temporary.unlink(missing_ok=True)
        return target

    def completed(self) -> list[dict]:
        with self.connection() as db:
            return [dict(row) for row in db.execute("""
                SELECT j.id AS chunk_id, j.recording_id, r.original_name, r.audio_path,
                       j.start, j.end, e.artifact_path, e.dimension, e.model_id,
                       e.model_revision, e.preprocessing_version
                FROM jobs j JOIN recordings r ON r.id=j.recording_id
                JOIN embeddings e ON e.chunk_id=j.id WHERE j.state='completed'
                ORDER BY j.recording_id, j.start
            """)]
