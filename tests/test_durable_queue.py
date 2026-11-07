"""Failure-boundary regressions for the durable queue (no CLAP downloads)."""
from dataclasses import replace
from contextlib import contextmanager
import math
import multiprocessing
import sqlite3
import time
from threading import Barrier
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock, patch

import numpy as np

from ensemble_atlas.coordinator import LocalCoordinator
from ensemble_atlas.storage import Storage
from ensemble_atlas.worker import run_worker
from test_vertical_slice import AudioTestCase, FakeEmbedder


def pause_after_artifact(config, job, ready):
    storage = Storage(config)
    original = storage.persist_embedding
    def persist(*args):
        path = original(*args)
        ready.set()
        time.sleep(60)
        return path
    storage.persist_embedding = persist
    LocalCoordinator(storage, clock=lambda: 1000.0).complete(
        job["job_id"], job["attempt_token"], [1, 0, 0])


class DurableQueueTests(AudioTestCase):
    def setUp(self):
        super().setUp()
        self.now = 1000.0
        self.coordinator = LocalCoordinator(self.storage, clock=lambda: self.now)
        self.coordinator.ingest(self.audio_file(duration=1))
        self.coordinator.register("one")
        self.job = self.coordinator.claim("one")

    def complete(self, vector=(1, 0, 0), **kwargs):
        return self.coordinator.complete(self.job["job_id"], self.job["attempt_token"], vector, **kwargs)

    def test_accepted_attempt_acknowledges_retry_without_revalidating_payload(self):
        self.assertTrue(self.complete())
        with patch.object(self.storage, "persist_embedding", side_effect=AssertionError("rewritten")):
            self.assertTrue(self.complete([float("nan")]))
        self.assertEqual(len(self.storage.completed()), 1)

    def test_stale_completion_is_rejected_before_payload_validation(self):
        self.now += self.config.lease_seconds
        self.assertFalse(self.complete([float("nan")]))

    def test_heartbeat_uses_one_time_for_validation_and_extension(self):
        expiration = self.job["lease_expiration"]
        self.coordinator.clock = Mock(
            side_effect=[expiration - 0.01, expiration + 10])
        self.assertTrue(self.coordinator.heartbeat(self.job["job_id"], self.job["attempt_token"]))
        with self.storage.connection() as db:
            renewed = db.execute("SELECT lease_expiration FROM jobs WHERE id=?", (self.job["job_id"],)).fetchone()[0]
        self.assertEqual(renewed, expiration - 0.01 + self.config.lease_seconds)

    def test_invalid_elapsed_time_cannot_poison_persisted_status(self):
        for elapsed in (-1, math.inf, math.nan):
            with self.subTest(elapsed=elapsed), self.assertRaises(ValueError):
                self.complete(elapsed_seconds=elapsed)
        self.assertEqual(self.storage.completed(), [])

    def test_expiration_during_artifact_write_cannot_publish_result(self):
        original = self.storage.persist_embedding
        def persist(*args):
            path = original(*args)
            self.now += self.config.lease_seconds
            return path
        with patch.object(self.storage, "persist_embedding", side_effect=persist):
            self.assertFalse(self.complete())
        reopened = LocalCoordinator(Storage(self.config), clock=lambda: self.now)
        self.assertEqual(reopened.status()["counts"]["completed"], 0)
        current = reopened.claim("one")
        self.assertTrue(reopened.complete(current["job_id"], current["attempt_token"], [0, 1, 0]))
        np.testing.assert_array_equal(np.load(self.storage.completed()[0]["artifact_path"]), [0, 1, 0])

    def test_concurrent_duplicate_completions_accept_one_immutable_artifact(self):
        barrier = Barrier(2)
        original = self.storage.persist_embedding
        def persist(*args):
            path = original(*args)
            barrier.wait(timeout=10)
            return path
        with patch.object(self.storage, "persist_embedding", side_effect=persist):
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(self.complete, vector) for vector in ([1, 0, 0], [0, 1, 0])]
                self.assertTrue(all(f.result(timeout=15) for f in futures))
        self.assertEqual(len(Storage(self.config).completed()), 1)

    def test_database_failure_rolls_back_result_and_job_together(self):
        original = self.storage.connection
        @contextmanager
        def connection(*, write=False):
            with original(write=write) as db:
                yield db
                if write:
                    raise sqlite3.OperationalError("simulated commit failure")
        with patch.object(self.storage, "connection", side_effect=connection):
            with self.assertRaises(sqlite3.OperationalError):
                self.complete()
        self.assertEqual(Storage(self.config).completed(), [])
        self.assertEqual(self.coordinator.status()["counts"]["leased"], 1)
        self.assertTrue(self.complete())

    def test_kill_after_rename_before_database_acceptance_is_recoverable(self):
        context = multiprocessing.get_context("spawn")
        ready = context.Event()
        process = context.Process(target=pause_after_artifact, args=(self.config, self.job, ready))
        try:
            process.start()
            self.assertTrue(ready.wait(20), "child never persisted its artifact")
            process.terminate()
            process.join(10)
            self.assertFalse(process.is_alive())
            self.assertEqual(Storage(self.config).completed(), [])
            self.now += self.config.lease_seconds
            recovered = LocalCoordinator(Storage(self.config), clock=lambda: self.now)
            job = recovered.claim("one")
            self.assertTrue(recovered.complete(job["job_id"], job["attempt_token"], [0, 1, 0]))
            self.assertFalse(self.complete())
            self.assertEqual(len(self.storage.completed()), 1)
            np.testing.assert_array_equal(np.load(self.storage.completed()[0]["artifact_path"]), [0, 1, 0])
        finally:
            if process.is_alive():
                process.terminate()
                process.join(5)

    def test_lowering_attempt_limit_does_not_strand_pending_job(self):
        self.coordinator.fail(self.job["job_id"], self.job["attempt_token"], "retry")
        reopened = LocalCoordinator(Storage(replace(self.config, max_attempts=1)), clock=lambda: self.now)
        self.assertIsNone(reopened.claim("one"))
        self.assertEqual(reopened.status()["counts"]["failed"], 1)

    def test_heartbeat_error_exits_worker_with_recoverable_lease(self):
        # Release the setup claim, then simulate a renewal transport/DB failure.
        self.coordinator.fail(self.job["job_id"], self.job["attempt_token"], "setup")
        config = replace(self.config, lease_seconds=1, heartbeat_seconds=0.01)
        coordinator = LocalCoordinator(Storage(config))
        class SlowEmbedder(FakeEmbedder):
            def audio(self, samples):
                time.sleep(0.05)
                return super().audio(samples)
        with patch.object(coordinator, "heartbeat", side_effect=OSError("unavailable")):
            summary = run_worker(coordinator, config, "one", SlowEmbedder(), lambda _: None)
        self.assertEqual(summary["completed"], 0)
        self.assertEqual(summary["discarded"], 1)
        self.assertEqual(coordinator.status()["counts"]["leased"], 1)


class QueueConfigTests(AudioTestCase):
    def test_nonfinite_timings_are_rejected(self):
        for value in (math.inf, math.nan):
            with self.subTest(value=value), self.assertRaises(ValueError):
                replace(self.config, lease_seconds=value)

    def test_attempt_limit_requires_positive_integer(self):
        for value in (True, 1.5, math.inf):
            with self.subTest(value=value), self.assertRaises(ValueError):
                replace(self.config, max_attempts=value)
