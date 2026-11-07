from dataclasses import replace
from pathlib import Path
import multiprocessing
from tempfile import TemporaryDirectory
import time
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf
from typer.testing import CliRunner

from ensemble_atlas.audio import read_window, windows
from ensemble_atlas.cli import app, main, timestamp
from ensemble_atlas.config import Config
from ensemble_atlas.coordinator import LocalCoordinator
from ensemble_atlas.embeddings import normalize
from ensemble_atlas.search import search
from ensemble_atlas.storage import Storage
from ensemble_atlas.worker import run_worker


class FakeEmbedder:
    def audio(self, samples):
        # distinguish a silent window from a tone, with no downloaded model
        return np.array([1, 0, 0] if np.max(np.abs(samples)) < 0.01 else [0, 1, 0], dtype=np.float32)

    def text(self, query):
        return np.array([0, 1, 0], dtype=np.float32)


def claim_in_process(config, worker_id, gate, results, hold=False):
    coordinator = LocalCoordinator(Storage(config), clock=lambda: 1000.0)
    coordinator.register(worker_id)
    gate.wait(10)
    results.put(coordinator.claim(worker_id))
    if hold:
        time.sleep(60)


class AudioTestCase(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config = Config(data_dir=self.root / "data", dimension=3)
        self.storage = Storage(self.config)
        self.coordinator = LocalCoordinator(self.storage)

    def audio_file(self, duration=11, rate=16000):
        samples = np.zeros(round(duration * rate), dtype=np.float32)
        start = min(len(samples), 10 * rate)
        samples[start:] = 0.5 * np.sin(2 * np.pi * 440 * np.arange(len(samples) - start) / rate)
        path = self.root / "recording.wav"
        sf.write(path, samples, rate)
        return path


class VerticalSliceTests(AudioTestCase):
    def test_ingest_worker_search_survives_new_instances_and_missing_source(self):
        source = self.audio_file()
        result = self.coordinator.ingest(source)
        source.unlink()
        self.assertEqual(result["chunks"], 3)
        summary = run_worker(self.coordinator, self.config, "one", FakeEmbedder(), lambda _: None)
        self.assertEqual(summary["completed"], 3)
        reopened = Storage(self.config)
        hits = search(reopened, FakeEmbedder(), "tone", top_k=2)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["score"], 1)
        self.assertEqual({match["start"] for match in hits[0]["matches"][:2]}, {5, 10})
        self.assertTrue(all(match["end"] == 11 for match in hits[0]["matches"][:2]))
        self.assertEqual(LocalCoordinator(reopened).status()["counts"]["completed"], 3)
        self.assertEqual(reopened.completed()[0]["original_name"], "recording.wav")
        for row in reopened.completed():
            self.assertAlmostEqual(float(np.linalg.norm(np.load(row["artifact_path"]))), 1)

    def test_tail_policy(self):
        for duration, expected in [(1, [(0, 1)]), (5, [(0, 5)]), (10, [(0, 10), (5, 10)]),
                                   (11, [(0, 10), (5, 11), (10, 11)])]:
            self.assertEqual(list(windows(duration * 16000, 16000)), expected)

    def test_short_stereo_resamples_to_mono_and_zero_pads(self):
        source = self.root / "stereo.wav"
        sf.write(source, np.full((22050, 2), [0.2, 0.6], dtype=np.float32), 22050, subtype="FLOAT")
        samples = read_window(str(source), 0, 1)
        self.assertEqual(samples.shape, (480000,))
        self.assertEqual(samples.dtype, np.float32)
        self.assertAlmostEqual(float(np.mean(samples[100:47000])), 0.4, places=4)
        self.assertTrue(np.all(samples[48000:] == 0))

    def test_empty_invalid_and_missing_inputs_leave_no_recordings(self):
        empty = self.root / "empty.wav"
        sf.write(empty, np.empty(0), 48000)
        broken = self.root / "broken.wav"
        broken.write_bytes(b"not audio")
        for source in (empty, broken, self.root / "missing.wav"):
            with self.subTest(source=source), self.assertRaises(ValueError):
                self.coordinator.ingest(source)
        self.assertEqual(self.coordinator.status()["recordings"], 0)
        self.assertEqual(list((self.config.data_dir / "audio").iterdir()), [])

    def test_incompatible_index_rejected(self):
        with self.assertRaisesRegex(ValueError, "Incompatible"):
            Storage(replace(self.config, model_revision="different"))

    def test_search_checks_each_embedding_contract_and_file(self):
        self.coordinator.ingest(self.audio_file(duration=1))
        run_worker(self.coordinator, self.config, "one", FakeEmbedder(), lambda _: None)
        with self.storage.connection(write=True) as db:
            db.execute("UPDATE embeddings SET preprocessing_version='wrong'")
        with self.assertRaisesRegex(ValueError, "incompatible"):
            search(self.storage, FakeEmbedder(), "tone")
        with self.storage.connection(write=True) as db:
            db.execute("UPDATE embeddings SET preprocessing_version=?", (self.config.preprocessing_version,))
        Path(self.storage.completed()[0]["artifact_path"]).unlink()
        with self.assertRaisesRegex(ValueError, "Cannot read embedding"):
            search(self.storage, FakeEmbedder(), "tone")

    def test_invalid_vectors_are_rejected(self):
        for vector in ([0, 0, 0], [1, 2], [1, np.nan, 0], [1, np.inf, 0]):
            with self.subTest(vector=vector), self.assertRaises(ValueError):
                normalize(vector, 3)

    def test_search_validation_and_no_work_need_no_model(self):
        self.assertEqual(search(self.storage, FakeEmbedder(), "tone"), [])
        for query, top_k in [(" ", 5), ("tone", 0)]:
            with self.assertRaises(ValueError):
                search(self.storage, FakeEmbedder(), query, top_k)
        with self.assertRaises(ValueError):
            search(self.storage, FakeEmbedder(), "tone", windows_per_recording=0)
        runner = CliRunner()
        for command in (["search", "tone"], ["worker"], ["status"]):
            with patch("ensemble_atlas.cli.ClapEmbedder", side_effect=AssertionError("unexpected model load")):
                result = runner.invoke(app, ["--data-dir", str(self.root / "empty"), *command])
            self.assertEqual(result.exit_code, 0, result.output)

    def test_cli_full_workflow_and_argument_validation(self):
        runner = CliRunner()
        arguments = ["--data-dir", str(self.root / "cli")]
        # CLI's normal contract has 512 dimensions; use an adapter fake.
        class CliFake(FakeEmbedder):
            def audio(self, samples):
                return np.pad(super().audio(samples), (0, 509))
            def text(self, query):
                return np.pad(super().text(query), (0, 509))
        fixture = self.audio_file()
        result = runner.invoke(app, [*arguments, "ingest", str(fixture)])
        self.assertEqual(result.exit_code, 0, result.output)
        with patch("ensemble_atlas.cli.ClapEmbedder", return_value=CliFake()):
            result = runner.invoke(app, [*arguments, "worker", "--id", "test"])
            self.assertEqual(result.exit_code, 0, result.output)
            result = runner.invoke(app, [*arguments, "search", "tone", "--top-k", "1"])
            self.assertEqual(result.exit_code, 0, result.output)
            self.assertIn("cosine 1.0000", result.output)
            self.assertIn("00:00:05.000", result.output)
            self.assertIn("recording.wav", result.output)
            self.assertEqual(result.output.count("recording.wav ["), 1)
            self.assertIn("00:00:10.000", result.output)
            result = runner.invoke(app, [*arguments, "search", "tone", "--windows-per-recording", "1"])
            self.assertEqual(result.exit_code, 0, result.output)
            self.assertIn("00:00:05.000", result.output)
            self.assertNotIn("00:00:10.000", result.output)
        result = runner.invoke(app, [*arguments, "search", "tone", "--top-k", "0"])
        self.assertEqual(result.exit_code, 2)
        result = runner.invoke(app, [*arguments, "search", "tone", "--windows-per-recording", "0"])
        self.assertEqual(result.exit_code, 2)

    def test_recordings_rank_by_maximum_window_before_top_k_is_applied(self):
        source = self.audio_file()
        # Identical filenames still represent distinct recordings. The first has
        # several strong windows; the second has the best single window but a
        # lower average, and must rank first.
        scores = [[0.9, 0.85, 0.8], [0.1, 0.95, 0.0], [0.7, 0.6, 0.5]]
        ids = [self.coordinator.ingest(source)["recording_id"] for _ in scores]
        by_id = dict(zip(ids, scores))
        self.coordinator.register("scorer")
        while (job := self.coordinator.claim("scorer")) is not None:
            recording_id, window = job["job_id"].rsplit("-", 1)
            score = by_id[recording_id][int(window)]
            vector = [np.sqrt(1 - score ** 2), score, 0]
            self.coordinator.complete(job["job_id"], job["attempt_token"], vector)
        hits = search(self.storage, FakeEmbedder(), "tone", top_k=2, windows_per_recording=2)
        self.assertEqual([hit["recording_id"] for hit in hits], [ids[1], ids[0]])
        np.testing.assert_allclose([hit["score"] for hit in hits], [0.95, 0.9])
        self.assertTrue(all(len(hit["matches"]) == 2 for hit in hits))
        self.assertEqual([match["start"] for match in hits[0]["matches"]], [5, 0])
        self.assertEqual(hits[0]["matches"][0]["end"], 11)
        all_hits = search(self.storage, FakeEmbedder(), "tone", top_k=10, windows_per_recording=10)
        self.assertEqual(len(all_hits), 3)
        self.assertTrue(all(len(hit["matches"]) == 3 for hit in all_hits))

    def test_main_formats_actionable_errors_without_tracebacks(self):
        runner = CliRunner()
        import typer
        command = typer.Typer()
        @command.command()
        def missing():
            main()
        with patch("ensemble_atlas.cli.app", side_effect=ValueError("Recording is empty.")):
            result = runner.invoke(command, [])
        self.assertEqual(result.exit_code, 1)
        self.assertIn("Error: Recording is empty.", result.output)
        self.assertNotIn("Traceback", result.output)

    def test_timestamp_rounding(self):
        self.assertEqual(timestamp(59.9999), "00:01:00.000")
        self.assertEqual(timestamp(3661.123), "01:01:01.123")


class LeaseBoundaryTests(AudioTestCase):
    def setUp(self):
        super().setUp()
        self.now = 1000.0
        self.coordinator = LocalCoordinator(self.storage, clock=lambda: self.now)
        self.coordinator.ingest(self.audio_file(duration=1))
        self.coordinator.register("one")
        self.coordinator.register("two")

    def test_expired_token_cannot_heartbeat_fail_or_overwrite(self):
        old = self.coordinator.claim("one")
        self.now += self.config.lease_seconds
        self.assertFalse(self.coordinator.heartbeat(old["job_id"], old["attempt_token"]))
        current = self.coordinator.claim("two")
        self.assertNotEqual(old["attempt_token"], current["attempt_token"])
        self.assertFalse(self.coordinator.complete(old["job_id"], old["attempt_token"], [1, 0, 0]))
        self.assertFalse(self.coordinator.fail(old["job_id"], old["attempt_token"], "late"))
        self.assertTrue(self.coordinator.complete(current["job_id"], current["attempt_token"], [0, 1, 0]))
        accepted = self.storage.completed()[0]["artifact_path"]
        self.assertTrue(self.coordinator.complete(current["job_id"], current["attempt_token"], [1, 0, 0]))
        np.testing.assert_array_equal(np.load(accepted), [0, 1, 0])
        self.assertEqual(len(self.storage.completed()), 1)

    def test_retries_exhaust_after_three_claims(self):
        for _ in range(3):
            job = self.coordinator.claim("one")
            self.assertTrue(self.coordinator.fail(job["job_id"], job["attempt_token"], "test failure"))
        self.assertIsNone(self.coordinator.claim("one"))
        self.assertEqual(self.coordinator.status()["counts"]["failed"], 1)
        self.assertEqual(self.coordinator.status()["retries"], 2)

    def test_last_expiration_becomes_permanent_failure(self):
        for _ in range(3):
            self.assertIsNotNone(self.coordinator.claim("one"))
            self.now += self.config.lease_seconds
        self.assertIsNone(self.coordinator.claim("two"))
        self.assertEqual(self.coordinator.status()["counts"]["failed"], 1)

    def test_one_active_chunk_per_worker(self):
        self.coordinator.claim("one")
        with self.assertRaisesRegex(ValueError, "already has an active"):
            self.coordinator.claim("one")

    def test_worker_heartbeats_during_inference(self):
        config = replace(self.config, lease_seconds=0.3, heartbeat_seconds=0.03)
        coordinator = LocalCoordinator(Storage(config))
        calls = []
        original = coordinator.heartbeat
        def heartbeat(*args):
            calls.append(args)
            return original(*args)
        coordinator.heartbeat = heartbeat
        class SlowEmbedder(FakeEmbedder):
            def audio(self, samples):
                time.sleep(0.45)
                return super().audio(samples)
        summary = run_worker(coordinator, config, "slow", SlowEmbedder(), lambda _: None)
        self.assertEqual(summary["completed"], 1)
        self.assertGreaterEqual(len(calls), 3)

    def test_persist_failure_never_marks_chunk_completed(self):
        job = self.coordinator.claim("one")
        with patch.object(self.storage, "persist_embedding", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                self.coordinator.complete(job["job_id"], job["attempt_token"], [1, 0, 0])
        self.assertEqual(self.storage.completed(), [])
        self.assertEqual(self.coordinator.status()["counts"]["leased"], 1)

    def test_two_processes_cannot_claim_same_active_job(self):
        context = multiprocessing.get_context("spawn")
        gate, results = context.Event(), context.Queue()
        processes = [context.Process(target=claim_in_process, args=(self.config, worker, gate, results))
                     for worker in ("process-one", "process-two")]
        try:
            for process in processes:
                process.start()
            gate.set()
            claims = [results.get(timeout=20) for _ in processes]
            self.assertEqual(sum(job is not None for job in claims), 1)
            for process in processes:
                process.join(10)
                self.assertEqual(process.exitcode, 0)
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
                    process.join(5)
            results.close()

    def test_terminated_process_leaves_recoverable_assignment(self):
        context = multiprocessing.get_context("spawn")
        gate, results = context.Event(), context.Queue()
        process = context.Process(target=claim_in_process,
                                  args=(self.config, "terminated", gate, results, True))
        try:
            process.start()
            gate.set()
            abandoned = results.get(timeout=20)
            process.terminate()
            process.join(10)
            self.now += self.config.lease_seconds
            replacement = LocalCoordinator(Storage(self.config), clock=lambda: self.now)
            current = replacement.claim("two")
            self.assertEqual(current["job_id"], abandoned["job_id"])
            self.assertNotEqual(current["attempt_token"], abandoned["attempt_token"])
            self.assertTrue(replacement.complete(current["job_id"], current["attempt_token"], [0, 1, 0]))
        finally:
            if process.is_alive():
                process.terminate()
                process.join(5)
            results.close()


if __name__ == "__main__":
    unittest.main()
