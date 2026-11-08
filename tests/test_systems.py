"""Live status and end-to-end systems checks, using generated audio/fake vectors."""
from dataclasses import replace
from io import StringIO
from threading import Event
from unittest.mock import patch

from rich.console import Console
from typer.testing import CliRunner

from ensemble_atlas.cli import app, status_view
from ensemble_atlas.coordinator import LocalCoordinator
from ensemble_atlas.storage import Storage
from ensemble_atlas.worker import run_worker
from systems_demo import run_phase
from test_vertical_slice import AudioTestCase, FakeEmbedder


class SystemsTests(AudioTestCase):
    def test_two_process_workers_complete_and_recover_terminated_assignment(self):
        fixture = self.audio_file(duration=21)
        for recovery in (False, True):
            with self.subTest(recovery=recovery):
                result = run_phase(self.root / str(recovery), fixture, workers=2, fake=True,
                                   recovery=recovery, cache_dir=self.root, lease_seconds=1,
                                   heartbeat_seconds=0.05, timeout=40)
                self.assertEqual(result["chunks"], 5)
                self.assertGreater(result["chunks_per_second"], 0)
                self.assertEqual(sum(result["completed_by_worker"].values()), 5)
                if recovery:
                    self.assertTrue(result["reassignment_and_completion_checks_passed"])
                    self.assertEqual(result["retries"], 1)
                else:
                    self.assertEqual(set(result["completed_by_worker"]), {"worker-1", "worker-2"})
                    self.assertTrue(all(result["completed_by_worker"].values()))

    def test_wait_mode_exits_on_empty_and_failed_queue(self):
        self.assertEqual(run_worker(self.coordinator, self.config, "idle", FakeEmbedder(),
                                   lambda _: None, wait_for_leases=True)["completed"], 0)
        self.coordinator.ingest(self.audio_file(duration=1))
        self.coordinator.register("failed")
        for _ in range(self.config.max_attempts):
            job = self.coordinator.claim("failed")
            self.coordinator.fail(job["job_id"], job["attempt_token"], "test")
        self.assertEqual(run_worker(self.coordinator, self.config, "idle", FakeEmbedder(),
                                   lambda _: None, wait_for_leases=True)["completed"], 0)

    def test_default_worker_still_exits_with_outstanding_lease(self):
        self.coordinator.ingest(self.audio_file(duration=1))
        self.coordinator.register("other")
        self.coordinator.claim("other")
        result = run_worker(self.coordinator, self.config, "idle", FakeEmbedder(), lambda _: None)
        self.assertEqual(result["completed"], 0)
        self.assertEqual(self.coordinator.status()["counts"]["leased"], 1)

    def test_heartbeat_after_success_does_not_stop_queue_drain(self):
        self.coordinator.ingest(self.audio_file())
        config = replace(self.config, lease_seconds=2, heartbeat_seconds=0.01)
        coordinator = LocalCoordinator(Storage(config))
        finished_heartbeat = Event()
        complete, heartbeat = coordinator.complete, coordinator.heartbeat
        def racing_heartbeat(*args):
            result = heartbeat(*args)
            if not result:
                finished_heartbeat.set()
            return result
        def delayed_completion(*args):
            finished_heartbeat.clear()
            result = complete(*args)
            self.assertTrue(finished_heartbeat.wait(5))
            return result
        with patch.object(coordinator, "heartbeat", side_effect=racing_heartbeat), \
             patch.object(coordinator, "complete", side_effect=delayed_completion):
            result = run_worker(coordinator, config, "racing", FakeEmbedder(), lambda _: None)
        self.assertEqual(result["completed"], 3)

    def test_status_identifies_expired_lease_without_reclaiming(self):
        self.coordinator.ingest(self.audio_file(duration=1))
        now = 1000.0
        coordinator = LocalCoordinator(self.storage, clock=lambda: now)
        coordinator.register("[red]literal")
        coordinator.claim("[red]literal")
        now += self.config.lease_seconds
        snapshot = coordinator.status()
        output = StringIO()
        Console(file=output, width=220).print(status_view(snapshot))
        self.assertIn("expired lease", output.getvalue())
        self.assertIn("[red]literal", output.getvalue())
        self.assertEqual(snapshot["counts"]["leased"], 1)
        self.assertEqual(snapshot["retries"], 0)
        self.assertEqual(snapshot["recording_progress"][0]["completed"], 0)

    def test_watch_refreshes_recording_progress_and_interrupts_cleanly(self):
        self.coordinator.ingest(self.audio_file(duration=1))
        snapshots = []
        original = LocalCoordinator.status
        def snapshot(coordinator):
            result = original(coordinator)
            snapshots.append(result)
            return result
        calls = 0
        def sleep(_):
            nonlocal calls
            calls += 1
            if calls == 1:
                run_worker(self.coordinator, self.config, "one", FakeEmbedder(), lambda _: None)
            else:
                raise KeyboardInterrupt
        # CLI Config normally expects 512 dimensions; use the fixture contract.
        with patch("ensemble_atlas.cli.Config", return_value=self.config), \
             patch.object(LocalCoordinator, "status", snapshot), \
             patch("ensemble_atlas.cli.time.sleep", side_effect=sleep):
            result = CliRunner().invoke(app, ["status", "--watch", "--interval", "0.05"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual([s["counts"]["completed"] for s in snapshots], [0, 1])
        self.assertEqual(snapshots[-1]["recording_progress"][0]["completed"], 1)

    def test_invalid_watch_interval_rejected_before_model_loading(self):
        for value in ("0", "nan", "inf"):
            with self.subTest(value=value), patch("ensemble_atlas.cli.ClapEmbedder", side_effect=AssertionError("model loaded")):
                result = CliRunner().invoke(app, ["--data-dir", str(self.config.data_dir), "status", "--watch", "--interval", value])
            self.assertNotEqual(result.exit_code, 0)
