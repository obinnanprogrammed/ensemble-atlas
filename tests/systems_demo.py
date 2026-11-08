"""Measure one/two workers and demonstrate termination/reassignment on fresh indexes.

Uses real CPU CLAP by default. --fake exercises scheduling without a model;
its timings are NOT inference benchmarks. Existing indexes are never modified.
"""
import argparse
from dataclasses import replace
import json
import math
import multiprocessing
from multiprocessing.connection import wait
from pathlib import Path
import resource
import sys
from tempfile import TemporaryDirectory
import time

import numpy as np
import soundfile as sf

from ensemble_atlas.audio import inspect_audio, windows
from ensemble_atlas.config import Config
from ensemble_atlas.coordinator import LocalCoordinator
from ensemble_atlas.embeddings import ClapEmbedder
from ensemble_atlas.storage import Storage
from ensemble_atlas.worker import run_worker


class DemoEmbedder:
    def audio(self, samples):
        time.sleep(0.05)
        return np.array([1, 0, 0], dtype=np.float32)


def demo_worker(config, cache_dir, worker_id, gate, events, fake, victim, recovery_claimed):
    try:
        started = time.monotonic()
        embedder = DemoEmbedder() if fake else ClapEmbedder(replace(config, data_dir=cache_dir))
        coordinator = LocalCoordinator(Storage(config))
        claim = coordinator.claim
        def observed_claim(label):
            job = claim(label)
            if job is not None:
                if victim:
                    # Finish releasing shared Event locks before advertising the
                    # safe termination point; killing during set() can strand peers.
                    recovery_claimed.set()
                events.send({"event": "claim", "worker": label, "job": job})
                if victim:
                    # Hold after a real durable claim, before decoding/heartbeats.
                    # The controller terminates only this child, never an external PID.
                    time.sleep(300)
                    raise RuntimeError("Controller did not terminate the victim")
            return job
        coordinator.claim = observed_claim
        events.send({"event": "ready", "worker": worker_id, "load_seconds": time.monotonic() - started})
        if not gate.wait(300):
            raise TimeoutError("Start gate was not released")
        if recovery_claimed is not None and not victim and not recovery_claimed.wait(300):
            raise TimeoutError("Victim did not claim an assignment")
        summary = run_worker(coordinator, config, worker_id, embedder, lambda _: None,
                             wait_for_leases=True, poll_seconds=0.05)
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        events.send({"event": "done", "worker": worker_id, "summary": summary,
                    "peak_rss_mib": peak / (1024 ** 2 if sys.platform == "darwin" else 1024)})
    except BaseException as error:
        events.send({"event": "error", "worker": worker_id, "error": repr(error)})
        raise


def run_phase(root, fixture, *, workers, fake, cache_dir, recovery=False,
              lease_seconds=5.0, heartbeat_seconds=0.5, cpu_threads=2, timeout=300):
    config = Config(data_dir=root, dimension=3 if fake else 512, lease_seconds=lease_seconds,
                    heartbeat_seconds=heartbeat_seconds, cpu_threads=cpu_threads)
    storage = Storage(config)
    coordinator = LocalCoordinator(storage)
    chunks = coordinator.ingest(fixture)["chunks"]
    context = multiprocessing.get_context("spawn")
    gate = context.Event()
    # Independent pipes keep termination of one child from stranding another
    # child's Queue feeder lock or damaging a shared result channel.
    channels = [context.Pipe(duplex=False) for _ in range(workers)]
    readers = {reader for reader, _ in channels}
    recovery_claimed = context.Event() if recovery else None
    processes = [context.Process(target=demo_worker,
                 args=(config, cache_dir, f"worker-{i + 1}", gate, channels[i][1], fake, recovery and i == 0, recovery_claimed))
                 for i in range(workers)]
    loaded, finished, claims = {}, {}, []
    abandoned = None
    started = time.monotonic()
    deadline = started + timeout
    indexing_started = None
    try:
        for process in processes:
            process.start()
        for _, writer in channels:
            writer.close()
        while len(finished) < workers - int(recovery):
            if time.monotonic() >= deadline:
                raise TimeoutError("Demo timed out waiting for worker progress")
            ready = wait(readers, timeout=0.1)
            if not ready:
                for i, process in enumerate(processes):
                    if process.exitcode is not None and f"worker-{i + 1}" not in finished:
                        if recovery and i == 0 and abandoned is not None:
                            continue
                        if process.exitcode != 0:
                            raise RuntimeError(f"Worker {i + 1} exited with code {process.exitcode}")
                continue
            reader = ready[0]
            try:
                message = reader.recv()
            except EOFError:
                readers.remove(reader)
                continue
            kind, worker = message["event"], message["worker"]
            if kind == "error":
                raise RuntimeError(message["error"])
            if kind == "ready":
                loaded[worker] = message["load_seconds"]
                if len(loaded) == workers:
                    indexing_started = time.monotonic()
                    gate.set()
            elif kind == "claim":
                claims.append(message)
                if recovery and worker == "worker-1" and abandoned is None:
                    abandoned = message["job"]
                    processes[0].terminate()
                    processes[0].join(10)
                    if processes[0].is_alive():
                        raise RuntimeError("Victim did not terminate")
            elif kind == "done":
                finished[worker] = message
        for i, process in enumerate(processes):
            process.join(10)
            if not (recovery and i == 0) and process.exitcode != 0:
                raise RuntimeError(f"Worker {i + 1} failed to exit cleanly")
        elapsed = time.monotonic() - indexing_started
        # Reopen the index; validate persisted state rather than child counters alone.
        reopened = Storage(config)
        snapshot = LocalCoordinator(reopened).status()
        if snapshot["counts"] != {"pending": 0, "leased": 0, "completed": chunks, "failed": 0}:
            raise AssertionError(snapshot)
        rows = reopened.completed()
        if len(rows) != chunks or len({r["artifact_path"] for r in rows}) != chunks:
            raise AssertionError("Expected exactly one accepted embedding per chunk")
        for row in rows:
            vector = np.load(row["artifact_path"], allow_pickle=False)
            if vector.shape != (config.dimension,) or not np.isfinite(vector).all() or not np.isclose(np.linalg.norm(vector), 1):
                raise AssertionError("Invalid persisted embedding")
        distribution = {row["id"]: row["completed"] for row in snapshot["workers"]}
        if not recovery and any(distribution.get(f"worker-{i + 1}", 0) == 0 for i in range(workers)):
            raise AssertionError("Workload too small to demonstrate participation by every worker")
        recovered = None  # No reassignment check in the throughput-only phases.
        if recovery:
            if abandoned is None or snapshot["retries"] < 1:
                raise AssertionError("No terminated assignment was retried")
            replacements = [m["job"] for m in claims if m["job"]["job_id"] == abandoned["job_id"]
                            and m["worker"] == "worker-2"]
            if not replacements or replacements[-1]["attempt_token"] == abandoned["attempt_token"]:
                raise AssertionError("Terminated assignment was not replaced")
            accepted = next(row for row in rows if row["chunk_id"] == abandoned["job_id"])
            before = np.load(accepted["artifact_path"], allow_pickle=False)
            alternate = np.zeros(config.dimension, dtype=np.float32)
            alternate[-1] = 1
            current = LocalCoordinator(reopened)
            if current.complete(abandoned["job_id"], abandoned["attempt_token"], alternate.tolist()):
                raise AssertionError("Stale completion was accepted")
            if not current.complete(abandoned["job_id"], replacements[-1]["attempt_token"], alternate.tolist()):
                raise AssertionError("Accepted attempt retry was not acknowledged")
            if not np.array_equal(before, np.load(accepted["artifact_path"], allow_pickle=False)) or len(reopened.completed()) != chunks:
                raise AssertionError("Completion retry changed persisted results")
            recovered = True
        return {"workers": workers, "recovery": recovery, "chunks": chunks,
                "indexing_seconds": elapsed, "chunks_per_second": chunks / elapsed,
                "startup_seconds": indexing_started - started, "model_load_seconds": loaded,
                "mean_completed_chunk_seconds": snapshot["chunk_seconds"] / chunks,
                "worker_peak_rss_mib": {w: m["peak_rss_mib"] for w, m in finished.items()},
                "completed_by_worker": distribution, "retries": snapshot["retries"],
                "reassignment_and_completion_checks_passed": recovered}
    finally:
        for process in processes:
            if process.pid is not None:
                if process.is_alive():
                    process.terminate()
                process.join(10)
                if process.is_alive():
                    process.kill()
                    process.join(5)
        for reader, writer in channels:
            reader.close()
            writer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, help="Optional recording with at least four windows; otherwise generate 30s of tone/silence.")
    parser.add_argument("--fake", action="store_true", help="Scheduling validation only, NOT a CLAP performance measurement.")
    parser.add_argument("--cache-dir", type=Path, default=Path("data"), help="Existing CLAP cache root; default data/models.")
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=300, help="Maximum wall seconds per phase, including model loading.")
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be finite and positive")
    Config(cpu_threads=args.cpu_threads)
    with TemporaryDirectory(prefix="atlas-systems-") as temporary:
        root = Path(temporary)
        fixture = args.fixture
        if fixture is None:
            fixture = root / "synthetic-tone-silence.wav"
            samples = np.zeros(30 * 16000, dtype=np.float32)
            samples[10 * 16000:20 * 16000] = 0.3 * np.sin(2 * np.pi * 440 * np.arange(10 * 16000) / 16000)
            sf.write(fixture, samples, 16000)
        frames, rate = inspect_audio(fixture)
        if len(list(windows(frames, rate))) < 4:
            parser.error("Use a recording longer than 15 seconds (at least four windows)")
        reports = []
        for label, workers, recovery in (("one", 1, False), ("two", 2, False), ("recovery", 2, True)):
            print(f"Running {label}: {workers} worker(s), {'fake' if args.fake else 'CPU CLAP'}", file=sys.stderr, flush=True)
            reports.append(run_phase(root / label, fixture, workers=workers, recovery=recovery, fake=args.fake,
                                     cache_dir=args.cache_dir.resolve(), cpu_threads=args.cpu_threads, timeout=args.timeout,
                                     lease_seconds=0.5 if args.fake else 5.0,
                                     heartbeat_seconds=0.05 if args.fake else 0.5))
        print(json.dumps({"backend": "fake (not an inference benchmark)" if args.fake else "CPU CLAP",
                          "fixture": str(args.fixture.resolve()) if args.fixture else "generated 30s tone/silence",
                          "cpu_threads_per_worker": args.cpu_threads,
                          "timing": "Indexing excludes ingestion and synchronized model startup; includes decoding, inference, writes and worker shutdown. Recovery includes lease expiry wait.",
                          "chunk_timing": "Mean accepted-attempt decoding plus inference; excludes result persistence and retried computation.",
                          "memory": "Per-process peak RSS; includes imports/model load. Recovery excludes terminated worker; peaks are not simultaneous combined memory.",
                          "runs": reports}, indent=2))


if __name__ == "__main__":
    main()
