"""Run multiple queue workers behind one model-owning inference process."""

from __future__ import annotations

from multiprocessing import Event, Process, Queue
from queue import Empty
from threading import Lock
from typing import Callable
import os
import uuid

import numpy as np

from .config import Config
from .coordinator import LocalCoordinator
from .embeddings import ClapEmbedder
from .storage import Storage
from .worker import run_worker


def _inference_loop(config: Config, requests: Queue, responses: list[Queue], stop: Event,
                    ready: Event, failed: Event,
                    embedder_factory: Callable[[Config], object]) -> None:
    try:
        embedder = embedder_factory(config)
        ready.set()
        while not stop.is_set():
            try:
                request_id, response_index, kind, value = requests.get(timeout=0.2)
            except Empty:
                continue
            try:
                vector = getattr(embedder, kind)(value)
                response = (request_id, "ok", vector)
            except Exception as error:
                response = (request_id, "error", f"{type(error).__name__}: {error}")
            responses[response_index].put(response)
    except Exception:
        failed.set()
        raise
    finally:
        ready.clear()


class ProcessEmbedder:
    """Client for the model process; one instance belongs to one queue worker."""

    def __init__(self, requests: Queue, responses: Queue, response_index: int,
                 ready: Event, failed: Event, broker_pid: int):
        self._requests = requests
        self._responses = responses
        self._response_index = response_index
        self._ready = ready
        self._failed = failed
        self._broker_pid = broker_pid
        self._lock = Lock()

    def _request(self, kind: str, value: object) -> np.ndarray:
        while not self._ready.is_set():
            if self._failed.is_set() or not self._process_exists():
                raise RuntimeError("The shared inference process failed during startup.")
            self._failed.wait(0.2)
        request_id = uuid.uuid4().hex
        self._requests.put((request_id, self._response_index, kind, value))
        while True:
            try:
                response_id, status, result = self._responses.get(timeout=0.2)
            except Empty:
                if self._failed.is_set() or not self._ready.is_set() or not self._process_exists():
                    raise RuntimeError("The shared inference process exited.")
                continue
            if response_id != request_id:
                raise RuntimeError("Shared inference response did not match its request.")
            if status == "error":
                raise RuntimeError(str(result))
            return np.asarray(result)

    def _process_exists(self) -> bool:
        try:
            os.kill(self._broker_pid, 0)
        except (OSError, ProcessLookupError):
            return False
        return True

    def audio(self, samples: np.ndarray) -> np.ndarray:
        with self._lock:
            return self._request("audio", samples)

    def text(self, query: str) -> np.ndarray:
        with self._lock:
            return self._request("text", query)


def _run_queue_worker(config: Config, worker_id: str, requests: Queue,
                      responses: Queue, response_index: int, ready: Event, failed: Event,
                      broker_pid: int, results: Queue, wait_for_leases: bool) -> None:
    try:
        coordinator = LocalCoordinator(Storage(config))
        summary = run_worker(coordinator, config, worker_id,
                             ProcessEmbedder(requests, responses, response_index, ready, failed, broker_pid),
                             report=lambda _: None,
                             wait_for_leases=wait_for_leases)
        results.put((worker_id, "ok", summary))
    except Exception as error:
        results.put((worker_id, "error", f"{type(error).__name__}: {error}"))


def run_shared_workers(config: Config, worker_ids: list[str], *,
                       wait_for_leases: bool = False,
                       embedder_factory: Callable[[Config], object] = ClapEmbedder) -> dict:
    """Drain work with several lightweight workers and one model process."""
    if not worker_ids or len(set(worker_ids)) != len(worker_ids):
        raise ValueError("Shared inference requires at least one distinct worker ID.")

    requests: Queue = Queue()
    results: Queue = Queue()
    stop = Event()
    ready = Event()
    failed = Event()
    response_queues = [Queue() for _ in worker_ids]
    broker = Process(target=_inference_loop,
                     args=(config, requests, response_queues, stop, ready, failed,
                           embedder_factory), daemon=True)
    broker.start()
    workers = [Process(target=_run_queue_worker,
                        args=(config, worker_id, requests, response_queue, index, ready, failed,
                              broker.pid, results, wait_for_leases), daemon=True)
               for index, (worker_id, response_queue) in enumerate(zip(worker_ids, response_queues))]
    for process in workers:
        process.start()

    summaries = {}
    try:
        for _ in workers:
            worker_id, status, result = results.get()
            if status == "error":
                raise RuntimeError(f"Shared worker {worker_id} failed: {result}")
            summaries[worker_id] = result
        for process in workers:
            process.join(timeout=5)
            if process.is_alive():
                raise RuntimeError(f"Shared worker {process.pid} did not exit.")
        elapsed = max((item["elapsed_seconds"] for item in summaries.values()), default=0)
        completed = sum(item["completed"] for item in summaries.values())
        return {
            "workers": summaries,
            "completed": completed,
            "failed_attempts": sum(item["failed_attempts"] for item in summaries.values()),
            "discarded": sum(item["discarded"] for item in summaries.values()),
            "elapsed_seconds": elapsed,
            "chunks_per_second": completed / elapsed if elapsed > 0 else 0.0,
        }
    finally:
        stop.set()
        for process in workers:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
        if broker.is_alive():
            broker.terminate()
        broker.join(timeout=5)
        for queue in response_queues:
            queue.close()
        requests.close()
        results.close()
