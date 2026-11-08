from threading import Event, Thread
import math
import time
from typing import Callable

from .audio import read_window
from .config import Config
from .coordinator import Coordinator
from .embeddings import Embedder


def run_worker(coordinator: Coordinator, config: Config, worker_id: str, embedder: Embedder,
               report: Callable[[str], None] = print, *, wait_for_leases: bool = False,
               poll_seconds: float = 0.5) -> dict:
    """Drain available work and exit; each heartbeat uses a fresh DB connection."""
    if not math.isfinite(poll_seconds) or poll_seconds <= 0:
        raise ValueError("Poll interval must be finite and positive.")
    coordinator.register(worker_id)
    completed = failed = discarded = 0
    started = time.monotonic()
    while True:
        job = coordinator.claim(worker_id)
        if job is None:
            if not wait_for_leases:
                break
            counts = coordinator.status()["counts"]
            if not (counts["pending"] or counts["leased"]):
                break
            time.sleep(poll_seconds)
            continue
        stop = Event()
        lost = Event()

        def heartbeat():
            while not stop.wait(config.heartbeat_seconds):
                try:
                    if not coordinator.heartbeat(job["job_id"], job["attempt_token"]):
                        lost.set()
                        return
                except Exception:
                    # Without proof of a renewed lease, discard this computation.
                    lost.set()
                    return

        thread = Thread(target=heartbeat, daemon=True)
        thread.start()
        chunk_started = time.monotonic()
        accepted = False
        try:
            samples = read_window(job["audio_path"], job["start"], job["end"])
            vector = embedder.audio(samples)
            if not lost.is_set() and coordinator.complete(job["job_id"], job["attempt_token"], vector.tolist(),
                                                        time.monotonic() - chunk_started):
                accepted = True
                completed += 1
                report(f"Completed {job['job_id']} ({job['start']:.3f}–{job['end']:.3f}s)")
            else:
                discarded += 1
                report(f"Discarded result for expired/replaced assignment {job['job_id']}")
        except Exception as error:
            coordinator.fail(job["job_id"], job["attempt_token"], str(error))
            failed += 1
            report(f"Attempt failed for {job['job_id']}: {error}")
        finally:
            stop.set()
            thread.join()
        if lost.is_set() and not accepted:
            # A failed renewal may still leave our lease active. Exit rather than
            # claiming again under the same ID; scheduling recovers it on expiry.
            break
    elapsed = time.monotonic() - started
    return {"completed": completed, "failed_attempts": failed, "discarded": discarded, "elapsed_seconds": elapsed,
            "chunks_per_second": completed / elapsed if elapsed > 0 else 0.0}
