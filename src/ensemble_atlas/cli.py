from datetime import datetime, timezone
from pathlib import Path
import math
import sqlite3
import time
from typing import Annotated

from rich.console import Console, Group
from rich.live import Live
from rich.text import Text
from rich.table import Table
import typer

from .config import Config
from .coordinator import LocalCoordinator
from .embeddings import ClapEmbedder
from .search import search as search_index
from .search import validate_query
from .shared_inference import run_shared_workers
from .storage import Storage
from .worker import run_worker


app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False,
                  help="Search inside local audio recordings with CLAP.")
console = Console(highlight=False)


@app.callback()
def setup(ctx: typer.Context,
          data_dir: Annotated[Path, typer.Option(envvar="ATLAS_DATA_DIR", help="Managed audio, SQLite, embeddings and model cache.")] = Path("data")):
    ctx.obj = Config(data_dir=data_dir)


@app.command()
def ingest(ctx: typer.Context, recording: Path):
    """Copy and validate a recording, then enqueue 10s windows at a 5s stride."""
    result = LocalCoordinator(Storage(ctx.obj)).ingest(recording)
    console.print(f"Ingested {recording.name}: {result['recording_id']}\n"
                  f"{result['duration']:.3f}s; queued {result['chunks']} chunk(s).", markup=False)


@app.command()
def worker(ctx: typer.Context,
           worker_id: Annotated[str, typer.Option("--id", help="Unique label for this worker.")] = "worker-1",
           count: Annotated[int, typer.Option(min=1, help="Workers sharing one inference process.")] = 1,
           single_inference: Annotated[bool, typer.Option(help="Use one CLAP process for all workers.")] = False,
           lease_seconds: Annotated[float, typer.Option(min=0.1)] = 120.0,
           heartbeat_seconds: Annotated[float, typer.Option(min=0.01)] = 20.0,
           max_attempts: Annotated[int, typer.Option(min=1, help="Total claims allowed per chunk, including retries.")] = 3,
           wait_for_leases: Annotated[bool, typer.Option(help="Wait for outstanding leases and recover expired work; exit when the queue is terminal.")] = False,
           cpu_threads: Annotated[int, typer.Option(min=1)] = 2):
    """Process available chunks, then exit."""
    if not worker_id.strip():
        raise ValueError("Worker ID cannot be blank.")
    if count > 1 and not single_inference:
        raise ValueError("--count greater than one requires --single-inference.")
    if count > 1 and worker_id != "worker-1":
        raise ValueError("Use --id only when running one worker.")
    config = Config(data_dir=ctx.obj.data_dir, lease_seconds=lease_seconds,
                    heartbeat_seconds=heartbeat_seconds, max_attempts=max_attempts, cpu_threads=cpu_threads)
    coordinator = LocalCoordinator(Storage(config))
    counts = coordinator.status()["counts"]
    if not (counts["pending"] or counts["leased"]):
        console.print("No work available. Ingest a recording first, or inspect atlas status.")
        return
    if single_inference:
        console.print(f"Loading one shared CLAP process for {count} worker(s) (first use downloads the checkpoint).")
        worker_ids = [worker_id] if count == 1 else [f"worker-{i}" for i in range(1, count + 1)]
        summary = run_shared_workers(config, worker_ids, wait_for_leases=wait_for_leases)
    else:
        console.print("Loading CLAP on CPU (first use downloads the checkpoint).")
        embedder = ClapEmbedder(config)
        summary = run_worker(coordinator, config, worker_id, embedder,
                             lambda message: console.print(message, markup=False),
                             wait_for_leases=wait_for_leases)
    console.print(f"Completed {summary['completed']} chunk(s) in {summary['elapsed_seconds']:.2f}s "
                  f"({summary['chunks_per_second']:.3f} chunks/s); {summary['failed_attempts']} failed attempt(s); "
                  f"{summary['discarded']} discarded result(s).")
    if summary["failed_attempts"] or summary["discarded"] or coordinator.status()["counts"]["failed"]:
        raise typer.Exit(1)


def timestamp(seconds: float) -> str:
    milliseconds = round(seconds * 1000)
    minutes, remainder = divmod(milliseconds, 60_000)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02}:{minutes:02}:{remainder // 1000:02}.{remainder % 1000:03}"


@app.command()
def search(ctx: typer.Context, query: Annotated[str | None, typer.Argument()] = None,
           audio: Annotated[Path | None, typer.Option(help="Query clip of at most 10 seconds; shorter clips are zero-padded.")] = None,
           top_k: Annotated[int, typer.Option(min=1, help="Number of recordings to return.")] = 5,
           windows_per_recording: Annotated[int, typer.Option(min=1, help="Best windows to show per recording.")] = 3):
    """Rank recordings by their best window and show matching timestamps."""
    validate_query(query, audio)
    storage = Storage(ctx.obj)
    if not storage.completed():
        console.print("No completed embeddings. Run atlas ingest and atlas worker first.")
        return
    hits = search_index(storage, ClapEmbedder(ctx.obj), query, top_k, windows_per_recording, audio=audio)
    console.print("Recordings ranked by best window. Cosine similarity is a ranking score.")
    for index, hit in enumerate(hits, 1):
        console.print(f"{index}. {hit['original_name']} [{hit['recording_id']}]\n"
                      f"   Best cosine {hit['score']:.4f}\n"
                      f"   {hit['audio_path']}", markup=False, soft_wrap=True)
        for match in hit["matches"]:
            console.print(f"   {timestamp(match['start'])} – {timestamp(match['end'])} | cosine {match['score']:.4f}",
                          markup=False, soft_wrap=True)


def status_view(snapshot: dict) -> Group:
    """Render a read-only DB snapshot; lease expiry is not proof of process death."""
    heading = Text(f"Recordings: {snapshot['recordings']} | Retries: {snapshot['retries']} | "
                   f"Completed chunk time: {snapshot['chunk_seconds']:.2f}s")
    counts = Text(" | ".join(f"{state}: {count}" for state, count in snapshot["counts"].items()))
    table = Table("Worker", "Last seen (UTC)", "Assignment", "Leased chunk", "Lease expires (UTC)", "Completed")
    for row in snapshot["workers"]:
        last_seen = datetime.fromtimestamp(row["last_seen"], timezone.utc).isoformat(timespec="seconds")
        expiration = (datetime.fromtimestamp(row["lease_expiration"], timezone.utc).isoformat(timespec="seconds")
                      if row["lease_expiration"] is not None else "—")
        assignment = ("expired lease" if row["lease_expiration"] <= snapshot["sampled_at"] else "leased") if row["active_job"] else "no lease"
        table.add_row(Text(row["id"]), last_seen, assignment, row["active_job"] or "—", expiration,
                      str(row["completed"]))
    recordings = Table("Recording", "ID", "Completed / total", "Failed")
    for row in snapshot["recording_progress"]:
        recordings.add_row(Text(row["original_name"]), row["id"], f"{row['completed']} / {row['total']}", str(row["failed"]))
    errors = [Text(f"{row['id']} ({row['state']}): {row['last_error']}") for row in snapshot["errors"]]
    return Group(heading, counts, recordings, table, *errors)


@app.command()
def status(ctx: typer.Context,
           watch: Annotated[bool, typer.Option(help="Refresh until Ctrl-C; does not change queue state.")] = False,
           interval: Annotated[float, typer.Option(min=0.05, help="Seconds between live snapshots.")] = 1.0):
    """Show recording progress, leases, worker activity, retries, and errors."""
    if not math.isfinite(interval):
        raise ValueError("Status interval must be finite.")
    coordinator = LocalCoordinator(Storage(ctx.obj))
    if not watch:
        console.print(status_view(coordinator.status()))
        return
    try:
        with Live(status_view(coordinator.status()), console=console, auto_refresh=False) as live:
            while True:
                time.sleep(interval)
                live.update(status_view(coordinator.status()), refresh=True)
    except KeyboardInterrupt:
        pass


def main():
    try:
        app()
    except (ValueError, OSError, RuntimeError, sqlite3.Error) as error:
        console.print(f"Error: {error}", style="red", markup=False)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
