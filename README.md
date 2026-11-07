# EnsembleAtlas

Search inside audio recordings using natural language or an example sound, backed by a recoverable pool of embedding workers.

EnsembleAtlas is a solo, 24-hour TAMU Datathon 2026 project. The initial implementation is a local Python CLI running on one laptop. Its systems focus is explicit job scheduling, process isolation, durable state, and worker failure recovery. A Go coordinator is a later extension.

## Project status

This README is the development specification, not a description of completed software. Commands, modules, and schemas below are planned interfaces. Update implementation status and installation instructions as functionality lands; do not imply that unimplemented commands work.

## Scope and decisions

- Start entirely in Python, with a CLI as the user interface.
- Run independent worker processes; start with one and then demonstrate two.
- Use pretrained LAION CLAP, initially `laion/clap-htsat-unfused`.
- Split recordings into timestamped, overlapping windows and index their embeddings.
- Support text search first, followed by audio-example search.
- Persist metadata and job state in SQLite; keep audio and embedding artifacts on disk.
- Keep coordination behind an explicit interface so a later Go service can replace it.
- Defer the web UI, FastAPI server, remote workers, and Go implementation.
- Do not train a model or split one model across devices.

Multiple workers on one machine are a local demonstration of a distributed-style architecture. Do not claim multi-machine execution or throughput improvements without measuring them.

## Intended user workflow

The following commands are proposed, not currently implemented:

```bash
# Ingest a recording and enqueue its windows.
atlas ingest rehearsal.wav

# Run each worker in its own terminal.
atlas worker --id worker-1
atlas worker --id worker-2

# Inspect recording progress, worker activity, and retries.
atlas status

# Find matching segments.
atlas search "brass playing a slow melody" --top-k 5
atlas search --audio example.wav --top-k 5
```

Search output should include recording ID/name, start and end timestamps, and cosine similarity. Treat similarity as a ranking score, not a probability or confidence percentage. Timestamped playback is a later usability improvement; the CLI MVP can return file paths and timestamps.

## Audio and embedding pipeline

1. Copy the input into managed local storage and validate that it can be decoded.
2. Decode, convert to mono, and resample to 48 kHz for the selected CLAP checkpoint.
3. Create logical 10-second windows with a 5-second stride. Store timestamps rather than duplicating overlapping audio files.
4. Workers read their assigned windows, apply the checkpoint's processor, and compute embeddings in inference mode.
5. L2-normalize embeddings and persist accepted results.
6. Embed a text or audio query with the same checkpoint and preprocessing contract, then rank completed chunks by dot product of normalized vectors.

Define a deterministic tail policy: emit windows starting at 0, 5, 10, ... seconds while the start is before the recording ends; pad short windows to 10 seconds for inference. Store the actual unpadded end timestamp. Reject empty or undecodable recordings with a clear CLI error.

Use the model's text and audio projection outputs in their shared embedding space. Record model ID/revision and preprocessing version with each index; reject searches across incompatible versions. Confirm the installed library's feature-return API during implementation and pin the working dependency versions.

Start with exact NumPy similarity search over completed embeddings. A vector database is unnecessary for this demo. Adjacent overlapping hits may be merged for presentation after basic search works.

## Architecture and module ownership

Suggested layout:

```text
pyproject.toml
src/ensemble_atlas/
    cli.py              # Commands and terminal output
    config.py           # Paths, model, window and lease settings
    coordinator.py      # Scheduling interface and lifecycle rules
    storage.py          # SQLite transactions and artifact persistence
    audio.py            # Decode, resample, and slice recordings
    embeddings.py       # CLAP loading, preprocessing, normalization
    worker.py           # Claim, heartbeat, infer, complete loop
    search.py           # Query embedding and similarity ranking
tests/
data/                   # Runtime artifacts; excluded from version control
```

Use a `src` package with an `atlas` console entry point. Suggested tools are Typer for CLI commands, Rich for status output, PyTorch/Transformers for CLAP, NumPy for ranking, and Python's SQLite support. Choose and pin audio decoding/resampling dependencies after verifying the checkpoint on the development laptop. Document any external decoder requirement in the eventual setup instructions.

| Component | Owns |
| --- | --- |
| CLI | Argument validation, command dispatch, formatting, exit codes |
| Coordinator | Registration, claims, leases, retries, result acceptance |
| Storage adapter | Database transactions, schema, file persistence |
| Worker | Model lifetime, audio extraction, inference, heartbeat loop |
| Search | Query embeddings, compatibility checks, ranked results |

Workers call the coordinator interface; they do not issue SQL directly. Initially, every CLI/worker process instantiates a local coordinator backed by the same SQLite database. There is no required coordinator daemon. Database state is authoritative across processes; an in-memory queue or Python lock alone is insufficient.

## Coordinator contract

Keep request and response fields serializable so this interface can later cross an HTTP boundary:

| Operation | Essential behavior |
| --- | --- |
| `register(worker_id)` | Record worker identity and last-seen time |
| `claim(worker_id)` | Atomically lease one eligible chunk or return no work |
| `heartbeat(job_id, attempt_token)` | Extend only the current, unexpired lease |
| `complete(job_id, attempt_token, embedding)` | Validate and atomically accept one current result |
| `fail(job_id, attempt_token, error)` | Record an error and retry or permanently fail the chunk |
| `status()` | Report counts, worker activity, and timings |

The exact Python signatures can evolve while preserving these semantics. Generate a fresh opaque attempt token on every claim. Worker IDs are labels; attempt tokens identify authority to complete a particular assignment.

## Job lifecycle and concurrency

The normal path is `pending -> leased -> completed`. Expired leases return to `pending`; retry exhaustion produces `failed`. Use a configurable finite attempt limit, initially three total claims per chunk.

- Claim inside a short SQLite write transaction, such as `BEGIN IMMEDIATE`, so two processes cannot lease the same pending chunk.
- Reclaim expired leases as part of scheduling, without depending on the dead worker to report failure.
- Accept heartbeat, failure, and completion only when state, token, and lease expiration match the active assignment.
- Duplicate completion of an already accepted attempt is an idempotent acknowledgement. A different or expired attempt cannot overwrite its result.
- Keep inference and audio decoding outside database transactions.
- Enable foreign keys, configure a busy timeout, and use WAL on the local filesystem. Open database connections within their owning processes/threads.
- Heartbeats must continue during inference, for example through a background thread with its own database connection. Make interval and lease length configurable.
- Stopping a worker must leave leased work recoverable after expiration.

This is at-least-once execution with one accepted result per chunk, not exactly-once computation. Workers may compute the same chunk after a retry.

Persist embedding files through a temporary file and atomic rename before committing their database reference. Use attempt-specific artifact paths so stale workers cannot overwrite accepted files. A crash may leave an unreferenced artifact, but must not expose a completed chunk with a missing embedding.

## Minimal persisted data

| Entity | Essential fields |
| --- | --- |
| Recording | ID, original name, managed audio path, duration, ingestion state |
| Chunk/job | Stable ID, recording ID, start/end, state, attempt count, token, worker ID, lease expiration, last error |
| Embedding | Unique chunk ID, artifact path, dimension, model/revision, preprocessing version |
| Worker | ID, last seen, optional active job |

Use schema constraints for unique chunk results and valid relationships. Store shared lease timestamps in UTC wall-clock time; use a monotonic clock for in-process elapsed timing. Do not infer worker death solely from the absence of a completed result.

## Resource management

Load CLAP once per worker, not once per chunk. Start with CPU inference unless a usable accelerator is confirmed. Two processes duplicate model memory and can oversubscribe CPU threads; measure memory, chunk latency, and throughput before increasing concurrency. On a single GPU, start with one inference worker and consider batching before additional model processes.

Keep chunk claims bounded: initially one active chunk per worker. A long recording should not cause all decoded windows to remain in memory. Text search may initially load a separate model process; document its memory cost and avoid running it concurrently with indexing if the laptop cannot sustain both.

## Implementation milestones

1. **Vertical slice:** ingest one recording, process with one worker, persist embeddings, and return text-search timestamps.
2. **Durable queue:** atomic claims, lease tokens, heartbeats, retries, and restart-safe state.
3. **Systems demo:** two workers, live CLI status, worker termination, reassignment, and complete indexing.
4. **Audio queries:** embed a query clip and retrieve matching recording segments.
5. **Polish:** useful errors, setup instructions, reproducible commands, and measured resource usage.

The first milestone must work before adding more infrastructure.

## Validation and demo acceptance

- Concurrent claims never hand the same active lease to two workers.
- Terminate a worker after it claims a job; another worker eventually completes that chunk.
- A late result from the terminated/replaced attempt cannot overwrite the accepted result.
- Retrying completion does not create duplicate searchable embeddings.
- Restart processes and retain recording, queue, and completed-result state.
- Empty input and decoding failures produce actionable errors; a short clip follows the documented padding policy.
- A known matching query returns interpretable timestamps from the test recording. Semantic relevance needs a manual check, not just a shape assertion.
- Report elapsed indexing time and completed chunks per second for one versus two workers; make no assumed speedup claim.

Use lightweight fake embeddings for scheduling/failure tests so those checks do not require downloading CLAP. Run a separate real-model smoke test on a small audio fixture.

## Later Go coordinator

Replace the local coordinator adapter with an HTTP client implementing the same operations. Go then owns scheduling, durable job metadata, leases, and result acceptance; Python retains audio processing and inference. Centralize database access in the Go service at that point.

Remote workers require an audio-fetch endpoint or shared object storage; local filesystem paths alone are not portable. Keep query embedding/ranking in Python initially and add a service boundary when needed. A web UI, phone recording client, remote workers, and approximate vector search are extensions after the CLI is complete.

## Guidance for development agents

Treat the scope and ownership boundaries above as the starting contract. Keep changes small and runnable. Preserve the Python-first CLI workflow, implement recovery semantics explicitly, and update this README when an interface changes. Do not add Go, a web server, cloud infrastructure, or a vector database without a new project decision. Clearly separate implemented functionality from planned behavior in documentation and demos.
