# EnsembleAtlas

Search inside audio recordings using natural language or an example sound, backed by a recoverable pool of embedding workers.

EnsembleAtlas is a solo, 24-hour TAMU Datathon 2026 project. The initial implementation is a local Python CLI running on one laptop. Its systems focus is explicit job scheduling, process isolation, durable state, and worker failure recovery. A Go coordinator is a later extension.

## Project status

**Milestones 1–3 are implemented:** the local Python CLI ingests recordings, indexes their timestamped windows with independent CPU CLAP workers, persists embeddings, and searches them with text. The durable SQLite queue supports atomic claims, attempt tokens, lease renewal, bounded retries, and recovery across worker/process restarts. `atlas ingest`, `atlas worker`, `atlas search TEXT`, and `atlas status` (snapshot or live) are available.

The systems demo runs one- and two-worker comparisons and terminates a claimed worker to demonstrate reassignment and complete indexing. Audio-example queries, playback, and the later service/UI extensions remain future work. The remaining sections describe the development contract; only functionality identified above is delivered.

## Setup and first search

Tested on Linux with Python 3.12.3 and CPU inference. From the repository root:

```bash
python3.12 -m venv eavenv
source eavenv/bin/activate
# Install the smaller CPU build before resolving the remaining dependencies.
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.lock
python -m pip install --no-build-isolation --no-deps -e .

atlas ingest audio-files/rehearsal.wav
atlas worker --id worker-1
atlas status
atlas search "brass playing a slow melody" --top-k 5
```

Supply your own recording; `audio-files/` contains no bundled demo. `requirements.lock` records the tested Linux CPU environment, including transitive dependencies. Direct dependencies are pinned in `pyproject.toml`: Typer, Rich, PyTorch, Transformers, NumPy, and the approved SoundFile/SciPy audio dependencies. No additional test framework is needed.

SoundFile decodes the managed input; SciPy performs polyphase resampling. WAV and FLAC are recommended. Other formats depend on the installed libsndfile build. The tested Linux SoundFile wheel bundles libsndfile, so no FFmpeg or external decoder was required; source/platform-specific SoundFile installations may require system libsndfile. Invalid, empty, and non-finite audio is rejected before jobs are published.

All processes must use the same data directory on a local filesystem. The default is `./data` relative to the current directory; set `ATLAS_DATA_DIR` or use `atlas --data-dir /absolute/path ...` to make the location explicit. It holds SQLite, managed audio copies, `.npy` embeddings, and the Hugging Face model cache. These artifacts are ignored by Git. Keep managed files in place after indexing; deleting the original source recording is safe.

First use downloads the approximately 615 MB checkpoint plus processor files. The model is pinned to `laion/clap-htsat-unfused` revision `8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a`. The installed Transformers 4.52.4 API was checked: `ClapProcessor` takes `audios=`, and `get_audio_features`/`get_text_features` return projected tensors of shape `(batch, 512)`. Both paths use inference mode and L2 normalization. Model/revision, dimension, and preprocessing version are checked against the index contract; use a separate data directory for a different contract.

Workers load one CPU model each, drain currently available jobs, and by default exit when no claim is available. `--wait-for-leases` keeps an idle worker polling while pending or leased jobs remain, allowing it to reclaim expired assignments automatically; it exits when all current jobs are completed or failed. Neither mode waits for future ingests after the queue becomes terminal. Without that option, restart `atlas worker` after an interrupted worker's lease expires to reclaim its work. Defaults are a 120-second lease, a 20-second background heartbeat, and three total attempts; `--lease-seconds` and `--heartbeat-seconds` configure the worker timings. `--cpu-threads` defaults to two. `Ctrl-C`/termination leaves unfinished leases recoverable.

`atlas status` shows recording progress, persisted counts, retries, per-worker completed counts, worker last-seen/lease times in UTC, and last errors. `atlas status --watch --interval 0.5` refreshes the display until Ctrl-C (default interval: one second). Status labels describe assignment authority: `leased`, `expired lease`, or `no lease`; they do not claim a worker is alive or dead. Reading status does not reclaim work; scheduling does that on the next claim. Completed chunk time is the sum of accepted attempts' decoding/inference time, not wall-clock throughput.

`atlas worker --max-attempts 3` configures the total claim limit, including the first attempt. Use the same attempt limit across workers sharing an index. Counts survive restarts; failed jobs remain terminal. Lowering the limit marks exhausted pending jobs failed on the next claim. Lease timings must be finite and the heartbeat interval must be shorter than the lease. If heartbeat renewal fails, the worker discards its computation and exits with a nonzero CLI status; its outstanding lease remains recoverable after expiration. Each concurrent worker needs a distinct `--id`.

Queue writes use SQLite WAL, `BEGIN IMMEDIATE`, foreign keys, a ten-second busy timeout, and explicit `synchronous=FULL`. Embedding writes flush the file and directory before publishing the database reference; first-created artifact directories are also synced in their parent. Completion checks assignment authority again after the file write. Accepted-attempt retries acknowledge the original result without rewriting it. A crash or a losing completion can leave an unreferenced artifact on disk; automatic artifact cleanup is not implemented. These guarantees assume a local filesystem that honors SQLite locking, atomic rename, and fsync.

Text search loads its own model, so run it after the indexing worker exits on a memory-constrained laptop. Search groups completed windows by recording ID and ranks each recording by its highest window cosine score. `--top-k` counts recordings (default five), and `--windows-per-recording` controls how many best-scoring timestamps appear beneath each recording (default three). Each recording appears once with its name, full ID, managed path, and best score; its windows show actual unpadded start/end timestamps and individual cosine scores, ordered best first. Overlapping windows remain separate within their recording. There is no minimum similarity threshold, so the displayed windows are the best available candidates rather than guaranteed semantic matches. Embeddings are exact-search ranked with NumPy; scores are not confidence percentages. This presentation change works with existing indexes and requires no re-ingestion or re-embedding. Audio-example search (`--audio`) is not yet implemented.

## Running validation

```bash
source eavenv/bin/activate
python -m unittest discover -s tests -v
python -m pip check
```

The lightweight tests use fake embeddings and generated audio. They cover the CLI flow, recording grouping and ranking by maximum window score, per-recording window limits, tail/padding/resampling policy, validation errors, persistent search results, incompatible metadata, atomic cross-process claims, process termination and reassignment, stale and concurrent duplicate completions, heartbeat renewal during inference and renewal failure, retry limits across restarts, invalid timings, artifact-write failure, lease expiration during artifact writes, database rollback during completion, and process termination after artifact rename before database acceptance. Systems tests also run two independent worker processes, terminate one after a claim, verify the survivor completes the entire index, check late and duplicate completions, refresh live status across a completion, and exercise a heartbeat racing with successful completion. They do not download CLAP.

Run the separate real-model smoke test with an identifiable clip no longer than ten seconds:

```bash
python tests/smoke_clap.py --fixture /path/to/short-dog-bark.wav \
    --query "a dog barking" --distractor "a person playing a violin"
```

This caches the real checkpoint under `data/models` and uses a temporary index. It places the event after ten seconds of silence, indexes all windows, reopens the database, and checks that the matching query retrieves the event over silence and an unrelated query. After the checkpoint is cached, `HF_HUB_OFFLINE=1` runs it without network access. Listen to the fixture and inspect its timestamps for a manual semantic relevance check; passing a vector-shape check alone is insufficient.

Development smoke result: [ESC-50's `1-100032-A-0.wav`](https://github.com/karolpiczak/ESC-50/blob/master/audio/1-100032-A-0.wav), a five-second dog-bark fixture, was downloaded outside the repository. See the dataset's [attribution and license](https://github.com/karolpiczak/ESC-50#license). With ten seconds of silence prepended, “a dog barking” retrieved 5–15s (0.5280) and 10–15s (0.4776), ahead of silence at 0–10s (0.2523); the violin query's best score was 0.2210. One worker indexed three chunks in 1.75s (1.716 chunks/s), excluding the 22.44s cached model load and ingestion. This is one small fixture on the development machine; see the separate systems comparison below.

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

The currently implemented text-search workflow is:

```bash
# Ingest a recording and enqueue its windows.
atlas ingest rehearsal.wav

# Run one worker until the available queue is drained.
atlas worker --id worker-1

# Inspect recording progress, worker activity, and retries.
atlas status

# Find five recordings, each with up to three best-matching windows.
atlas search "brass playing a slow melody" --top-k 5
```

The planned audio-example interface is `atlas search --audio example.wav --top-k 5`.

## Systems demo and worker comparison

Ingest a recording, then start workers in separate terminals using the same absolute data directory:

```bash
atlas --data-dir /absolute/path/data ingest rehearsal.wav

# Terminal A
atlas --data-dir /absolute/path/data worker --id worker-1 --wait-for-leases \
    --lease-seconds 5 --heartbeat-seconds 1

# Terminal B
atlas --data-dir /absolute/path/data worker --id worker-2 --wait-for-leases \
    --lease-seconds 5 --heartbeat-seconds 1

# Terminal C
atlas --data-dir /absolute/path/data status --watch --interval 0.5
```

When worker-1 has a leased chunk, interrupt it with Ctrl-C in terminal A. Worker-2 drains available chunks, waits for outstanding leases, reclaims the interrupted assignment after expiration, and exits when the queue is terminal. Watch retries increase and the recording reach complete indexing. Use a sufficiently long recording to interrupt before completion; short jobs can finish before a manual interruption. Choose leases long enough for the laptop's scheduling delays. Each process loads its own CLAP model; two workers duplicate model memory and may be slower than one.

For a deterministic demonstration and a measured comparison, run:

```bash
source eavenv/bin/activate
# Scheduling checks only: generated audio, fake vectors, no checkpoint download.
python tests/systems_demo.py --fake

# Actual CPU CLAP, using the existing cache; omit HF_HUB_OFFLINE=1 for first download.
HF_HUB_OFFLINE=1 python tests/systems_demo.py > data/systems-demo-report.json

# Optional real recording (longer than 15 seconds) and model cache root.
HF_HUB_OFFLINE=1 python tests/systems_demo.py --fixture rehearsal.wav --cache-dir data
```

The script uses separate temporary indexes for one worker, two workers, and two workers with a termination. It synchronizes loaded workers before timing indexing. In the recovery phase it pauses one child immediately after a durable claim, terminates that child, and verifies the survivor reclaims the same job with a fresh token and completes every chunk. It reopens the database, validates all accepted files, rejects the terminated attempt's late completion, and acknowledges a duplicate accepted completion without changing its artifact or adding results. Child errors and phase timeouts fail the script; it cleans up only its own processes and temporary indexes. Existing recordings/indexes are untouched. The default fixture is generated 30-second mono tone/silence (six windows); it measures systems behavior, not semantic search relevance.

JSON reports include wall-clock indexing time and chunks/s, separate startup/model-load times, mean accepted-attempt decoding/inference time, per-process peak RSS, worker participation, and recovery checks. Indexing time excludes ingestion and model loading, and includes result persistence, idle polling and child shutdown; recovery time includes waiting for lease expiration. Peak RSS includes imports/model load, is measured separately for each surviving process, and must not be added as if the peaks occurred simultaneously. Fake-mode timings are scheduling checks, not CLAP performance results. This is local multiprocessing, not multi-machine execution.

Development measurement (2026-11-07): Linux/Python 3.12, AMD Ryzen 5 7520U, approximately 7 GiB system RAM, CPU CLAP with two PyTorch threads per worker, cached checkpoint, generated 30-second tone/silence fixture, six windows per fresh index. The automated test suite was stopped before this comparison.

| Run | Completed | Indexing seconds | Chunks/s | Startup seconds | Mean decode/inference seconds | Peak worker RSS (MiB) |
| --- | --- | --- | --- | --- | --- | --- |
| One worker | 6 | 4.218 | 1.423 | 23.954 | 0.409 | 559.6 |
| Two workers | 3 + 3 | 3.550 | 1.690 | 11.264 | 0.586 | 561.6 / 558.1 |
| Terminate one of two; survivor recovers | 6 | 3.918 | 1.531 | 8.484 | 0.404 | 557.2 (survivor only) |

The recovery run recorded one retry, completed all six chunks, rejected the old token, and preserved the accepted embedding on duplicate completion. Two workers were faster in this small comparison; this single run is not a general speedup claim. Startup varies with filesystem/model caches, and longer recordings, CPU contention, and memory pressure can change the outcome. The raw report is generated at `data/systems-demo-report.json`; rerun the command to measure your machine.

Search output should include recording ID/name, start and end timestamps, and cosine similarity. Treat similarity as a ranking score, not a probability or confidence percentage. Timestamped playback is a later usability improvement; the CLI MVP can return file paths and timestamps.

## Audio and embedding pipeline

1. Copy the input into managed local storage and validate that it can be decoded.
2. Decode, convert to mono, and resample to 48 kHz for the selected CLAP checkpoint.
3. Create logical 10-second windows with a 5-second stride. Store timestamps rather than duplicating overlapping audio files.
4. Workers read their assigned windows, apply the checkpoint's processor, and compute embeddings in inference mode.
5. L2-normalize embeddings and persist accepted results.
6. Embed a text or audio query with the same checkpoint and preprocessing contract, then rank completed chunks by dot product of normalized vectors.

Define a deterministic tail policy: emit windows starting at 0, 5, 10, ... seconds while the start is before the recording ends; pad short windows to 10 seconds for inference. Store the actual unpadded end timestamp. Reject empty or undecodable recordings with a clear CLI error.

The vertical slice validates the managed copy in bounded blocks, creates logical window timestamps, and decodes/converts/resamples each requested source window to mono 48 kHz on demand. This keeps decoded audio bounded to one window. Tail windows are explicitly zero-padded before the CLAP processor, rather than relying on its default repeat-padding behavior. The contract identifier is `mono-polyphase-48k-window10-stride5-zero-pad-v1`.

Use the model's text and audio projection outputs in their shared embedding space. Record model ID/revision and preprocessing version with each index; reject searches across incompatible versions. Confirm the installed library's feature-return API during implementation and pin the working dependency versions.

Use exact NumPy similarity search over completed embeddings. Group results by recording, rank recordings by their best window, and show their highest-scoring windows underneath. A vector database is unnecessary for this demo. Adjacent overlapping windows may be merged for presentation in a later improvement.

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

1. **Vertical slice — implemented:** ingest one recording, process with one worker, persist embeddings, and return text-search timestamps.
2. **Durable queue — implemented:** atomic claims, lease tokens, heartbeats, retries, and restart-safe state.
3. **Systems demo — implemented:** two workers, live CLI status, worker termination, reassignment, complete indexing, and measured one/two-worker comparisons.
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
