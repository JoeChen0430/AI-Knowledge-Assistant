# AI Knowledge Assistant

A local-first RAG platform over your own documents (PDF, DOCX, TXT, MD), built
to keep answering when parts of it fail and to ask a human before doing
anything risky. Everything runs on your machine: Ollama for models, Chroma for
vectors, SQLite or Postgres for state.

**Stack:** Python · FastAPI · LlamaIndex (parsing/chunking) · ChromaDB · Ollama · SQLAlchemy · Streamlit · Docker Compose · OpenTelemetry · Prometheus

## What it does

**Incremental ingestion with blue/green index swaps** — `aka/ingest/`
- Every file is content-hashed (SHA-256); only changed files are parsed and
  embedded. Vectors for unchanged documents are copied into the new index
  version instead of being recomputed.
- Each rebuild writes a *new* Chroma collection. It is smoke-checked (chunk
  count, a probe query) and only then flipped to ACTIVE in one transaction, so
  queries never see an empty or half-built index. A failed build leaves the
  previous version serving; one retired version is kept for rollback.
- Builds run in a worker off a durable DB-backed queue: leases with heartbeat,
  retries with backoff, a dead-letter state, and one build per tenant at a time.

**Resilience: circuit breakers, time budgets, a fallback chain** — `aka/llm/`, `aka/rag/pipeline.py`
- Per-model circuit breakers (closed → open → half-open) for every LLM tier
  and the embedding model, so a dead backend fails fast instead of costing a
  timeout per request.
- A per-request time budget; each model call's timeout is clamped to what is
  left of it.
- Degradation ladder: **primary model → smaller model → stale cached answer →
  retrieval-only (the matching passages)**. If the embedding model is down,
  retrieval falls back to keyword search (CJK-aware), and a cached answer is
  served only if every chunk it cited is still in the index unchanged.
- Every degraded response says so (`degraded`, `degrade_reason`, `mode`).
- A fault injector (`aka/llm/faults.py`) fails a configurable share of model
  calls with the same exceptions a real outage produces.

**Confidence policy engine with human review** — `aka/hitl/`
- Every consequential action is a proposal evaluated by a rule: ALLOW, REVIEW
  or DENY. Rules fail closed.
- Memory writes: auto-commit above a confidence threshold, drop below a floor,
  queue the middle band; contradictions always go to a human.
- Answers: refused below a grounding floor, shown-but-flagged and queued in a
  middle band, released above it.
- Humans resolve the queue, and 👍/👎 on released answers or "forget" on
  stored memories are verdicts too. The audit log links each human verdict to
  the automatic one it overrides, so the **override rate** is a query, and
  `eval/calibrate_policy.py` tunes the thresholds from it.

Also: tiered memory (recent turns, rolling session summary, durable facts
with decay), JWT auth with tenant isolation, prompt-injection fencing,
OpenTelemetry traces and Prometheus metrics (`/metrics`). Documents and
questions can be in Chinese: chunking splits on CJK punctuation, and grounding
and keyword fallback match CJK bigrams.

A diagrammed walkthrough of the design — overview, one worked example with
real numbers, a deep-dive per module, failure modes and pitfalls — is in
[`docs/architecture-dossier.html`](docs/architecture-dossier.html) (open it in a browser).

## Results

Measured, not projected. The raw output behind each table is committed under
[`docs/results/`](docs/results/).

### Resilience under injected failure

[`eval/fault_injection.py`](eval/fault_injection.py) with real models
(llama3 → llama3.2:1b, nomic-embed-text) on an Apple M4 with 24 GB: 500
requests per condition, 30% of chat and embedding calls failing (half as
immediate errors, half as timeouts), seed 42.

| Condition | Success | Answered | Degraded | p95 latency |
|---|---|---|---|---|
| No faults | 100.0% | 98.4% | 2.2% | 31.7 s |
| 30% of model calls failing | **100.0%** | 98.0% | 30.4% | 32.2 s |
| 30% failing, resilience off (`--naive`) | 60.8% | 59.6% | 50.2% | 36.2 s |

Under faults, 443 requests were answered by generation (about 99 of them by
the fallback model), 34 from the stale-answer cache and 13 with the retrieved
passages only; the other 10 were policy refusals made on real evidence. With
resilience off, 110 requests had nothing to show and 86 more would have needed
a fallback the naive setup does not have.

**Caveat.** The corpus behind this run is 14 documents (25 chunks), so it has
25 distinct questions, each asked about 20 times. Repeats let the answer cache
and the query-vector cache absorb faults, which flatters the resilient
condition; a larger corpus with more distinct questions is the stricter test.

### Not measured yet

- **Ingest** — full rebuild vs. single-document update with real embeddings on
  a large corpus (`eval/ingest_benchmark.py --synthetic-docs 4000`).
- **Override rate** — needs a period of real review-queue and feedback use
  before `eval/calibrate_policy.py` has human verdicts to report.

## Architecture

| Component | Role |
|---|---|
| `api` (FastAPI, `aka/api/`) | chat, documents, index status, review queue, stats, health |
| `worker` (`aka/ingest/worker.py`) | claims reindex jobs, builds and swaps index versions |
| `ui` (Streamlit, `app.py`) | thin HTTP client of the API |
| SQLite / Postgres | documents, chunks, index versions, jobs, conversations, memory, approvals, audit log |
| Chroma | one collection per index version |
| Ollama | generation chain + embeddings |

A request: embed the question → assemble memory → retrieve from the ACTIVE
index (tenant-filtered) → rerank → build the prompt with untrusted-content
fencing → generate down the fallback chain → policy gate → respond.

## Quick start (local)

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env    # single-user mode (AUTH_ENABLED=false); edit as needed

ollama pull llama3 && ollama pull llama3.2:1b && ollama pull nomic-embed-text
ollama serve            # if the Ollama app is not already running

# API (runs the ingest worker in-process by default)
uvicorn aka.api.main:app --port 8080
# UI, in another terminal
streamlit run app.py
```

Put documents in `data/` (or upload them in the UI), then build the index from
the UI's Documents page, `POST /index/rebuild`, or `python -m aka.cli ingest`.
API docs: http://localhost:8080/docs.

## Quick start (Docker)

```bash
docker compose up --build     # api :8080, ui :8501, postgres, chroma, worker
```

Compose uses Ollama on the host by default (`host.docker.internal:11434`); on
macOS a container cannot use the GPU, so host Ollama is much faster. To run
Ollama in a container instead:
`OLLAMA_BASE_URL=http://ollama:11434 docker compose --profile ollama up --build`.
`./data` is mounted into the containers; state lives in named volumes
(`docker compose down -v` wipes it).

## Measuring it

The numbers come from these scripts, run on your hardware and corpus. Each
one works in a throwaway workspace and never touches `./data`, `./var` or
`./chroma_db`. Results are written to `eval/results/`, after every stage, so an
interrupted run keeps what finished.

With real models these runs take hours (llama3 answers in roughly 15–20 s per
request on an M4), so prefix them with `caffeinate -i` on macOS — a sleeping
laptop pauses the run — and don't run two at once: they share one Ollama.

```bash
# 1. Single-document update vs. full rebuild, and index availability during each
#    (~12k chunks; two full builds, roughly 30-35 min on an M-series Mac)
caffeinate -i python eval/ingest_benchmark.py --synthetic-docs 4000

# 2. Success rate with 30% of model calls failing (chat + embedding)
caffeinate -i python eval/fault_injection.py --corpus ./data --rate 0.3 --requests 500 --naive --max-tokens 200

# 3. Override rate, and thresholds that would have agreed with the humans
python eval/calibrate_policy.py --until 2026-11-01   # period before a change
python eval/calibrate_policy.py --since 2026-11-01   # period after
```

- **Ingest benchmark:** builds the corpus with blue/green (the cost of a full
  rebuild), changes one document and runs the incremental update, then times
  the original strategy on the same corpus — drop the collection, re-embed
  everything, insert in LlamaIndex's 2048-node batches. A probe thread queries
  the index every 50 ms throughout and records how long it was empty or
  incomplete. Synthetic documents are fine for timing: embedding cost depends
  on token count, not content.
- **Fault injection:** indexes the corpus, then runs the same question set
  with no faults (`--skip-baseline` drops this), with `--rate` faults, and
  (`--naive`) with every resilience feature disabled. Success = an answer, a
  cached answer, the passages, or a policy refusal made on real evidence;
  failure = an error, nothing to show, or a refusal caused only by a fault
  losing the retrieval. `--max-tokens` caps answer length — availability does
  not need long answers. The script prints how many distinct questions the
  corpus yields and warns when repeats would flatter the result.
- **Calibration:** reads the audit log of a running deployment. The measured
  override rate is real; the threshold replay is a projection over the same
  human verdicts. Labels come only from people using the review queue and
  feedback buttons.

`--backend fake` runs scripts 1 and 2 without a model server (a deterministic
stand-in, `aka/llm/fake.py`), useful for checking the harness itself.

## Configuration

All settings are environment variables (or `.env`); see `.env.example` and
`aka/settings.py`. The ones you are most likely to touch:

| Setting | Default | |
|---|---|---|
| `LLM_MODEL` / `LLM_FALLBACK_MODELS` | `llama3` / `llama3.2:1b` | generation chain |
| `EMBED_MODEL` | `nomic-embed-text` | try `bge-m3` for Chinese-heavy corpora |
| `REQUEST_BUDGET_S` | `120` | total time per chat request |
| `BREAKER_FAILURE_THRESHOLD` | `5` | consecutive failures before a breaker opens |
| `ANSWER_MIN_GROUNDING` / `ANSWER_REVIEW_GROUNDING` | `0.15` / `0.40` | refuse / review / release bands |
| `MEMORY_AUTO_COMMIT_CONFIDENCE` / `MEMORY_REJECT_CONFIDENCE` | `0.80` / `0.35` | memory write bands |
| `AUTH_ENABLED` | `true` | `false` = single-user local mode |
| `FAULT_INJECTION_RATE` | `0` | chaos testing; refused when `ENV=production` |

With auth on: create users with `python -m aka.cli create-user --email you@example.com --role admin`,
or bootstrap the first admin through `POST /auth/bootstrap` with `BOOTSTRAP_ADMIN_TOKEN`.
Roles: `user` < `reviewer` (review queue, stats) < `admin` (rebuild, rollback, users).

## Development

```bash
pytest -q        # runs offline: fake model backend, temp SQLite + Chroma
```

```
aka/
  api/        FastAPI app, routes, request dependencies
  ingest/     hashing, chunking, incremental pipeline, blue/green index, queue, worker
  llm/        Ollama client, router, breaker, embedding retries, cache, fault injector, fake
  rag/        request pipeline, prompt assembly, grounding
  memory/     L0/L1/L2 memory, extraction, decay
  hitl/       policy engine, approvals, audit, effects, calibration
  security/   auth, path containment, injection checks, secret scanning
  obs/        logging, metrics, tracing
  db/         models, sessions
eval/         benchmark and evaluation scripts
tests/        pytest suite
```

## Privacy

Models, vectors and data stay on your machine. Nothing is sent to an external
API.
