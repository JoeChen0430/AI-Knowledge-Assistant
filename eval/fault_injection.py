"""Resilience under injected failure.

Builds an index from a corpus (no faults), then sends the same question set
through the full RAG pipeline twice: once with no faults, once with a share of
model calls (chat and embedding) failing at random. Reports how many requests
still got a usable response and which rung of the degradation ladder served it.

    # real models (Ollama must be running), 30% of calls fail
    python eval/fault_injection.py --corpus ./data --rate 0.3 --requests 500

    # no model server needed
    python eval/fault_injection.py --backend fake --synthetic-docs 40 --rate 0.3

Success = the user got an answer, a cached answer, the matching passages, or
a policy refusal ("not in your documents"). Failure = the request errored or
the system had nothing at all to show ("unavailable"). ``--naive`` adds a run
with every resilience feature off (no fallback model, no retries, no keyword
fallback, no cache) where only a freshly generated answer counts, which is
roughly what the original app could do.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.common import (  # noqa: E402
    copy_corpus,
    describe_latency,
    make_backend,
    make_workspace,
    question_from_chunk,
    synthesize_corpus,
    write_json,
)

SUCCESS_OUTCOMES = {"answered", "policy_refused"}


def build_questions(settings, n: int, repeat_share: float, seed: int) -> list[str]:
    """Questions drawn from the indexed chunks, with a share of repeats so the
    cache tier sees realistic traffic (people re-ask)."""
    from aka.db.models import Chunk
    from aka.db.session import session_scope
    from aka.ingest.vectorstore import IndexManager
    from sqlalchemy import select

    rng = random.Random(seed)
    with session_scope(settings) as session:
        store = IndexManager(session, settings).active_store(settings.DEFAULT_TENANT)
        ids = list(session.scalars(select(Chunk.id).where(Chunk.tenant_id == settings.DEFAULT_TENANT)))
        if store is None or not ids:
            raise SystemExit("the index is empty; nothing to ask about")
        sample = rng.sample(ids, k=min(len(ids), max(1, n)))
        texts = store.collection.get(ids=sample, include=["documents"])["documents"]

    unique = [q for q in (question_from_chunk(t or "", rng) for t in texts) if q]
    if not unique:
        raise SystemExit("could not derive questions from the corpus")

    questions: list[str] = []
    for i in range(n):
        if questions and rng.random() < repeat_share:
            questions.append(rng.choice(questions))
        else:
            questions.append(unique[i % len(unique)])
    return questions


def run_condition(settings, inner_client, questions, *, rate, seed, timeout_delay, naive) -> dict:
    from aka.db.session import session_scope
    from aka.llm.breaker import all_breakers, clear_breaker_registry
    from aka.llm.cache import AnswerCache
    from aka.llm.embedding import ResilientEmbedder
    from aka.llm.faults import FaultInjectingClient
    from aka.llm.router import ModelRouter
    from aka.rag.pipeline import RagPipeline

    clear_breaker_registry()
    run_settings = settings.model_copy()
    if naive:
        run_settings.LLM_FALLBACK_MODELS = []
        run_settings.EMBED_RETRY_ATTEMPTS = 1
        run_settings.LEXICAL_FALLBACK_ENABLED = False
        run_settings.QUERY_EMBED_CACHE_SIZE = 0

    client = FaultInjectingClient(
        inner_client, rate=rate, seed=seed, timeout_delay_s=timeout_delay
    )
    router = ModelRouter(run_settings, client=client)
    embedder = ResilientEmbedder(client, run_settings)
    cache = AnswerCache(
        ttl_s=run_settings.ANSWER_CACHE_TTL_S,
        max_entries=0 if naive else run_settings.ANSWER_CACHE_MAX_ENTRIES,
    )

    outcomes, modes, retrieval = Counter(), Counter(), Counter()
    degraded = tiers = 0
    latencies: list[float] = []
    started = time.perf_counter()
    for i, question in enumerate(questions, 1):
        with session_scope(run_settings) as session:
            pipeline = RagPipeline(session, run_settings, router=router, embedder=embedder, cache=cache)
            result = pipeline.answer(
                question, tenant_id=run_settings.DEFAULT_TENANT, user_id="eval"
            )
        outcome = result.outcome
        if naive and outcome == "answered" and result.mode != "generated":
            outcome = "fallback_needed"
        outcomes[outcome] += 1
        modes[result.mode] += 1
        retrieval[result.retrieval_mode] += 1
        degraded += result.degraded
        tiers += result.model_tier > 0
        latencies.append(result.latency_s)
        if i % 50 == 0:
            ok = sum(outcomes[o] for o in SUCCESS_OUTCOMES)
            print(f"  {i}/{len(questions)}  success so far {ok / i:.1%}", flush=True)

    n = len(questions)
    ok = sum(outcomes[o] for o in SUCCESS_OUTCOMES)
    return {
        "fault_rate": rate,
        "naive": naive,
        "requests": n,
        "success_rate": round(ok / n, 4),
        "answered_rate": round(outcomes["answered"] / n, 4),
        "outcomes": dict(outcomes),
        "modes": dict(modes),
        "retrieval_modes": dict(retrieval),
        "degraded_share": round(degraded / n, 4),
        "fallback_model_share": round(tiers / n, 4),
        "latency": describe_latency(latencies),
        "wall_s": round(time.perf_counter() - started, 1),
        "faults": client.fault_summary(),
        "breakers_at_end": all_breakers(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", choices=["ollama", "fake"], default="ollama")
    parser.add_argument("--corpus", type=Path, default=Path("data"), help="documents to index (copied)")
    parser.add_argument("--synthetic-docs", type=int, default=0, help="generate N docs instead of --corpus")
    parser.add_argument("--rate", type=float, default=0.3, help="share of model calls that fail")
    parser.add_argument("--requests", type=int, default=300)
    parser.add_argument("--repeat-share", type=float, default=0.3, help="share of repeated questions")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout-delay", type=float, default=None,
                        help="seconds an injected timeout waits (default 0.5 ollama, 0.02 fake)")
    parser.add_argument("--naive", action="store_true", help="also run with resilience disabled")
    parser.add_argument("--workdir", type=Path, default=None, help="workspace (default: temp dir)")
    parser.add_argument("--out", default="eval/results/fault_injection.json")
    args = parser.parse_args()

    overrides = {"REQUEST_BUDGET_S": 120}
    if args.backend == "fake":
        # Fake embeddings are bag-of-words; see tests/test_api.py.
        overrides.update(RETRIEVAL_MIN_SCORE=0.05, LLM_MODEL="fake-large", LLM_FALLBACK_MODELS="fake-small")
    settings, root = make_workspace(args.workdir, **overrides)
    timeout_delay = args.timeout_delay if args.timeout_delay is not None else (
        0.02 if args.backend == "fake" else 0.5
    )

    data_dir = Path(settings.DATA_DIR)
    if args.synthetic_docs:
        synthesize_corpus(data_dir, docs=args.synthetic_docs, words_per_doc=600, seed=args.seed)
    else:
        if not args.corpus.exists():
            raise SystemExit(f"corpus not found: {args.corpus}")
        copied = copy_corpus(args.corpus, data_dir, settings.ALLOWED_EXTENSIONS)
        print(f"copied {copied} files from {args.corpus}")

    backend = make_backend(args.backend)
    print(f"workspace: {root}\nmodels: {settings.LLM_MODEL} -> {settings.LLM_FALLBACK_MODELS}, "
          f"embed {settings.EMBED_MODEL}")

    from aka.db.session import session_scope
    from aka.ingest.pipeline import IncrementalIngestor

    print("indexing (no faults)...")
    with session_scope(settings) as session:
        report = IncrementalIngestor(session, settings, client=backend).run(settings.DEFAULT_TENANT)
    print(f"  {report.chunks_embedded} chunks in {report.duration_s:.1f}s")

    questions = build_questions(settings, args.requests, args.repeat_share, args.seed)
    conditions = [("baseline", 0.0, False), (f"faults@{args.rate:.0%}", args.rate, False)]
    if args.naive:
        conditions.append((f"naive faults@{args.rate:.0%}", args.rate, True))

    results = {}
    for name, rate, naive in conditions:
        print(f"\n== {name} ==")
        results[name] = run_condition(
            settings, backend, questions, rate=rate, seed=args.seed,
            timeout_delay=timeout_delay, naive=naive,
        )

    print("\ncondition                success  answered  degraded  p95 latency  outcomes")
    for name, r in results.items():
        print(f"{name:<24} {r['success_rate']:>7.1%}  {r['answered_rate']:>8.1%}  "
              f"{r['degraded_share']:>8.1%}  {r['latency'].get('p95_s', 0):>10.2f}s  {r['outcomes']}")

    write_json(args.out, {
        "backend": args.backend,
        "models": {"chain": [settings.LLM_MODEL, *settings.LLM_FALLBACK_MODELS], "embed": settings.EMBED_MODEL},
        "corpus": str(args.corpus) if not args.synthetic_docs else f"synthetic:{args.synthetic_docs}",
        "chunks": report.chunks_embedded,
        "seed": args.seed,
        "repeat_share": args.repeat_share,
        "timeout_delay_s": timeout_delay,
        "results": results,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
