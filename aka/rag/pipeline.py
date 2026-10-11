"""The request pipeline.

Seven traced stages, in order:

    1 embed_query      question -> vector (retried; keyword fallback if down)
    2 memory.assemble  L0 turns + L1 summary + L2 facts
    3 retrieve         L3 corpus, tenant-filtered
    4 rerank           score floor + diversity
    5 prompt.build     structured assembly with untrusted fencing
    6 generate         router (primary -> fallback -> cache -> extractive)
    7 policy.release   grounding gate + output-side injection check

Degradation ladder, in the order it is attempted:

    a. fallback model            (handled inside ModelRouter)
    b. stale cached answer       same question, same evidence ids
    c. extractive passages       retrieval succeeded, generation did not
    d. honest refusal            nothing to stand on

and, on the retrieval side, when the embedding model is down: a cached answer
whose evidence is verified unchanged, else keyword retrieval feeding the same
generation ladder.

Every one of those sets ``degraded=True`` and a human-readable reason that the
API returns and the UI shows. Silently serving a degraded answer as if it were
a normal one is the thing this design is trying to avoid.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from aka.db.models import new_id
from aka.hitl import approvals as approval_queue
from aka.hitl.policy import Action, PolicyEngine, Proposal, Verdict
from aka.ingest.vectorstore import IndexManager, SearchHit
from aka.llm.cache import AnswerCache, get_answer_cache
from aka.llm.client import LLMError
from aka.llm.embedding import EmbeddingUnavailable, ResilientEmbedder
from aka.llm.router import AllModelsUnavailable, ModelRouter, get_router
from aka.memory.tiers import MemoryContext, TieredMemory
from aka.obs.logging import get_logger
from aka.obs.metrics import (
    chat_latency,
    chat_requests,
    degradations,
    retrieval_results,
    retrieval_top_score,
    timed,
)
from aka.obs.telemetry import (
    ATTR_CHUNK_IDS,
    ATTR_DEGRADE_REASON,
    ATTR_DEGRADED,
    ATTR_GROUNDING,
    ATTR_HITL_DECISION,
    ATTR_HITL_RULE,
    ATTR_SCORES,
    ATTR_TOP_SCORE,
    current_trace_id,
    set_attrs,
    stage,
)
from aka.rag.prompts import (
    REFUSAL_TEXT,
    build_extractive_answer,
    build_prompt,
    cited_indices,
    estimate_grounding,
    lexical_terms,
)
from aka.security.injection import assess, check_output
from aka.settings import Settings, get_settings

log = get_logger("aka.rag")

# What happened to a request, from the user's point of view. The resilience
# eval counts "unavailable" and "error" as failures; a policy refusal is the
# system working as designed.
OUTCOME_ANSWERED = "answered"
OUTCOME_POLICY_REFUSED = "policy_refused"
OUTCOME_UNAVAILABLE = "unavailable"
OUTCOME_ERROR = "error"


@dataclass
class AnswerResult:
    answer: str
    conversation_id: str
    trace_id: str | None = None
    message_id: str | None = None   # the assistant message; target of feedback

    degraded: bool = False
    degrade_reason: str | None = None
    mode: str = "generated"      # generated | cached | extractive | refused
    outcome: str = OUTCOME_ANSWERED
    retrieval_mode: str = "vector"   # vector | lexical | none
    model: str | None = None
    model_tier: int = 0

    evidence: list[SearchHit] = field(default_factory=list)
    citations: list[int] = field(default_factory=list)
    grounding: float = 0.0
    memory: MemoryContext | None = None

    policy_rule: str | None = None
    policy_verdict: str | None = None
    approval_id: str | None = None
    needs_review: bool = False
    review_reason: str | None = None

    latency_s: float = 0.0
    prompt_tokens: int = 0
    token_breakdown: dict = field(default_factory=dict)

    def mark_degraded(self, reason: str | None) -> None:
        """Accumulate reasons: a request can degrade at more than one stage
        (keyword retrieval *and* a fallback model), and the user should see both."""
        self.degraded = True
        if reason and reason not in (self.degrade_reason or ""):
            self.degrade_reason = f"{self.degrade_reason}; {reason}" if self.degrade_reason else reason

    def as_dict(self) -> dict:
        """Shape consumed by the API and the 'why this answer' panel."""
        return {
            "answer": self.answer,
            "conversation_id": self.conversation_id,
            "message_id": self.message_id,
            "trace_id": self.trace_id,
            "degraded": self.degraded,
            "degrade_reason": self.degrade_reason,
            "mode": self.mode,
            "outcome": self.outcome,
            "retrieval_mode": self.retrieval_mode,
            "model": self.model,
            "model_tier": self.model_tier,
            "grounding": round(self.grounding, 4),
            "citations": self.citations,
            "policy": {"rule": self.policy_rule, "verdict": self.policy_verdict},
            "approval_id": self.approval_id,
            "needs_review": self.needs_review,
            "review_reason": self.review_reason,
            "latency_s": round(self.latency_s, 3),
            "prompt_tokens": self.prompt_tokens,
            "token_breakdown": self.token_breakdown,
            "evidence": [
                {
                    "n": i + 1,
                    "chunk_id": hit.chunk_id,
                    "source": hit.source,
                    "score": round(hit.score, 4),
                    "excerpt": hit.text[:400],
                }
                for i, hit in enumerate(self.evidence)
            ],
            "memory": self.memory.as_dict() if self.memory else None,
        }


class RagPipeline:
    def __init__(
        self,
        session: Session,
        settings: Settings | None = None,
        router: ModelRouter | None = None,
        embed_client=None,
        embedder: ResilientEmbedder | None = None,
        cache: AnswerCache | None = None,
    ):
        self.session = session
        self.settings = settings or get_settings()
        self.router = router or get_router(self.settings)
        self.embedder = embedder or ResilientEmbedder(embed_client or self.router.client, self.settings)
        self.memory = TieredMemory(session, self.settings)
        self.index = IndexManager(session, self.settings)
        self.policy = PolicyEngine(self.settings)
        # `is not None`, not `or`: an empty cache has len() 0 and is falsy.
        self.cache = cache if cache is not None else get_answer_cache()

    # ----------------------------------------------------------------------
    def answer(
        self,
        question: str,
        *,
        tenant_id: str,
        user_id: str,
        conversation_id: str | None = None,
    ) -> AnswerResult:
        started = time.perf_counter()
        deadline = time.monotonic() + self.settings.REQUEST_BUDGET_S

        convo = self.memory.ensure_conversation(
            conversation_id=conversation_id,
            tenant_id=tenant_id,
            user_id=user_id,
            title=question[:80],
        )
        result = AnswerResult(
            answer="",
            conversation_id=convo.id,
            trace_id=current_trace_id(),
            message_id=new_id(),
        )

        # A question is user input, not document content, but the same
        # heuristics are informative — a user pasting an injection payload is
        # worth logging even though we do not block it.
        question_assessment = assess(question, surface="question")
        if question_assessment.flagged:
            log.info("rag.question_flagged", rules=question_assessment.as_dict()["rules"])

        self.memory.append_message(
            conversation_id=convo.id,
            tenant_id=tenant_id,
            role="user",
            content=question,
            trace_id=result.trace_id,
        )
        # Release the write lock before the slow part. On SQLite an open write
        # transaction held across generation would block every other writer
        # (other chats, the ingest worker) for the length of an LLM call.
        self.session.commit()

        try:
            # --- 1. embed --------------------------------------------------
            query_vector = self._embed_query(question, result, deadline)

            # --- 2. memory -------------------------------------------------
            memory_ctx = self.memory.build_context(
                tenant_id=tenant_id,
                user_id=user_id,
                conversation_id=convo.id,
                query_vector=query_vector,
            )
            result.memory = memory_ctx
            self.session.commit()  # reinforcement counters; same reason as above

            cached = None if query_vector is not None else self._verified_cached_answer(
                tenant_id, question
            )
            if cached is not None:
                # --- retrieval degraded, but a verified answer exists -------
                prompt_nonce = self._serve_cached(result, *cached)
            else:
                # --- 3/4. retrieve + rerank --------------------------------
                hits = self._retrieve(tenant_id, query_vector, question, result)
                result.evidence = hits

                # --- 5. prompt ---------------------------------------------
                with stage("prompt.build"), timed("prompt.build"):
                    prompt = build_prompt(question, hits=hits, memory=memory_ctx)
                result.prompt_tokens = prompt.prompt_tokens
                result.token_breakdown = prompt.token_breakdown
                prompt_nonce = prompt.nonce

                # --- 6. generate (with degradation ladder) -----------------
                cache_key = self.cache.make_key(tenant_id, question, [h.chunk_id for h in hits])
                self._generate(result, prompt, cache_key, deadline=deadline)

            # --- 7. policy gate --------------------------------------------
            self._release(result, question, tenant_id, prompt_nonce)

            if cached is None:
                self._maybe_cache(result, tenant_id, question, cache_key)

        except Exception as exc:
            log.exception("rag.pipeline_error", error_type=type(exc).__name__)
            self.session.rollback()
            result.answer = (
                "Something went wrong answering that. The failure has been recorded."
            )
            result.mark_degraded(f"pipeline error: {type(exc).__name__}")
            result.mode = "refused"
            result.outcome = OUTCOME_ERROR
            chat_requests.labels(outcome="error", degraded="true").inc()
        else:
            chat_requests.labels(
                outcome=result.outcome, degraded=str(result.degraded).lower()
            ).inc()

        result.latency_s = time.perf_counter() - started
        chat_latency.observe(result.latency_s)

        self.memory.append_message(
            conversation_id=convo.id,
            tenant_id=tenant_id,
            role="assistant",
            content=result.answer,
            trace_id=result.trace_id,
            message_id=result.message_id,
            meta={
                "mode": result.mode,
                "outcome": result.outcome,
                "degraded": result.degraded,
                "degrade_reason": result.degrade_reason,
                "retrieval_mode": result.retrieval_mode,
                "model": result.model,
                "grounding": round(result.grounding, 4),
                "policy": {"rule": result.policy_rule, "verdict": result.policy_verdict},
                "needs_review": result.needs_review,
                "chunk_ids": [h.chunk_id for h in result.evidence],
            },
        )
        return result

    # ----------------------------------------------------------------------
    def _embed_query(
        self, question: str, result: AnswerResult, deadline: float
    ) -> list[float] | None:
        """Query vector, or None when the embedding model cannot be reached."""
        with stage("embed_query"), timed("embed_query"):
            try:
                return self.embedder.embed_query(question, deadline=deadline)
            except EmbeddingUnavailable as exc:
                lexical = self.settings.LEXICAL_FALLBACK_ENABLED
                result.retrieval_mode = "lexical" if lexical else "none"
                result.mark_degraded(
                    "embedding model unavailable; "
                    + ("matched documents by keyword instead" if lexical else "retrieval skipped")
                )
                degradations.labels(reason="embed_unavailable", mode=result.retrieval_mode).inc()
                log.warning("rag.embed_unavailable", error=str(exc))
                return None

    def _retrieve(
        self,
        tenant_id: str,
        query_vector: list[float] | None,
        question: str,
        result: AnswerResult,
    ) -> list[SearchHit]:
        with stage("retrieve", **{"aka.retrieval_mode": result.retrieval_mode}), timed("retrieve"):
            store = self.index.active_store(tenant_id)
            if store is None:
                retrieval_results.labels(hit="no_index").inc()
                log.warning("rag.no_active_index", tenant=tenant_id)
                return []
            # Tenant filter is applied server-side, never trusted from the client.
            where = {"tenant_id": tenant_id}
            if query_vector is not None:
                raw = store.query(query_vector, k=self.settings.RETRIEVAL_TOP_K, where=where)
            elif result.retrieval_mode == "lexical":
                raw = store.keyword_search(
                    lexical_terms(question), k=self.settings.RETRIEVAL_TOP_K, where=where
                )
            else:
                raw = []

        with stage("rerank"), timed("rerank"):
            kept = [h for h in raw if h.score >= self.settings.RETRIEVAL_MIN_SCORE]
            kept.sort(key=lambda h: h.score, reverse=True)
            kept = self._diversify(kept)[: self.settings.RETRIEVAL_RERANK_TO]

        if raw:
            retrieval_top_score.observe(max(h.score for h in raw))
        retrieval_results.labels(hit="hit" if kept else "miss").inc()

        set_attrs(
            **{
                ATTR_CHUNK_IDS: [h.chunk_id for h in kept],
                ATTR_SCORES: [round(h.score, 4) for h in kept],
                ATTR_TOP_SCORE: max((h.score for h in raw), default=0.0),
            }
        )
        return kept

    @staticmethod
    def _diversify(hits: list[SearchHit], per_document: int = 2) -> list[SearchHit]:
        """Cap chunks per document so one long file cannot crowd out the rest."""
        seen: dict[str, int] = {}
        out = []
        for hit in hits:
            key = hit.document_id or hit.source
            if seen.get(key, 0) >= per_document:
                continue
            seen[key] = seen.get(key, 0) + 1
            out.append(hit)
        return out

    # --- cache tier ---------------------------------------------------------
    def _verified_cached_answer(self, tenant_id: str, question: str) -> tuple[dict, float] | None:
        """The last good answer to this exact question, if every chunk it was
        built from is still in the active index (chunk ids are content hashes,
        so present means unchanged)."""
        value, age_s = self.cache.get_by_question(self.cache.question_key(tenant_id, question))
        if not value:
            return None
        ids = [e["chunk_id"] for e in value.get("evidence", [])]
        store = self.index.active_store(tenant_id)
        if not ids or store is None or store.existing_ids(ids) != set(ids):
            return None
        return value, age_s

    def _serve_cached(self, result: AnswerResult, value: dict, age_s: float) -> str:
        result.answer = value["answer"]
        result.model = value.get("model")
        result.mode = "cached"
        result.evidence = [
            SearchHit(
                chunk_id=e["chunk_id"], text=e["text"], score=e["score"], metadata=e.get("metadata", {})
            )
            for e in value.get("evidence", [])
        ]
        result.mark_degraded(
            f"served a cached answer from {int(age_s // 60)} min ago; its sources are unchanged"
        )
        degradations.labels(reason="embed_unavailable", mode="cached").inc()
        return value.get("nonce", "")

    def _maybe_cache(self, result: AnswerResult, tenant_id: str, question: str, cache_key: str) -> None:
        # Only cache full-quality, released answers: caching a fallback's
        # output (or one the gate held back) would let a transient outage
        # poison later responses.
        if result.mode != "generated" or result.degraded:
            return
        if result.policy_verdict != Verdict.ALLOW.value:
            return
        self.cache.set(
            cache_key,
            {
                "answer": result.answer,
                "model": result.model,
                "evidence": [
                    {"chunk_id": h.chunk_id, "text": h.text, "score": h.score, "metadata": h.metadata}
                    for h in result.evidence
                ],
            },
            question_key=self.cache.question_key(tenant_id, question),
        )

    # ----------------------------------------------------------------------
    def _generate(self, result: AnswerResult, prompt, cache_key: str, *, deadline: float) -> None:
        try:
            generation = self.router.generate(prompt.messages, deadline=deadline)
        except AllModelsUnavailable as exc:
            self._degrade_without_generation(result, prompt, cache_key, exc)
            return

        result.answer = generation.text.strip()
        result.model = generation.model
        result.model_tier = generation.tier
        result.mode = "generated"
        if generation.degraded:
            result.mark_degraded(generation.degrade_reason)

    def _degrade_without_generation(
        self, result: AnswerResult, prompt, cache_key: str, exc: AllModelsUnavailable
    ) -> None:
        log.error("rag.all_models_unavailable", attempts=len(exc.attempts))

        cached, age_s = self.cache.get(cache_key, allow_stale=True)
        if cached:
            result.answer = cached["answer"]
            result.model = cached.get("model")
            result.mode = "cached"
            result.mark_degraded(
                f"model unavailable; served a cached answer from {int(age_s // 60)} min ago"
            )
            degradations.labels(reason="all_models_down", mode="cached").inc()
            return

        if prompt.evidence:
            result.answer = build_extractive_answer("", prompt.evidence)
            result.mode = "extractive"
            result.mark_degraded("model unavailable; showing matching passages only")
            degradations.labels(reason="all_models_down", mode="extractive").inc()
            return

        result.answer = (
            "The language model is unavailable and nothing in your documents "
            "matched this question closely enough to show."
        )
        result.mode = "refused"
        result.outcome = OUTCOME_UNAVAILABLE
        result.mark_degraded("model unavailable and no matching evidence")
        degradations.labels(reason="all_models_down", mode="refused").inc()

    # ----------------------------------------------------------------------
    def _release(self, result: AnswerResult, question: str, tenant_id: str, nonce: str) -> None:
        """Gate the answer on grounding and output-side checks."""
        with stage("policy.release"), timed("policy.release"):
            # Degraded non-generated modes are already labelled and are literal
            # document text; putting them through the grounding gate would
            # refuse the very fallback that exists to avoid refusing.
            if result.mode in ("extractive", "refused"):
                result.policy_verdict = Verdict.ALLOW.value
                result.policy_rule = f"answer_release.{result.mode}_passthrough"
                return

            output_flags = check_output(result.answer, nonce=nonce) if nonce else []
            result.grounding = estimate_grounding(result.answer, result.evidence)
            result.citations = cited_indices(result.answer, len(result.evidence))

            proposal = Proposal(
                action=Action.ANSWER_RELEASE,
                tenant_id=tenant_id,
                subject_type="message",
                subject_id=result.message_id,
                title=question[:120],
                grounding=result.grounding,
                has_citations=bool(result.citations) or bool(result.evidence),
                output_flags=output_flags,
                payload={
                    "question": question[:500],
                    "answer": result.answer[:1500],
                    "mode": result.mode,
                    "trace_id": result.trace_id,
                    "sources": sorted({h.source for h in result.evidence}),
                },
            )
            decision = self.policy.evaluate(proposal)
            result.policy_verdict = decision.verdict.value
            result.policy_rule = decision.rule

            set_attrs(
                **{
                    ATTR_GROUNDING: round(result.grounding, 4),
                    ATTR_HITL_DECISION: decision.verdict.value,
                    ATTR_HITL_RULE: decision.rule,
                    ATTR_DEGRADED: result.degraded,
                    ATTR_DEGRADE_REASON: result.degrade_reason,
                }
            )

            if decision.verdict is Verdict.REVIEW:
                # The answer is still shown, but flagged and queued so a human
                # confirms or rejects it.
                approval = approval_queue.open_approval(
                    self.session,
                    proposal=proposal,
                    decision=decision,
                    settings=self.settings,
                )
                result.approval_id = approval.id
                result.needs_review = True
                result.review_reason = decision.reason
                if output_flags:
                    result.mark_degraded(decision.reason)
                log.warning("rag.answer_flagged", rule=decision.rule, approval=approval.id)
                return

            # ALLOW and DENY are automatic decisions; audit them so a later
            # human verdict (feedback) has something to agree or disagree with.
            approval_queue.record_auto_decision(self.session, proposal=proposal, decision=decision)

            if decision.verdict is Verdict.DENY:
                # Refuse rather than serve an unsupported answer.
                log.info(
                    "rag.answer_refused",
                    rule=decision.rule,
                    grounding=round(result.grounding, 4),
                )
                result.answer = REFUSAL_TEXT
                result.mode = "refused"
                result.outcome = OUTCOME_POLICY_REFUSED
                result.mark_degraded(decision.reason)
                degradations.labels(reason="ungrounded", mode="refused").inc()

    # ----------------------------------------------------------------------
    def learn_from_turn(
        self,
        *,
        tenant_id: str,
        user_id: str,
        conversation_id: str,
        question: str,
        answer: str,
        deadline: float | None = None,
    ) -> list[dict]:
        """Post-turn memory work: fold old turns into the L1 summary, then
        extract candidate facts and promote them to L2 through the gate.

        Runs after the answer is returned, never in the request's critical
        path: remembering is not worth making the user wait for, and a failure
        here must not fail the turn. Commits after each step so a slow model
        call never sits inside an open write transaction.
        """
        from aka.memory.extraction import extract_facts

        outcomes: list[dict] = []
        try:
            self.memory.condense_if_needed(conversation_id, self.router, deadline=deadline)
            self.session.commit()
        except Exception:
            self.session.rollback()
            log.exception("rag.condense_failed")

        try:
            candidates = extract_facts(
                self.router,
                user_message=question,
                assistant_message=answer,
                deadline=deadline,
            )
            for candidate in candidates:
                try:
                    embedding = self.embedder.embed([candidate.content], deadline=deadline)[0]
                except LLMError:
                    embedding = None

                item, decision = self.memory.promote(
                    candidate,
                    tenant_id=tenant_id,
                    user_id=user_id,
                    router=self.router,
                    embedding=embedding,
                    conversation_id=conversation_id,
                    deadline=deadline,
                )

                proposal = Proposal(
                    action=Action.MEMORY_WRITE,
                    tenant_id=tenant_id,
                    subject_type="memory_item",
                    subject_id=item.id if item else None,
                    title=candidate.content[:120],
                    confidence=candidate.confidence,
                    conflicts_with=(
                        decision.conflicts_with.id if decision.conflicts_with else None
                    ),
                    payload={"evidence": candidate.evidence[:300], "content": candidate.content[:500]},
                )
                policy_decision = self.policy.evaluate(proposal)

                if item is not None and policy_decision.verdict is Verdict.REVIEW:
                    approval_queue.open_approval(
                        self.session,
                        proposal=proposal,
                        decision=policy_decision,
                        settings=self.settings,
                    )
                elif item is not None:
                    approval_queue.record_auto_decision(
                        self.session, proposal=proposal, decision=policy_decision
                    )
                self.session.commit()

                outcomes.append(
                    {
                        "content": candidate.content,
                        "action": decision.action,
                        "verdict": policy_decision.verdict.value,
                        "memory_id": item.id if item else None,
                    }
                )
        except Exception:
            self.session.rollback()
            log.exception("rag.learn_failed")

        return outcomes
