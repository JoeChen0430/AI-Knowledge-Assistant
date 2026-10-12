"""Prompt assembly and grounding measurement.

The prompt has a strict layout so that a trace can attribute every token to a
tier, and so that untrusted document text is always structurally separated from
instructions (see aka/security/injection.py).

    system   : role + refusal policy + untrusted-content preamble
    system   : L2 durable facts about the user
    system   : L1 session summary
    messages : L0 recent turns, verbatim
    user     : untrusted evidence block + the question
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from aka.ingest.hashing import estimate_tokens
from aka.ingest.vectorstore import SearchHit
from aka.llm.client import ChatMessage
from aka.memory.tiers import MemoryContext
from aka.security.injection import build_untrusted_preamble, new_nonce, wrap_untrusted

SYSTEM_ROLE = """\
You are a personal knowledge assistant. Answer strictly from the evidence and \
the remembered facts provided below.

Rules:
- If the evidence does not support an answer, say so plainly. Never fill gaps \
with general knowledge and never guess.
- Cite the source of each claim as [1], [2] matching the numbered evidence.
- Be concise. Prefer quoting the document over paraphrasing when precision matters.
- Remembered facts describe the user, not the documents. Do not cite them."""

REFUSAL_TEXT = (
    "I don't have enough in your documents to answer that reliably. "
    "You could try rephrasing, or add a document that covers it."
)

_STOPWORDS = frozenset(
    """a an the of to in on at for with and or but is are was were be been being as by from
    it its this that these those i you he she they we my your our their not no do does did
    have has had will would can could should may might must if then than so such about into
    over under between""".split()
)
_WORD = re.compile(r"\w+", re.UNICODE)


@dataclass
class BuiltPrompt:
    messages: list[ChatMessage]
    nonce: str
    evidence: list[SearchHit]
    prompt_tokens: int
    # Per-tier accounting, surfaced in the trace and the "why this answer" panel.
    token_breakdown: dict[str, int]


def format_evidence(hits: list[SearchHit], nonce: str) -> str:
    if not hits:
        return wrap_untrusted("(no matching documents)", nonce)
    blocks = []
    for i, hit in enumerate(hits, start=1):
        blocks.append(f"[{i}] source: {hit.source} (score {hit.score:.3f})\n{hit.text}")
    return wrap_untrusted("\n\n".join(blocks), nonce)


def format_facts(memory: MemoryContext) -> str:
    if not memory.facts:
        return ""
    lines = [
        f"- {hit.item.content} (confidence {hit.item.confidence:.2f})"
        for hit in memory.facts
    ]
    return "Remembered about this user:\n" + "\n".join(lines)


def build_prompt(
    question: str,
    *,
    hits: list[SearchHit],
    memory: MemoryContext,
) -> BuiltPrompt:
    nonce = new_nonce()
    breakdown: dict[str, int] = {}

    system_parts = [SYSTEM_ROLE, build_untrusted_preamble(nonce)]
    breakdown["system"] = sum(estimate_tokens(p) for p in system_parts)

    facts_block = format_facts(memory)
    if facts_block:
        system_parts.append(facts_block)
        breakdown["l2_facts"] = estimate_tokens(facts_block)

    if memory.session_summary:
        summary_block = f"Earlier in this conversation:\n{memory.session_summary}"
        system_parts.append(summary_block)
        breakdown["l1_summary"] = estimate_tokens(summary_block)

    messages = [ChatMessage(role="system", content="\n\n".join(system_parts))]

    # L0 turns go in as real conversation turns, not as a flattened blob, so
    # the model sees turn structure.
    for turn in memory.recent_turns:
        if turn.role in ("user", "assistant"):
            messages.append(turn)
    breakdown["l0_turns"] = sum(estimate_tokens(t.content) for t in memory.recent_turns)

    evidence_block = format_evidence(hits, nonce)
    breakdown["evidence"] = estimate_tokens(evidence_block)

    messages.append(
        ChatMessage(
            role="user",
            content=f"{evidence_block}\n\nQuestion: {question}",
        )
    )
    breakdown["question"] = estimate_tokens(question)

    return BuiltPrompt(
        messages=messages,
        nonce=nonce,
        evidence=hits,
        prompt_tokens=sum(breakdown.values()),
        token_breakdown=breakdown,
    )


# --------------------------------------------------------------------------
# Grounding
# --------------------------------------------------------------------------

_CJK_RUN = re.compile(r"[぀-ヿ㐀-䶿一-鿿]+")


def _cjk_bigrams(text: str) -> list[str]:
    """CJK has no spaces, so ``\\w+`` would turn a whole sentence into one
    "word" that never matches anything. Overlapping bigrams are the standard
    tokenization for CJK keyword matching."""
    out: list[str] = []
    for run in _CJK_RUN.findall(text or ""):
        out.extend(run[i : i + 2] for i in range(len(run) - 1))
    return out


def _content_words(text: str) -> list[str]:
    latin = _CJK_RUN.sub(" ", text or "")
    words = [
        w for w in (m.group(0).casefold() for m in _WORD.finditer(latin))
        if w not in _STOPWORDS and len(w) > 1
    ]
    return words + _cjk_bigrams(text)


def lexical_terms(question: str, limit: int = 8) -> list[str]:
    """Search terms for keyword retrieval when no query vector is available.

    Latin words are kept whole, longest first as the most specific; CJK runs
    contribute their bigrams.
    """
    latin = _CJK_RUN.sub(" ", question or "")
    words = [
        w for w in (m.group(0).casefold() for m in _WORD.finditer(latin))
        if w not in _STOPWORDS and len(w) > 2 and not w.isdigit()
    ]
    terms = sorted(dict.fromkeys(words), key=len, reverse=True) + _cjk_bigrams(question)
    return list(dict.fromkeys(terms))[:limit]


def lexical_support(answer: str, hits: list[SearchHit]) -> float:
    """Fraction of the answer's content words that appear in the evidence.

    A blunt instrument, and it says so: it measures overlap, not entailment. It
    is here because it is cheap, deterministic and monotonic — good enough to
    separate "answered from the documents" from "answered from the model's
    memory", which is the decision the policy gate actually makes. A
    model-based faithfulness judge would be more accurate and is what the
    offline eval uses; this runs on every request.
    """
    words = _content_words(answer)
    if not words:
        return 0.0
    evidence = set()
    for hit in hits:
        evidence.update(_content_words(hit.text))
    if not evidence:
        return 0.0
    return sum(1 for w in words if w in evidence) / len(words)


def estimate_grounding(answer: str, hits: list[SearchHit]) -> float:
    """Combine retrieval confidence with answer/evidence overlap.

    Multiplicative, not a weighted sum. A weighted sum lets a high retrieval
    score carry an answer that ignores the evidence entirely — which is exactly
    the failure this gate exists to catch, since retrieval succeeding says
    nothing about whether the model used what it retrieved. Overlap is
    therefore a necessary condition, and retrieval quality only modulates it.
    """
    if not hits or not answer.strip():
        return 0.0
    top_score = max(h.score for h in hits)
    return lexical_support(answer, hits) * (0.5 + 0.5 * top_score)


def cited_indices(answer: str, evidence_count: int) -> list[int]:
    """Which [n] markers the answer actually used."""
    found = {int(m) for m in re.findall(r"\[(\d{1,2})\]", answer or "")}
    return sorted(i for i in found if 1 <= i <= evidence_count)


def build_extractive_answer(question: str, hits: list[SearchHit]) -> str:
    """Last-resort degraded answer: no generation, just the evidence.

    Explicitly labelled so the user is never shown retrieved text as if the
    assistant had reasoned about it.
    """
    if not hits:
        return REFUSAL_TEXT
    lines = [
        "The language model is unavailable, so here are the most relevant "
        "passages from your documents instead of a written answer:",
        "",
    ]
    for i, hit in enumerate(hits, start=1):
        excerpt = hit.text.strip()
        if len(excerpt) > 600:
            excerpt = excerpt[:600].rsplit(" ", 1)[0] + "…"
        lines.append(f"**[{i}] {hit.source}** (relevance {hit.score:.2f})\n{excerpt}\n")
    return "\n".join(lines)
