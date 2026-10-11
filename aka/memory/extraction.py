"""Turning conversation into durable facts.

Two LLM-backed steps, both defensive about model output:

* **extract** — pull candidate facts out of a turn, each with a self-reported
  confidence and the verbatim span that justifies it.
* **detect_contradiction** — decide whether a candidate conflicts with an
  existing memory, which is what routes a write to human review instead of
  silently overwriting what the user told us before.

A model that returns garbage yields zero candidates rather than an exception:
memory extraction is best-effort and must never break the chat turn.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from aka.llm.client import ChatMessage
from aka.obs.logging import get_logger
from aka.obs.telemetry import stage

log = get_logger("aka.memory.extraction")

_JSON_BLOCK = re.compile(r"\{.*\}|\[.*\]", re.DOTALL)

EXTRACTION_PROMPT = """\
You extract durable facts about the user from a conversation turn.

Return ONLY a JSON array. Each element:
  {"content": "<fact, third person, self-contained>",
   "kind": "fact" | "preference" | "identity",
   "confidence": <0.0-1.0>,
   "evidence": "<verbatim quote from the user's message>"}

Rules:
- Only facts about the USER that stay true beyond this conversation.
- Skip questions, one-off requests, and anything about the documents.
- Each fact must stand alone without the conversation for context.
- If the turn contains no durable fact, return [].
- confidence reflects how clearly the user stated it, not how plausible it is.

Return [] rather than guessing."""

CONTRADICTION_PROMPT = """\
Decide whether a new statement contradicts an existing one about the same user.

Return ONLY JSON: {"contradicts": true|false, "confidence": <0.0-1.0>, \
"reason": "<one short sentence>"}

"contradicts" is true only when both cannot be true at the same time.
An addition, refinement, or unrelated fact is NOT a contradiction."""


@dataclass
class FactCandidate:
    content: str
    kind: str = "fact"
    confidence: float = 0.5
    evidence: str = ""

    def is_valid(self) -> bool:
        return bool(self.content.strip()) and 0.0 <= self.confidence <= 1.0


@dataclass
class ContradictionVerdict:
    contradicts: bool
    confidence: float
    reason: str


def _parse_json(text: str):
    """Models wrap JSON in prose and fences; salvage the payload."""
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = _JSON_BLOCK.search(text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None


def extract_facts(
    router,
    *,
    user_message: str,
    assistant_message: str = "",
    max_facts: int = 5,
    deadline: float | None = None,
) -> list[FactCandidate]:
    """Best-effort extraction. Returns [] on any model or parse failure."""
    if not user_message.strip():
        return []

    turn = f"User: {user_message.strip()}"
    if assistant_message.strip():
        turn += f"\nAssistant: {assistant_message.strip()[:600]}"

    with stage("memory.extract"):
        try:
            result = router.generate(
                [
                    ChatMessage(role="system", content=EXTRACTION_PROMPT),
                    ChatMessage(role="user", content=turn),
                ],
                deadline=deadline,
                max_tokens=512,
                temperature=0.0,
            )
        except Exception as exc:
            log.warning("memory.extract_failed", error=str(exc))
            return []

    parsed = _parse_json(result.text)
    if not isinstance(parsed, list):
        log.debug("memory.extract_unparseable", sample=(result.text or "")[:200])
        return []

    candidates: list[FactCandidate] = []
    for entry in parsed[:max_facts]:
        if not isinstance(entry, dict):
            continue
        try:
            confidence = float(entry.get("confidence", 0.5))
        except (TypeError, ValueError):
            confidence = 0.5
        candidate = FactCandidate(
            content=str(entry.get("content", "")).strip(),
            kind=str(entry.get("kind", "fact")).strip() or "fact",
            confidence=max(0.0, min(1.0, confidence)),
            evidence=str(entry.get("evidence", "")).strip(),
        )
        # An "extracted" fact whose evidence is absent from the turn is a
        # hallucination; drop it rather than store it.
        if candidate.is_valid() and _evidence_supported(candidate, user_message):
            candidates.append(candidate)

    log.debug("memory.extracted", count=len(candidates))
    return candidates


def _evidence_supported(candidate: FactCandidate, user_message: str) -> bool:
    if not candidate.evidence:
        return True  # no claim of a quote; confidence gating still applies
    needle = re.sub(r"\s+", " ", candidate.evidence.casefold()).strip()
    haystack = re.sub(r"\s+", " ", user_message.casefold())
    if needle in haystack:
        return True
    # Tolerate light paraphrase: most content words must be present.
    words = [w for w in re.findall(r"\w+", needle) if len(w) > 2]
    if not words:
        return True
    present = sum(1 for w in words if w in haystack)
    return present / len(words) >= 0.6


def detect_contradiction(
    router,
    *,
    new_fact: str,
    existing_fact: str,
    deadline: float | None = None,
) -> ContradictionVerdict:
    """Conservative: an unparseable or failed check reports no contradiction,
    because a false positive would queue a needless human review while the
    confidence gate still catches genuinely uncertain writes."""
    with stage("memory.contradiction_check"):
        try:
            result = router.generate(
                [
                    ChatMessage(role="system", content=CONTRADICTION_PROMPT),
                    ChatMessage(
                        role="user",
                        content=f"Existing: {existing_fact}\nNew: {new_fact}",
                    ),
                ],
                deadline=deadline,
                max_tokens=200,
                temperature=0.0,
            )
        except Exception as exc:
            log.warning("memory.contradiction_check_failed", error=str(exc))
            return ContradictionVerdict(False, 0.0, "check unavailable")

    parsed = _parse_json(result.text)
    if not isinstance(parsed, dict):
        return ContradictionVerdict(False, 0.0, "unparseable verdict")

    try:
        confidence = float(parsed.get("confidence", 0.5))
    except (TypeError, ValueError):
        confidence = 0.5
    return ContradictionVerdict(
        contradicts=bool(parsed.get("contradicts", False)),
        confidence=max(0.0, min(1.0, confidence)),
        reason=str(parsed.get("reason", ""))[:300],
    )


SUMMARY_PROMPT = """\
Condense this conversation excerpt into a compact third-person summary.

Keep: decisions made, facts the user stated, unresolved questions, and any
constraint the user set. Drop: pleasantries, retrieved document text, and
anything already obvious from the newer messages.

Write at most 150 words. No preamble."""


def summarize_turns(router, *, transcript: str, previous_summary: str = "", deadline=None) -> str:
    """Fold older turns (plus any prior summary) into one L1 summary."""
    if not transcript.strip():
        return previous_summary

    body = transcript if not previous_summary else (
        f"Summary so far:\n{previous_summary}\n\nNew messages:\n{transcript}"
    )
    with stage("memory.summarize"):
        try:
            result = router.generate(
                [
                    ChatMessage(role="system", content=SUMMARY_PROMPT),
                    ChatMessage(role="user", content=body),
                ],
                deadline=deadline,
                max_tokens=400,
                temperature=0.1,
            )
        except Exception as exc:
            log.warning("memory.summarize_failed", error=str(exc))
            # Keeping the old summary is strictly better than losing context.
            return previous_summary

    return result.text.strip() or previous_summary
