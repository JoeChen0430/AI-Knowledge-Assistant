"""The policy engine — where "not blindly automatic" is actually implemented.

Every consequential action is described as a :class:`Proposal` and evaluated by
a rule. A rule returns ALLOW, REVIEW or DENY. The engine itself is a pure
function of the proposal plus settings, so the whole governance model is
testable without a database, an LLM, or a network.

Rules deliberately fail *closed* on the risky side: an unknown action type is
sent to review rather than allowed, and any rule that raises is treated as
REVIEW rather than swallowed. A governance layer that silently no-ops when it
breaks is worse than no governance layer, because it looks like it is working.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

from aka.obs.logging import get_logger
from aka.obs.metrics import hitl_decisions
from aka.settings import Settings

log = get_logger("aka.hitl.policy")


class Verdict(str, Enum):
    ALLOW = "allow"      # proceed automatically
    REVIEW = "review"    # queue for a human, do not proceed yet
    DENY = "deny"        # refuse outright, no queue


class Action(str, Enum):
    MEMORY_WRITE = "memory_write"
    DOCUMENT_DELETE = "document_delete"
    INDEX_REBUILD = "index_rebuild"
    ANSWER_RELEASE = "answer_release"
    DOCUMENT_INGEST = "document_ingest"


@dataclass
class Proposal:
    """A thing the system wants to do, plus the evidence for doing it."""

    action: Action
    tenant_id: str
    subject_id: str | None = None
    subject_type: str = ""
    title: str = ""
    confidence: float | None = None
    payload: dict = field(default_factory=dict)

    # Signals populated by the caller from upstream stages.
    conflicts_with: str | None = None
    grounding: float | None = None
    has_citations: bool = True
    scan_severity: str | None = None
    injection_flagged: bool = False
    output_flags: list[str] = field(default_factory=list)
    irreversible: bool = False
    actor_is_human: bool = False


@dataclass
class Decision:
    verdict: Verdict
    rule: str
    reason: str
    confidence: float | None = None
    # What the reviewer needs to see to decide, already redacted.
    review_context: dict = field(default_factory=dict)

    @property
    def requires_human(self) -> bool:
        return self.verdict is Verdict.REVIEW


def contradicts(system_verdict: str, human_approved: bool) -> bool:
    """Whether a human verdict overrides the engine's.

    * ALLOW then rejected: the engine let through something a human refused.
    * REVIEW then approved: the engine held back something a human was happy
      with; the review was unnecessary.
    * DENY then approved: the engine refused something a human wanted.

    REVIEW then rejected, and ALLOW then approved, are agreement.
    """
    if system_verdict == Verdict.ALLOW.value:
        return not human_approved
    if system_verdict in (Verdict.REVIEW.value, Verdict.DENY.value):
        return human_approved
    return False

    def as_dict(self) -> dict:
        return {
            "verdict": self.verdict.value,
            "rule": self.rule,
            "reason": self.reason,
            "confidence": self.confidence,
        }


Rule = Callable[[Proposal, Settings], Decision]


# --------------------------------------------------------------------------
# Rules
# --------------------------------------------------------------------------

def rule_memory_write(p: Proposal, s: Settings) -> Decision:
    """Long-term memory writes. Contradictions always see a human."""
    confidence = p.confidence if p.confidence is not None else 0.0

    if p.conflicts_with:
        return Decision(
            Verdict.REVIEW,
            "memory_write.contradiction",
            "contradicts an existing memory; a human decides which is current",
            confidence,
            {"conflicts_with": p.conflicts_with},
        )
    if confidence < s.MEMORY_REJECT_CONFIDENCE:
        return Decision(
            Verdict.DENY, "memory_write.low_confidence",
            f"confidence {confidence:.2f} below floor {s.MEMORY_REJECT_CONFIDENCE:.2f}",
            confidence,
        )
    if confidence >= s.MEMORY_AUTO_COMMIT_CONFIDENCE:
        return Decision(
            Verdict.ALLOW, "memory_write.high_confidence",
            f"confidence {confidence:.2f} at or above auto-commit threshold",
            confidence,
        )
    return Decision(
        Verdict.REVIEW, "memory_write.uncertain",
        f"confidence {confidence:.2f} in the review band",
        confidence,
    )


def rule_document_delete(p: Proposal, s: Settings) -> Decision:
    """Destructive document ops.

    Reversible deletes (soft delete to trash with an undo window) are allowed
    when a human initiated them — the human *is* the confirmation. An
    irreversible delete, or one the system proposes on its own, needs review.
    """
    if p.irreversible:
        return Decision(
            Verdict.REVIEW, "document_delete.irreversible",
            "permanent deletion always requires explicit approval",
            review_context={"path": p.payload.get("rel_path")},
        )
    if not p.actor_is_human:
        return Decision(
            Verdict.REVIEW, "document_delete.machine_initiated",
            "the system does not delete a user's documents unprompted",
            review_context={"path": p.payload.get("rel_path")},
        )
    return Decision(
        Verdict.ALLOW, "document_delete.reversible",
        f"soft delete, recoverable for {s.UNDO_WINDOW_HOURS}h",
    )


def rule_index_rebuild(p: Proposal, s: Settings) -> Decision:
    """Rebuilds are safe by construction — blue/green means a bad build never
    replaces a good one — so this is allowed and merely audited."""
    return Decision(
        Verdict.ALLOW, "index_rebuild.blue_green",
        "new index is smoke-checked before the swap; failure leaves the current index serving",
    )


def rule_answer_release(p: Proposal, s: Settings) -> Decision:
    """Gate on whether the answer is actually supported by retrieved evidence.

    This is the anti-hallucination node, with three bands:

    * below ``ANSWER_MIN_GROUNDING`` the answer is denied and the caller
      degrades to an honest "I don't have this in your documents";
    * below ``ANSWER_REVIEW_GROUNDING`` it is shown, flagged as weakly
      supported, and queued so a human confirms or rejects it;
    * above that it is released.

    The human verdicts on the middle band are what the override rate is
    computed from, and therefore what the two thresholds are tuned against.
    """
    grounding = p.grounding if p.grounding is not None else 0.0

    if p.output_flags:
        return Decision(
            Verdict.REVIEW, "answer_release.output_flagged",
            f"output-side check fired: {', '.join(p.output_flags)}",
            grounding,
            review_context={"flags": p.output_flags},
        )
    if not p.has_citations or grounding < s.ANSWER_MIN_GROUNDING:
        return Decision(
            Verdict.DENY, "answer_release.ungrounded",
            f"grounding {grounding:.2f} below {s.ANSWER_MIN_GROUNDING:.2f}; refuse rather than guess",
            grounding,
        )
    if grounding < s.ANSWER_REVIEW_GROUNDING:
        return Decision(
            Verdict.REVIEW, "answer_release.low_grounding",
            f"grounding {grounding:.2f} below review threshold {s.ANSWER_REVIEW_GROUNDING:.2f}; "
            "shown with a warning and queued for review",
            grounding,
            review_context={"grounding": round(grounding, 4)},
        )
    return Decision(
        Verdict.ALLOW, "answer_release.grounded",
        f"grounding {grounding:.2f} with citations",
        grounding,
    )


def rule_document_ingest(p: Proposal, s: Settings) -> Decision:
    """Quarantine anything carrying credentials before it becomes retrievable."""
    if p.scan_severity == "high":
        return Decision(
            Verdict.REVIEW, "document_ingest.secret_detected",
            "high-severity secret or PII detected; quarantined pending approval",
            review_context={"findings": p.payload.get("scan_summary")},
        )
    if p.injection_flagged:
        return Decision(
            Verdict.REVIEW, "document_ingest.injection_suspected",
            "document contains instruction-like text targeting the assistant",
            review_context={"signals": p.payload.get("injection_rules")},
        )
    return Decision(Verdict.ALLOW, "document_ingest.clean", "no high-severity findings")


_RULES: dict[Action, Rule] = {
    Action.MEMORY_WRITE: rule_memory_write,
    Action.DOCUMENT_DELETE: rule_document_delete,
    Action.INDEX_REBUILD: rule_index_rebuild,
    Action.ANSWER_RELEASE: rule_answer_release,
    Action.DOCUMENT_INGEST: rule_document_ingest,
}


class PolicyEngine:
    def __init__(self, settings: Settings):
        self.settings = settings

    def evaluate(self, proposal: Proposal) -> Decision:
        if not self.settings.HITL_ENABLED:
            # Escape hatch for benchmarking the ungoverned baseline. Loud on
            # purpose — this should never be how it runs in front of a user.
            log.warning("hitl.disabled", action=proposal.action.value)
            return Decision(Verdict.ALLOW, "hitl.disabled", "policy engine disabled by config")

        rule = _RULES.get(proposal.action)
        if rule is None:
            decision = Decision(
                Verdict.REVIEW, "unknown_action",
                f"no rule for action '{proposal.action}'; failing closed to review",
            )
        else:
            try:
                decision = rule(proposal, self.settings)
            except Exception as exc:
                # A broken rule must not become an implicit ALLOW.
                log.exception("hitl.rule_error", action=proposal.action.value)
                decision = Decision(
                    Verdict.REVIEW, f"{proposal.action.value}.rule_error",
                    f"policy rule raised {type(exc).__name__}; failing closed to review",
                )

        hitl_decisions.labels(rule=decision.rule, decision=decision.verdict.value).inc()
        log.info(
            "hitl.decision",
            action=proposal.action.value,
            verdict=decision.verdict.value,
            rule=decision.rule,
            subject=proposal.subject_id,
        )
        return decision
