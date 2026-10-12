"""Threshold calibration from human verdicts.

The audit log pairs every automatic decision with the signal it was made on
(memory-write confidence, answer grounding) and, where a human later weighed
in, the human's verdict. That is a labelled dataset. Replaying it under
candidate thresholds shows which settings would have agreed with the humans
most often, i.e. minimised the override rate.

Caveats the report states rather than hides:

* Only items a human saw are labelled: queued items and auto-released ones
  that got feedback. Anything the engine refused outright was never shown to
  anyone, so loosening a refusal threshold cannot be scored from this data.
* With few labels the "best" thresholds overfit; below ``MIN_LABELS`` per
  family the current thresholds are kept.
* The replay is a projection. The real before/after number comes from
  measuring the override rate over a period after the new thresholds ship.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from aka.db.models import ActorType, AuditLog, utcnow
from aka.hitl.policy import Verdict, contradicts
from aka.settings import Settings

MIN_LABELS = 20

MEMORY = "memory_write"
ANSWER = "answer_release"

# Rules whose verdict does not depend on a threshold; replaying them under
# different thresholds would be meaningless.
_UNTUNABLE = {"memory_write.contradiction", "answer_release.output_flagged"}


@dataclass(frozen=True)
class Thresholds:
    memory_auto: float
    memory_reject: float
    answer_min: float
    answer_review: float

    @classmethod
    def from_settings(cls, s: Settings) -> "Thresholds":
        return cls(
            memory_auto=s.MEMORY_AUTO_COMMIT_CONFIDENCE,
            memory_reject=s.MEMORY_REJECT_CONFIDENCE,
            answer_min=s.ANSWER_MIN_GROUNDING,
            answer_review=s.ANSWER_REVIEW_GROUNDING,
        )

    def as_env(self) -> dict[str, str]:
        return {
            "MEMORY_AUTO_COMMIT_CONFIDENCE": f"{self.memory_auto:.2f}",
            "MEMORY_REJECT_CONFIDENCE": f"{self.memory_reject:.2f}",
            "ANSWER_MIN_GROUNDING": f"{self.answer_min:.2f}",
            "ANSWER_REVIEW_GROUNDING": f"{self.answer_review:.2f}",
        }


@dataclass(frozen=True)
class Labelled:
    family: str        # memory_write | answer_release
    rule: str
    signal: float      # confidence or grounding the engine decided on
    approved: bool     # the human verdict
    system_verdict: str


def verdict_for(family: str, signal: float, t: Thresholds) -> str:
    """The engine's verdict for one signal value, mirroring aka.hitl.policy."""
    if family == MEMORY:
        if signal < t.memory_reject:
            return Verdict.DENY.value
        if signal >= t.memory_auto:
            return Verdict.ALLOW.value
        return Verdict.REVIEW.value
    if signal < t.answer_min:
        return Verdict.DENY.value
    if signal < t.answer_review:
        return Verdict.REVIEW.value
    return Verdict.ALLOW.value


@dataclass
class Replay:
    labelled: int
    overrides: int
    queued: int  # items these thresholds would send to review

    @property
    def override_rate(self) -> float:
        return self.overrides / self.labelled if self.labelled else 0.0

    @property
    def review_share(self) -> float:
        return self.queued / self.labelled if self.labelled else 0.0

    def as_dict(self) -> dict:
        return {
            "labelled": self.labelled,
            "overrides": self.overrides,
            "override_rate": round(self.override_rate, 4),
            "review_share": round(self.review_share, 4),
        }


def replay(items: list[Labelled], t: Thresholds) -> Replay:
    overrides = queued = 0
    for item in items:
        verdict = verdict_for(item.family, item.signal, t)
        queued += verdict == Verdict.REVIEW.value
        overrides += contradicts(verdict, item.approved)
    return Replay(labelled=len(items), overrides=overrides, queued=queued)


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------

def labelled_decisions(
    session: Session,
    *,
    tenant_id: str,
    since: datetime | None = None,
    until: datetime | None = None,
    window_days: int = 90,
) -> list[Labelled]:
    """One row per human verdict, joined to the automatic decision it judged."""
    since = since or utcnow() - timedelta(days=window_days)
    q = select(AuditLog).where(
        AuditLog.tenant_id == tenant_id,
        AuditLog.actor_type == ActorType.HUMAN,
        AuditLog.created_at >= since,
    )
    if until is not None:
        q = q.where(AuditLog.created_at < until)

    out: list[Labelled] = []
    for human in session.scalars(q.order_by(AuditLog.created_at)):
        system = _system_decision_for(session, human)
        if system is None or system.rule in _UNTUNABLE:
            continue
        if system.action not in (MEMORY, ANSWER):
            continue
        signal = (system.payload or {}).get("confidence")
        if signal is None:
            continue
        out.append(
            Labelled(
                family=system.action,
                rule=system.rule or "",
                signal=float(signal),
                approved=human.decision == "approved",
                system_verdict=system.decision,
            )
        )
    return out


def _system_decision_for(session: Session, human: AuditLog) -> AuditLog | None:
    if human.overrides_audit_id:
        return session.get(AuditLog, human.overrides_audit_id)
    return session.scalars(
        select(AuditLog)
        .where(
            AuditLog.tenant_id == human.tenant_id,
            AuditLog.subject_id == human.subject_id,
            AuditLog.actor_type == ActorType.SYSTEM,
            AuditLog.decision.in_([v.value for v in Verdict]),
            AuditLog.created_at <= human.created_at,
        )
        .order_by(AuditLog.created_at.desc())
    ).first()


# --------------------------------------------------------------------------
# Search
# --------------------------------------------------------------------------

def _steps(lo: float, hi: float, step: float = 0.05) -> list[float]:
    n = int(round((hi - lo) / step))
    return [round(lo + i * step, 2) for i in range(n + 1)]


@dataclass
class CalibrationResult:
    current: Thresholds
    recommended: Thresholds
    before: dict[str, dict]
    after: dict[str, dict]
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "current": asdict(self.current),
            "recommended": asdict(self.recommended),
            "recommended_env": self.recommended.as_env(),
            "replay_before": self.before,
            "replay_after": self.after,
            "notes": self.notes,
        }


def _pick(candidates, score, current):
    """Fewest overrides; ties broken by the smallest move from current
    settings, so the recommendation never changes more than the data supports."""
    return min(candidates, key=lambda c: (score(c), sum(abs(a - b) for a, b in zip(c, current))))


def calibrate(items: list[Labelled], current: Thresholds) -> CalibrationResult:
    notes: list[str] = []
    memory = [i for i in items if i.family == MEMORY]
    answers = [i for i in items if i.family == ANSWER]

    rec = {
        "memory_auto": current.memory_auto,
        "memory_reject": current.memory_reject,
        "answer_min": current.answer_min,
        "answer_review": current.answer_review,
    }

    if len(memory) >= MIN_LABELS:
        grid = [(a, r) for a in _steps(0.5, 0.95) for r in _steps(0.1, 0.5) if r < a]

        def mem_score(c):
            t = Thresholds(c[0], c[1], current.answer_min, current.answer_review)
            return replay(memory, t).overrides

        rec["memory_auto"], rec["memory_reject"] = _pick(
            grid, mem_score, (current.memory_auto, current.memory_reject)
        )
    else:
        notes.append(f"memory_write: {len(memory)} labels (< {MIN_LABELS}); thresholds kept")

    if len(answers) >= MIN_LABELS:
        grid = [(m, r) for m in _steps(0.05, 0.4) for r in _steps(0.15, 0.8) if m < r]

        def ans_score(c):
            t = Thresholds(current.memory_auto, current.memory_reject, c[0], c[1])
            return replay(answers, t).overrides

        rec["answer_min"], rec["answer_review"] = _pick(
            grid, ans_score, (current.answer_min, current.answer_review)
        )
    else:
        notes.append(f"answer_release: {len(answers)} labels (< {MIN_LABELS}); thresholds kept")

    recommended = Thresholds(**rec)
    notes.append(
        "only items a human saw are labelled; refusals were never shown, so "
        "the refuse floors are scored on queued/released items only"
    )

    def summary(t: Thresholds) -> dict[str, dict]:
        return {
            "overall": replay(items, t).as_dict(),
            MEMORY: replay(memory, t).as_dict(),
            ANSWER: replay(answers, t).as_dict(),
        }

    return CalibrationResult(
        current=current,
        recommended=recommended,
        before=summary(current),
        after=summary(recommended),
        notes=notes,
    )
