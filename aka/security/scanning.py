"""Secret and PII detection at ingest.

Purpose is not to be a DLP product — it is to make "this document contains
credentials" a decision a human makes rather than one the pipeline makes
silently. Findings quarantine the document and open an approval.

Tuned for few false negatives on high-severity secrets (keys, tokens) and to
stay quiet on low-signal patterns, because every false positive costs a human
a review.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from enum import Enum

from aka.obs.metrics import scan_findings


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True)
class Rule:
    name: str
    pattern: re.Pattern
    severity: Severity
    kind: str  # secret | pii
    # Some matches are only meaningful if they look random rather than English.
    entropy_floor: float | None = None


def _luhn_ok(digits: str) -> bool:
    """Card-number checksum — turns a loose 16-digit pattern into a real signal."""
    nums = [int(d) for d in digits if d.isdigit()][::-1]
    if len(nums) < 13:
        return False
    total = 0
    for i, n in enumerate(nums):
        if i % 2:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


def shannon_entropy(text: str) -> float:
    if not text:
        return 0.0
    counts = Counter(text)
    length = len(text)
    return -sum((c / length) * math.log2(c / length) for c in counts.values())


RULES: tuple[Rule, ...] = (
    Rule("aws_access_key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"), Severity.HIGH, "secret"),
    Rule(
        "aws_secret_key",
        re.compile(r"(?i)aws_secret_access_key\s*[:=]\s*['\"]?([A-Za-z0-9/+=]{40})"),
        Severity.HIGH,
        "secret",
    ),
    Rule("github_token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b"), Severity.HIGH, "secret"),
    Rule("slack_token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"), Severity.HIGH, "secret"),
    Rule("openai_key", re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"), Severity.HIGH, "secret"),
    Rule("anthropic_key", re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}\b"), Severity.HIGH, "secret"),
    Rule(
        "private_key_block",
        re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |PGP )?PRIVATE KEY-----"),
        Severity.HIGH,
        "secret",
    ),
    Rule("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"),
         Severity.HIGH, "secret"),
    Rule(
        "generic_assigned_secret",
        re.compile(
            r"(?i)\b(?:api[_-]?key|secret|passwd|password|token)\b\s*[:=]\s*['\"]([^'\"\s]{12,})['\"]"
        ),
        Severity.MEDIUM,
        "secret",
        entropy_floor=3.0,
    ),
    Rule(
        "db_connection_string",
        re.compile(r"(?i)\b(?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis)://[^\s:@]+:[^\s@]+@"),
        Severity.HIGH,
        "secret",
    ),
    Rule("us_ssn", re.compile(r"\b(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b"),
         Severity.HIGH, "pii"),
    Rule("credit_card", re.compile(r"\b(?:\d[ -]?){13,19}\b"), Severity.HIGH, "pii"),
    Rule(
        "email",
        re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
        Severity.LOW,
        "pii",
    ),
    Rule(
        "phone_e164",
        re.compile(r"(?<![\d-])\+\d{1,3}[\s-]?\(?\d{2,4}\)?[\s-]?\d{3,4}[\s-]?\d{3,4}(?![\d-])"),
        Severity.LOW,
        "pii",
    ),
    Rule("tw_id", re.compile(r"\b[A-Z][12]\d{8}\b"), Severity.MEDIUM, "pii"),
)


@dataclass
class Finding:
    rule: str
    kind: str
    severity: Severity
    line: int
    excerpt: str  # already redacted

    def as_dict(self) -> dict:
        return {
            "rule": self.rule,
            "kind": self.kind,
            "severity": self.severity.value,
            "line": self.line,
            "excerpt": self.excerpt,
        }


@dataclass
class ScanResult:
    findings: list[Finding]

    @property
    def max_severity(self) -> Severity | None:
        for level in (Severity.HIGH, Severity.MEDIUM, Severity.LOW):
            if any(f.severity is level for f in self.findings):
                return level
        return None

    @property
    def should_quarantine(self) -> bool:
        """Only HIGH quarantines. Flagging every email address would make the
        review queue useless, which is worse than not having one."""
        return self.max_severity is Severity.HIGH

    def summary(self) -> str:
        if not self.findings:
            return "clean"
        counts = Counter(f.rule for f in self.findings)
        return ", ".join(f"{rule} x{n}" for rule, n in counts.most_common())

    def as_dict(self) -> dict:
        return {
            "findings": [f.as_dict() for f in self.findings],
            "max_severity": self.max_severity.value if self.max_severity else None,
            "should_quarantine": self.should_quarantine,
            "summary": self.summary(),
        }


def redact(text: str, keep: int = 4) -> str:
    """Never echo a live credential into logs, audit rows, or the review UI."""
    text = text.strip()
    if len(text) <= keep:
        return "*" * len(text)
    return f"{text[:keep]}{'*' * min(12, len(text) - keep)}"


def scan_text(text: str, *, max_findings: int = 50) -> ScanResult:
    findings: list[Finding] = []
    if not text:
        return ScanResult(findings)

    lines = text.splitlines()
    for line_no, line in enumerate(lines, start=1):
        if len(line) > 4000:
            line = line[:4000]
        for rule in RULES:
            for match in rule.pattern.finditer(line):
                value = match.group(1) if match.groups() else match.group(0)

                if rule.name == "credit_card" and not _luhn_ok(value):
                    continue
                if rule.entropy_floor is not None and shannon_entropy(value) < rule.entropy_floor:
                    # Looks like prose, e.g. password = "changeme please"
                    continue

                findings.append(
                    Finding(
                        rule=rule.name,
                        kind=rule.kind,
                        severity=rule.severity,
                        line=line_no,
                        excerpt=redact(value),
                    )
                )
                scan_findings.labels(kind=rule.name).inc()
                if len(findings) >= max_findings:
                    return ScanResult(findings)
    return ScanResult(findings)
