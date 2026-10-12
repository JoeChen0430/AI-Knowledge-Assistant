"""Measure the override rate and recommend thresholds from human verdicts.

Reads the audit log of a running deployment (whatever DATABASE_URL points at)
— nothing here generates labels; they come from people using the review queue
and the 👍/👎 buttons.

    # the trailing 30 days, overall and per rule, plus recommended thresholds
    python eval/calibrate_policy.py

    # before/after a threshold change on a given date
    python eval/calibrate_policy.py --until 2026-11-01      # period with old thresholds
    python eval/calibrate_policy.py --since 2026-11-01      # period with new thresholds

The measured rate is the real number. The "replay" is a projection: the same
human verdicts re-scored under candidate thresholds.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_date(value: str | None) -> datetime | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(value)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tenant", default=None, help="tenant id (default: DEFAULT_TENANT)")
    parser.add_argument("--since", default=None, help="ISO date, start of the period")
    parser.add_argument("--until", default=None, help="ISO date, end of the period (exclusive)")
    parser.add_argument("--window-days", type=int, default=30, help="used when --since is not given")
    parser.add_argument("--json", action="store_true", help="print the full report as JSON")
    args = parser.parse_args()

    from aka.db.session import session_scope
    from aka.hitl import audit
    from aka.hitl.calibration import Thresholds, calibrate, labelled_decisions
    from aka.settings import get_settings

    settings = get_settings()
    tenant = args.tenant or settings.DEFAULT_TENANT
    until = parse_date(args.until)
    since = parse_date(args.since) or (until or datetime.now(timezone.utc)) - timedelta(days=args.window_days)

    with session_scope(settings) as session:
        measured = audit.override_stats(session, tenant_id=tenant, since=since, until=until)
        rules = sorted({r["rule"] for r in audit.rule_breakdown(session, tenant_id=tenant, window_days=3650)})
        per_rule = {}
        for rule in rules:
            stats = audit.override_stats(session, tenant_id=tenant, rule=rule, since=since, until=until)
            if stats.total_human_decisions:
                per_rule[rule] = stats.as_dict()
        items = labelled_decisions(session, tenant_id=tenant, since=since, until=until)

    result = calibrate(items, Thresholds.from_settings(settings))
    period = f"{since:%Y-%m-%d} .. {until:%Y-%m-%d}" if until else f"since {since:%Y-%m-%d}"
    report = {
        "tenant": tenant,
        "period": period,
        "measured": {**measured.as_dict(), "per_rule": per_rule},
        "calibration": result.as_dict(),
    }
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0

    print(f"tenant {tenant}, {period}")
    print(f"\nmeasured override rate: {measured.rate:.1%} "
          f"({measured.overrides} of {measured.total_human_decisions} human verdicts)")
    for rule, stats in per_rule.items():
        print(f"  {rule:<36} {stats['override_rate']:>6.1%}  of {stats['human_decisions']}")

    print(f"\nreplay over {len(items)} labelled decisions")
    for label, block in (("current thresholds", result.before), ("recommended", result.after)):
        overall = block["overall"]
        print(f"  {label:<20} override {overall['override_rate']:>6.1%}   "
              f"would queue {overall['review_share']:>6.1%}")
    print("\nrecommended settings (.env):")
    for key, value in result.recommended.as_env().items():
        current = Thresholds.from_settings(settings).as_env()[key]
        marker = "" if value == current else f"   (was {current})"
        print(f"  {key}={value}{marker}")
    for note in result.notes:
        print(f"  note: {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
