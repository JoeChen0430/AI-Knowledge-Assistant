"""Operator commands.

    python -m aka.cli init-db
    python -m aka.cli ingest [--force-full]        # synchronous build, prints the report
    python -m aka.cli worker                       # run the ingest worker in the foreground
    python -m aka.cli create-user --email a@b.c --role reviewer
    python -m aka.cli token --email a@b.c          # mint a JWT for an existing user
    python -m aka.cli overrides [--window-days 30]  # override rate, overall and per rule
    python -m aka.cli rollback                     # re-activate the previous index version
"""

from __future__ import annotations

import argparse
import getpass
import json
import sys

from sqlalchemy import select

from aka.settings import get_settings


def _tenant(args) -> str:
    return args.tenant or get_settings().DEFAULT_TENANT


def cmd_init_db(args) -> int:
    from aka.db.session import init_db

    init_db()
    print("database ready")
    return 0


def cmd_ingest(args) -> int:
    from aka.db.session import init_db, session_scope
    from aka.ingest.pipeline import IncrementalIngestor

    init_db()
    with session_scope() as session:
        report = IncrementalIngestor(session).run(_tenant(args), force_full=args.force_full)
    print(json.dumps(report.as_dict(), indent=2, ensure_ascii=False))
    return 0 if (report.swapped or report.noop or report.scanned == 0) else 1


def cmd_worker(args) -> int:
    from aka.ingest.worker import main

    main()
    return 0


def cmd_create_user(args) -> int:
    from aka.db.session import init_db, session_scope
    from aka.security.auth import create_user

    password = sys.stdin.readline().rstrip("\n") if args.password_stdin else getpass.getpass("password: ")
    init_db()
    with session_scope() as session:
        user = create_user(
            session, tenant_id=_tenant(args), email=args.email, password=password, role=args.role
        )
        print(json.dumps({"user_id": user.id, "email": user.email, "role": user.role}))
    return 0


def cmd_token(args) -> int:
    from aka.db.models import User
    from aka.db.session import session_scope
    from aka.security.auth import issue_token, principal_from_user

    with session_scope() as session:
        user = session.scalars(
            select(User).where(User.tenant_id == _tenant(args), User.email == args.email.lower())
        ).first()
        if user is None:
            print("no such user", file=sys.stderr)
            return 1
        print(issue_token(principal_from_user(user)))
    return 0


def cmd_overrides(args) -> int:
    from aka.db.session import session_scope
    from aka.hitl import audit

    tenant = _tenant(args)
    with session_scope() as session:
        overall = audit.override_stats(session, tenant_id=tenant, window_days=args.window_days)
        rows = audit.rule_breakdown(session, tenant_id=tenant, window_days=args.window_days)
        per_rule = {
            row["rule"]: audit.override_stats(
                session, tenant_id=tenant, window_days=args.window_days, rule=row["rule"]
            ).as_dict()
            for row in rows
        }
    print(json.dumps({"overall": overall.as_dict(), "per_rule": per_rule}, indent=2))
    return 0


def cmd_rollback(args) -> int:
    from aka.db.session import session_scope
    from aka.ingest.vectorstore import IndexManager

    with session_scope() as session:
        version = IndexManager(session).rollback_to_previous(_tenant(args))
        if version is None:
            print("no previous version to roll back to", file=sys.stderr)
            return 1
        print(f"active version is now {version.id}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m aka.cli")
    parser.add_argument("--tenant", help="tenant id (default: DEFAULT_TENANT)")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init-db").set_defaults(fn=cmd_init_db)

    p = sub.add_parser("ingest")
    p.add_argument("--force-full", action="store_true", help="re-embed everything")
    p.set_defaults(fn=cmd_ingest)

    sub.add_parser("worker").set_defaults(fn=cmd_worker)

    p = sub.add_parser("create-user")
    p.add_argument("--email", required=True)
    p.add_argument("--role", default="user", choices=["user", "reviewer", "admin"])
    p.add_argument("--password-stdin", action="store_true")
    p.set_defaults(fn=cmd_create_user)

    p = sub.add_parser("token")
    p.add_argument("--email", required=True)
    p.set_defaults(fn=cmd_token)

    p = sub.add_parser("overrides")
    p.add_argument("--window-days", type=int, default=30)
    p.set_defaults(fn=cmd_overrides)

    sub.add_parser("rollback").set_defaults(fn=cmd_rollback)

    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
