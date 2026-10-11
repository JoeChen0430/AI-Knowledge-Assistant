"""Content addressing.

Incremental ingest rests on two hashes:

* ``file_digest``  — has this file changed at all? (skip parsing entirely)
* ``chunk_id``     — has this exact chunk been embedded before? (skip the
  embedding call, which is the expensive part)

Both are pure functions of content, never of mtime, so copying a file or
re-uploading an identical one is free.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import Path

_CHUNK_READ_SIZE = 1024 * 1024
_WS = re.compile(r"\s+")


def file_digest(path: Path | str) -> str:
    """SHA-256 of file bytes, streamed so large PDFs don't load into memory."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(_CHUNK_READ_SIZE):
            h.update(block)
    return h.hexdigest()


def bytes_digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def normalize_text(text: str) -> str:
    """Canonical form for hashing: NFKC, collapsed whitespace, stripped.

    Whitespace-only edits (a reflowed PDF export, CRLF churn) then produce the
    same chunk hash and cost nothing to re-ingest.
    """
    return _WS.sub(" ", unicodedata.normalize("NFKC", text)).strip()


def text_digest(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def chunk_id(tenant_id: str, document_id: str, ordinal: int, text: str) -> str:
    """Deterministic vector-store id.

    Includes the content hash so that re-ingesting a changed chunk writes to a
    *new* id — the stale vector is then removed by id diff rather than needing
    a full collection wipe.
    """
    return f"{document_id}:{ordinal:05d}:{text_digest(text)[:16]}"


def chunk_hash(text: str) -> str:
    return text_digest(text)


def estimate_tokens(text: str) -> int:
    """Cheap token estimate: ~4 chars/token for latin, ~1.6 for CJK.

    Used for memory budgets, where being within ~15% is enough and calling a
    real tokenizer on every turn is not worth the latency.
    """
    if not text:
        return 0
    cjk = sum(1 for ch in text if "一" <= ch <= "鿿" or "぀" <= ch <= "ヿ")
    latin = len(text) - cjk
    return int(cjk / 1.6 + latin / 4) + 1
