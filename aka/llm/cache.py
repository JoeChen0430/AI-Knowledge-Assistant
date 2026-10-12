"""TTL + LRU answer cache.

Two jobs: skip repeat work, and act as a degradation tier — when every model is
down, a slightly stale answer to the same question over the same evidence beats
an error page.

Entries are keyed on the question *and* the evidence chunk ids. A second index
by question alone serves the case where retrieval itself is degraded (no query
vector): the caller can still find the last good answer and verify that its
evidence chunks are still in the active index. Chunk ids are content hashes, so
"the ids still exist" means "the evidence text is unchanged".
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from aka.obs.metrics import answer_cache


def _normalize_question(question: str) -> str:
    return " ".join(question.strip().lower().split())


@dataclass
class CacheEntry:
    value: Any
    stored_at: float
    expires_at: float
    question_key: str | None = None

    def age_s(self, now: float) -> float:
        return now - self.stored_at


class AnswerCache:
    def __init__(self, ttl_s: int = 900, max_entries: int = 512, time_source=time.monotonic):
        self.ttl_s = ttl_s
        self.max_entries = max_entries
        self._time = time_source
        self._data: OrderedDict[str, CacheEntry] = OrderedDict()
        self._by_question: dict[str, str] = {}
        self._lock = threading.Lock()

    @staticmethod
    def make_key(tenant_id: str, question: str, evidence_ids: list[str]) -> str:
        """Key on the question *and* the evidence, so a re-indexed corpus does
        not keep serving an answer built from chunks that no longer exist."""
        h = hashlib.sha256()
        h.update(tenant_id.encode())
        h.update(b"\x00")
        h.update(_normalize_question(question).encode())
        h.update(b"\x00")
        for cid in sorted(evidence_ids):
            h.update(cid.encode())
            h.update(b"\x01")
        return h.hexdigest()

    @staticmethod
    def question_key(tenant_id: str, question: str) -> str:
        h = hashlib.sha256()
        h.update(tenant_id.encode())
        h.update(b"\x00")
        h.update(_normalize_question(question).encode())
        return h.hexdigest()

    def get(self, key: str, *, allow_stale: bool = False) -> tuple[Any | None, float]:
        """Return (value, age_seconds). ``allow_stale`` ignores expiry — used
        only on the degraded path, where the caller labels the answer as stale."""
        now = self._time()
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                answer_cache.labels(result="miss").inc()
                return None, 0.0
            if entry.expires_at < now and not allow_stale:
                self._drop(key)
                answer_cache.labels(result="expired").inc()
                return None, 0.0
            self._data.move_to_end(key)
            answer_cache.labels(result="stale_hit" if entry.expires_at < now else "hit").inc()
            return entry.value, entry.age_s(now)

    def get_by_question(self, question_key: str) -> tuple[Any | None, float]:
        """Latest answer to this question regardless of evidence (stale allowed).
        The caller must verify the evidence before serving it."""
        with self._lock:
            key = self._by_question.get(question_key)
        if key is None:
            answer_cache.labels(result="miss").inc()
            return None, 0.0
        return self.get(key, allow_stale=True)

    def set(self, key: str, value: Any, *, question_key: str | None = None) -> None:
        now = self._time()
        with self._lock:
            self._data[key] = CacheEntry(value, now, now + self.ttl_s, question_key)
            self._data.move_to_end(key)
            if question_key:
                self._by_question[question_key] = key
            while len(self._data) > self.max_entries:
                oldest = next(iter(self._data))
                self._drop(oldest)

    def _drop(self, key: str) -> None:
        """Remove one entry and its question pointer. Caller holds the lock."""
        entry = self._data.pop(key, None)
        if entry is not None and entry.question_key:
            if self._by_question.get(entry.question_key) == key:
                del self._by_question[entry.question_key]

    def invalidate_all(self) -> int:
        """Drop everything. Evidence-keyed entries stay correct across index
        swaps on their own; this is for operators and tests."""
        with self._lock:
            n = len(self._data)
            self._data.clear()
            self._by_question.clear()
            return n

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)


_default_cache: AnswerCache | None = None
_lock = threading.Lock()


def get_answer_cache() -> AnswerCache:
    global _default_cache
    with _lock:
        if _default_cache is None:
            from aka.settings import get_settings

            s = get_settings()
            _default_cache = AnswerCache(
                ttl_s=s.ANSWER_CACHE_TTL_S, max_entries=s.ANSWER_CACHE_MAX_ENTRIES
            )
        return _default_cache


def reset_answer_cache() -> None:
    global _default_cache
    with _lock:
        _default_cache = None
