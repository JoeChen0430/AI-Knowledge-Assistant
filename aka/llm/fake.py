"""Deterministic, offline stand-in for the Ollama client.

Used by the test suite and by ``--backend fake`` runs of the eval scripts, so
the whole pipeline (ingest, retrieval, generation, policy gate) can run with
no model server. It is not a model; it is a fixture with the same interface:

* embeddings are a hashing-trick bag of words (Latin words + CJK bigrams),
  so texts that share vocabulary are close, and identical texts are identical;
* chat answers by quoting the first sentence of the top evidence passage with
  a ``[1]`` citation, which is grounded by construction — or by returning
  the JSON the memory prompts ask for.

Optional per-call latency makes timing-sensitive evals (ingest benchmark,
time budgets) behave like a slow backend without one.
"""

from __future__ import annotations

import hashlib
import math
import re
import time
from typing import Iterator

from aka.llm.client import ChatMessage, Completion

_EVIDENCE = re.compile(r"\[1\] source: [^\n]*\n(.+?)(?:\n\n\[2\] source:|\n</untrusted-)", re.DOTALL)
_SENTENCE_END = re.compile(r"(?<=[.!?。！？])\s*")


class FakeOllama:
    DIM = 256

    def __init__(
        self,
        *,
        embed_latency_s: float = 0.0,
        chat_latency_s: float = 0.0,
        answer_override: str | None = None,
    ):
        self.embed_latency_s = embed_latency_s
        self.chat_latency_s = chat_latency_s
        self.answer_override = answer_override
        self.embed_calls = 0
        self.embedded_texts = 0
        self.chat_calls = 0

    # --- embeddings ---------------------------------------------------------
    @classmethod
    def vector(cls, text: str) -> list[float]:
        from aka.rag.prompts import _content_words  # local: avoids an import cycle

        counts = [0.0] * cls.DIM
        for token in _content_words(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "little") % cls.DIM
            sign = 1.0 if digest[4] & 1 else -1.0
            counts[bucket] += sign
        norm = math.sqrt(sum(v * v for v in counts))
        if norm == 0.0:
            counts[0], norm = 1.0, 1.0
        return [v / norm for v in counts]

    def embed(self, texts, *, model=None) -> list[list[float]]:
        self.embed_calls += 1
        self.embedded_texts += len(texts)
        if self.embed_latency_s:
            time.sleep(self.embed_latency_s * len(texts))
        return [self.vector(t) for t in texts]

    def embed_one(self, text, *, model=None) -> list[float]:
        return self.embed([text], model=model)[0]

    # --- generation ---------------------------------------------------------
    def _respond(self, messages: list[ChatMessage]) -> str:
        system = messages[0].content if messages else ""
        if "extract durable facts" in system:
            return "[]"
        if "contradicts an existing one" in system:
            return '{"contradicts": false, "confidence": 0.1, "reason": "fake backend"}'
        if "Condense this conversation" in system:
            return "Earlier turns discussed the user's documents."
        if self.answer_override is not None:
            return self.answer_override

        match = _EVIDENCE.search(messages[-1].content if messages else "")
        if not match:
            return "I don't have enough in your documents to answer that."
        passage = " ".join(match.group(1).split())
        first = _SENTENCE_END.split(passage, maxsplit=1)[0][:300]
        return f"{first} [1]"

    def chat(self, messages, *, model, timeout=None, max_tokens=None, temperature=0.2) -> Completion:
        self.chat_calls += 1
        if self.chat_latency_s:
            time.sleep(self.chat_latency_s)
        text = self._respond(list(messages))
        return Completion(text=text, model=model, tokens_in=100, tokens_out=max(1, len(text) // 4))

    def chat_stream(self, messages, *, model, timeout=None, max_tokens=None, temperature=0.2) -> Iterator[str]:
        text = self.chat(messages, model=model).text
        for i in range(0, len(text), 16):
            yield text[i : i + 16]

    # --- introspection --------------------------------------------------------
    def ping(self) -> bool:
        return True

    def available_models(self) -> list[str]:
        return ["fake"]

    def close(self) -> None:
        pass
