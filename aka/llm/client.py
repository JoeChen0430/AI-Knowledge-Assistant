"""Ollama HTTP client.

Deliberately talks to Ollama's REST API directly instead of going through
LlamaIndex's LLM wrapper: the router needs per-call control over timeouts,
token accounting and streaming to make the fallback chain and the traces
meaningful, and the framework wrapper hides all three.

Sync on purpose. The workload is I/O-bound with low concurrency, and FastAPI
runs ``def`` handlers in a threadpool, so sync code avoids an entire class of
blocking-inside-async bugs for no practical throughput cost here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterator

import httpx

from aka.obs.logging import get_logger
from aka.settings import Settings, get_settings

log = get_logger("aka.llm.client")


class LLMError(RuntimeError):
    """Any failure talking to the model backend."""


class LLMTimeout(LLMError):
    pass


class LLMUnavailable(LLMError):
    """Backend unreachable — the signal the breaker counts."""


@dataclass
class ChatMessage:
    role: str
    content: str

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}


@dataclass
class Completion:
    text: str
    model: str
    tokens_in: int = 0
    tokens_out: int = 0
    latency_s: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)


class OllamaClient:
    def __init__(self, settings: Settings | None = None, client: httpx.Client | None = None):
        self.settings = settings or get_settings()
        self.base_url = self.settings.OLLAMA_BASE_URL.rstrip("/")
        self._client = client or httpx.Client(
            base_url=self.base_url,
            timeout=httpx.Timeout(self.settings.LLM_TIMEOUT_S, connect=5.0),
        )

    # --- lifecycle --------------------------------------------------------
    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "OllamaClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # --- introspection ----------------------------------------------------
    def available_models(self) -> list[str]:
        try:
            resp = self._client.get("/api/tags", timeout=5.0)
            resp.raise_for_status()
            return [m["name"] for m in resp.json().get("models", [])]
        except Exception as exc:
            raise LLMUnavailable(f"cannot list models: {exc}") from exc

    def ping(self) -> bool:
        try:
            self._client.get("/api/tags", timeout=3.0).raise_for_status()
            return True
        except Exception:
            return False

    # --- generation -------------------------------------------------------
    def chat(
        self,
        messages: list[ChatMessage],
        *,
        model: str,
        timeout: float | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.2,
    ) -> Completion:
        payload = {
            "model": model,
            "messages": [m.to_dict() for m in messages],
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens or self.settings.LLM_MAX_TOKENS,
            },
        }
        data = self._post("/api/chat", payload, timeout or self.settings.LLM_TIMEOUT_S)
        return Completion(
            text=(data.get("message") or {}).get("content", ""),
            model=model,
            tokens_in=data.get("prompt_eval_count", 0) or 0,
            tokens_out=data.get("eval_count", 0) or 0,
            latency_s=(data.get("total_duration", 0) or 0) / 1e9,
            meta={"done_reason": data.get("done_reason")},
        )

    def chat_stream(
        self,
        messages: list[ChatMessage],
        *,
        model: str,
        timeout: float | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.2,
    ) -> Iterator[str]:
        """Yield content deltas. Final usage counters land in ``.usage`` on the
        generator object once exhausted."""
        payload = {
            "model": model,
            "messages": [m.to_dict() for m in messages],
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens or self.settings.LLM_MAX_TOKENS,
            },
        }
        try:
            with self._client.stream(
                "POST", "/api/chat", json=payload, timeout=timeout or self.settings.LLM_TIMEOUT_S
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    chunk = json.loads(line)
                    if chunk.get("error"):
                        raise LLMError(chunk["error"])
                    delta = (chunk.get("message") or {}).get("content", "")
                    if delta:
                        yield delta
                    if chunk.get("done"):
                        break
        except httpx.TimeoutException as exc:
            raise LLMTimeout(f"stream timed out after {timeout}s") from exc
        except httpx.HTTPError as exc:
            raise LLMUnavailable(f"stream transport failure: {exc}") from exc

    # --- embeddings -------------------------------------------------------
    def embed(self, texts: list[str], *, model: str | None = None) -> list[list[float]]:
        """Batch embed. Falls back to the legacy single-input endpoint on
        Ollama builds that predate /api/embed."""
        if not texts:
            return []
        model = model or self.settings.EMBED_MODEL
        try:
            data = self._post(
                "/api/embed",
                {"model": model, "input": texts},
                self.settings.EMBED_TIMEOUT_S * max(1, len(texts) / 8),
            )
            vectors = data.get("embeddings")
            if vectors:
                return vectors
        except LLMError:
            log.debug("embed.batch_endpoint_unavailable", model=model)

        out = []
        for text in texts:
            data = self._post(
                "/api/embeddings",
                {"model": model, "prompt": text},
                self.settings.EMBED_TIMEOUT_S,
            )
            vec = data.get("embedding")
            if not vec:
                raise LLMError(f"embedding endpoint returned no vector for model {model}")
            out.append(vec)
        return out

    def embed_one(self, text: str, *, model: str | None = None) -> list[float]:
        vectors = self.embed([text], model=model)
        if not vectors:
            raise LLMError("no embedding returned")
        return vectors[0]

    # --- transport --------------------------------------------------------
    def _post(self, path: str, payload: dict, timeout: float) -> dict:
        try:
            resp = self._client.post(path, json=payload, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except httpx.TimeoutException as exc:
            raise LLMTimeout(f"{path} timed out after {timeout}s") from exc
        except httpx.HTTPStatusError as exc:
            body = exc.response.text[:300]
            if exc.response.status_code >= 500:
                raise LLMUnavailable(f"{path} -> {exc.response.status_code}: {body}") from exc
            raise LLMError(f"{path} -> {exc.response.status_code}: {body}") from exc
        except httpx.HTTPError as exc:
            raise LLMUnavailable(f"{path} transport failure: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise LLMError(f"{path} returned non-JSON") from exc
