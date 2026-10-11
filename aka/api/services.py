"""Process-wide collaborators shared by every request.

The breaker registry, the answer cache and the query-vector cache only work if
every request sees the same instances, so they are built once per app and
handed to each request's pipeline instead of being created per call.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.orm import Session

from aka.db.session import session_scope
from aka.ingest.worker import EmbeddedWorker
from aka.llm.cache import AnswerCache
from aka.llm.embedding import ResilientEmbedder
from aka.llm.faults import build_client
from aka.llm.router import ModelRouter
from aka.obs.logging import get_logger
from aka.rag.pipeline import RagPipeline
from aka.settings import Settings

log = get_logger("aka.api.services")


@dataclass
class AppServices:
    settings: Settings
    client: Any
    router: ModelRouter
    embedder: ResilientEmbedder
    cache: AnswerCache
    worker: EmbeddedWorker | None = field(default=None)

    @classmethod
    def build(cls, settings: Settings, llm_client: Any = None) -> "AppServices":
        client = llm_client or build_client(settings)
        router = ModelRouter(settings, client=client)
        return cls(
            settings=settings,
            client=client,
            router=router,
            embedder=ResilientEmbedder(client, settings),
            cache=AnswerCache(
                ttl_s=settings.ANSWER_CACHE_TTL_S, max_entries=settings.ANSWER_CACHE_MAX_ENTRIES
            ),
        )

    def pipeline(self, session: Session) -> RagPipeline:
        return RagPipeline(
            session,
            self.settings,
            router=self.router,
            embedder=self.embedder,
            cache=self.cache,
        )

    def learn_in_background(
        self, *, tenant_id: str, user_id: str, conversation_id: str, question: str, answer: str
    ) -> None:
        """Post-turn memory work with its own session (the request's is closed)."""
        try:
            with session_scope(self.settings) as session:
                self.pipeline(session).learn_from_turn(
                    tenant_id=tenant_id,
                    user_id=user_id,
                    conversation_id=conversation_id,
                    question=question,
                    answer=answer,
                )
        except Exception:
            log.exception("api.learn_failed", conversation=conversation_id)

    def start_worker(self) -> None:
        if self.worker is None:
            self.worker = EmbeddedWorker(self.settings, client=self.client)
            self.worker.start()

    def stop_worker(self) -> None:
        if self.worker is not None:
            self.worker.stop()
            self.worker = None
