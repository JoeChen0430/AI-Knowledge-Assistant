"""Document loading and chunking.

Keeps LlamaIndex only for what it is genuinely good at — format-specific
readers and sentence-aware splitting — and returns plain dataclasses so the
rest of the pipeline has no framework dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter

from aka.ingest.hashing import chunk_hash, estimate_tokens, normalize_text
from aka.obs.logging import get_logger
from aka.settings import Settings, get_settings

log = get_logger("aka.chunker")


class UnsupportedDocument(ValueError):
    pass


@dataclass
class ParsedChunk:
    ordinal: int
    text: str
    hash: str
    token_count: int
    meta: dict = field(default_factory=dict)


def _splitter(settings: Settings) -> SentenceSplitter:
    return SentenceSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        # Chinese sentence terminators, so CJK documents don't get split
        # mid-sentence on whitespace that isn't there.
        secondary_chunking_regex="[。！？；\n]",
    )


def load_text(path: Path, settings: Settings | None = None) -> str:
    """Extract raw text from one file."""
    settings = settings or get_settings()
    suffix = path.suffix.lower()
    if suffix not in settings.ALLOWED_EXTENSIONS:
        raise UnsupportedDocument(f"unsupported extension: {suffix}")

    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="replace")

    docs = SimpleDirectoryReader(input_files=[str(path)]).load_data()
    return "\n\n".join(d.text for d in docs if d.text)


def chunk_text(text: str, settings: Settings | None = None) -> list[ParsedChunk]:
    settings = settings or get_settings()
    normalized = normalize_text(text)
    if not normalized:
        return []

    pieces = _splitter(settings).split_text(normalized)
    chunks: list[ParsedChunk] = []
    for ordinal, piece in enumerate(pieces):
        piece = piece.strip()
        if not piece:
            continue
        chunks.append(
            ParsedChunk(
                ordinal=ordinal,
                text=piece,
                hash=chunk_hash(piece),
                token_count=estimate_tokens(piece),
            )
        )
    return chunks


def parse_document(path: Path, settings: Settings | None = None) -> list[ParsedChunk]:
    settings = settings or get_settings()
    text = load_text(path, settings)
    chunks = chunk_text(text, settings)
    log.debug("chunker.parsed", path=path.name, chunks=len(chunks))
    return chunks
