"""Shared plumbing for the eval scripts.

Every eval runs in its own workspace (temp dir by default): its own SQLite
file, Chroma directory and copy of the corpus, so nothing touches the real
knowledge base under ./data, ./var and ./chroma_db.
"""

from __future__ import annotations

import json
import os
import random
import re
import shutil
import statistics
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("AKA_EAGER_SETTINGS", "0")


def make_workspace(root: str | Path | None, **overrides: Any):
    """Point every storage setting at ``root`` and return fresh Settings.

    Call before anything imports ``aka``: logging is configured from the
    environment on first import.
    """
    root = Path(root) if root else Path(tempfile.mkdtemp(prefix="aka-eval-"))
    root.mkdir(parents=True, exist_ok=True)
    env = {
        "ENV": "eval",
        "DATA_DIR": str(root / "data"),
        "DB_DIR": str(root / "chroma"),
        "DATABASE_URL": f"sqlite:///{root / 'aka.db'}",
        "TRASH_DIR": str(root / "trash"),
        "QUARANTINE_DIR": str(root / "quarantine"),
        "OTEL_ENABLED": "false",
        "LOG_LEVEL": os.environ.get("LOG_LEVEL", "WARNING"),
        "AUTH_ENABLED": "false",
        "EMBEDDED_WORKER": "false",
        "MEMORY_EXTRACTION_ENABLED": "false",
    }
    env.update({k: str(v) for k, v in overrides.items()})
    os.environ.update(env)

    from aka.db.session import init_db, reset_engine_for_tests
    from aka.ingest.vectorstore import reset_chroma_for_tests
    from aka.llm.breaker import clear_breaker_registry
    from aka.settings import get_settings, reset_settings_cache

    reset_settings_cache()
    reset_engine_for_tests()
    reset_chroma_for_tests()
    clear_breaker_registry()
    settings = get_settings()
    init_db(settings)
    return settings, root


def copy_corpus(source: Path, dest: Path, allowed: set[str]) -> int:
    """Copy ingestable files, keeping relative paths. Returns the file count."""
    count = 0
    for path in sorted(source.rglob("*")):
        if path.is_file() and not path.name.startswith(".") and path.suffix.lower() in allowed:
            target = dest / path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            count += 1
    return count


_VOCAB = (
    "system cluster replica latency throughput budget schedule incident review "
    "deploy rollback storage index vector query cache policy approval memory "
    "network region failover monitor alert dashboard tenant quota release "
    "migration backup restore checksum pipeline worker queue lease timeout "
    "capacity forecast audit compliance contract invoice customer roadmap"
).split()


def synthesize_corpus(dest: Path, *, docs: int, words_per_doc: int, seed: int = 7) -> int:
    """Plain-text documents with a distinct topic word each, so questions can
    target one document. Good enough for timing; not for answer quality."""
    rng = random.Random(seed)
    dest.mkdir(parents=True, exist_ok=True)
    for i in range(docs):
        topic = f"project{i:05d}"
        sentences = []
        written = 0
        while written < words_per_doc:
            n = rng.randint(8, 18)
            words = [rng.choice(_VOCAB) for _ in range(n)]
            words.insert(rng.randrange(len(words)), topic)
            sentences.append(" ".join(words).capitalize() + ".")
            written += n + 1
        (dest / f"doc_{i:05d}.txt").write_text(" ".join(sentences), encoding="utf-8")
    return docs


_CJK = re.compile(r"[㐀-䶿一-鿿]{6,}")
_WORD = re.compile(r"[A-Za-z][A-Za-z\-]{3,}")


def question_from_chunk(text: str, rng: random.Random) -> str | None:
    """A question whose answer is in this chunk: built from its own salient
    words, which is what a user who remembers a document would type."""
    cjk = _CJK.findall(text)
    if cjk:
        run = rng.choice(cjk)
        start = rng.randrange(0, max(1, len(run) - 8))
        return f"我的文件裡關於「{run[start : start + 8]}」說了什麼？"
    words = list(dict.fromkeys(w.lower() for w in _WORD.findall(text)))
    if len(words) < 3:
        return None
    picked = rng.sample(words, k=min(4, len(words)))
    return f"What do my documents say about {' '.join(picked)}?"


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = max(0, min(len(ordered) - 1, int(round(pct / 100 * (len(ordered) - 1)))))
    return ordered[k]


def describe_latency(values: list[float]) -> dict:
    if not values:
        return {}
    return {
        "p50_s": round(percentile(values, 50), 3),
        "p95_s": round(percentile(values, 95), 3),
        "max_s": round(max(values), 3),
        "mean_s": round(statistics.fmean(values), 3),
    }


def write_json(path: str | Path | None, payload: dict) -> None:
    if not path:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {path}")


def make_backend(kind: str, *, embed_latency_s: float = 0.0, chat_latency_s: float = 0.0):
    """Real Ollama client, or the deterministic fake."""
    if kind == "fake":
        from aka.llm.fake import FakeOllama

        return FakeOllama(embed_latency_s=embed_latency_s, chat_latency_s=chat_latency_s)
    from aka.llm.client import OllamaClient
    from aka.settings import get_settings

    client = OllamaClient(get_settings())
    if not client.ping():
        raise SystemExit(
            f"Ollama is not reachable at {get_settings().OLLAMA_BASE_URL}. "
            "Start it (ollama serve) or use --backend fake."
        )
    return client
