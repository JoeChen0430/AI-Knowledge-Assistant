"""Single-document update cost, and whether the index ever goes empty.

Compares the original ingest strategy with the blue/green incremental one on
the same corpus, while a probe thread queries the knowledge base every
``--probe-interval`` seconds the whole time:

  initial       first blue/green build of the corpus (= the cost of a full rebuild)
  incremental   the same corpus after one document is edited or added
  legacy        drop the collection, re-parse and re-embed every document
                (what the first version did on every upload or delete),
                on the same edited corpus
  noop          a rerun with nothing changed

Two full builds in total (initial and legacy), so expect roughly twice the
full-rebuild time. A heartbeat line every 30s shows a long build is moving.

    # real embeddings (Ollama must be running). ~4000 synthetic docs of 800
    # words is ~12k chunks — around 14 minutes per full build on an M-series Mac
    python eval/ingest_benchmark.py --synthetic-docs 4000

    # or a real document folder, if it is large enough
    python eval/ingest_benchmark.py --corpus ~/Documents/notes

    # timing shape only, no model server
    python eval/ingest_benchmark.py --backend fake --synthetic-docs 2000 --fake-embed-ms 20

Don't run it while another Ollama job (e.g. fault_injection.py) is running:
both share the same model server and the timings would be meaningless.

Each probe classifies what a query would have seen:

  complete   the index holds at least as many chunks as before the change
  partial    it answers, but from part of the corpus (a rebuild in progress)
  empty      no collection, an empty one, or the query failed

The legacy run mirrors LlamaIndex's ``VectorStoreIndex.from_documents``,
which embeds and inserts in batches of 2048 nodes, so a small corpus stays
empty for the whole rebuild and a large one is partial for most of it.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.common import (  # noqa: E402
    copy_corpus,
    make_backend,
    make_workspace,
    synthesize_corpus,
    write_json,
)


COMPLETE, PARTIAL, EMPTY = "complete", "partial", "empty"


class Prober:
    """Queries the knowledge base on a timer and records what it saw."""

    def __init__(self, probe, interval_s: float):
        self.probe = probe  # () -> COMPLETE | PARTIAL | EMPTY
        self.interval_s = interval_s
        self.counts = {COMPLETE: 0, PARTIAL: 0, EMPTY: 0}
        self.errors = 0
        # Longest continuous stretch in each bad state.
        self.longest = {PARTIAL: 0.0, EMPTY: 0.0, "not_complete": 0.0}
        self._since: dict[str, float | None] = {PARTIAL: None, EMPTY: None, "not_complete": None}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _track(self, key: str, active: bool, now: float) -> None:
        since = self._since[key]
        if active and since is None:
            self._since[key] = now
        elif not active and since is not None:
            self.longest[key] = max(self.longest[key], now - since)
            self._since[key] = None

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                state = self.probe()
            except Exception:
                self.errors += 1
                state = EMPTY
            now = time.perf_counter()
            self.counts[state] += 1
            self._track(PARTIAL, state == PARTIAL, now)
            self._track(EMPTY, state == EMPTY, now)
            self._track("not_complete", state != COMPLETE, now)
            self._stop.wait(self.interval_s)

    def __enter__(self) -> "Prober":
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join(timeout=10)
        now = time.perf_counter()
        for key in self._since:
            self._track(key, False, now)

    @property
    def longest_not_complete_s(self) -> float:
        return self.longest["not_complete"]

    def as_dict(self) -> dict:
        probes = sum(self.counts.values())
        return {
            "probes": probes,
            "counts": dict(self.counts),
            "errors": self.errors,
            "longest_empty_s": round(self.longest[EMPTY], 3),
            "longest_partial_s": round(self.longest[PARTIAL], 3),
            "longest_not_complete_s": round(self.longest["not_complete"], 3),
            "complete_share": round(self.counts[COMPLETE] / probes, 4) if probes else 0.0,
        }


# --------------------------------------------------------------------------
# legacy strategy
# --------------------------------------------------------------------------

LEGACY_COLLECTION = "legacy_benchmark"
LLAMAINDEX_INSERT_BATCH = 2048  # VectorStoreIndex's default insert_batch_size


def legacy_rebuild(settings, backend, progress: dict | None = None) -> dict:
    """Delete-and-rebuild, as the original ingest.py did: same parser, same
    chunker, embeddings inserted LLAMAINDEX_INSERT_BATCH nodes at a time.
    ``progress['embedded']`` is updated as it goes, for the heartbeat."""
    from aka.ingest.chunker import parse_document
    from aka.ingest.pipeline import EMBED_BATCH, scan_data_dir
    from aka.ingest.vectorstore import get_chroma

    chroma = get_chroma(settings)
    started = time.perf_counter()
    try:
        chroma.delete_collection(LEGACY_COLLECTION)
    except Exception:
        pass
    collection = chroma.get_or_create_collection(LEGACY_COLLECTION, metadata={"hnsw:space": "cosine"})

    buffer: list[tuple[str, str, list[float]]] = []
    chunks = 0

    def flush() -> None:
        if buffer:
            collection.upsert(
                ids=[b[0] for b in buffer],
                documents=[b[1] for b in buffer],
                embeddings=[b[2] for b in buffer],
            )
            buffer.clear()

    for rel_path, path in scan_data_dir(settings).items():
        try:
            parsed = parse_document(path, settings)
        except Exception:
            continue
        for start in range(0, len(parsed), EMBED_BATCH):
            batch = parsed[start : start + EMBED_BATCH]
            vectors = backend.embed([c.text for c in batch])
            buffer.extend(
                (f"{rel_path}:{c.ordinal}", c.text, v) for c, v in zip(batch, vectors)
            )
            chunks += len(batch)
            if progress is not None:
                progress["embedded"] = chunks
            if len(buffer) >= LLAMAINDEX_INSERT_BATCH:
                flush()
    flush()
    return {"duration_s": round(time.perf_counter() - started, 3), "chunks_embedded": chunks}


def legacy_probe(settings, vector, expected: int):
    from aka.ingest.vectorstore import get_chroma

    chroma = get_chroma(settings)

    def probe() -> str:
        try:
            collection = chroma.get_collection(LEGACY_COLLECTION)
            count = collection.count()
            if count == 0:
                return EMPTY
            got = collection.query(query_embeddings=[vector], n_results=1)
        except Exception:
            return EMPTY
        if not (got["ids"] and got["ids"][0]):
            return EMPTY
        return COMPLETE if count >= expected else PARTIAL

    return probe


# --------------------------------------------------------------------------
# blue/green strategy
# --------------------------------------------------------------------------

def bluegreen_probe(settings, vector, expected: int):
    from aka.db.session import get_session_factory
    from aka.ingest.vectorstore import IndexManager

    factory = get_session_factory(settings)

    def probe() -> str:
        session = factory()
        try:
            store = IndexManager(session, settings).active_store(settings.DEFAULT_TENANT)
            if store is None or not store.query(vector, k=1):
                return EMPTY
            return COMPLETE if store.count() >= expected else PARTIAL
        finally:
            session.close()

    return probe


class Heartbeat:
    """A progress line every 30s, so a 14-minute build visibly moves."""

    def __init__(self, label, count=None, every: float = 30.0):
        self.label, self.count, self.every = label, count, every
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._started = 0.0

    def _run(self) -> None:
        while not self._stop.wait(self.every):
            extra = ""
            if self.count is not None:
                try:
                    extra = f", {self.count()} chunks so far"
                except Exception:
                    pass
            minutes = (time.perf_counter() - self._started) / 60
            print(f"  ... {self.label}: {minutes:.1f} min{extra}", flush=True)

    def __enter__(self) -> "Heartbeat":
        self._started = time.perf_counter()
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join(timeout=5)


def building_count(settings) -> int:
    """Chunks written so far into the BUILDING version's collection."""
    from sqlalchemy import select

    from aka.db.models import IndexStatus, IndexVersion
    from aka.db.session import get_session_factory
    from aka.ingest.vectorstore import get_chroma

    session = get_session_factory(settings)()
    try:
        name = session.scalars(
            select(IndexVersion.collection_name).where(
                IndexVersion.tenant_id == settings.DEFAULT_TENANT,
                IndexVersion.status == IndexStatus.BUILDING,
            )
        ).first()
    finally:
        session.close()
    return get_chroma(settings).get_collection(name).count() if name else 0


def active_count(settings) -> int:
    from aka.db.session import session_scope
    from aka.ingest.vectorstore import IndexManager

    with session_scope(settings) as session:
        store = IndexManager(session, settings).active_store(settings.DEFAULT_TENANT)
        return store.count() if store else 0


def run_ingest(settings, backend) -> dict:
    from aka.db.session import session_scope
    from aka.ingest.pipeline import IncrementalIngestor

    with session_scope(settings) as session:
        report = IncrementalIngestor(session, settings, client=backend).run(settings.DEFAULT_TENANT)
    return report.as_dict()


def edit_one_document(data_dir: Path, target: str | None) -> str:
    """Change one document: append to a text file if there is one, otherwise
    add a new small document (docx/pdf cannot be edited in place here)."""
    if target:
        path = data_dir / target
        if not path.exists():
            raise SystemExit(f"--edit target not found in corpus: {target}")
    else:
        path = next(
            (p for p in sorted(data_dir.rglob("*")) if p.suffix.lower() in (".txt", ".md")), None
        )
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    if path is not None and path.suffix.lower() in (".txt", ".md"):
        with path.open("a", encoding="utf-8") as fh:
            fh.write(f"\n\nBenchmark edit at {stamp}: the failover drill moved to Thursday.\n")
        return f"edited {path.relative_to(data_dir)}"
    added = data_dir / "benchmark_added_note.txt"
    added.write_text(f"Benchmark note added at {stamp}. The failover drill moved to Thursday.\n", encoding="utf-8")
    return f"added {added.name}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", choices=["ollama", "fake"], default="ollama")
    parser.add_argument("--corpus", type=Path, default=Path("data"))
    parser.add_argument("--synthetic-docs", type=int, default=0)
    parser.add_argument("--words-per-doc", type=int, default=800)
    parser.add_argument("--fake-embed-ms", type=float, default=0.0, help="fake latency per embedded chunk")
    parser.add_argument("--edit", default=None, help="corpus-relative file to modify (default: first .txt/.md)")
    parser.add_argument("--skip-legacy", action="store_true")
    parser.add_argument("--probe-interval", type=float, default=0.05)
    parser.add_argument("--workdir", type=Path, default=None)
    parser.add_argument("--out", default="eval/results/ingest_benchmark.json")
    args = parser.parse_args()

    settings, root = make_workspace(args.workdir)
    data_dir = Path(settings.DATA_DIR)
    if args.synthetic_docs:
        synthesize_corpus(data_dir, docs=args.synthetic_docs, words_per_doc=args.words_per_doc)
        corpus_label = f"synthetic:{args.synthetic_docs}x{args.words_per_doc}w"
    else:
        if not args.corpus.exists():
            raise SystemExit(f"corpus not found: {args.corpus}")
        copy_corpus(args.corpus, data_dir, settings.ALLOWED_EXTENSIONS)
        corpus_label = str(args.corpus)
    files = sum(1 for p in data_dir.rglob("*") if p.is_file())
    print(f"workspace: {root}\ncorpus: {corpus_label} ({files} files), backend: {args.backend}")

    backend = make_backend(args.backend, embed_latency_s=args.fake_embed_ms / 1000)
    probe_vector = backend.embed(["failover drill schedule"])[0]
    results: dict = {"corpus": corpus_label, "files": files, "backend": args.backend}

    print("\nblue/green: initial build (= the cost of a full rebuild)...")
    with Heartbeat("initial build", lambda: building_count(settings)):
        results["initial_build"] = run_ingest(settings, backend)
    full = results["initial_build"]
    rate = full["chunks_embedded"] / max(full["duration_s"], 1e-9)
    print(f"  {full['duration_s']:.1f}s for {full['chunks_embedded']} chunks ({rate:.1f} chunks/s)")
    write_json(args.out, results)

    print("blue/green: one document changed -> incremental update (probing)...")
    note = edit_one_document(data_dir, args.edit)
    expected = active_count(settings)
    with Prober(bluegreen_probe(settings, probe_vector, expected), args.probe_interval) as prober:
        update = run_ingest(settings, backend)
    results["incremental_update"] = {**update, "change": note, "availability": prober.as_dict()}
    print(f"  {update['duration_s']:.2f}s (embedded {update['chunks_embedded']}, "
          f"reused {update['chunks_reused']}); empty for {prober.longest[EMPTY]:.2f}s, "
          f"incomplete for {prober.longest_not_complete_s:.2f}s")
    write_json(args.out, results)

    if not args.skip_legacy:
        print(f"\nlegacy: drop the collection and rebuild everything, same corpus (probing; "
              f"expect ~{full['duration_s'] / 60:.1f} min)...")
        progress = {"embedded": 0}
        with Prober(legacy_probe(settings, probe_vector, expected), args.probe_interval) as prober, \
                Heartbeat("legacy rebuild", lambda: progress["embedded"]):
            legacy = legacy_rebuild(settings, backend, progress)
        results["legacy_rebuild"] = {**legacy, "change": note, "availability": prober.as_dict()}
        print(f"  {legacy['duration_s']:.1f}s; empty for {prober.longest[EMPTY]:.1f}s, "
              f"incomplete for {prober.longest_not_complete_s:.1f}s")
        write_json(args.out, results)

    print("blue/green: nothing changed...")
    results["noop"] = run_ingest(settings, backend)
    print(f"  {results['noop']['duration_s']:.2f}s")

    print("\nsummary                         time      longest empty   longest incomplete")
    rows = []
    if "legacy_rebuild" in results:
        rows.append(("legacy full rebuild", results["legacy_rebuild"]))
    rows.append(("blue/green incremental", results["incremental_update"]))
    for label, r in rows:
        a = r["availability"]
        print(f"  {label:<26} {r['duration_s']:>8.2f}s   {a['longest_empty_s']:>12.2f}s   "
              f"{a['longest_not_complete_s']:>16.2f}s")
    print(f"  incremental time breakdown: {results['incremental_update']['timings']}")

    write_json(args.out, results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
