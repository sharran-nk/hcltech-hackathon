"""Persistent, crash-safe store for chunk metadata + FAISS + BM25.

On disk (DATA_DIR/index):
    versions/v000007/        one complete, self-contained index version
        chunks.json            chunk metadata + text   (source of truth for BM25)
        vectors.npy            float32 embeddings      (source of truth for FAISS)
        faiss.index            serialised FAISS index  (rebuilt from vectors.npy if damaged)
        manifest.json          version, counts, model name, sha256 checksums
    CURRENT                  text file naming the active version (replaced atomically)

Safety rules
- ONE lock serialises all writers. Readers use an immutable in-memory snapshot, so a search
  never sees a half-updated index.
- A new version is written to a temp dir, re-loaded and verified, renamed into place, and only
  then does CURRENT switch (os.replace is atomic). Any failure leaves the old version active.
- The previous version is kept as a backup; load() falls back to it if the current one is corrupt.
- BM25 is rebuilt from chunks.json at load time (rank_bm25 has no safe on-disk format, and
  pickle would be a code-execution risk).
- Single-process design (Streamlit threads). Multi-process access would need a file lock.
"""
import hashlib
import json
import logging
import os
import re
import shutil
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

import config
from chunking import Chunk
from embeddings import EMBEDDING_DIM
from textutil import tokenize

logger = logging.getLogger(__name__)

_VERSION_RE = re.compile(r"v(\d{6})")
KEEP_VERSIONS = 2  # current + one backup


class IndexUpdateError(Exception):
    """An update failed; the previously active index is still in use."""


class IndexLoadError(Exception):
    """No usable index version could be loaded; a rebuild is required."""


@dataclass(frozen=True)
class IndexSnapshot:
    """Immutable view of one index version. Safe to read from many threads."""
    version: str
    chunks: tuple
    vectors: np.ndarray
    faiss_index: object
    bm25: Optional[BM25Okapi]
    pos_by_id: dict

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)

    def doc_ids(self) -> set:
        return {c.doc_id for c in self.chunks}


# ------------------------------------------------------------------ snapshot building
def build_snapshot(version: str, chunks: list, vectors: np.ndarray, faiss_index=None) -> IndexSnapshot:
    vectors = np.ascontiguousarray(vectors, dtype="float32")
    if vectors.ndim != 2 or vectors.shape[0] != len(chunks):
        raise ValueError("vectors must have one row per chunk")
    if len(chunks) and vectors.shape[1] != EMBEDDING_DIM:
        raise ValueError(f"vectors must have {EMBEDDING_DIM} columns")
    ids = [c.chunk_id for c in chunks]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate chunk_id")
    if faiss_index is None:
        faiss_index = faiss.IndexFlatIP(EMBEDDING_DIM)  # exact search; vectors are normalised => cosine
        if len(chunks):
            faiss_index.add(vectors)
    bm25 = None
    if chunks:  # empty tokens would break BM25 statistics, so use a placeholder token
        bm25 = BM25Okapi([tokenize(c.text) or ["__empty__"] for c in chunks])
    return IndexSnapshot(version, tuple(chunks), vectors, faiss_index, bm25, {c.chunk_id: i for i, c in enumerate(chunks)})


def verify_snapshot(s: IndexSnapshot) -> list:
    """Return a list of consistency problems (empty list = consistent)."""
    problems = []
    if s.faiss_index.ntotal != len(s.chunks):
        problems.append(f"FAISS has {s.faiss_index.ntotal} vectors but there are {len(s.chunks)} chunks")
    if s.vectors.shape[0] != len(s.chunks):
        problems.append("vector rows != chunk count")
    bm25_size = s.bm25.corpus_size if s.bm25 else 0
    if bm25_size != len(s.chunks):
        problems.append(f"BM25 has {bm25_size} documents but there are {len(s.chunks)} chunks")
    if len(s.pos_by_id) != len(s.chunks):
        problems.append("chunk id map out of sync")
    return problems


def assert_document_absent(s: IndexSnapshot, doc_id: str) -> None:
    """Deletion check: no chunk of doc_id may remain in metadata, FAISS or BM25."""
    leftovers = [c.chunk_id for c in s.chunks if c.doc_id == doc_id]
    if leftovers:
        raise IndexUpdateError(f"Chunks of {doc_id} still present: {leftovers}")
    problems = verify_snapshot(s)  # FAISS/BM25 sizes equal the remaining chunk list => nothing extra remains
    if problems:
        raise IndexUpdateError("; ".join(problems))


# ------------------------------------------------------------------ disk format
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _write_version_files(directory: Path, s: IndexSnapshot) -> None:
    directory.mkdir(parents=True, exist_ok=False)
    np.save(directory / "vectors.npy", s.vectors)
    (directory / "chunks.json").write_text(
        json.dumps([c.to_dict() for c in s.chunks], ensure_ascii=False), encoding="utf-8")
    (directory / "faiss.index").write_bytes(faiss.serialize_index(s.faiss_index).tobytes())
    manifest = {
        "version": s.version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "chunk_count": len(s.chunks),
        "doc_ids": sorted(s.doc_ids()),
        "embedding_model": config.EMBEDDING_MODEL,
        "dim": EMBEDDING_DIM,
        "sha256": {"chunks.json": _sha256(directory / "chunks.json"),
                   "vectors.npy": _sha256(directory / "vectors.npy")},
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _read_version(directory: Path) -> IndexSnapshot:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["sha256"].items():
        if _sha256(directory / name) != expected:
            raise ValueError(f"checksum mismatch for {name}")
    if manifest.get("embedding_model") != config.EMBEDDING_MODEL:
        logger.warning("Index was built with %s but config uses %s: rebuild recommended.",
                       manifest.get("embedding_model"), config.EMBEDDING_MODEL)
    chunks = [Chunk(**d) for d in json.loads((directory / "chunks.json").read_text(encoding="utf-8"))]
    vectors = np.load(directory / "vectors.npy", allow_pickle=False)
    faiss_index = None
    try:
        candidate = faiss.deserialize_index(np.frombuffer((directory / "faiss.index").read_bytes(), dtype="uint8"))
        if candidate.ntotal == len(chunks):
            faiss_index = candidate
        else:
            logger.warning("faiss.index size mismatch in %s: rebuilding from vectors.npy", directory.name)
    except Exception:
        logger.warning("faiss.index unreadable in %s: rebuilding from vectors.npy", directory.name)
    return build_snapshot(manifest["version"], chunks, vectors, faiss_index)  # None => rebuilt from vectors


# ------------------------------------------------------------------ the store
class IndexStore:
    def __init__(self, index_dir: Path = None):
        self.dir = Path(index_dir or config.INDEX_DIR)
        self.versions_dir = self.dir / "versions"
        self._lock = threading.Lock()  # one lock around ALL index updates
        self._snapshot = build_snapshot("v000000", [], np.zeros((0, EMBEDDING_DIM), dtype="float32"))

    # -- reading (lock-free: snapshots are immutable)
    @property
    def snapshot(self) -> IndexSnapshot:
        return self._snapshot

    # -- startup
    def load(self) -> IndexSnapshot:
        """Load CURRENT, falling back to the backup version if it is damaged."""
        with self._lock:
            self.versions_dir.mkdir(parents=True, exist_ok=True)
            for leftover in self.dir.glob("building-*"):  # crashed earlier update
                shutil.rmtree(leftover, ignore_errors=True)
            names = sorted((p.name for p in self.versions_dir.iterdir() if _VERSION_RE.fullmatch(p.name)), reverse=True)
            if not names:
                return self._snapshot
            current = self._read_current()
            order = ([current] if current in names else []) + [n for n in names if n != current]
            errors = []
            for name in order:
                try:
                    snap = _read_version(self.versions_dir / name)
                    problems = verify_snapshot(snap)
                    if problems:
                        raise ValueError("; ".join(problems))
                except Exception as exc:
                    errors.append(f"{name}: {type(exc).__name__}: {exc}")
                    logger.error("Index version %s unusable (%s)", name, exc)
                    continue
                if name != current:
                    logger.warning("Fell back to index version %s", name)
                    self._write_current(name)
                self._snapshot = snap
                return snap
            raise IndexLoadError("No usable index version. Rebuild required. " + " | ".join(errors))

    # -- updates (all go through _commit)
    def upsert_document(self, doc_id: str, chunks: list, vectors: np.ndarray) -> IndexSnapshot:
        """Add a document, or atomically REPLACE it: old chunks vanish and new ones appear in one switch."""
        if not chunks:
            raise IndexUpdateError("Refusing to index a document with no chunks.")
        if any(c.doc_id != doc_id for c in chunks):
            raise IndexUpdateError("All chunks must belong to doc_id.")
        vectors = np.ascontiguousarray(vectors, dtype="float32")
        with self._lock:
            base = self._snapshot
            keep = [i for i, c in enumerate(base.chunks) if c.doc_id != doc_id]
            new_chunks = [base.chunks[i] for i in keep] + list(chunks)
            new_vectors = np.vstack([base.vectors[keep], vectors]) if keep else vectors
            return self._commit(new_chunks, new_vectors)

    def remove_document(self, doc_id: str) -> int:
        """Remove all chunks of doc_id from metadata, FAISS and BM25. Returns number removed."""
        with self._lock:
            base = self._snapshot
            keep = [i for i, c in enumerate(base.chunks) if c.doc_id != doc_id]
            removed = len(base.chunks) - len(keep)
            if removed == 0:
                return 0
            new_chunks = [base.chunks[i] for i in keep]
            new_vectors = base.vectors[keep] if keep else np.zeros((0, EMBEDDING_DIM), dtype="float32")
            self._commit(new_chunks, new_vectors, must_not_contain=doc_id)
            return removed

    def rebuild_from_chunks(self, chunks: list, embedder) -> IndexSnapshot:
        """Recovery path: re-embed the given chunks and replace the whole index
        (e.g. after the embedding model changed, or after re-extracting documents from S3)."""
        vectors = embedder.embed_texts([c.text for c in chunks])
        with self._lock:
            return self._commit(list(chunks), vectors)

    # -- internals
    def _commit(self, chunks: list, vectors: np.ndarray, must_not_contain: str = None) -> IndexSnapshot:
        """Caller holds the lock. Build -> verify -> write temp -> reload+verify -> rename -> switch CURRENT."""
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        previous = self._snapshot.version
        new_version = self._next_version()
        build_dir = self.dir / f"building-{uuid.uuid4().hex}"
        final_dir = self.versions_dir / new_version
        try:
            snap = build_snapshot(new_version, chunks, vectors)
            if must_not_contain:
                assert_document_absent(snap, must_not_contain)
            problems = verify_snapshot(snap)
            if problems:
                raise IndexUpdateError("; ".join(problems))
            _write_version_files(build_dir, snap)
            reloaded = _read_version(build_dir)  # prove what is on disk is loadable and consistent
            if verify_snapshot(reloaded) or [c.chunk_id for c in reloaded.chunks] != [c.chunk_id for c in snap.chunks]:
                raise IndexUpdateError("Written index failed verification.")
            os.replace(build_dir, final_dir)
            try:
                self._write_current(new_version)
            except Exception:
                shutil.rmtree(final_dir, ignore_errors=True)
                raise
        except Exception as exc:
            shutil.rmtree(build_dir, ignore_errors=True)
            logger.error("Index update failed, keeping %s: %s", previous, type(exc).__name__)
            if isinstance(exc, IndexUpdateError):
                raise
            raise IndexUpdateError(f"Index update failed ({type(exc).__name__}); the previous index is still active.") from exc
        self._snapshot = snap  # atomic reference swap: readers move to the new version
        self._prune({new_version, previous})
        return snap

    def _next_version(self) -> str:
        nums = [int(m.group(1)) for p in self.versions_dir.iterdir() if (m := _VERSION_RE.fullmatch(p.name))]
        return f"v{(max(nums) if nums else 0) + 1:06d}"

    def _read_current(self) -> Optional[str]:
        try:
            return (self.dir / "CURRENT").read_text(encoding="utf-8").strip()
        except OSError:
            return None

    def _write_current(self, version: str) -> None:
        tmp = self.dir / f"CURRENT.{uuid.uuid4().hex}.tmp"
        tmp.write_text(version, encoding="utf-8")
        os.replace(tmp, self.dir / "CURRENT")  # atomic pointer switch

    def _prune(self, keep: set) -> None:
        for p in self.versions_dir.iterdir():
            if _VERSION_RE.fullmatch(p.name) and p.name not in keep:
                shutil.rmtree(p, ignore_errors=True)
