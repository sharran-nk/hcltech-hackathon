"""Document lifecycle (core): upload -> S3 -> queue -> extract/chunk/embed/index, with persistent status.

Phase 8 scope: submit_upload, persistent registry, single-worker queue, seed loading, listing.
Phase 9 adds: duplicate detection, retry, deletion, replacement, failure-recovery tests.

Rules enforced here
- Every admin action starts with auth.require_admin(token): the check is server-side, not in the UI.
- Validation happens BEFORE anything is stored. The original goes to S3 (never kept on EC2).
- Exactly ONE document is processed at a time (single worker thread + queue).
- A document is searchable only after IndexStore.upsert_document succeeds (atomic). If anything
  fails the indexes are untouched, the original stays in S3, and status becomes "failed" with a reason.
- Status lives in a JSON registry (atomic writes + one backup copy) so it survives restarts.
"""
import json
import logging
import os
import queue
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import chunking
import config
from auth import AuthManager
from document_processor import (DocumentProcessingError, ExtractionError, content_hash, extract_text,
                                temp_workspace, validate_upload)
from embeddings import EmbeddingError
from index_store import IndexStore, IndexUpdateError, assert_document_absent
from s3_storage import S3Storage, S3StorageError, new_document_id, sanitize_filename

logger = logging.getLogger(__name__)

QUEUED, PROCESSING, INDEXED, FAILED, DELETE_FAILED = "queued", "processing", "indexed", "failed", "delete_failed"
ACTIVE_STATUSES = (QUEUED, PROCESSING)


class DocumentManagerError(Exception):
    """A refused or incomplete admin action. str(exc) is safe to show to the admin."""


class DuplicateDocumentError(DocumentManagerError):
    pass


class DocumentNotFoundError(DocumentManagerError):
    pass


class DocumentBusyError(DocumentManagerError):
    pass


class DeleteIncompleteError(DocumentManagerError):
    """Removed from search, but the stored original could not be deleted yet."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ------------------------------------------------------------------ persistent registry
class DocumentRegistry:
    """documents.json: one record per document. Atomic replace + a .bak copy of the previous version."""

    def __init__(self, path: Path = None):
        self.path = Path(path or config.METADATA_DB_PATH)
        self._lock = threading.RLock()
        self._docs = {}
        self._meta = {}   # small persisted flags, e.g. {"seeded": true}

    def load(self) -> None:
        with self._lock:
            for candidate in (self.path, self.path.with_suffix(".json.bak")):
                try:
                    data = json.loads(candidate.read_text(encoding="utf-8"))
                    self._docs = {d["doc_id"]: d for d in data["documents"]}
                    self._meta = dict(data.get("meta", {}))
                    if candidate != self.path:
                        logger.warning("Registry restored from backup %s", candidate.name)
                    return
                except FileNotFoundError:
                    continue
                except (ValueError, KeyError, OSError):
                    logger.error("Registry file %s unreadable", candidate.name)
            self._docs, self._meta = {}, {}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({"documents": list(self._docs.values()), "meta": self._meta},
                                  indent=2, ensure_ascii=False), encoding="utf-8")
        if self.path.exists():
            shutil.copyfile(self.path, self.path.with_suffix(".json.bak"))
        os.replace(tmp, self.path)

    def put(self, record: dict) -> None:
        with self._lock:
            self._docs[record["doc_id"]] = dict(record)
            self._save()

    def update(self, doc_id: str, **fields) -> dict:
        with self._lock:
            self._docs[doc_id].update(fields, updated_at=_now())
            self._save()
            return dict(self._docs[doc_id])

    def remove(self, doc_id: str) -> None:
        with self._lock:
            self._docs.pop(doc_id, None)
            self._save()

    def get_meta(self, key: str, default=None):
        with self._lock:
            return self._meta.get(key, default)

    def set_meta(self, key: str, value) -> None:
        with self._lock:
            self._meta[key] = value
            self._save()

    def get(self, doc_id: str) -> Optional[dict]:
        with self._lock:
            d = self._docs.get(doc_id)
            return dict(d) if d else None

    def all(self) -> list:
        with self._lock:
            return sorted((dict(d) for d in self._docs.values()), key=lambda d: d["created_at"], reverse=True)


# ------------------------------------------------------------------ manager
class DocumentManager:
    def __init__(self, storage: S3Storage, store: IndexStore, embedder, auth: AuthManager,
                 registry: DocumentRegistry = None):
        self.storage, self.store, self.embedder, self.auth = storage, store, embedder, auth
        self.registry = registry or DocumentRegistry()
        self._queue: "queue.Queue[Optional[str]]" = queue.Queue()
        self._worker: Optional[threading.Thread] = None
        self._admin_lock = threading.RLock()  # serialises duplicate-check+register, and delete, so they cannot race

    # -- startup
    def start(self) -> None:
        self.store.load()
        self.registry.load()
        for rec in self.registry.all():  # work interrupted by a crash/restart cannot be trusted
            if rec["status"] in ACTIVE_STATUSES:
                self.registry.update(rec["doc_id"], status=FAILED, error="Processing was interrupted by a restart. Retry is needed.")
        self._seed()
        self._worker = threading.Thread(target=self._worker_loop, name="doc-worker", daemon=True)
        self._worker.start()

    def stop(self) -> None:
        if self._worker:
            self._queue.put(None)
            self._worker.join(timeout=10)

    def _seed(self) -> None:
        """Index the six docs.json articles at FIRST startup only. A persisted flag stops them being
        re-created after an admin deletes one."""
        if self.registry.get_meta("seeded"):
            return
        indexed = self.store.snapshot.doc_ids()
        for doc in chunking.load_seed_docs():
            doc_id = doc["doc_id"]
            existing = self.registry.get(doc_id)
            if doc_id not in indexed:
                chunks = chunking.chunk_seed_doc(doc)
                self.store.upsert_document(doc_id, chunks, self.embedder.embed_texts([c.text for c in chunks]))
            if not existing or existing["status"] != INDEXED or doc_id not in indexed:
                chunks = [c for c in self.store.snapshot.chunks if c.doc_id == doc_id]
                self.registry.put({
                    "doc_id": doc_id, "filename": doc["title"], "kind": "seed", "size": len(doc["content"].encode("utf-8")),
                    "sha256": content_hash(doc["content"].encode("utf-8")), "s3_key": None, "source": "seed",
                    "status": INDEXED, "error": "", "chunk_count": len(chunks), "page_count": 1, "warnings": [],
                    "created_at": _now(), "updated_at": _now(),
                })
        self.registry.set_meta("seeded", True)

    # -- admin actions (all start with require_admin)
    def submit_upload(self, token: str, filename: str, data: bytes) -> dict:
        """Validate -> reject duplicates -> store original in S3 -> register -> queue. Returns the registry record.
        Raises NotAuthenticatedError, ValidationError, DuplicateDocumentError or S3StorageError;
        in those cases nothing is stored and nothing is registered."""
        self.auth.require_admin(token)
        kind = validate_upload(filename, data)
        safe_name = sanitize_filename(filename)
        digest = content_hash(data)
        with self._admin_lock:  # check + register are one step, so two identical simultaneous uploads cannot both pass
            existing = self._find_duplicate(digest)
            if existing:
                raise DuplicateDocumentError(
                    f"This file was already uploaded as '{existing['filename']}' (status: {existing['status']}). "
                    "Nothing was stored.")
            doc_id = new_document_id()
            key = self.storage.make_key(doc_id, safe_name)
            self.storage.upload_bytes(data, key)  # raises S3StorageError -> nothing registered
            record = {
                "doc_id": doc_id, "filename": safe_name, "kind": kind, "size": len(data), "sha256": digest,
                "s3_key": key, "source": "upload", "status": QUEUED, "error": "", "chunk_count": 0, "page_count": 0,
                "warnings": [], "created_at": _now(), "updated_at": _now(),
            }
            self.registry.put(record)
            self._queue.put(doc_id)
        return record

    def _find_duplicate(self, digest: str) -> Optional[dict]:
        """Same content hash = same file, whatever it is called. Documents being deleted do not count."""
        for rec in self.registry.all():
            if rec["sha256"] == digest and rec["status"] != DELETE_FAILED:
                return rec
        return None

    def delete_document(self, token: str, doc_id: str) -> None:
        """Remove a document everywhere. Order matters:
          1. remove its chunks from metadata + FAISS + BM25 (one atomic, verified index update)
          2. delete the original from S3
          3. delete the registry record
        After step 1 the content can no longer be retrieved, even if step 2 fails (then the record is
        kept as 'delete_failed' and deleting again finishes the job). If step 1 fails, nothing changed."""
        self.auth.require_admin(token)
        with self._admin_lock:
            rec = self.registry.get(doc_id)
            if rec is None:
                raise DocumentNotFoundError("That document no longer exists.")
            if rec["status"] in ACTIVE_STATUSES:
                raise DocumentBusyError("This document is still being processed. Try again when it has finished.")

            self.store.remove_document(doc_id)                   # raises IndexUpdateError -> nothing else touched
            assert_document_absent(self.store.snapshot, doc_id)  # no chunk left in metadata, FAISS or BM25
            if rec["s3_key"]:                                    # seed articles have no S3 object
                try:
                    self.storage.delete(rec["s3_key"])
                except S3StorageError as exc:
                    self.registry.update(doc_id, status=DELETE_FAILED, error=str(exc), chunk_count=0)
                    raise DeleteIncompleteError(
                        "The document is no longer searchable, but its stored file could not be deleted yet. "
                        "Delete it again to finish.") from exc
            self.registry.remove(doc_id)
            logger.info("Deleted document %s", doc_id)

    def list_documents(self, token: str, refresh: bool = True) -> list:
        self.auth.require_admin(token, refresh=refresh)
        return self.registry.all()

    # -- background processing: ONE document at a time
    def wait_until_idle(self) -> None:
        self._queue.join()

    def _worker_loop(self) -> None:
        while True:
            doc_id = self._queue.get()
            try:
                if doc_id is None:
                    return
                self._process(doc_id)
            except Exception:  # never let the worker thread die
                logger.exception("Unexpected worker error")
            finally:
                self._queue.task_done()

    def _process(self, doc_id: str) -> None:
        rec = self.registry.get(doc_id)
        self.registry.update(doc_id, status=PROCESSING, error="")
        try:
            with temp_workspace() as ws:  # temp files are removed even if we raise
                local = self.storage.download_to_path(rec["s3_key"], ws / f"input{Path(rec['filename']).suffix.lower()}")
                extraction = extract_text(local, rec["kind"])
                chunks = chunking.chunk_extraction(doc_id, rec["filename"], extraction)
                if not chunks:
                    raise ExtractionError("The document produced no indexable content.")
                vectors = self.embedder.embed_texts([c.text for c in chunks])
            self.store.upsert_document(doc_id, chunks, vectors)  # atomic: searchable only after this succeeds
        except (DocumentProcessingError, S3StorageError, EmbeddingError, IndexUpdateError) as exc:
            logger.warning("Processing failed for %s: %s", doc_id, type(exc).__name__)
            self.registry.update(doc_id, status=FAILED, error=str(exc))
            return
        except Exception:
            logger.exception("Unexpected processing failure for %s", doc_id)
            self.registry.update(doc_id, status=FAILED, error="Unexpected error while processing (see server log).")
            return
        self.registry.update(doc_id, status=INDEXED, error="", chunk_count=len(chunks),
                             page_count=extraction.page_count, warnings=extraction.warnings)
