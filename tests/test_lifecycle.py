"""Phase 9 (scoped): duplicate detection and deletion."""
import threading
from types import SimpleNamespace

import numpy as np
import pytest

import document_manager as dm
from auth import NotAuthenticatedError
from document_manager import (DeleteIncompleteError, DocumentBusyError, DocumentNotFoundError,
                              DuplicateDocumentError)
from document_processor import ExtractionError
from index_store import IndexUpdateError, verify_snapshot
from rag_pipeline import REFUSAL, RagPipeline
from retrieval import Retriever
from s3_storage import S3StorageError
from tests.conftest import BUCKET, TEST_PASSWORD
from tests.test_document_processor import digital_pdf

ZEPHYR = digital_pdf(("Zephyr archive keeps quarterly exports.", "Zephyr reports are stored for 90 days."))
OTHER = digital_pdf(("Quasar module handles nightly synchronisation jobs.",))


def s3_keys(client):
    return [o["Key"] for o in client.list_objects_v2(Bucket=BUCKET).get("Contents", [])]


def login(mgr):
    return mgr.auth.login(TEST_PASSWORD)


def upload(mgr, name="z.pdf", data=ZEPHYR):
    rec = mgr.submit_upload(login(mgr), name, data)
    mgr.wait_until_idle()
    return mgr.registry.get(rec["doc_id"])


def assert_consistent(mgr):
    """The invariant: searchable documents == registry documents marked indexed; all index parts agree."""
    snap = mgr.store.snapshot
    assert verify_snapshot(snap) == []
    assert snap.doc_ids() == {r["doc_id"] for r in mgr.registry.all() if r["status"] == dm.INDEXED}


def retrievable_doc_ids(mgr, embedder, query):
    """Doc ids that EITHER retriever can return for this query (checked at the raw index level too)."""
    from textutil import tokenize
    snap = mgr.store.snapshot
    found = {c.chunk.doc_id for c in Retriever(mgr.store, embedder).search(query)}
    if snap.chunk_count:
        scores = snap.bm25.get_scores(tokenize(query))
        found |= {snap.chunks[i].doc_id for i in np.argsort(-scores)[:snap.chunk_count] if scores[i] > 0}
        _, ids = snap.faiss_index.search(embedder.embed_query(query).reshape(1, -1), snap.chunk_count)
        found |= {snap.chunks[i].doc_id for i in ids[0] if i >= 0}
    return found


# ================================================================== duplicate detection
def test_identical_file_is_rejected_and_nothing_is_stored(make_manager, fake_s3):
    mgr = make_manager()
    first = upload(mgr)
    with pytest.raises(DuplicateDocumentError, match="already uploaded as 'z.pdf'"):
        mgr.submit_upload(login(mgr), "z.pdf", ZEPHYR)
    assert s3_keys(fake_s3) == [first["s3_key"]]            # no second S3 object
    assert len(mgr.registry.all()) == 7                      # 6 seed + 1 upload, no new record
    assert_consistent(mgr)


def test_duplicate_is_detected_by_content_not_filename(make_manager):
    mgr = make_manager()
    upload(mgr, "original.pdf")
    with pytest.raises(DuplicateDocumentError):
        mgr.submit_upload(login(mgr), "renamed copy.pdf", ZEPHYR)


def test_same_filename_but_different_content_is_accepted_with_unique_keys(make_manager, fake_s3):
    mgr = make_manager()
    a, b = upload(mgr, "report.pdf", ZEPHYR), upload(mgr, "report.pdf", OTHER)
    assert a["doc_id"] != b["doc_id"] and a["s3_key"] != b["s3_key"] and len(s3_keys(fake_s3)) == 2
    assert a["status"] == b["status"] == dm.INDEXED
    assert_consistent(mgr)


def test_duplicate_of_a_failed_document_is_still_rejected(make_manager, monkeypatch):
    mgr = make_manager()
    monkeypatch.setattr(dm, "extract_text", lambda *a, **k: (_ for _ in ()).throw(ExtractionError("No readable text.")))
    failed = upload(mgr)
    assert failed["status"] == dm.FAILED
    with pytest.raises(DuplicateDocumentError, match="failed"):
        mgr.submit_upload(login(mgr), "again.pdf", ZEPHYR)


def test_duplicate_detection_survives_a_restart(make_manager):
    mgr = make_manager()
    upload(mgr)
    again = make_manager()
    with pytest.raises(DuplicateDocumentError):
        again.submit_upload(login(again), "z.pdf", ZEPHYR)


def test_simultaneous_identical_uploads_accept_exactly_one(make_manager, fake_s3):
    mgr = make_manager()
    token = login(mgr)
    results, lock = [], threading.Lock()

    def go():
        try:
            mgr.submit_upload(token, "z.pdf", ZEPHYR)
            outcome = "accepted"
        except DuplicateDocumentError:
            outcome = "duplicate"
        with lock:
            results.append(outcome)

    threads = [threading.Thread(target=go) for _ in range(6)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    mgr.wait_until_idle()
    assert sorted(results) == ["accepted"] + ["duplicate"] * 5
    assert len(s3_keys(fake_s3)) == 1
    assert_consistent(mgr)


def test_duplicate_check_needs_admin_first(make_manager):
    mgr = make_manager()
    upload(mgr)
    with pytest.raises(NotAuthenticatedError):  # must not reveal whether a file already exists
        mgr.submit_upload(None, "z.pdf", ZEPHYR)


def test_same_file_can_be_uploaded_again_after_it_was_deleted(make_manager):
    mgr = make_manager()
    first = upload(mgr)
    mgr.delete_document(login(mgr), first["doc_id"])
    second = upload(mgr)
    assert second["status"] == dm.INDEXED and second["doc_id"] != first["doc_id"]


# ================================================================== deletion
def test_delete_removes_s3_registry_and_every_chunk_from_both_retrievers(make_manager, fake_s3, embedder):
    mgr = make_manager()
    doc = upload(mgr)
    query = "Zephyr archive quarterly exports reports stored 90 days"
    assert doc["doc_id"] in retrievable_doc_ids(mgr, embedder, query)         # sanity: it WAS retrievable
    before = mgr.store.snapshot.chunk_count
    mine = sum(1 for c in mgr.store.snapshot.chunks if c.doc_id == doc["doc_id"])
    assert mine >= 1

    mgr.delete_document(login(mgr), doc["doc_id"])

    snap = mgr.store.snapshot
    assert mgr.registry.get(doc["doc_id"]) is None
    assert s3_keys(fake_s3) == []
    assert not any(c.doc_id == doc["doc_id"] for c in snap.chunks)             # metadata
    assert snap.chunk_count == before - mine
    assert snap.faiss_index.ntotal == snap.chunk_count                         # FAISS: exactly its vectors are gone
    assert snap.bm25.corpus_size == snap.chunk_count                           # BM25: exactly its chunks are gone
    assert doc["doc_id"] not in retrievable_doc_ids(mgr, embedder, query)      # neither retriever can return it
    assert_consistent(mgr)


def test_delete_is_persisted_across_restart(make_manager, fake_s3, embedder):
    mgr = make_manager()
    doc = upload(mgr)
    mgr.delete_document(login(mgr), doc["doc_id"])
    again = make_manager()
    assert again.registry.get(doc["doc_id"]) is None and doc["doc_id"] not in again.store.snapshot.doc_ids()
    assert doc["doc_id"] not in retrievable_doc_ids(again, embedder, "Zephyr archive quarterly exports")
    assert_consistent(again)


def test_deleted_seed_article_is_not_answerable_and_not_recreated_on_restart(make_manager, embedder):
    llm_calls = []
    mgr = make_manager()
    token = login(mgr)
    q3 = "What's the maximum number of rows I can export to CSV?"

    def pipeline(m):
        return RagPipeline(Retriever(m.store, embedder), SimpleNamespace(chat=lambda s, u: llm_calls.append(1) or "x"))

    mgr.delete_document(token, "D-03")
    assert pipeline(mgr).answer(q3).answer == REFUSAL and llm_calls == []        # evidence is gone => exact refusal
    again = make_manager()                                                        # restart: seeding must NOT bring it back
    assert "D-03" not in again.store.snapshot.doc_ids() and again.registry.get("D-03") is None
    assert pipeline(again).answer(q3).answer == REFUSAL
    assert_consistent(again)


def test_deleting_one_document_leaves_the_others_answerable(make_manager, embedder):
    mgr = make_manager()
    mgr.delete_document(login(mgr), "D-03")
    llm = SimpleNamespace(chat=lambda s, u: "Password reset links expire after 15 minutes. [1]")
    r = RagPipeline(Retriever(mgr.store, embedder), llm).answer("How long do I have to use a password reset link before it expires?")
    assert r.mode == "llm" and r.citations[0].doc_id == "D-01"


def test_delete_requires_admin_and_changes_nothing_otherwise(make_manager, fake_s3):
    mgr = make_manager()
    doc = upload(mgr)
    for token in (None, "", "forged"):
        with pytest.raises(NotAuthenticatedError):
            mgr.delete_document(token, doc["doc_id"])
    assert mgr.registry.get(doc["doc_id"]) and s3_keys(fake_s3) == [doc["s3_key"]]
    assert doc["doc_id"] in mgr.store.snapshot.doc_ids()


def test_deleting_unknown_or_already_deleted_document(make_manager):
    mgr = make_manager()
    token = login(mgr)
    with pytest.raises(DocumentNotFoundError):
        mgr.delete_document(token, "does-not-exist")
    doc = upload(mgr)
    mgr.delete_document(token, doc["doc_id"])
    with pytest.raises(DocumentNotFoundError):
        mgr.delete_document(token, doc["doc_id"])


def test_cannot_delete_while_processing_then_can_afterwards(make_manager):
    mgr = make_manager()
    token = login(mgr)
    started, gate = threading.Event(), threading.Event()
    real = mgr.embedder

    def slow(texts):
        started.set()
        gate.wait(15)
        return real.embed_texts(texts)

    mgr.embedder = SimpleNamespace(embed_texts=slow)
    rec = mgr.submit_upload(token, "z.pdf", ZEPHYR)
    assert started.wait(15)
    with pytest.raises(DocumentBusyError):
        mgr.delete_document(token, rec["doc_id"])
    gate.set()
    mgr.wait_until_idle()
    mgr.delete_document(token, rec["doc_id"])
    assert mgr.registry.get(rec["doc_id"]) is None
    assert_consistent(mgr)


def test_failed_document_can_be_deleted_and_its_original_is_removed_from_s3(make_manager, fake_s3, monkeypatch):
    mgr = make_manager()
    monkeypatch.setattr(dm, "extract_text", lambda *a, **k: (_ for _ in ()).throw(ExtractionError("No readable text.")))
    failed = upload(mgr)
    assert failed["status"] == dm.FAILED and s3_keys(fake_s3) == [failed["s3_key"]]
    mgr.delete_document(login(mgr), failed["doc_id"])
    assert mgr.registry.get(failed["doc_id"]) is None and s3_keys(fake_s3) == []
    assert_consistent(mgr)


def test_s3_failure_after_index_removal_is_safe_and_can_be_finished(make_manager, fake_s3, embedder, monkeypatch):
    mgr = make_manager()
    doc = upload(mgr)
    token = login(mgr)
    real_delete = mgr.storage.delete
    monkeypatch.setattr(mgr.storage, "delete", lambda key: (_ for _ in ()).throw(S3StorageError("S3 delete failed (503).")))

    with pytest.raises(DeleteIncompleteError, match="no longer searchable"):
        mgr.delete_document(token, doc["doc_id"])
    rec = mgr.registry.get(doc["doc_id"])
    assert rec["status"] == dm.DELETE_FAILED and "503" in rec["error"]
    assert doc["doc_id"] not in retrievable_doc_ids(mgr, embedder, "Zephyr archive quarterly exports")  # already unsearchable
    assert s3_keys(fake_s3) == [doc["s3_key"]]                                                         # file still stored
    assert_consistent(mgr)

    monkeypatch.setattr(mgr.storage, "delete", real_delete)                                            # S3 recovers
    mgr.delete_document(token, doc["doc_id"])
    assert mgr.registry.get(doc["doc_id"]) is None and s3_keys(fake_s3) == []


def test_document_pending_deletion_does_not_block_reuploading_the_same_file(make_manager, monkeypatch):
    mgr = make_manager()
    doc = upload(mgr)
    monkeypatch.setattr(mgr.storage, "delete", lambda key: (_ for _ in ()).throw(S3StorageError("S3 delete failed (503).")))
    with pytest.raises(DeleteIncompleteError):
        mgr.delete_document(login(mgr), doc["doc_id"])
    again = upload(mgr)
    assert again["status"] == dm.INDEXED


def test_index_removal_failure_changes_nothing(make_manager, fake_s3, embedder, monkeypatch):
    mgr = make_manager()
    doc = upload(mgr)
    monkeypatch.setattr(mgr.store, "remove_document", lambda doc_id: (_ for _ in ()).throw(IndexUpdateError("disk full")))
    with pytest.raises(IndexUpdateError):
        mgr.delete_document(login(mgr), doc["doc_id"])
    assert mgr.registry.get(doc["doc_id"])["status"] == dm.INDEXED                    # still a normal document
    assert s3_keys(fake_s3) == [doc["s3_key"]]                                         # original untouched
    assert doc["doc_id"] in retrievable_doc_ids(mgr, embedder, "Zephyr archive quarterly exports")
    assert_consistent(mgr)


def test_uploads_and_deletes_running_together_stay_consistent(make_manager, fake_s3):
    mgr = make_manager()
    token = login(mgr)
    victim = upload(mgr, "victim.pdf", ZEPHYR)
    errors = []

    def up(i):
        try:
            mgr.submit_upload(token, f"new{i}.pdf", digital_pdf((f"Document {i} explains feature bravo{i} in detail.",)))
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    def delete(doc_id):
        try:
            mgr.delete_document(token, doc_id)
        except DocumentBusyError:
            pass
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=up, args=(i,)) for i in range(4)] + \
              [threading.Thread(target=delete, args=(victim["doc_id"],)), threading.Thread(target=delete, args=("D-02",))]
    [t.start() for t in threads]
    [t.join() for t in threads]
    mgr.wait_until_idle()
    assert errors == []
    assert mgr.registry.get(victim["doc_id"]) is None and mgr.registry.get("D-02") is None
    assert len(s3_keys(fake_s3)) == 4
    assert_consistent(mgr)
