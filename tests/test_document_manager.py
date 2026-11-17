import json
import threading
from types import SimpleNamespace

import pytest
from PIL import Image

import document_manager as dm
from auth import NotAuthenticatedError
from document_processor import ExtractionError, ValidationError
from index_store import verify_snapshot
from s3_storage import S3StorageError
from tests.conftest import BUCKET, TEST_PASSWORD, make_auth
from tests.test_document_processor import SENTENCE, digital_pdf, img_bytes, text_image

ZEPHYR = digital_pdf(("Zephyr archive keeps quarterly exports.", "Zephyr reports are stored for 90 days."))


def s3_keys(client):
    return [o["Key"] for o in client.list_objects_v2(Bucket=BUCKET).get("Contents", [])]


def login(mgr):
    return mgr.auth.login(TEST_PASSWORD)


# ------------------------------------------------------------------ startup / seed
def test_seed_articles_indexed_and_registered_at_first_startup(make_manager):
    mgr = make_manager()
    assert mgr.store.snapshot.chunk_count == 13 and len(mgr.store.snapshot.doc_ids()) == 6
    docs = mgr.registry.all()
    assert {d["doc_id"] for d in docs} == {f"D-0{i}" for i in range(1, 7)}
    assert all(d["status"] == dm.INDEXED and d["source"] == "seed" for d in docs)


def test_seeding_is_idempotent_across_restarts(make_manager, tmp_path):
    first = make_manager()
    version = first.store.snapshot.version
    second = make_manager()  # same index dir + registry file
    assert second.store.snapshot.version == version and len(second.registry.all()) == 6


# ------------------------------------------------------------------ authorisation
def test_upload_requires_admin_and_stores_nothing(make_manager, fake_s3):
    mgr = make_manager()
    for token in (None, "", "forged-token"):
        with pytest.raises(NotAuthenticatedError):
            mgr.submit_upload(token, "a.pdf", ZEPHYR)
    assert s3_keys(fake_s3) == [] and len(mgr.registry.all()) == 6


def test_listing_requires_admin(make_manager):
    mgr = make_manager()
    with pytest.raises(NotAuthenticatedError):
        mgr.list_documents(None)
    assert len(mgr.list_documents(login(mgr))) == 6


def test_expired_session_cannot_upload(make_manager, fake_s3):
    class Clock:
        t = 0.0
        def __call__(self): return self.t
    clock = Clock()
    mgr = make_manager(clock=clock)
    token = login(mgr)
    clock.t += 31 * 60
    with pytest.raises(NotAuthenticatedError):
        mgr.submit_upload(token, "a.pdf", ZEPHYR)
    assert s3_keys(fake_s3) == []


# ------------------------------------------------------------------ validation / S3 failure
@pytest.mark.parametrize("name,data", [("virus.exe", b"MZ\x90"), ("empty.pdf", b""), ("fake.pdf", b"MZ not a pdf"),
                                        ("corrupt.pdf", b"%PDF-1.4 garbage")])
def test_invalid_upload_rejected_before_anything_is_stored(make_manager, fake_s3, name, data):
    mgr = make_manager()
    token = login(mgr)
    with pytest.raises(ValidationError):
        mgr.submit_upload(token, name, data)
    assert s3_keys(fake_s3) == [] and len(mgr.registry.all()) == 6


def test_s3_upload_failure_registers_nothing(make_manager, fake_s3, monkeypatch):
    mgr = make_manager()
    token = login(mgr)
    monkeypatch.setattr(mgr.storage, "upload_bytes", lambda *a, **k: (_ for _ in ()).throw(S3StorageError("S3 upload failed (503).")))
    with pytest.raises(S3StorageError):
        mgr.submit_upload(token, "a.pdf", ZEPHYR)
    assert len(mgr.registry.all()) == 6 and mgr.store.snapshot.chunk_count == 13


# ------------------------------------------------------------------ happy path
def test_pdf_upload_flows_to_searchable(make_manager, fake_s3, tmp_path):
    mgr = make_manager()
    token = login(mgr)
    rec = mgr.submit_upload(token, "../../Zephyr guide (v1).pdf", ZEPHYR)
    assert rec["status"] == dm.QUEUED and rec["filename"] == "Zephyr_guide_v1_.pdf"  # path stripped, unsafe chars replaced
    mgr.wait_until_idle()
    final = mgr.registry.get(rec["doc_id"])
    assert final["status"] == dm.INDEXED and final["chunk_count"] >= 1 and final["page_count"] == 2 and final["error"] == ""
    assert s3_keys(fake_s3) == [final["s3_key"]]
    assert fake_s3.get_object(Bucket=BUCKET, Key=final["s3_key"])["Body"].read() == ZEPHYR   # original is in S3, byte for byte
    assert any("Zephyr" in c.text and c.doc_id == rec["doc_id"] for c in mgr.store.snapshot.chunks)
    assert verify_snapshot(mgr.store.snapshot) == []
    assert not list((tmp_path / "tmp").iterdir())                                           # temp files cleaned up


def test_image_upload_is_ocrd_and_indexed(make_manager):
    mgr = make_manager()
    rec = mgr.submit_upload(login(mgr), "photo.png", img_bytes(text_image(SENTENCE)))
    mgr.wait_until_idle()
    assert mgr.registry.get(rec["doc_id"])["status"] == dm.INDEXED
    assert any("10,000" in c.text for c in mgr.store.snapshot.chunks if c.doc_id == rec["doc_id"])


def test_document_is_not_searchable_until_indexing_finishes(make_manager):
    mgr = make_manager()
    token = login(mgr)
    started, gate = threading.Event(), threading.Event()
    real = mgr.embedder

    def slow_embed(texts):
        started.set()
        gate.wait(15)
        return real.embed_texts(texts)

    mgr.embedder = SimpleNamespace(embed_texts=slow_embed)
    rec = mgr.submit_upload(token, "z.pdf", ZEPHYR)
    assert started.wait(15)
    assert mgr.registry.get(rec["doc_id"])["status"] == dm.PROCESSING
    assert rec["doc_id"] not in mgr.store.snapshot.doc_ids()          # NOT searchable yet
    gate.set()
    mgr.wait_until_idle()
    assert rec["doc_id"] in mgr.store.snapshot.doc_ids() and mgr.registry.get(rec["doc_id"])["status"] == dm.INDEXED


# ------------------------------------------------------------------ failures after upload
def test_processing_failure_keeps_original_marks_failed_and_leaves_index_unchanged(make_manager, fake_s3, monkeypatch):
    mgr = make_manager()
    token = login(mgr)
    version = mgr.store.snapshot.version
    monkeypatch.setattr(dm, "extract_text", lambda *a, **k: (_ for _ in ()).throw(ExtractionError("No readable text could be extracted.")))
    rec = mgr.submit_upload(token, "z.pdf", ZEPHYR)
    mgr.wait_until_idle()
    final = mgr.registry.get(rec["doc_id"])
    assert final["status"] == dm.FAILED and "No readable text" in final["error"]
    assert final["s3_key"] in s3_keys(fake_s3)                        # original retained so the admin can retry
    assert mgr.store.snapshot.version == version and rec["doc_id"] not in mgr.store.snapshot.doc_ids()


def test_embedding_failure_marks_failed_and_worker_survives(make_manager):
    mgr = make_manager()
    token = login(mgr)
    real = mgr.embedder
    mgr.embedder = SimpleNamespace(embed_texts=lambda t: (_ for _ in ()).throw(dm.EmbeddingError("Embedding generation failed: MemoryError")))
    bad = mgr.submit_upload(token, "a.pdf", ZEPHYR)
    mgr.wait_until_idle()
    assert mgr.registry.get(bad["doc_id"])["status"] == dm.FAILED
    mgr.embedder = real
    good = mgr.submit_upload(token, "b.pdf", digital_pdf(("Another document about quarterly billing archives.",)))
    mgr.wait_until_idle()
    assert mgr.registry.get(good["doc_id"])["status"] == dm.INDEXED   # the worker thread did not die


def test_unexpected_error_gives_generic_message_without_internals(make_manager, monkeypatch):
    mgr = make_manager()
    monkeypatch.setattr(dm, "extract_text", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("secret internal path C:\\x")))
    rec = mgr.submit_upload(login(mgr), "a.pdf", ZEPHYR)
    mgr.wait_until_idle()
    err = mgr.registry.get(rec["doc_id"])["error"]
    assert "Unexpected error" in err and "secret" not in err


def test_index_update_failure_marks_failed_and_old_index_still_works(make_manager, monkeypatch):
    mgr = make_manager()
    import index_store as ix
    monkeypatch.setattr(ix, "_write_version_files", lambda d, s: (_ for _ in ()).throw(OSError("disk full")))
    rec = mgr.submit_upload(login(mgr), "a.pdf", ZEPHYR)
    mgr.wait_until_idle()
    assert mgr.registry.get(rec["doc_id"])["status"] == dm.FAILED
    assert mgr.store.snapshot.chunk_count == 13


# ------------------------------------------------------------------ persistence
def test_status_survives_restart(make_manager):
    mgr = make_manager()
    rec = mgr.submit_upload(login(mgr), "z.pdf", ZEPHYR)
    mgr.wait_until_idle()
    again = make_manager()
    saved = again.registry.get(rec["doc_id"])
    assert saved["status"] == dm.INDEXED and saved["chunk_count"] >= 1
    assert rec["doc_id"] in again.store.snapshot.doc_ids()


def test_documents_interrupted_by_restart_are_marked_failed(make_manager, tmp_path):
    mgr = make_manager()
    mgr.stop()
    path = tmp_path / "documents.json"
    data = json.loads(path.read_text())
    ghost = dict(data["documents"][0], doc_id="ghost", filename="ghost.pdf", status="processing", source="upload", created_at="2099-01-01T00:00:00+00:00")
    data["documents"].append(ghost)
    path.write_text(json.dumps(data))
    again = make_manager()
    rec = again.registry.get("ghost")
    assert rec["status"] == dm.FAILED and "interrupted" in rec["error"]


def test_corrupt_registry_falls_back_to_backup(make_manager, tmp_path):
    mgr = make_manager()
    mgr.submit_upload(login(mgr), "z.pdf", ZEPHYR)
    mgr.wait_until_idle()
    mgr.stop()
    (tmp_path / "documents.json").write_text("{not json")
    again = make_manager()
    assert len(again.registry.all()) >= 6


# ------------------------------------------------------------------ simultaneous uploads
def test_simultaneous_uploads_are_processed_one_at_a_time(make_manager, fake_s3):
    mgr = make_manager()
    token = login(mgr)
    lock, state = threading.Lock(), {"active": 0, "max": 0}
    original = mgr._process

    def tracked(doc_id):
        with lock:
            state["active"] += 1
            state["max"] = max(state["max"], state["active"])
        try:
            original(doc_id)
        finally:
            with lock:
                state["active"] -= 1

    mgr._process = tracked
    ids, errors = [], []

    def upload(i):
        try:
            ids.append(mgr.submit_upload(token, f"doc{i}.pdf", digital_pdf((f"Document number {i} describes feature alpha{i}.",)))["doc_id"])
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=upload, args=(i,)) for i in range(5)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    mgr.wait_until_idle()
    assert errors == [] and state["max"] == 1
    assert all(mgr.registry.get(i)["status"] == dm.INDEXED for i in ids) and len(set(ids)) == 5
    assert len(s3_keys(fake_s3)) == 5 and len(set(s3_keys(fake_s3))) == 5
    assert verify_snapshot(mgr.store.snapshot) == [] and set(ids) <= mgr.store.snapshot.doc_ids()
