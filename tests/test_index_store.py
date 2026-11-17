import threading

import numpy as np
import pytest

import index_store as ix
from chunking import Chunk
from index_store import IndexLoadError, IndexStore, IndexUpdateError, verify_snapshot
from retrieval import Retriever


def filled_store(tmp_path, seed, docs=("D-01", "D-02", "D-03", "D-04", "D-05", "D-06")):
    store = IndexStore(tmp_path / "index")
    store.load()
    for d in docs:
        store.upsert_document(d, *seed[d])
    return store


def versions(store):
    return sorted(p.name for p in store.versions_dir.iterdir())


def hits(store, embedder, q):
    return {c.chunk.doc_id for c in Retriever(store, embedder).search(q)}


# ------------------------------------------------------------------ persistence
def test_empty_store_is_usable(tmp_path):
    s = IndexStore(tmp_path / "i")
    assert s.load().chunk_count == 0 and verify_snapshot(s.snapshot) == []


def test_add_persist_and_reload(tmp_path, seed):
    store = filled_store(tmp_path, seed)
    assert store.snapshot.chunk_count == 13 and store.snapshot.doc_ids() == set(seed)
    fresh = IndexStore(tmp_path / "index")
    snap = fresh.load()
    assert [c.chunk_id for c in snap.chunks] == [c.chunk_id for c in store.snapshot.chunks]
    assert snap.faiss_index.ntotal == 13 and snap.bm25.corpus_size == 13
    assert verify_snapshot(snap) == []
    assert np.array_equal(snap.vectors, store.snapshot.vectors)


def test_metadata_persisted_with_chunks(tmp_path, seed):
    store = filled_store(tmp_path, seed, ("D-03",))
    c = IndexStore(tmp_path / "index")
    c.load()
    first = c.snapshot.chunks[0]
    assert (first.doc_id, first.filename, first.chunk_id) == ("D-03", "Exporting Reports to CSV", "D-03-c0")


def test_only_current_and_one_backup_kept(tmp_path, seed):
    store = filled_store(tmp_path, seed)  # 6 commits
    assert len(versions(store)) == 2
    assert (store.dir / "CURRENT").read_text() == store.snapshot.version


def test_duplicate_chunk_ids_rejected(tmp_path, seed):
    store = filled_store(tmp_path, seed, ("D-01",))
    clash = [Chunk("D-09", "x.pdf", "D-01-c0", 0, None, "other text")]  # same chunk_id, different doc
    with pytest.raises(IndexUpdateError):
        store.upsert_document("D-09", clash, seed["D-02"][1][:1])
    assert store.snapshot.doc_ids() == {"D-01"}


def test_empty_or_mismatched_input_rejected(tmp_path, seed):
    store = filled_store(tmp_path, seed, ("D-01",))
    with pytest.raises(IndexUpdateError):
        store.upsert_document("D-02", [], np.zeros((0, 384), dtype="float32"))
    with pytest.raises(IndexUpdateError):
        store.upsert_document("D-02", seed["D-02"][0], seed["D-02"][1][:0])  # vector count mismatch
    with pytest.raises(IndexUpdateError):
        store.upsert_document("D-99", seed["D-02"][0], seed["D-02"][1])      # wrong doc_id on chunks
    assert store.snapshot.doc_ids() == {"D-01"}


# ------------------------------------------------------------------ failed updates keep the old index
def test_failure_midwrite_keeps_old_index_and_cleans_up(tmp_path, seed, monkeypatch, embedder):
    store = filled_store(tmp_path, seed, ("D-01", "D-02"))
    before_version = store.snapshot.version
    before_versions = versions(store)

    def half_written(directory, snap):
        directory.mkdir(parents=True)
        (directory / "chunks.json").write_text("{partial")
        raise OSError("disk full")

    monkeypatch.setattr(ix, "_write_version_files", half_written)
    with pytest.raises(IndexUpdateError):
        store.upsert_document("D-03", *seed["D-03"])
    monkeypatch.undo()

    assert store.snapshot.version == before_version and store.snapshot.doc_ids() == {"D-01", "D-02"}
    assert versions(store) == before_versions
    assert not list(store.dir.glob("building-*"))
    assert "D-03" not in hits(store, embedder, "maximum rows CSV export")
    fresh = IndexStore(tmp_path / "index")
    assert fresh.load().doc_ids() == {"D-01", "D-02"}


def test_failure_switching_current_keeps_old_index(tmp_path, seed, monkeypatch):
    store = filled_store(tmp_path, seed, ("D-01",))
    monkeypatch.setattr(store, "_write_current", lambda v: (_ for _ in ()).throw(OSError("nope")))
    with pytest.raises(IndexUpdateError):
        store.upsert_document("D-02", *seed["D-02"])
    monkeypatch.undo()
    assert store.snapshot.doc_ids() == {"D-01"} and len(versions(store)) == 1


# ------------------------------------------------------------------ corruption + backup
def test_corrupt_current_falls_back_to_previous(tmp_path, seed):
    store = filled_store(tmp_path, seed, ("D-01", "D-02"))
    current, = [store.snapshot.version]
    (store.versions_dir / current / "vectors.npy").write_bytes(b"garbage")
    fresh = IndexStore(tmp_path / "index")
    snap = fresh.load()
    assert snap.doc_ids() == {"D-01"} and snap.version != current  # the backup (before D-02 was added)
    assert (fresh.dir / "CURRENT").read_text() == snap.version     # pointer repaired


def test_damaged_faiss_file_is_rebuilt_from_vectors(tmp_path, seed):
    store = filled_store(tmp_path, seed)
    (store.versions_dir / store.snapshot.version / "faiss.index").write_bytes(b"\x00\x01broken")
    snap = IndexStore(tmp_path / "index").load()
    assert snap.faiss_index.ntotal == 13 and verify_snapshot(snap) == []


def test_all_versions_corrupt_raises_clear_error(tmp_path, seed):
    store = filled_store(tmp_path, seed, ("D-01", "D-02"))
    for v in store.versions_dir.iterdir():
        (v / "chunks.json").write_text("[]")  # checksum no longer matches
    with pytest.raises(IndexLoadError, match="Rebuild required"):
        IndexStore(tmp_path / "index").load()


def test_leftover_building_dir_removed_on_load(tmp_path, seed):
    store = filled_store(tmp_path, seed, ("D-01",))
    (store.dir / "building-crashed").mkdir()
    IndexStore(tmp_path / "index").load()
    assert not list(store.dir.glob("building-*"))


def test_rebuild_from_chunks_recovers_index(tmp_path, seed, embedder):
    store = filled_store(tmp_path, seed)
    chunks = list(store.snapshot.chunks)
    store.rebuild_from_chunks(chunks, embedder)
    assert store.snapshot.chunk_count == 13 and verify_snapshot(store.snapshot) == []
    assert "D-03" in hits(store, embedder, "maximum number of rows CSV export")


# ------------------------------------------------------------------ deletion
def test_delete_removes_every_chunk_from_both_retrievers(tmp_path, seed, embedder):
    store = filled_store(tmp_path, seed)
    snap_before = store.snapshot
    assert any(c.doc_id == "D-03" for c in snap_before.chunks)
    assert store.remove_document("D-03") == 3

    snap = store.snapshot
    assert "D-03" not in snap.doc_ids()
    assert snap.faiss_index.ntotal == snap.chunk_count == 10      # FAISS lost exactly D-03's vectors
    assert snap.bm25.corpus_size == 10                            # BM25 lost exactly D-03's chunks
    assert verify_snapshot(snap) == []
    # neither retriever can return it, whatever the query
    for q in ["CSV export 10,000 rows", "export reports download icon", "Exports retained 7 days"]:
        assert "D-03" not in hits(store, embedder, q)
        scores = snap.bm25.get_scores(ix.tokenize(q))
        assert all(snap.chunks[i].doc_id != "D-03" for i in np.argsort(-scores)[:5])
        _, ids = snap.faiss_index.search(embedder.embed_query(q).reshape(1, -1), 10)
        assert all(snap.chunks[i].doc_id != "D-03" for i in ids[0] if i >= 0)


def test_delete_persists_across_restart(tmp_path, seed, embedder):
    store = filled_store(tmp_path, seed)
    store.remove_document("D-03")
    fresh = IndexStore(tmp_path / "index")
    snap = fresh.load()
    assert "D-03" not in snap.doc_ids() and snap.chunk_count == 10
    assert "D-03" not in hits(fresh, embedder, "CSV export 10,000 rows")


def test_delete_missing_doc_is_noop(tmp_path, seed):
    store = filled_store(tmp_path, seed, ("D-01",))
    version = store.snapshot.version
    assert store.remove_document("nope") == 0 and store.snapshot.version == version


def test_delete_last_document_leaves_valid_empty_index(tmp_path, seed):
    store = filled_store(tmp_path, seed, ("D-02",))
    store.remove_document("D-02")
    snap = IndexStore(tmp_path / "index").load()
    assert snap.chunk_count == 0 and verify_snapshot(snap) == []


def test_assert_document_absent_catches_leftovers(tmp_path, seed):
    store = filled_store(tmp_path, seed, ("D-01",))
    with pytest.raises(IndexUpdateError):
        ix.assert_document_absent(store.snapshot, "D-01")


# ------------------------------------------------------------------ replacement
def test_replace_swaps_old_for_new_in_one_step(tmp_path, seed, embedder):
    store = filled_store(tmp_path, seed, ("D-01", "D-03"))
    text = "Password reset links now expire after 30 minutes. Request a new link if it expired."
    new = [Chunk("D-01", "Resetting Your Password", "D-01-c0", 0, None, text)]
    store.upsert_document("D-01", new, embedder.embed_texts([text]))
    snap = store.snapshot
    assert [c.text for c in snap.chunks if c.doc_id == "D-01"] == [text]
    assert not any("15 minutes" in c.text for c in snap.chunks) and verify_snapshot(snap) == []
    assert snap.chunk_count == 1 + 3  # new D-01 chunk + untouched D-03 chunks


def test_failed_replace_keeps_old_content(tmp_path, seed, embedder, monkeypatch):
    store = filled_store(tmp_path, seed, ("D-01",))
    monkeypatch.setattr(ix, "_write_version_files", lambda d, s: (_ for _ in ()).throw(OSError("boom")))
    text = "Totally different content that must not replace the old one."
    with pytest.raises(IndexUpdateError):
        store.upsert_document("D-01", [Chunk("D-01", "x", "D-01-c0", 0, None, text)], embedder.embed_texts([text]))
    monkeypatch.undo()
    assert any("15 minutes" in c.text for c in store.snapshot.chunks)


# ------------------------------------------------------------------ concurrency
def test_simultaneous_updates_do_not_corrupt_state(tmp_path, seed):
    store = IndexStore(tmp_path / "index")
    store.load()
    errors = []

    def add(doc):
        try:
            store.upsert_document(doc, *seed[doc])
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=add, args=(d,)) for d in seed]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert errors == []
    assert store.snapshot.doc_ids() == set(seed) and store.snapshot.chunk_count == 13
    assert verify_snapshot(store.snapshot) == [] and len(versions(store)) == 2
    assert IndexStore(tmp_path / "index").load().chunk_count == 13


def test_readers_never_fail_during_writes(tmp_path, seed, embedder):
    store = filled_store(tmp_path, seed, ("D-01",))
    retriever, stop, errors = Retriever(store, embedder), threading.Event(), []

    def read():
        while not stop.is_set():
            try:
                retriever.search("password reset link")
            except Exception as exc:  # pragma: no cover
                errors.append(exc)

    t = threading.Thread(target=read)
    t.start()
    for d in ("D-02", "D-03", "D-04", "D-05", "D-06"):
        store.upsert_document(d, *seed[d])
    store.remove_document("D-02")
    stop.set()
    t.join()
    assert errors == []
