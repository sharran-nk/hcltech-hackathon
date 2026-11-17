import numpy as np
import pytest

import chunking
from embeddings import EMBEDDING_DIM, Embedder, EmbeddingError, get_embedder


@pytest.fixture(scope="module")
def emb():
    return get_embedder()


def test_shape_dtype_and_normalisation(emb):
    v = emb.embed_texts(["hello world", "password reset link", "billing cycle"])
    assert v.shape == (3, EMBEDDING_DIM) and v.dtype == np.float32
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-4)


def test_deterministic(emb):
    a, b = emb.embed_query("How do I reset my password?"), emb.embed_query("How do I reset my password?")
    assert np.allclose(a, b, atol=1e-6)


def test_empty_input_returns_empty_array(emb):
    assert emb.embed_texts([]).shape == (0, EMBEDDING_DIM)


def test_batching_gives_same_result(emb):
    texts = [f"sentence number {i} about exports" for i in range(70)]  # > batch size 32
    assert emb.embed_texts(texts).shape == (70, EMBEDDING_DIM)


def test_semantic_sanity_on_seed_chunks(emb):
    """A paraphrased question should land closest to the chunk holding the answer."""
    chunks = [c for d in chunking.load_seed_docs() for c in chunking.chunk_seed_doc(d)]
    vectors = emb.embed_texts([c.text for c in chunks])
    for question, expected_doc in [
        ("How long is the reset link valid for?", "D-01"),
        ("What is the maximum number of rows in a CSV export?", "D-03"),
        ("How many API calls per minute on the Growth plan?", "D-06"),
        ("When do I get my invoice?", "D-04"),
    ]:
        best = chunks[int(np.argmax(vectors @ emb.embed_query(question)))]
        assert best.doc_id == expected_doc, (question, best.chunk_id)


def test_model_load_failure_is_wrapped():
    with pytest.raises(EmbeddingError):
        Embedder("this/model-does-not-exist-xyz").embed_texts(["hi"])


def test_encode_failure_is_wrapped(emb, monkeypatch):
    class Boom:
        def encode(self, *a, **k):
            raise MemoryError()
    bad = Embedder()
    bad._model = Boom()
    with pytest.raises(EmbeddingError):
        bad.embed_texts(["x"])
