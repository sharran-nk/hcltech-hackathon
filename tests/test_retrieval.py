import json

import pytest

import config
from index_store import IndexStore
from retrieval import Retriever
from textutil import tokenize


@pytest.fixture(scope="module")
def retriever(tmp_path_factory, seed, embedder):
    store = IndexStore(tmp_path_factory.mktemp("idx") / "index")
    store.load()
    for doc_id in seed:
        store.upsert_document(doc_id, *seed[doc_id])
    return Retriever(store, embedder)


def questions():
    return json.loads(config.EXAMPLE_QUESTIONS_PATH.read_text(encoding="utf-8"))["example_questions"]


# ------------------------------------------------------------------ tokenizer
def test_tokenizer_lowercases_and_drops_stopwords():
    assert tokenize("How do I reset my password?") == ["reset", "password"]
    assert tokenize("What's the maximum number of rows?") == ["maximum", "number", "row"]  # 1-letter noise dropped, plural folded


def test_tokenizer_plural_folding_and_brand_stopword():
    assert tokenize("reset links") == tokenize("reset link") == ["reset", "link"]
    assert tokenize("Solstice API limits") == ["api", "limit"]
    assert tokenize("access class status") == ["access", "class", "statu"]  # 'ss' kept; crude but consistent both sides
    assert tokenize("7 days and 10,000 rows") == ["7", "day", "10", "000", "row"]  # digits survive


def test_tokenizer_folds_word_forms_so_they_match():
    groups = [("invite", "inviting", "invited", "invites"), ("export", "exported", "exports", "exporting"),
              ("expire", "expires", "expired"), ("arrive", "arrived"), ("role", "roles"), ("fail", "fails", "failed")]
    for g in groups:
        assert len({tuple(tokenize(w)) for w in g}) == 1, g
    assert tokenize("password") == ["password"] and tokenize("sso") == ["sso"]


# ------------------------------------------------------------------ the five example questions
@pytest.mark.parametrize("q", [q for q in questions() if q["expected_doc_id"]], ids=lambda q: q["question_id"])
def test_expected_document_is_among_candidates(retriever, q):
    cands = retriever.search(q["question"])
    assert q["expected_doc_id"] in {c.chunk.doc_id for c in cands}, [c.chunk.chunk_id for c in cands]


def test_q2_answer_chunk_is_retrieved(retriever):
    cands = retriever.search("How long do I have to use a password reset link before it expires?")
    assert any("15 minutes" in c.chunk.text for c in cands)


def test_q3_and_q4_fact_chunks_are_retrieved(retriever):
    assert any("10,000 rows" in c.chunk.text for c in retriever.search("What's the maximum number of rows I can export to CSV?"))
    assert any("60 requests per minute" in c.chunk.text for c in retriever.search("What are the API rate limits for the Growth plan?"))


def test_q5_salesforce_still_returns_candidates_so_evidence_check_is_needed(retriever):
    """Retrieval alone cannot refuse: it returns related-but-unsupporting passages."""
    cands = retriever.search("Can I connect Solstice to Salesforce?")
    assert cands and not any("salesforce" in c.chunk.text.lower() for c in cands)


# ------------------------------------------------------------------ keyword vs semantic
def test_keyword_only_query_found_by_bm25(retriever):
    cands = retriever.search("429")
    top = next(c for c in cands if c.bm25_rank == 1)
    assert top.chunk.doc_id == "D-06" and top.bm25_score > 0


def test_semantic_paraphrase_without_shared_keywords_found_by_faiss(retriever):
    q = "How many days can I still download a file I generated?"  # doc says "retained for 7 days"
    cands = retriever.search(q)
    assert "D-03" in {c.chunk.doc_id for c in cands if c.faiss_rank is not None}


def test_no_keyword_overlap_gives_no_bm25_candidates_but_faiss_still_answers(retriever):
    cands = retriever.search("zzzz qqqq")
    assert all(c.bm25_rank is None for c in cands) and all(c.faiss_rank is not None for c in cands)


# ------------------------------------------------------------------ merge / dedupe / separate scores
def test_candidates_are_deduplicated_by_chunk_id(retriever):
    cands = retriever.search("password reset link expires")
    ids = [c.chunk.chunk_id for c in cands]
    assert len(ids) == len(set(ids))
    assert len(ids) <= config.BM25_TOP_K + config.FAISS_TOP_K


def test_chunk_found_by_both_has_both_ranks_and_sorts_first(retriever):
    cands = retriever.search("password reset link expires")
    both = [c for c in cands if len(c.found_by) == 2]
    assert both and cands[0] in both


def test_scores_are_kept_separately_on_their_own_scales(retriever):
    for c in retriever.search("CSV export rows limit"):
        assert -1.001 <= c.faiss_score <= 1.001  # cosine
        assert c.bm25_score >= 0                 # BM25: unbounded, not comparable to cosine


def test_empty_question_and_empty_index(tmp_path, embedder):
    empty = IndexStore(tmp_path / "e")
    empty.load()
    assert Retriever(empty, embedder).search("anything") == []
    assert Retriever(empty, embedder).search("   ") == []


def test_top_k_is_respected(retriever):
    assert len(retriever.search("export", bm25_k=1, faiss_k=1)) <= 2
