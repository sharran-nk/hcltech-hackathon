import json

import pytest

import config
from chunking import Chunk
from evidence import EvidenceNotCalibrated, EvidenceThresholds, check_evidence, term_coverage
from index_store import IndexStore
from retrieval import Candidate, Retriever
from textutil import tokenize

T = EvidenceThresholds(bm25_min=3.5, faiss_min=0.46, min_coverage=0.4)


def cand(text, bm25, faiss, cid="X-c0"):
    return Candidate(Chunk("X", "Doc X", cid, 0, None, text), bm25, 1, faiss, 1)


# ------------------------------------------------------------------ rule mechanics
def test_term_coverage_uses_normalised_content_words():
    terms = set(tokenize("How long until the reset link expires?"))
    assert terms == set(tokenize("long until reset link expire")) and len(terms) == 5
    assert term_coverage(terms, "Password reset links expire after 15 minutes.") == pytest.approx(3 / 5)  # reset, link, expire; "long", "until" missing
    assert term_coverage(set(), "anything") == 0.0


def test_bm25_and_faiss_are_judged_on_separate_scales():
    text = "CSV exports are capped at 10,000 rows."
    q = "csv export rows"
    assert check_evidence(q, [cand(text, bm25=4.0, faiss=0.10)], T).sufficient     # BM25 alone passes
    assert check_evidence(q, [cand(text, bm25=0.5, faiss=0.60)], T).sufficient     # FAISS alone passes
    assert not check_evidence(q, [cand(text, bm25=3.4, faiss=0.45)], T).sufficient  # both just below their OWN thresholds
    # a BM25-sized number must never satisfy the cosine threshold or vice versa
    assert not check_evidence(q, [cand(text, bm25=0.46, faiss=-1.0)], T).sufficient


def test_high_scores_without_term_overlap_are_not_evidence():
    r = check_evidence("connect salesforce", [cand("Enterprise plans support SSO via SAML 2.0.", 9.0, 0.9)], T)
    assert not r.sufficient and "term-overlap" in r.reason


def test_passes_only_with_both_signals():
    text = "CSV exports are capped at 10,000 rows."
    assert not check_evidence("csv export rows", [cand(text, 0.1, 0.1)], T).sufficient   # overlap fine, scores low


def test_no_candidates_or_no_content_words():
    assert not check_evidence("csv export", [], T).sufficient
    assert not check_evidence("the of and", [cand("x", 9, 1)], T).sufficient


def test_supporting_passages_are_capped_and_best_covered_first():
    cands = [cand(f"password reset link {i}", 9, 0.9, f"X-c{i}") for i in range(6)]
    cands.append(cand("password reset link expire", 9, 0.9, "X-best"))
    r = check_evidence("password reset link expire", cands, EvidenceThresholds(3.5, 0.46, 0.4, max_passages=3))
    assert len(r.supporting) == 3 and r.supporting[0].chunk.chunk_id == "X-best"


# ------------------------------------------------------------------ configuration
def test_uncalibrated_thresholds_raise(monkeypatch):
    monkeypatch.setattr(config, "BM25_MIN_SCORE", None)
    with pytest.raises(EvidenceNotCalibrated):
        EvidenceThresholds.from_config()


def test_saved_thresholds_load_from_config():
    t = EvidenceThresholds.from_config()
    saved = json.loads(config.EVIDENCE_THRESHOLDS_PATH.read_text())
    assert (t.bm25_min, t.faiss_min, t.min_coverage) == (saved["bm25_min"], saved["faiss_min"], saved["min_coverage"])


# ------------------------------------------------------------------ against the seed corpus
@pytest.fixture(scope="module")
def retriever(tmp_path_factory, seed, embedder):
    store = IndexStore(tmp_path_factory.mktemp("ev") / "index")
    store.load()
    for d in seed:
        store.upsert_document(d, *seed[d])
    return Retriever(store, embedder)


EXAMPLES = json.loads(config.EXAMPLE_QUESTIONS_PATH.read_text(encoding="utf-8"))["example_questions"]
FACTS = {"Q-1": "Forgot password", "Q-2": "15 minutes", "Q-3": "10,000 rows", "Q-4": "60 requests per minute"}


@pytest.mark.parametrize("q", EXAMPLES, ids=lambda q: q["question_id"])
def test_five_example_questions(retriever, q):
    result = check_evidence(q["question"], retriever.search(q["question"]))  # real calibrated thresholds
    if q["expected_doc_id"] is None:
        assert not result.sufficient, "Salesforce must be refused by the evidence check"
    else:
        assert result.sufficient
        assert all(c.chunk.doc_id == q["expected_doc_id"] for c in result.supporting)
        assert any(FACTS[q["question_id"]] in c.chunk.text for c in result.supporting)


def test_salesforce_is_refused_for_lack_of_term_overlap_not_a_lucky_score(retriever):
    result = check_evidence("Can I connect Solstice to Salesforce?", retriever.search("Can I connect Solstice to Salesforce?"))
    assert max(p.coverage for p in result.passages) == 0.0


def test_labelled_eval_set_meets_calibrated_targets(retriever):
    qs = json.loads(config.EVAL_QUESTIONS_PATH.read_text(encoding="utf-8"))["questions"]
    missed, leaked = [], []
    for q in qs:
        r = check_evidence(q["question"], retriever.search(q["question"]))
        if q["expected_doc_id"]:
            if not any(q["expected_fact"].lower() in c.chunk.text.lower() for c in r.supporting):
                missed.append(q["id"])
        elif r.sufficient:
            leaked.append(q["id"])
    assert missed == []                                    # every answerable question reaches the LLM
    assert set(leaked) <= {"U-05", "U-07", "U-11"}         # known near-misses, left to the LLM prompt
    assert len(leaked) <= 3
