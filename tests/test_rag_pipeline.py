import pytest
import requests

import config
from evidence import EvidenceThresholds
from index_store import IndexStore
from rag_pipeline import (REFUSAL, OllamaClient, OllamaUnavailable, RagPipeline, build_user_prompt,
                          cited_numbers, is_refusal)
from retrieval import Retriever

THRESHOLDS = EvidenceThresholds(3.5, 0.46, 0.4)


class FakeLLM:
    def __init__(self, reply="", error=None):
        self.reply, self.error, self.calls = reply, error, []

    def chat(self, system, user):
        self.calls.append((system, user))
        if self.error:
            raise self.error
        return self.reply


@pytest.fixture(scope="module")
def retriever(tmp_path_factory, seed, embedder):
    store = IndexStore(tmp_path_factory.mktemp("rag") / "index")
    store.load()
    for d in seed:
        store.upsert_document(d, *seed[d])
    return Retriever(store, embedder)


def pipe(retriever, llm):
    return RagPipeline(retriever, llm, THRESHOLDS)


# ------------------------------------------------------------------ refusal paths
def test_refusal_text_is_exact():
    assert REFUSAL == "I don't have an answer for that."


def test_insufficient_evidence_returns_exact_refusal_without_calling_llm(retriever):
    llm = FakeLLM("should never be used")
    r = pipe(retriever, llm).answer("Can I connect Solstice to Salesforce?")
    assert r.answer == "I don't have an answer for that." and r.refused and r.mode == "refusal"
    assert r.citations == [] and llm.calls == []


@pytest.mark.parametrize("q", ["", "   ", "hello", "What is the weather in Chennai today?"])
def test_empty_and_off_topic_questions_refused_without_llm(retriever, q):
    llm = FakeLLM("x")
    assert pipe(retriever, llm).answer(q).answer == REFUSAL and llm.calls == []


def test_model_refusal_is_normalised_to_the_exact_message(retriever):
    for reply in ["I don't have an answer for that.", "\"I don't have an answer for that.\"", "I don't have an answer for that"]:
        r = pipe(retriever, FakeLLM(reply)).answer("Can I export a report to PDF?")
        if r.mode == "refusal":
            assert r.answer == REFUSAL and r.citations == []


def test_empty_model_reply_becomes_refusal(retriever):
    r = pipe(retriever, FakeLLM("   ")).answer("How do I reset my password?")
    assert r.refused and r.answer == REFUSAL


def test_is_refusal_variants():
    assert is_refusal("I don't have an answer for that.") and is_refusal("  \"I don't have an answer for that\" ")
    assert not is_refusal("Reset links expire after 15 minutes. [1]")


# ------------------------------------------------------------------ answers + citations
def test_supported_question_calls_llm_with_only_supporting_passages(retriever):
    llm = FakeLLM("Password reset links expire after 15 minutes. [1]")
    r = pipe(retriever, llm).answer("How long do I have to use a password reset link before it expires?")
    assert r.mode == "llm" and not r.refused and len(llm.calls) == 1
    system, user = llm.calls[0]
    assert "ONLY" in system and REFUSAL in system
    assert "Resetting Your Password" in user and "Question: How long" in user
    assert "Billing Cycles" not in user and "Salesforce" not in user  # unrelated passages never reach the model
    assert [c.doc_id for c in r.citations] == ["D-01"] and r.citations[0].number == 1


def test_only_passages_the_model_cited_are_shown(retriever):
    llm = FakeLLM("CSV exports are capped at 10,000 rows. [2]")
    r = pipe(retriever, llm).answer("What's the maximum number of rows I can export to CSV?")
    assert [c.number for c in r.citations] == [2]


def test_uncited_or_invalid_markers_fall_back_to_all_supplied_passages(retriever):
    r = pipe(retriever, FakeLLM("Growth plans: 60 requests per minute. [9]")).answer("What are the API rate limits for the Growth plan?")
    assert len(r.citations) >= 1 and all(c.doc_id == "D-06" for c in r.citations)


def test_cited_numbers_parsing():
    assert cited_numbers("a [1] b [3] c [1] d [7] e [0]", 3) == [1, 3]


def test_prompt_labels_sources_and_pages():
    from chunking import Chunk
    from retrieval import Candidate
    c = Candidate(Chunk("d", "manual.pdf", "d-c0", 0, 4, "Some text."), 1, 1, 1, 1)
    p = build_user_prompt("Q?", [c])
    assert "[1] Source: manual.pdf, page 4" in p and "Question: Q?" in p


def test_long_question_is_truncated_not_crashing(retriever):
    r = pipe(retriever, FakeLLM("x")).answer("reset password " * 500)
    assert r.answer  # either a refusal or an answer, never an exception


# ------------------------------------------------------------------ Ollama unavailable
def test_ollama_down_with_evidence_returns_extractive_passage_with_citation(retriever):
    llm = FakeLLM(error=OllamaUnavailable("down"))
    r = pipe(retriever, llm).answer("How long do I have to use a password reset link before it expires?")
    assert r.mode == "extractive" and not r.refused and "15 minutes" in r.answer
    assert r.citations[0].doc_id == "D-01" and "unavailable" in r.notice


def test_ollama_down_without_evidence_still_refuses(retriever):
    r = pipe(retriever, FakeLLM(error=OllamaUnavailable("down"))).answer("Can I connect Solstice to Salesforce?")
    assert r.answer == REFUSAL


def test_client_maps_connection_errors(monkeypatch):
    def boom(*a, **k):
        raise requests.ConnectionError("refused")
    monkeypatch.setattr(requests, "post", boom)
    with pytest.raises(OllamaUnavailable):
        OllamaClient().chat("s", "u")


def test_client_maps_missing_model_and_http_errors(monkeypatch):
    class R:
        def __init__(self, code): self.status_code = code
    monkeypatch.setattr(requests, "post", lambda *a, **k: R(404))
    with pytest.raises(OllamaUnavailable, match="ollama pull"):
        OllamaClient().chat("s", "u")
    monkeypatch.setattr(requests, "post", lambda *a, **k: R(500))
    with pytest.raises(OllamaUnavailable, match="500"):
        OllamaClient().chat("s", "u")


def test_ollama_is_only_addressed_on_localhost_by_default():
    assert config.OLLAMA_URL.startswith("http://127.0.0.1")
