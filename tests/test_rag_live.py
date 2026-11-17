"""Live tests against the real Ollama + qwen2.5:7b. Skipped automatically if Ollama is not available."""
import json

import pytest
import requests

import config
from index_store import IndexStore
from rag_pipeline import REFUSAL, RagPipeline
from retrieval import Retriever


def ollama_ready() -> bool:
    try:
        tags = requests.get(f"{config.OLLAMA_URL}/api/tags", timeout=2).json()
        return any(m["name"] == config.OLLAMA_MODEL for m in tags.get("models", []))
    except Exception:
        return False


pytestmark = [pytest.mark.llm, pytest.mark.skipif(not ollama_ready(), reason="Ollama / qwen2.5:7b not available")]

EXAMPLES = json.loads(config.EXAMPLE_QUESTIONS_PATH.read_text(encoding="utf-8"))["example_questions"]
FACT = {"Q-1": "forgot password", "Q-2": "15 minutes", "Q-3": "10,000", "Q-4": "60"}


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory, seed, embedder):
    store = IndexStore(tmp_path_factory.mktemp("live") / "index")
    store.load()
    for d in seed:
        store.upsert_document(d, *seed[d])
    return RagPipeline(Retriever(store, embedder))


@pytest.mark.parametrize("q", EXAMPLES, ids=lambda q: q["question_id"])
def test_five_example_questions_end_to_end(pipeline, q):
    r = pipeline.answer(q["question"])
    if q["expected_doc_id"] is None:
        assert r.answer == REFUSAL and r.refused and r.citations == []
    else:
        assert r.mode == "llm" and not r.refused
        assert any(c.doc_id == q["expected_doc_id"] for c in r.citations)
        assert FACT[q["question_id"]] in r.answer.lower()
